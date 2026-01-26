/*
 * gemma3_tokenizer.c - SentencePiece BPE tokenizer for Gemma 3
 *
 * Parses SentencePiece protobuf format and implements BPE encoding
 * with byte fallback for the 262K vocabulary.
 */

#include "gemma3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ============================================================================
 * Constants
 * ========================================================================== */

#define MAX_TOKEN_LEN 512
#define HASH_SIZE 524288  /* Must be power of 2, > vocab_size */

/* Special tokens for Gemma 3 */
#define GEMMA3_TOKEN_PAD 0
#define GEMMA3_TOKEN_EOS 1
#define GEMMA3_TOKEN_BOS 2
#define GEMMA3_TOKEN_UNK 3

/* Chat template tokens */
#define GEMMA3_TOKEN_START_TURN 106  /* <start_of_turn> */
#define GEMMA3_TOKEN_END_TURN 107    /* <end_of_turn> */

/* ============================================================================
 * Data Structures
 * ========================================================================== */

typedef struct {
    char *piece;       /* The string representation */
    float score;       /* BPE merge score */
    int type;          /* Token type: 1=normal, 2=unknown, 3=control, 4=user_defined, 6=byte */
} vocab_entry;

typedef struct {
    int left;
    int right;
    int merged;
} bpe_merge;

struct gemma3_tokenizer {
    vocab_entry *vocab;
    int vocab_size;

    /* Hash table for piece -> id lookup */
    int *piece_to_id;
    char **id_to_piece;

    /* Byte fallback tokens (256 entries for <0x00> - <0xFF>) */
    int byte_tokens[256];

    /* Special token IDs */
    int bos_id;
    int eos_id;
    int pad_id;
    int unk_id;

    /* Chat tokens */
    int start_turn_id;
    int end_turn_id;
};

/* ============================================================================
 * Protobuf Parsing (minimal, just for SentencePiece model)
 * ========================================================================== */

/* Protobuf wire types */
#define PB_VARINT 0
#define PB_64BIT 1
#define PB_LENDELIM 2
#define PB_32BIT 5

static uint64_t pb_read_varint(const uint8_t **ptr, const uint8_t *end) {
    uint64_t result = 0;
    int shift = 0;
    while (*ptr < end) {
        uint8_t byte = *(*ptr)++;
        result |= (uint64_t)(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) break;
        shift += 7;
        if (shift > 63) break;  /* Overflow protection */
    }
    return result;
}

static float pb_read_float(const uint8_t **ptr, const uint8_t *end) {
    if (*ptr + 4 > end) return 0.0f;
    float val;
    memcpy(&val, *ptr, 4);
    *ptr += 4;
    return val;
}

/* ============================================================================
 * Hash Table for Piece Lookup
 * ========================================================================== */

static uint32_t hash_string(const char *str) {
    uint32_t hash = 5381;
    while (*str) {
        hash = ((hash << 5) + hash) + (uint8_t)*str++;
    }
    return hash;
}

static int ht_lookup(gemma3_tokenizer *tok, const char *piece) {
    uint32_t idx = hash_string(piece) & (HASH_SIZE - 1);
    uint32_t start = idx;

    while (tok->piece_to_id[idx] >= 0) {
        int id = tok->piece_to_id[idx];
        if (strcmp(tok->vocab[id].piece, piece) == 0) {
            return id;
        }
        idx = (idx + 1) & (HASH_SIZE - 1);
        if (idx == start) break;  /* Table full */
    }
    return -1;  /* Not found */
}

static void ht_insert(gemma3_tokenizer *tok, const char *piece, int id) {
    uint32_t idx = hash_string(piece) & (HASH_SIZE - 1);
    uint32_t start = idx;

    while (tok->piece_to_id[idx] >= 0) {
        idx = (idx + 1) & (HASH_SIZE - 1);
        if (idx == start) return;  /* Table full */
    }
    tok->piece_to_id[idx] = id;
}

/* ============================================================================
 * SentencePiece Model Loading
 * ========================================================================== */

gemma3_tokenizer *gemma3_tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    /* Read entire file */
    uint8_t *data = (uint8_t *)malloc(file_size);
    if (!data) {
        fclose(f);
        return NULL;
    }
    if (fread(data, 1, file_size, f) != (size_t)file_size) {
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    /* Allocate tokenizer */
    gemma3_tokenizer *tok = (gemma3_tokenizer *)calloc(1, sizeof(gemma3_tokenizer));
    if (!tok) {
        free(data);
        return NULL;
    }

    /* Pre-allocate vocab (we'll resize if needed) */
    int vocab_capacity = GEMMA3_VOCAB_SIZE + 1024;
    tok->vocab = (vocab_entry *)calloc(vocab_capacity, sizeof(vocab_entry));
    tok->piece_to_id = (int *)malloc(HASH_SIZE * sizeof(int));
    memset(tok->piece_to_id, -1, HASH_SIZE * sizeof(int));

    if (!tok->vocab || !tok->piece_to_id) {
        free(tok->vocab);
        free(tok->piece_to_id);
        free(tok);
        free(data);
        return NULL;
    }

    /* Initialize byte tokens to -1 */
    for (int i = 0; i < 256; i++) {
        tok->byte_tokens[i] = -1;
    }

    /* Parse protobuf - SentencePiece model format */
    const uint8_t *ptr = data;
    const uint8_t *end = data + file_size;
    int piece_idx = 0;

    while (ptr < end) {
        uint64_t tag = pb_read_varint(&ptr, end);
        int field = tag >> 3;
        int wire_type = tag & 7;

        if (field == 1 && wire_type == PB_LENDELIM) {
            /* SentencePiece (repeated) */
            uint64_t msg_len = pb_read_varint(&ptr, end);
            const uint8_t *msg_end = ptr + msg_len;

            char *piece = NULL;
            float score = 0.0f;
            int type = 1;  /* default: normal */

            while (ptr < msg_end) {
                uint64_t inner_tag = pb_read_varint(&ptr, msg_end);
                int inner_field = inner_tag >> 3;
                int inner_wire = inner_tag & 7;

                if (inner_field == 1 && inner_wire == PB_LENDELIM) {
                    /* piece (string) */
                    uint64_t str_len = pb_read_varint(&ptr, msg_end);
                    piece = (char *)malloc(str_len + 1);
                    if (piece) {
                        memcpy(piece, ptr, str_len);
                        piece[str_len] = '\0';
                    }
                    ptr += str_len;
                } else if (inner_field == 2 && inner_wire == PB_32BIT) {
                    /* score (float) */
                    score = pb_read_float(&ptr, msg_end);
                } else if (inner_field == 3 && inner_wire == PB_VARINT) {
                    /* type (enum) */
                    type = pb_read_varint(&ptr, msg_end);
                } else {
                    /* Skip unknown field */
                    if (inner_wire == PB_VARINT) {
                        pb_read_varint(&ptr, msg_end);
                    } else if (inner_wire == PB_LENDELIM) {
                        uint64_t len = pb_read_varint(&ptr, msg_end);
                        ptr += len;
                    } else if (inner_wire == PB_32BIT) {
                        ptr += 4;
                    } else if (inner_wire == PB_64BIT) {
                        ptr += 8;
                    }
                }
            }

            if (piece && piece_idx < vocab_capacity) {
                tok->vocab[piece_idx].piece = piece;
                tok->vocab[piece_idx].score = score;
                tok->vocab[piece_idx].type = type;

                /* Add to hash table */
                ht_insert(tok, piece, piece_idx);

                /* Check for byte token: <0xNN> */
                if (type == 6 || (strlen(piece) == 6 && piece[0] == '<' &&
                    piece[1] == '0' && piece[2] == 'x' && piece[5] == '>')) {
                    unsigned int byte_val;
                    if (sscanf(piece, "<0x%02X>", &byte_val) == 1 ||
                        sscanf(piece, "<0x%02x>", &byte_val) == 1) {
                        tok->byte_tokens[byte_val] = piece_idx;
                    }
                }

                piece_idx++;
            } else {
                free(piece);
            }
        } else {
            /* Skip other fields */
            if (wire_type == PB_VARINT) {
                pb_read_varint(&ptr, end);
            } else if (wire_type == PB_LENDELIM) {
                uint64_t len = pb_read_varint(&ptr, end);
                ptr += len;
            } else if (wire_type == PB_32BIT) {
                ptr += 4;
            } else if (wire_type == PB_64BIT) {
                ptr += 8;
            }
        }
    }

    tok->vocab_size = piece_idx;
    free(data);

    /* Find special tokens */
    tok->pad_id = ht_lookup(tok, "<pad>");
    tok->eos_id = ht_lookup(tok, "<eos>");
    tok->bos_id = ht_lookup(tok, "<bos>");
    tok->unk_id = ht_lookup(tok, "<unk>");

    /* Fallback for special tokens */
    if (tok->pad_id < 0) tok->pad_id = GEMMA3_TOKEN_PAD;
    if (tok->eos_id < 0) tok->eos_id = GEMMA3_TOKEN_EOS;
    if (tok->bos_id < 0) tok->bos_id = GEMMA3_TOKEN_BOS;
    if (tok->unk_id < 0) tok->unk_id = GEMMA3_TOKEN_UNK;

    /* Find chat tokens */
    tok->start_turn_id = ht_lookup(tok, "<start_of_turn>");
    tok->end_turn_id = ht_lookup(tok, "<end_of_turn>");

    if (tok->start_turn_id < 0) tok->start_turn_id = GEMMA3_TOKEN_START_TURN;
    if (tok->end_turn_id < 0) tok->end_turn_id = GEMMA3_TOKEN_END_TURN;

    return tok;
}

void gemma3_tokenizer_free(gemma3_tokenizer *tok) {
    if (!tok) return;

    for (int i = 0; i < tok->vocab_size; i++) {
        free(tok->vocab[i].piece);
    }
    free(tok->vocab);
    free(tok->piece_to_id);
    free(tok->id_to_piece);
    free(tok);
}

/* ============================================================================
 * BPE Encoding
 * ========================================================================== */

/* Structure for tracking symbols during BPE */
typedef struct {
    int id;       /* Token ID (or -1 if this is a raw char) */
    int start;    /* Start position in original string */
    int len;      /* Length in bytes */
    int prev;     /* Previous symbol index (-1 if none) */
    int next;     /* Next symbol index (-1 if none) */
} bpe_symbol;

/* Find token ID for a string, with optional byte fallback */
static int find_piece(gemma3_tokenizer *tok, const char *str, int len) {
    /* Create null-terminated copy */
    char *tmp = (char *)malloc(len + 1);
    if (!tmp) return tok->unk_id;
    memcpy(tmp, str, len);
    tmp[len] = '\0';

    int id = ht_lookup(tok, tmp);
    free(tmp);

    return id;
}

/* Encode text to tokens using BPE */
int gemma3_tokenize(gemma3_tokenizer *tok, const char *text,
                    int *tokens, int max_tokens, int add_bos, int add_eos) {
    if (!tok || !text || !tokens || max_tokens <= 0) {
        return GEMMA3_ERR_INVALID_ARG;
    }

    int text_len = strlen(text);
    if (text_len == 0) {
        int n = 0;
        if (add_bos && n < max_tokens) tokens[n++] = tok->bos_id;
        if (add_eos && n < max_tokens) tokens[n++] = tok->eos_id;
        return n;
    }

    /* Allocate symbol array (one per byte initially) */
    int max_symbols = text_len + 2;
    bpe_symbol *symbols = (bpe_symbol *)malloc(max_symbols * sizeof(bpe_symbol));
    if (!symbols) return GEMMA3_ERR_OUT_OF_MEMORY;

    /* Initialize all symbol IDs to -1 (invalid) */
    for (int i = 0; i < max_symbols; i++) {
        symbols[i].id = -1;
        symbols[i].prev = -1;
        symbols[i].next = -1;
    }

    /* Initialize: each UTF-8 character becomes a symbol */
    int n_symbols = 0;
    int pos = 0;

    /* Handle spaces: SentencePiece uses ▁ (U+2581) for word boundaries */
    while (pos < text_len) {
        /* Determine UTF-8 character length */
        int char_len = 1;
        uint8_t c = (uint8_t)text[pos];
        if ((c & 0x80) == 0) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }

        /* Ensure we don't read past end */
        if (pos + char_len > text_len) char_len = text_len - pos;

        /* Check if we need to prepend ▁ (word boundary) */
        int prepend_space = 0;
        if (pos == 0 || text[pos - 1] == ' ') {
            prepend_space = 1;
        }

        /* Skip space characters, they're represented by ▁ on next token */
        if (text[pos] == ' ') {
            pos++;
            continue;
        }

        /* Try to find this character (with optional ▁ prefix) */
        char piece[16];
        int piece_len = 0;

        if (prepend_space) {
            /* ▁ is 3 bytes: E2 96 81 */
            piece[0] = (char)0xE2;
            piece[1] = (char)0x96;
            piece[2] = (char)0x81;
            memcpy(piece + 3, text + pos, char_len);
            piece_len = 3 + char_len;
        } else {
            memcpy(piece, text + pos, char_len);
            piece_len = char_len;
        }
        piece[piece_len] = '\0';

        int id = find_piece(tok, piece, piece_len);

        if (id >= 0) {
            symbols[n_symbols].id = id;
            symbols[n_symbols].start = pos;
            symbols[n_symbols].len = char_len;
            symbols[n_symbols].prev = n_symbols > 0 ? n_symbols - 1 : -1;
            symbols[n_symbols].next = -1;
            if (n_symbols > 0) symbols[n_symbols - 1].next = n_symbols;
            n_symbols++;
        } else {
            /* Byte fallback: encode each byte as <0xNN> */
            for (int b = 0; b < char_len; b++) {
                uint8_t byte = (uint8_t)text[pos + b];
                int byte_id = tok->byte_tokens[byte];
                if (byte_id < 0) byte_id = tok->unk_id;

                symbols[n_symbols].id = byte_id;
                symbols[n_symbols].start = pos + b;
                symbols[n_symbols].len = 1;
                symbols[n_symbols].prev = n_symbols > 0 ? n_symbols - 1 : -1;
                symbols[n_symbols].next = -1;
                if (n_symbols > 0) symbols[n_symbols - 1].next = n_symbols;
                n_symbols++;
            }
        }

        pos += char_len;
    }

    /* BPE merge loop: repeatedly merge the best pair */
    while (1) {
        /* Find best merge */
        float best_score = -1e30f;
        int best_i = -1;
        int best_merged_id = -1;

        for (int i = 0; i < max_symbols; i++) {
            if (symbols[i].id < 0) continue;  /* Deleted */
            int j = symbols[i].next;
            if (j < 0) continue;
            if (symbols[j].id < 0) continue;

            /* Create merged piece */
            const char *piece_i = tok->vocab[symbols[i].id].piece;
            const char *piece_j = tok->vocab[symbols[j].id].piece;
            if (!piece_i || !piece_j) continue;

            int len_i = strlen(piece_i);
            int len_j = strlen(piece_j);
            if (len_i + len_j >= MAX_TOKEN_LEN) continue;

            char merged[MAX_TOKEN_LEN];
            memcpy(merged, piece_i, len_i);
            memcpy(merged + len_i, piece_j, len_j);
            merged[len_i + len_j] = '\0';

            /* Look up merged piece */
            int merged_id = ht_lookup(tok, merged);
            if (merged_id < 0) continue;

            /* Check score (higher is better for merge priority) */
            float score = tok->vocab[merged_id].score;
            if (score > best_score) {
                best_score = score;
                best_i = i;
                best_merged_id = merged_id;
            }
        }

        if (best_i < 0) break;  /* No more merges */

        /* Apply best merge */
        int j = symbols[best_i].next;
        symbols[best_i].id = best_merged_id;
        symbols[best_i].len += symbols[j].len;
        symbols[best_i].next = symbols[j].next;
        if (symbols[j].next >= 0) {
            symbols[symbols[j].next].prev = best_i;
        }
        symbols[j].id = -1;  /* Mark as deleted */
    }

    /* Collect output tokens */
    int n_tokens = 0;
    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = tok->bos_id;
    }

    for (int i = 0; i < max_symbols && n_tokens < max_tokens; i++) {
        if (symbols[i].id >= 0) {
            tokens[n_tokens++] = symbols[i].id;
        }
    }

    if (add_eos && n_tokens < max_tokens) {
        tokens[n_tokens++] = tok->eos_id;
    }

    free(symbols);
    return n_tokens;
}

/* ============================================================================
 * Decoding
 * ========================================================================== */

const char *gemma3_decode_token(gemma3_tokenizer *tok, int token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) {
        return NULL;
    }
    return tok->vocab[token_id].piece;
}

char *gemma3_detokenize(gemma3_tokenizer *tok, const int *tokens, int num_tokens) {
    if (!tok || !tokens || num_tokens <= 0) {
        return NULL;
    }

    /* Calculate total length needed */
    size_t total_len = 0;
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->vocab_size) {
            const char *piece = tok->vocab[tokens[i]].piece;
            if (piece) total_len += strlen(piece);
        }
    }

    /* Allocate output buffer */
    char *output = (char *)malloc(total_len + 1);
    if (!output) return NULL;

    /* Build output string */
    char *ptr = output;
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->vocab_size) {
            const char *piece = tok->vocab[tokens[i]].piece;
            if (piece) {
                /* Handle ▁ -> space conversion */
                const char *src = piece;
                while (*src) {
                    /* Check for ▁ (0xE2 0x96 0x81) */
                    if ((uint8_t)src[0] == 0xE2 &&
                        (uint8_t)src[1] == 0x96 &&
                        (uint8_t)src[2] == 0x81) {
                        *ptr++ = ' ';
                        src += 3;
                    }
                    /* Check for byte token <0xNN> */
                    else if (src[0] == '<' && src[1] == '0' && src[2] == 'x' &&
                             src[5] == '>' && src[6] == '\0') {
                        unsigned int byte_val;
                        if (sscanf(src, "<0x%02X>", &byte_val) == 1 ||
                            sscanf(src, "<0x%02x>", &byte_val) == 1) {
                            *ptr++ = (char)byte_val;
                        }
                        src += 6;
                    } else {
                        *ptr++ = *src++;
                    }
                }
            }
        }
    }
    *ptr = '\0';

    /* Trim leading space if present */
    if (output[0] == ' ') {
        memmove(output, output + 1, strlen(output));
    }

    return output;
}

/* ============================================================================
 * Special Token Accessors
 * ========================================================================== */

int gemma3_bos_token(gemma3_tokenizer *tok) {
    return tok ? tok->bos_id : GEMMA3_TOKEN_BOS;
}

int gemma3_eos_token(gemma3_tokenizer *tok) {
    return tok ? tok->eos_id : GEMMA3_TOKEN_EOS;
}

int gemma3_pad_token(gemma3_tokenizer *tok) {
    return tok ? tok->pad_id : GEMMA3_TOKEN_PAD;
}

int gemma3_end_turn_token(gemma3_tokenizer *tok) {
    return tok ? tok->end_turn_id : GEMMA3_TOKEN_END_TURN;
}

int gemma3_start_turn_token(gemma3_tokenizer *tok) {
    return tok ? tok->start_turn_id : GEMMA3_TOKEN_START_TURN;
}

/* ============================================================================
 * Chat Template Formatting
 * ========================================================================== */

char *gemma3_format_chat(gemma3_tokenizer *tok, const gemma3_message *messages,
                         int num_msgs) {
    if (!tok || !messages || num_msgs <= 0) return NULL;

    /* Estimate buffer size */
    size_t buf_size = 1024;
    for (int i = 0; i < num_msgs; i++) {
        buf_size += strlen(messages[i].content) + 64;
    }

    char *buf = (char *)malloc(buf_size);
    if (!buf) return NULL;

    char *ptr = buf;
    *ptr = '\0';

    /* Gemma 3 chat format:
     * <bos><start_of_turn>user
     * {user_message}<end_of_turn>
     * <start_of_turn>model
     * {model_message}<end_of_turn>
     * ...
     */

    /* Add BOS at start */
    ptr += sprintf(ptr, "<bos>");

    for (int i = 0; i < num_msgs; i++) {
        const char *role_str;
        switch (messages[i].role) {
            case GEMMA3_ROLE_USER:
                role_str = "user";
                break;
            case GEMMA3_ROLE_MODEL:
                role_str = "model";
                break;
            case GEMMA3_ROLE_SYSTEM:
                role_str = "user";  /* System messages go as user with prefix */
                break;
            default:
                role_str = "user";
                break;
        }

        ptr += sprintf(ptr, "<start_of_turn>%s\n", role_str);

        if (messages[i].role == GEMMA3_ROLE_SYSTEM) {
            ptr += sprintf(ptr, "System: %s", messages[i].content);
        } else {
            ptr += sprintf(ptr, "%s", messages[i].content);
        }

        ptr += sprintf(ptr, "<end_of_turn>\n");
    }

    /* Add model turn start for generation */
    ptr += sprintf(ptr, "<start_of_turn>model\n");

    return buf;
}
