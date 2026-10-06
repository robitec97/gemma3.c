/*
 * gemma3_tokenizer.c - SentencePiece BPE tokenizer for Gemma 3
 *
 * Loads the SentencePiece protobuf (tokenizer.model) and reproduces the
 * Hugging Face `tokenizers` pipeline shipped with Gemma 3 (tokenizer.json):
 *
 *   1. Added tokens - every control / unknown / user-defined piece, e.g.
 *      <bos>, <start_of_turn>, <unused0>, "\n\n", "\t", "<b>" - are matched in
 *      the raw text first, leftmost-longest.
 *   2. In the remaining text every ' ' becomes U+2581 ('▁'). There is no
 *      dummy-prefix space.
 *   3. Each segment is split into code points and merged with BPE. Merge
 *      priority follows the HF converter's merge ranks: higher merged-piece
 *      score first, then longer left piece, then longer right piece, then the
 *      lower merged-piece id, then the leftmost position.
 *   4. Code points that are not in the vocabulary fall back to <0xNN> tokens.
 *
 * Encoding runs in O(n log n) using a linked list of symbols and a binary heap
 * of candidate merges with lazy invalidation.
 */

#include "gemma3_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ============================================================================
 * Constants
 * ========================================================================== */

/* Special tokens for Gemma 3 (fallbacks if lookup by name fails) */
#define GEMMA3_TOKEN_PAD 0
#define GEMMA3_TOKEN_EOS 1
#define GEMMA3_TOKEN_BOS 2
#define GEMMA3_TOKEN_UNK 3

/* Chat template tokens */
#define GEMMA3_TOKEN_START_TURN 105  /* <start_of_turn> */
#define GEMMA3_TOKEN_END_TURN 106    /* <end_of_turn> */

/* SentencePiece piece types */
#define SP_TYPE_NORMAL       1
#define SP_TYPE_UNKNOWN      2
#define SP_TYPE_CONTROL      3
#define SP_TYPE_USER_DEFINED 4
#define SP_TYPE_UNUSED       5
#define SP_TYPE_BYTE         6

/* Added tokens longer than this many bytes are not matched in raw text
 * (the longest in Gemma 3 is 93 bytes: 31 x '▁'). */
#define MAX_ADDED_LEN 256

/* User-defined pieces that HF marks as special (skipped when decoding) */
static const char *const k_special_user_pieces[] = {
    "<start_of_turn>", "<end_of_turn>", "<start_of_image>", "<end_of_image>",
    "<image_soft_token>",
};

/* ============================================================================
 * Data Structures
 * ========================================================================== */

typedef struct {
    uint32_t offset;   /* Offset of the NUL-terminated piece in piece_data */
    uint16_t len;      /* Length in bytes */
    uint16_t cplen;    /* Length in code points */
    float score;       /* SentencePiece score */
    uint8_t type;      /* SentencePiece piece type (SP_TYPE_*) */
    uint8_t added;     /* Matched verbatim in raw text before BPE */
    uint8_t special;   /* Produces no text when decoding */
} vocab_entry;

struct gemma3_tokenizer {
    vocab_entry *vocab;
    char *piece_data;        /* All piece strings, each NUL-terminated */
    int vocab_size;

    /* Open-addressing hash table: piece bytes -> id (-1 = empty slot) */
    int *table;
    uint32_t table_mask;

    /* Byte fallback tokens (256 entries for <0x00> - <0xFF>) */
    int byte_tokens[256];

    /* For each first byte, bitmask of the lengths of added tokens starting
     * with that byte (bit L-1 set => some added token has length L). */
    uint64_t added_lens[256][MAX_ADDED_LEN / 64];
    uint8_t added_first[256];

    /* Pieces containing '▁' after their first byte, bucketed by the byte that
     * precedes that '▁' (CSR layout). A '▁' in normalized text is a guaranteed
     * token boundary unless one of these pieces could cover it, so long
     * segments can be split there into independent BPE chunks. */
    int *span_ids;
    int *span_offs;          /* Byte offset of the '▁' inside the piece */
    int span_start[257];
    int split_ok;            /* 0 if the vocab has too many such pieces */

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
 * Hashing
 * ========================================================================== */

static inline uint32_t hash_bytes(const char *s, int len) {
    /* FNV-1a */
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static inline const char *piece_str(const gemma3_tokenizer *tok, int id) {
    return tok->piece_data + tok->vocab[id].offset;
}

/* Look up a piece by (ptr, len). Returns its id or -1. */
static inline int lookup(const gemma3_tokenizer *tok, const char *s, int len) {
    uint32_t idx = hash_bytes(s, len) & tok->table_mask;
    for (;;) {
        int id = tok->table[idx];
        if (id < 0) return -1;
        if (tok->vocab[id].len == len && memcmp(piece_str(tok, id), s, len) == 0) {
            return id;
        }
        idx = (idx + 1) & tok->table_mask;
    }
}

static void table_insert(gemma3_tokenizer *tok, int id) {
    const vocab_entry *v = &tok->vocab[id];
    uint32_t idx = hash_bytes(piece_str(tok, id), v->len) & tok->table_mask;
    while (tok->table[idx] >= 0) {
        int other = tok->table[idx];
        /* Keep the first occurrence of duplicate pieces (SentencePiece semantics) */
        if (tok->vocab[other].len == v->len &&
            memcmp(piece_str(tok, other), piece_str(tok, id), v->len) == 0) {
            return;
        }
        idx = (idx + 1) & tok->table_mask;
    }
    tok->table[idx] = id;
}

/* ============================================================================
 * UTF-8 helpers
 * ========================================================================== */

/* Length of the UTF-8 sequence starting at s (1 for invalid / truncated input). */
static inline int utf8_len(const uint8_t *s, size_t avail) {
    uint8_t c = s[0];
    int n;
    if (c < 0x80) return 1;
    else if ((c & 0xE0) == 0xC0) n = 2;
    else if ((c & 0xF0) == 0xE0) n = 3;
    else if ((c & 0xF8) == 0xF0) n = 4;
    else return 1;
    if ((size_t)n > avail) return 1;
    for (int i = 1; i < n; i++) {
        if ((s[i] & 0xC0) != 0x80) return 1;
    }
    return n;
}

static int count_code_points(const char *s, int len) {
    int n = 0;
    for (int i = 0; i < len; i++) {
        if (((uint8_t)s[i] & 0xC0) != 0x80) n++;
    }
    return n;
}

/* ============================================================================
 * Protobuf Parsing (minimal, just for SentencePiece model)
 * ========================================================================== */

/* Protobuf wire types */
#define PB_VARINT 0
#define PB_64BIT 1
#define PB_LENDELIM 2
#define PB_32BIT 5

static int pb_read_varint(const uint8_t **ptr, const uint8_t *end, uint64_t *out) {
    uint64_t result = 0;
    int shift = 0;
    while (*ptr < end && shift < 64) {
        uint8_t byte = *(*ptr)++;
        result |= (uint64_t)(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            *out = result;
            return 1;
        }
        shift += 7;
    }
    return 0;
}

/* Skip a field of the given wire type. Returns 0 on malformed input. */
static int pb_skip(const uint8_t **ptr, const uint8_t *end, int wire_type) {
    uint64_t v;
    switch (wire_type) {
        case PB_VARINT:
            return pb_read_varint(ptr, end, &v);
        case PB_LENDELIM:
            if (!pb_read_varint(ptr, end, &v) || v > (uint64_t)(end - *ptr)) return 0;
            *ptr += v;
            return 1;
        case PB_32BIT:
            if (end - *ptr < 4) return 0;
            *ptr += 4;
            return 1;
        case PB_64BIT:
            if (end - *ptr < 8) return 0;
            *ptr += 8;
            return 1;
        default:
            return 0;
    }
}

typedef struct {
    const uint8_t *str;
    int len;
    float score;
    int type;
} raw_piece;

/* Parse one SentencePiece message (ModelProto.pieces). */
static int pb_parse_piece(const uint8_t *p, const uint8_t *end, raw_piece *out) {
    out->str = NULL;
    out->len = 0;
    out->score = 0.0f;
    out->type = SP_TYPE_NORMAL;
    while (p < end) {
        uint64_t tag;
        if (!pb_read_varint(&p, end, &tag)) return 0;
        int field = (int)(tag >> 3);
        int wire = (int)(tag & 7);
        if (field == 1 && wire == PB_LENDELIM) {
            uint64_t len;
            if (!pb_read_varint(&p, end, &len) || len > (uint64_t)(end - p) || len > 65535) {
                return 0;
            }
            out->str = p;
            out->len = (int)len;
            p += len;
        } else if (field == 2 && wire == PB_32BIT) {
            if (end - p < 4) return 0;
            memcpy(&out->score, p, 4);
            p += 4;
        } else if (field == 3 && wire == PB_VARINT) {
            uint64_t type;
            if (!pb_read_varint(&p, end, &type)) return 0;
            out->type = (int)type;
        } else if (!pb_skip(&p, end, wire)) {
            return 0;
        }
    }
    return out->str != NULL;
}

/* Walk ModelProto counting pieces; if tok is set, also fill tok->vocab and
 * tok->piece_data. Returns the number of pieces, or -1 on malformed input. */
static int pb_parse_model(const uint8_t *data, size_t size, gemma3_tokenizer *tok,
                          size_t *piece_bytes) {
    const uint8_t *ptr = data;
    const uint8_t *end = data + size;
    int count = 0;
    size_t bytes = 0;

    while (ptr < end) {
        uint64_t tag;
        if (!pb_read_varint(&ptr, end, &tag)) return -1;
        int field = (int)(tag >> 3);
        int wire = (int)(tag & 7);

        if (field == 1 && wire == PB_LENDELIM) {
            uint64_t len;
            if (!pb_read_varint(&ptr, end, &len) || len > (uint64_t)(end - ptr)) return -1;
            raw_piece rp;
            if (!pb_parse_piece(ptr, ptr + len, &rp)) return -1;
            if (tok) {
                vocab_entry *v = &tok->vocab[count];
                v->offset = (uint32_t)bytes;
                v->len = (uint16_t)rp.len;
                v->cplen = (uint16_t)count_code_points((const char *)rp.str, rp.len);
                v->score = rp.score;
                v->type = (uint8_t)rp.type;
                memcpy(tok->piece_data + bytes, rp.str, rp.len);
                tok->piece_data[bytes + rp.len] = '\0';
            }
            bytes += (size_t)rp.len + 1;
            count++;
            ptr += len;
        } else if (!pb_skip(&ptr, end, wire)) {
            return -1;
        }
    }
    if (piece_bytes) *piece_bytes = bytes;
    return count;
}

/* ============================================================================
 * SentencePiece Model Loading
 * ========================================================================== */

static uint8_t *read_file(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long size = ftell(f);
    if (size <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    uint8_t *data = (uint8_t *)malloc((size_t)size);
    if (data && fread(data, 1, (size_t)size, f) != (size_t)size) {
        free(data);
        data = NULL;
    }
    fclose(f);
    *size_out = (size_t)size;
    return data;
}

static int is_special_user_piece(const char *s) {
    for (size_t i = 0; i < sizeof(k_special_user_pieces) / sizeof(k_special_user_pieces[0]); i++) {
        if (strcmp(s, k_special_user_pieces[i]) == 0) return 1;
    }
    return 0;
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

/* Parse "<0xNN>" -> byte value, or -1 */
static int parse_byte_piece(const char *s, int len) {
    if (len != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') return -1;
    int hi = hex_digit(s[3]), lo = hex_digit(s[4]);
    if (hi < 0 || lo < 0) return -1;
    return hi * 16 + lo;
}

static inline int is_space_marker(const char *s) {
    return (uint8_t)s[0] == 0xE2 && (uint8_t)s[1] == 0x96 && (uint8_t)s[2] == 0x81;
}

#define MAX_SPAN_ENTRIES 4096

/* Index the pieces that contain '▁' after their first byte (see struct). */
static void build_split_index(gemma3_tokenizer *tok) {
    int counts[256] = {0};
    int total = 0;
    for (int id = 0; id < tok->vocab_size; id++) {
        const char *p = piece_str(tok, id);
        int len = tok->vocab[id].len;
        for (int j = 1; j + 3 <= len; j++) {
            if (is_space_marker(p + j)) {
                counts[(uint8_t)p[j - 1]]++;
                total++;
            }
        }
    }
    tok->split_ok = 0;
    if (total > MAX_SPAN_ENTRIES) return;

    tok->span_ids = (int *)malloc((size_t)(total + 1) * sizeof(int));
    tok->span_offs = (int *)malloc((size_t)(total + 1) * sizeof(int));
    if (!tok->span_ids || !tok->span_offs) return;

    int fill[256];
    tok->span_start[0] = 0;
    for (int b = 0; b < 256; b++) {
        tok->span_start[b + 1] = tok->span_start[b] + counts[b];
        fill[b] = tok->span_start[b];
    }
    for (int id = 0; id < tok->vocab_size; id++) {
        const char *p = piece_str(tok, id);
        int len = tok->vocab[id].len;
        for (int j = 1; j + 3 <= len; j++) {
            if (is_space_marker(p + j)) {
                int k = fill[(uint8_t)p[j - 1]]++;
                tok->span_ids[k] = id;
                tok->span_offs[k] = j;
            }
        }
    }
    tok->split_ok = 1;
}

gemma3_tokenizer *gemma3_tokenizer_load(const char *path) {
    size_t size = 0;
    uint8_t *data = read_file(path, &size);
    if (!data) return NULL;

    /* Pass 1: count pieces and string bytes */
    size_t piece_bytes = 0;
    int count = pb_parse_model(data, size, NULL, &piece_bytes);
    if (count <= 0) {
        free(data);
        return NULL;
    }

    gemma3_tokenizer *tok = (gemma3_tokenizer *)calloc(1, sizeof(gemma3_tokenizer));
    if (!tok) {
        free(data);
        return NULL;
    }

    /* Room for one extra piece: <image_soft_token> (id 262144 in Gemma 3) is an
     * added token in tokenizer.json but missing from tokenizer.model. */
    static const char extra_piece[] = "<image_soft_token>";
    int capacity = count + 1;
    tok->vocab = (vocab_entry *)calloc((size_t)capacity, sizeof(vocab_entry));
    tok->piece_data = (char *)malloc(piece_bytes + sizeof(extra_piece));

    uint32_t table_size = 1;
    while (table_size < (uint32_t)capacity * 2) table_size <<= 1;
    tok->table = (int *)malloc((size_t)table_size * sizeof(int));
    tok->table_mask = table_size - 1;

    if (!tok->vocab || !tok->piece_data || !tok->table) {
        free(data);
        gemma3_tokenizer_free(tok);
        return NULL;
    }
    memset(tok->table, -1, (size_t)table_size * sizeof(int));

    /* Pass 2: fill vocab */
    pb_parse_model(data, size, tok, NULL);
    free(data);
    tok->vocab_size = count;

    for (int i = 0; i < 256; i++) tok->byte_tokens[i] = -1;

    for (int id = 0; id < tok->vocab_size; id++) {
        table_insert(tok, id);
    }

    if (lookup(tok, extra_piece, (int)strlen(extra_piece)) < 0) {
        int id = tok->vocab_size++;
        vocab_entry *v = &tok->vocab[id];
        v->offset = (uint32_t)piece_bytes;
        v->len = (uint16_t)strlen(extra_piece);
        v->cplen = v->len;
        v->score = 0.0f;
        v->type = SP_TYPE_USER_DEFINED;
        memcpy(tok->piece_data + piece_bytes, extra_piece, sizeof(extra_piece));
        table_insert(tok, id);
    }

    /* Classify pieces */
    for (int id = 0; id < tok->vocab_size; id++) {
        vocab_entry *v = &tok->vocab[id];
        const char *s = piece_str(tok, id);
        switch (v->type) {
            case SP_TYPE_UNKNOWN:
            case SP_TYPE_CONTROL:
                v->added = 1;
                v->special = 1;
                break;
            case SP_TYPE_USER_DEFINED:
                v->added = 1;
                v->special = (uint8_t)is_special_user_piece(s);
                break;
            case SP_TYPE_BYTE: {
                int b = parse_byte_piece(s, v->len);
                if (b >= 0 && tok->byte_tokens[b] < 0) tok->byte_tokens[b] = id;
                break;
            }
            default:
                break;
        }
        if (v->added && v->len > 0 && v->len <= MAX_ADDED_LEN &&
            lookup(tok, s, v->len) == id) {
            uint8_t first = (uint8_t)s[0];
            int bit = v->len - 1;
            tok->added_lens[first][bit / 64] |= (uint64_t)1 << (bit % 64);
            tok->added_first[first] = 1;
        }
    }

    build_split_index(tok);

    /* Find special tokens */
    tok->pad_id = lookup(tok, "<pad>", 5);
    tok->eos_id = lookup(tok, "<eos>", 5);
    tok->bos_id = lookup(tok, "<bos>", 5);
    tok->unk_id = lookup(tok, "<unk>", 5);
    if (tok->pad_id < 0) tok->pad_id = GEMMA3_TOKEN_PAD;
    if (tok->eos_id < 0) tok->eos_id = GEMMA3_TOKEN_EOS;
    if (tok->bos_id < 0) tok->bos_id = GEMMA3_TOKEN_BOS;
    if (tok->unk_id < 0) tok->unk_id = GEMMA3_TOKEN_UNK;

    tok->start_turn_id = lookup(tok, "<start_of_turn>", 15);
    tok->end_turn_id = lookup(tok, "<end_of_turn>", 13);
    if (tok->start_turn_id < 0) tok->start_turn_id = GEMMA3_TOKEN_START_TURN;
    if (tok->end_turn_id < 0) tok->end_turn_id = GEMMA3_TOKEN_END_TURN;

    return tok;
}

void gemma3_tokenizer_free(gemma3_tokenizer *tok) {
    if (!tok) return;
    free(tok->vocab);
    free(tok->piece_data);
    free(tok->table);
    free(tok->span_ids);
    free(tok->span_offs);
    free(tok);
}

/* ============================================================================
 * BPE Encoding
 * ========================================================================== */

/* A symbol in the working linked list. Symbols only ever grow by absorbing
 * their right neighbour, so (start, len) identifies a symbol's content. */
typedef struct {
    int id;          /* Token id */
    int start;       /* Byte offset in the normalized segment */
    int len;         /* Byte length (0 = merged away) */
    int cplen;       /* Length in code points */
    int prev;        /* Previous symbol index (-1 if none) */
    int next;        /* Next symbol index (-1 if none) */
    int mergeable;   /* 0 for byte-fallback symbols */
} bpe_symbol;

/* A candidate merge of symbols[pos] and its right neighbour. */
typedef struct {
    float score;     /* Score of the merged piece */
    int lcp;         /* Code points in left piece */
    int rcp;         /* Code points in right piece */
    int id;          /* Merged piece id */
    int pos;         /* Left symbol index */
    int llen;        /* Left byte length when queued (staleness check) */
    int rlen;        /* Right byte length when queued */
} bpe_candidate;

typedef struct {
    char *norm;            /* Normalized segment bytes */
    size_t norm_cap;
    bpe_symbol *syms;
    size_t syms_cap;
    bpe_candidate *heap;
    size_t heap_len;
    size_t heap_cap;
    /* Output */
    int *tokens;
    int n_tokens;
    int max_tokens;
    int oom;
} encode_state;

/* Merge priority, mirroring HF merge ranks for Gemma 3 (see file header). */
static inline int cand_before(const bpe_candidate *a, const bpe_candidate *b) {
    if (a->score != b->score) return a->score > b->score;
    if (a->lcp != b->lcp) return a->lcp > b->lcp;
    if (a->rcp != b->rcp) return a->rcp > b->rcp;
    if (a->id != b->id) return a->id < b->id;
    return a->pos < b->pos;
}

static void heap_push(encode_state *st, const bpe_candidate *c) {
    if (st->heap_len == st->heap_cap) {
        size_t cap = st->heap_cap ? st->heap_cap * 2 : 256;
        bpe_candidate *h = (bpe_candidate *)realloc(st->heap, cap * sizeof(bpe_candidate));
        if (!h) { st->oom = 1; return; }
        st->heap = h;
        st->heap_cap = cap;
    }
    size_t i = st->heap_len++;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!cand_before(c, &st->heap[parent])) break;
        st->heap[i] = st->heap[parent];
        i = parent;
    }
    st->heap[i] = *c;
}

static bpe_candidate heap_pop(encode_state *st) {
    bpe_candidate top = st->heap[0];
    bpe_candidate last = st->heap[--st->heap_len];
    size_t n = st->heap_len, i = 0;
    for (;;) {
        size_t l = 2 * i + 1;
        if (l >= n) break;
        size_t best = l;
        if (l + 1 < n && cand_before(&st->heap[l + 1], &st->heap[l])) best = l + 1;
        if (!cand_before(&st->heap[best], &last)) break;
        st->heap[i] = st->heap[best];
        i = best;
    }
    if (n > 0) st->heap[i] = last;
    return top;
}

/* Queue the merge of symbol `left` with its right neighbour, if it exists. */
static void try_add_candidate(const gemma3_tokenizer *tok, encode_state *st, int left) {
    if (left < 0) return;
    const bpe_symbol *l = &st->syms[left];
    if (l->next < 0) return;
    const bpe_symbol *r = &st->syms[l->next];
    if (!l->mergeable || !r->mergeable) return;
    /* Adjacent mergeable symbols are contiguous in the normalized buffer */
    int id = lookup(tok, st->norm + l->start, l->len + r->len);
    if (id < 0) return;
    bpe_candidate c = {
        tok->vocab[id].score, l->cplen, r->cplen, id, left, l->len, r->len
    };
    heap_push(st, &c);
}

static void emit_token(encode_state *st, int id) {
    if (st->n_tokens < st->max_tokens) st->tokens[st->n_tokens++] = id;
}

static int ensure_capacity(encode_state *st, size_t seg_len) {
    size_t need_norm = seg_len * 3 + 1;
    if (need_norm > st->norm_cap) {
        char *p = (char *)realloc(st->norm, need_norm);
        if (!p) return 0;
        st->norm = p;
        st->norm_cap = need_norm;
    }
    size_t need_syms = need_norm;  /* at most one symbol per normalized byte */
    if (need_syms > st->syms_cap) {
        bpe_symbol *s = (bpe_symbol *)realloc(st->syms, need_syms * sizeof(bpe_symbol));
        if (!s) return 0;
        st->syms = s;
        st->syms_cap = need_syms;
    }
    return 1;
}

/* Run BPE over norm[off, off + len) and emit the resulting tokens. */
static void bpe_chunk(const gemma3_tokenizer *tok, encode_state *st, size_t off, size_t len) {
    /* Initial symbols: one per code point, or one per byte for byte fallback */
    int n = 0;
    size_t pos = off, end = off + len;
    while (pos < end) {
        int cl = utf8_len((const uint8_t *)st->norm + pos, end - pos);
        int id = lookup(tok, st->norm + pos, cl);
        if (id >= 0) {
            bpe_symbol *s = &st->syms[n];
            s->id = id;
            s->start = (int)pos;
            s->len = cl;
            s->cplen = 1;
            s->mergeable = 1;
            s->prev = n - 1;
            s->next = -1;
            if (n > 0) st->syms[n - 1].next = n;
            n++;
        } else {
            for (int b = 0; b < cl; b++) {
                int bid = tok->byte_tokens[(uint8_t)st->norm[pos + b]];
                bpe_symbol *s = &st->syms[n];
                s->id = bid >= 0 ? bid : tok->unk_id;
                s->start = (int)(pos + b);
                s->len = 1;
                s->cplen = 1;
                s->mergeable = 0;
                s->prev = n - 1;
                s->next = -1;
                if (n > 0) st->syms[n - 1].next = n;
                n++;
            }
        }
        pos += cl;
    }

    /* Seed the queue with all adjacent pairs */
    st->heap_len = 0;
    for (int i = 0; i + 1 < n; i++) {
        try_add_candidate(tok, st, i);
    }

    /* Apply merges in priority order */
    while (st->heap_len > 0 && !st->oom) {
        bpe_candidate c = heap_pop(st);
        bpe_symbol *l = &st->syms[c.pos];
        if (l->len != c.llen || l->next < 0) continue;      /* stale */
        bpe_symbol *r = &st->syms[l->next];
        if (r->len != c.rlen) continue;                      /* stale */

        l->id = c.id;
        l->len += r->len;
        l->cplen += r->cplen;
        l->next = r->next;
        if (r->next >= 0) st->syms[r->next].prev = c.pos;
        r->len = 0;

        try_add_candidate(tok, st, l->prev);
        try_add_candidate(tok, st, c.pos);
    }

    /* Symbol 0 is never merged away, so the list starts there */
    for (int i = 0; i >= 0 && n > 0; i = st->syms[i].next) {
        emit_token(st, st->syms[i].id);
    }
}

/* True if no vocab piece can cover the '▁' starting at norm[b], i.e. the final
 * tokenization is guaranteed to have a token boundary at b. */
static int is_split_point(const gemma3_tokenizer *tok, const char *norm, size_t nlen, size_t b) {
    uint8_t prev = (uint8_t)norm[b - 1];
    for (int k = tok->span_start[prev]; k < tok->span_start[prev + 1]; k++) {
        int id = tok->span_ids[k];
        size_t off = (size_t)tok->span_offs[k];
        size_t plen = tok->vocab[id].len;
        if (off > b || b - off + plen > nlen) continue;
        if (memcmp(norm + b - off, piece_str(tok, id), plen) == 0) return 0;
    }
    return 1;
}

/* Encode one segment of raw text that contains no added tokens. */
static void encode_segment(const gemma3_tokenizer *tok, encode_state *st,
                           const char *text, size_t len) {
    if (len == 0) return;
    if (!ensure_capacity(st, len)) { st->oom = 1; return; }

    /* Normalize: ' ' -> U+2581 (E2 96 81) */
    size_t nlen = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] == ' ') {
            st->norm[nlen++] = (char)0xE2;
            st->norm[nlen++] = (char)0x96;
            st->norm[nlen++] = (char)0x81;
        } else {
            st->norm[nlen++] = text[i];
        }
    }

    /* Split at guaranteed token boundaries so each BPE run (and its heap)
     * stays small; merges never cross these points, so the result is exact. */
    size_t chunk = 0;
    if (tok->split_ok) {
        const char *p = st->norm + 1;
        const char *end = st->norm + nlen;
        while (p + 3 <= end && !st->oom) {
            p = (const char *)memchr(p, 0xE2, (size_t)(end - p));
            if (!p || p + 3 > end) break;
            size_t b = (size_t)(p - st->norm);
            if (is_space_marker(p) && is_split_point(tok, st->norm, nlen, b)) {
                bpe_chunk(tok, st, chunk, b - chunk);
                chunk = b;
            }
            p++;
        }
    }
    if (!st->oom) bpe_chunk(tok, st, chunk, nlen - chunk);
}

/* Longest added token starting at text (avail bytes). Returns id and sets *match_len. */
static int match_added(const gemma3_tokenizer *tok, const char *text, size_t avail,
                       int *match_len) {
    uint8_t first = (uint8_t)text[0];
    if (!tok->added_first[first]) return -1;
    const uint64_t *mask = tok->added_lens[first];
    int max_len = avail < MAX_ADDED_LEN ? (int)avail : MAX_ADDED_LEN;
    for (int len = max_len; len >= 1; len--) {
        int bit = len - 1;
        if (!(mask[bit / 64] & ((uint64_t)1 << (bit % 64)))) continue;
        int id = lookup(tok, text, len);
        if (id >= 0 && tok->vocab[id].added) {
            *match_len = len;
            return id;
        }
    }
    return -1;
}

/* Encode text to tokens */
int gemma3_tokenize(gemma3_tokenizer *tok, const char *text,
                    int *tokens, int max_tokens, int add_bos, int add_eos) {
    if (!tok || !text || !tokens || max_tokens <= 0) {
        return GEMMA3_ERR_INVALID_ARG;
    }

    encode_state st;
    memset(&st, 0, sizeof(st));
    st.tokens = tokens;
    st.max_tokens = max_tokens;

    if (add_bos) emit_token(&st, tok->bos_id);

    size_t len = strlen(text);
    size_t seg_start = 0, pos = 0;
    while (pos < len && st.n_tokens < st.max_tokens && !st.oom) {
        int mlen = 0;
        int id = match_added(tok, text + pos, len - pos, &mlen);
        if (id >= 0) {
            encode_segment(tok, &st, text + seg_start, pos - seg_start);
            emit_token(&st, id);
            pos += (size_t)mlen;
            seg_start = pos;
        } else {
            pos++;
        }
    }
    if (st.n_tokens < st.max_tokens && !st.oom) {
        encode_segment(tok, &st, text + seg_start, len - seg_start);
    }

    if (add_eos) emit_token(&st, tok->eos_id);

    free(st.norm);
    free(st.syms);
    free(st.heap);

    if (st.oom) return GEMMA3_ERR_OUT_OF_MEMORY;
    return st.n_tokens;
}

/* ============================================================================
 * Decoding
 * ========================================================================== */

const char *gemma3_decode_token(gemma3_tokenizer *tok, int token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) {
        return NULL;
    }
    return piece_str(tok, token_id);
}

/* Decoded byte length of a token (▁ -> ' ', <0xNN> -> byte, specials -> nothing). */
static int token_bytes(const gemma3_tokenizer *tok, int id, char *out) {
    const vocab_entry *v = &tok->vocab[id];
    if (v->special) return 0;
    const char *s = piece_str(tok, id);
    if (v->type == SP_TYPE_BYTE) {
        int b = parse_byte_piece(s, v->len);
        if (b >= 0) {
            if (out) out[0] = (char)b;
            return 1;
        }
    }
    int n = 0;
    for (int i = 0; i < v->len;) {
        if (i + 2 < v->len && (uint8_t)s[i] == 0xE2 && (uint8_t)s[i + 1] == 0x96 &&
            (uint8_t)s[i + 2] == 0x81) {
            if (out) out[n] = ' ';
            n++;
            i += 3;
        } else {
            if (out) out[n] = s[i];
            n++;
            i++;
        }
    }
    return n;
}

int gemma3_token_to_bytes(gemma3_tokenizer *tok, int token_id, char *out, int out_size) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size || (!out && out_size > 0) ||
        out_size < 0) {
        return GEMMA3_ERR_INVALID_ARG;
    }
    int n = token_bytes(tok, token_id, NULL);
    if (n > out_size) return GEMMA3_ERR_INVALID_ARG;
    token_bytes(tok, token_id, out);
    if (n < out_size) out[n] = '\0';
    return n;
}

char *gemma3_detokenize(gemma3_tokenizer *tok, const int *tokens, int num_tokens) {
    if (!tok || num_tokens < 0 || (!tokens && num_tokens > 0)) {
        return NULL;
    }

    size_t total = 0;
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->vocab_size) {
            total += (size_t)token_bytes(tok, tokens[i], NULL);
        }
    }

    char *output = (char *)malloc(total + 1);
    if (!output) return NULL;

    char *ptr = output;
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->vocab_size) {
            ptr += token_bytes(tok, tokens[i], ptr);
        }
    }
    *ptr = '\0';
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

/* Byte length of the whitespace code point at s (Python str.isspace), else 0. */
static int space_len_at(const uint8_t *s, size_t avail) {
    if (avail == 0) return 0;
    uint8_t c = s[0];
    if (c == ' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F)) return 1;
    int n = utf8_len(s, avail);
    if (n == 1) return 0;
    uint32_t cp;
    if (n == 2) cp = ((uint32_t)(c & 0x1F) << 6) | (s[1] & 0x3F);
    else if (n == 3) cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    else return 0;
    if (cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000) {
        return n;
    }
    return 0;
}

/* Trim leading/trailing whitespace like Jinja's `trim` filter (Python str.strip). */
static void trim_span(const char *s, size_t *start, size_t *end) {
    const uint8_t *u = (const uint8_t *)s;
    size_t b = 0, e = strlen(s);
    int n;
    while (b < e && (n = space_len_at(u + b, e - b)) > 0) b += (size_t)n;
    while (e > b) {
        size_t cp_start = e - 1;
        while (cp_start > b && (u[cp_start] & 0xC0) == 0x80) cp_start--;
        n = space_len_at(u + cp_start, e - cp_start);
        if (n > 0 && cp_start + (size_t)n == e) e = cp_start;
        else break;
    }
    *start = b;
    *end = e;
}

static char *append(char *dst, const char *src, size_t n) {
    memcpy(dst, src, n);
    return dst + n;
}

/*
 * Gemma 3 chat template (tokenizer_config.json):
 *
 *   <bos><start_of_turn>user
 *   {system}\n\n{user}<end_of_turn>
 *   <start_of_turn>model
 *   {model}<end_of_turn>
 *   ...
 *   <start_of_turn>model
 *
 * A system message is prepended (untrimmed, followed by a blank line) to the
 * next turn's content; user/model contents are trimmed. The template has no
 * system turn, so system messages after the first are handled the same way.
 */
char *gemma3_format_chat(gemma3_tokenizer *tok, const gemma3_message *messages,
                         int num_msgs) {
    if (!tok || !messages || num_msgs <= 0) return NULL;

    size_t buf_size = 64;
    for (int i = 0; i < num_msgs; i++) {
        buf_size += (messages[i].content ? strlen(messages[i].content) : 0) + 64;
    }

    char *buf = (char *)malloc(buf_size);
    if (!buf) return NULL;
    char *ptr = buf;

    static const char k_bos[] = "<bos>";
    static const char k_start[] = "<start_of_turn>";
    static const char k_end[] = "<end_of_turn>\n";
    ptr = append(ptr, k_bos, sizeof(k_bos) - 1);

    const char *prefix = NULL;  /* pending system content */
    for (int i = 0; i < num_msgs; i++) {
        const char *content = messages[i].content ? messages[i].content : "";

        if (messages[i].role == GEMMA3_ROLE_SYSTEM) {
            prefix = content;
            continue;
        }

        const char *role = messages[i].role == GEMMA3_ROLE_MODEL ? "model" : "user";
        ptr = append(ptr, k_start, sizeof(k_start) - 1);
        ptr = append(ptr, role, strlen(role));
        *ptr++ = '\n';

        if (prefix) {
            ptr = append(ptr, prefix, strlen(prefix));
            ptr = append(ptr, "\n\n", 2);
            prefix = NULL;
        }

        size_t b, e;
        trim_span(content, &b, &e);
        ptr = append(ptr, content + b, e - b);
        ptr = append(ptr, k_end, sizeof(k_end) - 1);
    }

    /* Add model turn start for generation */
    ptr = append(ptr, k_start, sizeof(k_start) - 1);
    ptr = append(ptr, "model\n", 6);
    *ptr = '\0';

    return buf;
}
