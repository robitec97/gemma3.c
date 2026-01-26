/*
 * main.c - CLI interface for Gemma 3 inference
 *
 * Usage:
 *   ./gemma3 -m <model_dir> -p "Your prompt here"
 *   ./gemma3 -m <model_dir> -i -s "System prompt"
 */

#include "gemma3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

/* ============================================================================
 * Configuration
 * ========================================================================== */

typedef struct {
    const char *model_dir;
    const char *prompt;
    const char *system_prompt;
    int interactive;
    int max_tokens;
    float temperature;
    int top_k;
    float top_p;
    int seed;
    int context_size;
    int verbose;
    int greedy;
    int verbose_tokens;
    int tokenize_mode;      /* --tokenize: print token IDs for prompt */
    int detokenize_mode;    /* --detokenize: decode token IDs */
    int logits_mode;        /* --logits: show top logits for single forward */
} cli_config;

static cli_config default_cli_config(void) {
    return (cli_config){
        .model_dir = NULL,
        .prompt = NULL,
        .system_prompt = "You are a helpful assistant.",
        .interactive = 0,
        .max_tokens = 512,
        .temperature = 0.7f,
        .top_k = 50,
        .top_p = 0.9f,
        .seed = -1,
        .context_size = 8192,
        .verbose = 0,
        .greedy = 0,
        .verbose_tokens = 0,
        .tokenize_mode = 0,
        .detokenize_mode = 0,
        .logits_mode = 0,
    };
}

/* ============================================================================
 * Signal Handling
 * ========================================================================== */

static volatile int g_interrupted = 0;

static void signal_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
}

/* ============================================================================
 * Streaming Callback
 * ========================================================================== */

/* Buffer to detect control tokens being generated character-by-character */
static char g_pending_buf[64];
static int g_pending_len = 0;

static void flush_pending(void) {
    for (int i = 0; i < g_pending_len; i++) {
        putchar(g_pending_buf[i]);
    }
    g_pending_len = 0;
}

static void reset_pending(void) {
    g_pending_len = 0;
}

static int stream_callback(int token_id, const char *token_str, void *user_data) {
    (void)user_data;

    if (g_interrupted) {
        return 1;  /* Stop generation */
    }

    /* Skip control tokens by ID: PAD(0), EOS(1), BOS(2), UNK(3) */
    if (token_id <= 3) {
        return 0;
    }

    /* Skip control tokens by string pattern */
    if (token_str && token_str[0] != '\0') {
        /* Skip any <...> control tokens like <end_of_turn>, <start_of_turn>, <bos>, etc. */
        size_t len = strlen(token_str);
        if (len >= 3 && token_str[0] == '<' && token_str[len-1] == '>') {
            return 0;
        }
    }

    /* Handle token output with control sequence detection */
    if (token_str && token_str[0] != '\0') {
        const char *ptr = token_str;
        while (*ptr) {
            /* Check for ▁ (0xE2 0x96 0x81) - sentencepiece space marker */
            if ((unsigned char)ptr[0] == 0xE2 &&
                (unsigned char)ptr[1] == 0x96 &&
                (unsigned char)ptr[2] == 0x81) {
                /* If we have pending content and see a space, flush it */
                if (g_pending_len > 0 && g_pending_buf[0] != '<') {
                    flush_pending();
                }
                if (g_pending_len == 0) {
                    putchar(' ');
                } else {
                    /* Space inside a potential control sequence - keep buffering */
                    if (g_pending_len < (int)sizeof(g_pending_buf) - 1) {
                        g_pending_buf[g_pending_len++] = ' ';
                    }
                }
                ptr += 3;
            } else if (ptr[0] == '<' && ptr[1] == '0' && ptr[2] == 'x') {
                /* Byte token - skip */
                while (*ptr && *ptr != '>') ptr++;
                if (*ptr == '>') ptr++;
            } else if (ptr[0] == '<') {
                /* Start of potential control sequence */
                flush_pending();  /* Flush any previous pending content */
                g_pending_buf[g_pending_len++] = '<';
                ptr++;
            } else if (g_pending_len > 0 && g_pending_buf[0] == '<') {
                /* We're inside a potential control sequence */
                if (ptr[0] == '>') {
                    /* End of control sequence - check if it's a known control token */
                    g_pending_buf[g_pending_len++] = '>';
                    g_pending_buf[g_pending_len] = '\0';

                    /* Check if this is a control token to suppress */
                    if (strstr(g_pending_buf, "end_of_turn") ||
                        strstr(g_pending_buf, "start_of_turn") ||
                        strstr(g_pending_buf, "bos") ||
                        strstr(g_pending_buf, "eos") ||
                        strstr(g_pending_buf, "pad")) {
                        /* Suppress this control token and signal to stop */
                        reset_pending();
                        fflush(stdout);
                        return 1;  /* Stop generation */
                    } else {
                        /* Not a known control token, print it */
                        flush_pending();
                    }
                    ptr++;
                } else {
                    /* Continue buffering the control sequence */
                    if (g_pending_len < (int)sizeof(g_pending_buf) - 2) {
                        g_pending_buf[g_pending_len++] = *ptr;
                    }
                    ptr++;
                }
            } else {
                /* Regular character - print directly */
                putchar(*ptr);
                ptr++;
            }
        }
        fflush(stdout);
    }

    return 0;
}

/* ============================================================================
 * Help and Usage
 * ========================================================================== */

static void print_usage(const char *prog) {
    fprintf(stderr, "Gemma 3 4B Inference - Pure C Implementation\n");
    fprintf(stderr, "Version: %s\n\n", gemma3_version());
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <path>      Path to model directory (required)\n");
    fprintf(stderr, "  -p, --prompt <text>     Input prompt for generation\n");
    fprintf(stderr, "  -i, --interactive       Interactive chat mode\n");
    fprintf(stderr, "  -s, --system <text>     System prompt for chat mode\n");
    fprintf(stderr, "  -n, --max-tokens <n>    Maximum tokens to generate (default: 512)\n");
    fprintf(stderr, "  -t, --temperature <f>   Sampling temperature (default: 0.7)\n");
    fprintf(stderr, "  -k, --top-k <n>         Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <f>             Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  --seed <n>              Random seed (-1 for random)\n");
    fprintf(stderr, "  --greedy                Force greedy decoding (deterministic)\n");
    fprintf(stderr, "  --verbose-tokens        Print token IDs during generation (to stderr)\n");
    fprintf(stderr, "  -c, --context <n>       Context size (default: 8192)\n");
    fprintf(stderr, "  -v, --verbose           Verbose output\n");
    fprintf(stderr, "  -h, --help              Show this help message\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Debug modes:\n");
    fprintf(stderr, "  --tokenize              Tokenize prompt and print token IDs\n");
    fprintf(stderr, "  --detokenize            Detokenize comma-separated IDs from prompt\n");
    fprintf(stderr, "  --logits                Run single forward pass and show top-20 logits\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -m ./gemma-3-4b-it -p \"Hello, how are you?\"\n", prog);
    fprintf(stderr, "  %s -m ./gemma-3-4b-it -i\n", prog);
    fprintf(stderr, "  %s -m ./gemma-3-4b-it -i -s \"You are a pirate.\"\n", prog);
    fprintf(stderr, "  %s -m ./gemma-3-4b-it -p \"Say OK\" --greedy --verbose-tokens\n", prog);
    fprintf(stderr, "  %s -m ./gemma-3-4b-it --tokenize -p \"Hello, world!\"\n", prog);
}

/* ============================================================================
 * Argument Parsing
 * ========================================================================== */

static int parse_args(int argc, char **argv, cli_config *config) {
    *config = default_cli_config();

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -m requires an argument\n");
                return 0;
            }
            config->model_dir = argv[i];
        } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--prompt") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -p requires an argument\n");
                return 0;
            }
            config->prompt = argv[i];
        } else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--interactive") == 0) {
            config->interactive = 1;
        } else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--system") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -s requires an argument\n");
                return 0;
            }
            config->system_prompt = argv[i];
        } else if (strcmp(arg, "-n") == 0 || strcmp(arg, "--max-tokens") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -n requires an argument\n");
                return 0;
            }
            config->max_tokens = atoi(argv[i]);
        } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--temperature") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -t requires an argument\n");
                return 0;
            }
            config->temperature = atof(argv[i]);
        } else if (strcmp(arg, "-k") == 0 || strcmp(arg, "--top-k") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -k requires an argument\n");
                return 0;
            }
            config->top_k = atoi(argv[i]);
        } else if (strcmp(arg, "--top-p") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: --top-p requires an argument\n");
                return 0;
            }
            config->top_p = atof(argv[i]);
        } else if (strcmp(arg, "--seed") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: --seed requires an argument\n");
                return 0;
            }
            config->seed = atoi(argv[i]);
        } else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--context") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -c requires an argument\n");
                return 0;
            }
            config->context_size = atoi(argv[i]);
        } else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            config->verbose = 1;
        } else if (strcmp(arg, "--greedy") == 0) {
            config->greedy = 1;
        } else if (strcmp(arg, "--verbose-tokens") == 0) {
            config->verbose_tokens = 1;
        } else if (strcmp(arg, "--tokenize") == 0) {
            config->tokenize_mode = 1;
        } else if (strcmp(arg, "--detokenize") == 0) {
            config->detokenize_mode = 1;
        } else if (strcmp(arg, "--logits") == 0) {
            config->logits_mode = 1;
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n", arg);
            return 0;
        }
    }

    if (!config->model_dir) {
        fprintf(stderr, "Error: Model directory (-m) is required\n");
        return 0;
    }

    /* Special modes only require prompt */
    if (config->tokenize_mode || config->detokenize_mode || config->logits_mode) {
        if (!config->prompt) {
            fprintf(stderr, "Error: -p (prompt) is required for debug modes\n");
            return 0;
        }
        return 1;
    }

    if (!config->interactive && !config->prompt) {
        fprintf(stderr, "Error: Either -p (prompt) or -i (interactive) is required\n");
        return 0;
    }

    return 1;
}

/* ============================================================================
 * Tokenize Mode
 * ========================================================================== */

static int run_tokenize_mode(gemma3_ctx *ctx, const cli_config *config) {
    gemma3_tokenizer *tok = gemma3_get_tokenizer(ctx);
    if (!tok) {
        fprintf(stderr, "Error: Failed to get tokenizer\n");
        return 1;
    }

    /* Allocate token buffer */
    int max_tokens = config->context_size;
    int *tokens = (int *)malloc(max_tokens * sizeof(int));
    if (!tokens) {
        fprintf(stderr, "Error: Out of memory\n");
        return 1;
    }

    /* Tokenize (with BOS, no EOS) */
    int n_tokens = gemma3_tokenize(tok, config->prompt, tokens, max_tokens, 1, 0);
    if (n_tokens < 0) {
        fprintf(stderr, "Error: Tokenization failed (code %d)\n", n_tokens);
        free(tokens);
        return 1;
    }

    /* Print token IDs */
    printf("Input: \"%s\"\n", config->prompt);
    printf("Token count: %d\n", n_tokens);
    printf("Token IDs: [");
    for (int i = 0; i < n_tokens; i++) {
        if (i > 0) printf(", ");
        printf("%d", tokens[i]);
    }
    printf("]\n");

    /* Print each token with its string representation */
    printf("\nToken breakdown:\n");
    for (int i = 0; i < n_tokens; i++) {
        const char *piece = gemma3_decode_token(tok, tokens[i]);
        printf("  %4d: %6d -> '%s'\n", i, tokens[i], piece ? piece : "(null)");
    }

    free(tokens);
    return 0;
}

/* ============================================================================
 * Detokenize Mode
 * ========================================================================== */

static int run_detokenize_mode(gemma3_ctx *ctx, const cli_config *config) {
    gemma3_tokenizer *tok = gemma3_get_tokenizer(ctx);
    if (!tok) {
        fprintf(stderr, "Error: Failed to get tokenizer\n");
        return 1;
    }

    /* Parse comma-separated token IDs from prompt */
    int max_tokens = 4096;
    int *tokens = (int *)malloc(max_tokens * sizeof(int));
    if (!tokens) {
        fprintf(stderr, "Error: Out of memory\n");
        return 1;
    }

    int n_tokens = 0;
    const char *p = config->prompt;
    while (*p && n_tokens < max_tokens) {
        /* Skip whitespace and commas */
        while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n') p++;
        if (!*p) break;

        /* Parse number */
        char *end;
        long val = strtol(p, &end, 10);
        if (end == p) {
            fprintf(stderr, "Error: Invalid token ID at position %ld\n", (long)(p - config->prompt));
            free(tokens);
            return 1;
        }
        tokens[n_tokens++] = (int)val;
        p = end;
    }

    if (n_tokens == 0) {
        fprintf(stderr, "Error: No token IDs provided\n");
        free(tokens);
        return 1;
    }

    /* Print token IDs */
    printf("Token IDs: [");
    for (int i = 0; i < n_tokens; i++) {
        if (i > 0) printf(", ");
        printf("%d", tokens[i]);
    }
    printf("]\n");

    /* Detokenize */
    char *text = gemma3_detokenize(tok, tokens, n_tokens);
    if (!text) {
        fprintf(stderr, "Error: Detokenization failed\n");
        free(tokens);
        return 1;
    }

    printf("Decoded text: \"%s\"\n", text);
    free(text);
    free(tokens);
    return 0;
}

/* ============================================================================
 * Logits Mode - Single forward pass for debugging
 * ========================================================================== */

static int run_logits_mode(gemma3_ctx *ctx, const cli_config *config) {
    gemma3_tokenizer *tok = gemma3_get_tokenizer(ctx);
    if (!tok) {
        fprintf(stderr, "Error: Failed to get tokenizer\n");
        return 1;
    }

    const gemma3_config *model_config = gemma3_get_config(ctx);
    int vocab_size = model_config->vocab_size;

    /* Allocate token buffer */
    int max_tokens = config->context_size;
    int *tokens = (int *)malloc(max_tokens * sizeof(int));
    float *logits = (float *)malloc(vocab_size * sizeof(float));
    if (!tokens || !logits) {
        fprintf(stderr, "Error: Out of memory\n");
        free(tokens);
        free(logits);
        return 1;
    }

    /* Tokenize (with BOS, no EOS) */
    int n_tokens = gemma3_tokenize(tok, config->prompt, tokens, max_tokens, 1, 0);
    if (n_tokens < 0) {
        fprintf(stderr, "Error: Tokenization failed (code %d)\n", n_tokens);
        free(tokens);
        free(logits);
        return 1;
    }

    printf("Input: \"%s\"\n", config->prompt);
    printf("Token count: %d\n", n_tokens);
    printf("Token IDs: [");
    for (int i = 0; i < n_tokens; i++) {
        if (i > 0) printf(", ");
        printf("%d", tokens[i]);
    }
    printf("]\n\n");

    /* Reset KV cache and run forward pass */
    gemma3_reset_cache(ctx);
    int err = gemma3_forward_batch(ctx, tokens, n_tokens, 0, logits);
    if (err != 0) {
        fprintf(stderr, "Error: Forward pass failed (code %d)\n", err);
        free(tokens);
        free(logits);
        return 1;
    }

    /* Find top-20 tokens by logit value */
    typedef struct { int id; float logit; } token_logit;
    token_logit top20[20];
    for (int i = 0; i < 20; i++) {
        top20[i].id = -1;
        top20[i].logit = -1e30f;
    }

    for (int i = 0; i < vocab_size; i++) {
        /* Find minimum in top20 */
        int min_idx = 0;
        for (int j = 1; j < 20; j++) {
            if (top20[j].logit < top20[min_idx].logit) {
                min_idx = j;
            }
        }
        /* Replace if current logit is higher */
        if (logits[i] > top20[min_idx].logit) {
            top20[min_idx].id = i;
            top20[min_idx].logit = logits[i];
        }
    }

    /* Sort top20 by logit descending */
    for (int i = 0; i < 19; i++) {
        for (int j = i + 1; j < 20; j++) {
            if (top20[j].logit > top20[i].logit) {
                token_logit tmp = top20[i];
                top20[i] = top20[j];
                top20[j] = tmp;
            }
        }
    }

    /* Print top-20 */
    printf("Top-20 next token predictions:\n");
    printf("%-6s  %-10s  %-10s  %s\n", "Rank", "Token ID", "Logit", "Token");
    printf("------  ----------  ----------  --------\n");
    for (int i = 0; i < 20; i++) {
        if (top20[i].id >= 0) {
            const char *piece = gemma3_decode_token(tok, top20[i].id);
            printf("%-6d  %-10d  %10.4f  '%s'\n",
                   i + 1, top20[i].id, top20[i].logit, piece ? piece : "(null)");
        }
    }

    free(tokens);
    free(logits);
    return 0;
}

/* ============================================================================
 * Single Prompt Mode
 * ========================================================================== */

static int run_single_prompt(gemma3_ctx *ctx, const cli_config *config) {
    gemma3_gen_params params = {
        .max_tokens = config->max_tokens,
        .temperature = config->temperature,
        .top_k = config->top_k,
        .top_p = config->top_p,
        .seed = config->seed,
        .stop_on_eos = 1,
        .greedy = config->greedy,
        .verbose_tokens = config->verbose_tokens,
    };

    g_interrupted = 0;
    reset_pending();  /* Reset control sequence buffer */

    /* Use chat interface to properly format prompt with chat template */
    gemma3_message messages[2];
    int num_messages = 0;

    /* Add system message if provided */
    if (config->system_prompt && strlen(config->system_prompt) > 0) {
        messages[num_messages].role = GEMMA3_ROLE_SYSTEM;
        messages[num_messages].content = config->system_prompt;
        num_messages++;
    }

    /* Add user message */
    messages[num_messages].role = GEMMA3_ROLE_USER;
    messages[num_messages].content = config->prompt;
    num_messages++;

    char *response = gemma3_chat(ctx, messages, num_messages, &params,
                                 stream_callback, NULL);
    printf("\n");

    if (!response) {
        fprintf(stderr, "Error: Generation failed: %s\n", gemma3_get_error());
        return 1;
    }

    free(response);
    return 0;
}

/* ============================================================================
 * Interactive Chat Mode
 * ========================================================================== */

#define MAX_INPUT_LEN 4096
#define MAX_MESSAGES 100

static int run_interactive(gemma3_ctx *ctx, const cli_config *config) {
    printf("Gemma 3 Interactive Chat\n");
    printf("Type 'quit' or 'exit' to end, 'clear' to reset conversation\n");
    printf("System: %s\n", config->system_prompt);
    printf("---\n\n");

    gemma3_message messages[MAX_MESSAGES];
    int num_messages = 0;

    /* Add system message */
    if (config->system_prompt && strlen(config->system_prompt) > 0) {
        messages[num_messages].role = GEMMA3_ROLE_SYSTEM;
        messages[num_messages].content = config->system_prompt;
        num_messages++;
    }

    char input[MAX_INPUT_LEN];

    while (1) {
        /* Prompt */
        printf("You: ");
        fflush(stdout);

        /* Read input */
        if (!fgets(input, sizeof(input), stdin)) {
            printf("\n");
            break;
        }

        /* Remove trailing newline */
        int len = strlen(input);
        if (len > 0 && input[len - 1] == '\n') {
            input[len - 1] = '\0';
            len--;
        }

        /* Skip empty input */
        if (len == 0) continue;

        /* Check for commands */
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            printf("Goodbye!\n");
            break;
        }

        if (strcmp(input, "clear") == 0) {
            num_messages = 0;
            if (config->system_prompt && strlen(config->system_prompt) > 0) {
                messages[num_messages].role = GEMMA3_ROLE_SYSTEM;
                messages[num_messages].content = config->system_prompt;
                num_messages++;
            }
            gemma3_reset_cache(ctx);
            printf("[Conversation cleared]\n\n");
            continue;
        }

        /* Check message limit */
        if (num_messages >= MAX_MESSAGES - 1) {
            printf("[Warning: Maximum messages reached, clearing history]\n");
            num_messages = 0;
            if (config->system_prompt && strlen(config->system_prompt) > 0) {
                messages[num_messages].role = GEMMA3_ROLE_SYSTEM;
                messages[num_messages].content = config->system_prompt;
                num_messages++;
            }
            gemma3_reset_cache(ctx);
        }

        /* Add user message */
        char *user_input = strdup(input);
        if (!user_input) {
            fprintf(stderr, "Error: Out of memory\n");
            break;
        }
        messages[num_messages].role = GEMMA3_ROLE_USER;
        messages[num_messages].content = user_input;
        num_messages++;

        /* Generate response */
        gemma3_gen_params params = {
            .max_tokens = config->max_tokens,
            .temperature = config->temperature,
            .top_k = config->top_k,
            .top_p = config->top_p,
            .seed = config->seed,
            .stop_on_eos = 1,
            .greedy = config->greedy,
            .verbose_tokens = config->verbose_tokens,
        };

        g_interrupted = 0;
        reset_pending();  /* Reset control sequence buffer */
        printf("\nGemma: ");
        fflush(stdout);

        char *response = gemma3_chat(ctx, messages, num_messages, &params,
                                     stream_callback, NULL);
        printf("\n\n");

        if (!response) {
            fprintf(stderr, "[Error: Generation failed: %s]\n\n", gemma3_get_error());
            /* Remove the failed user message */
            free((char *)messages[num_messages - 1].content);
            num_messages--;
            continue;
        }

        /* Add assistant response to history */
        messages[num_messages].role = GEMMA3_ROLE_MODEL;
        messages[num_messages].content = response;
        num_messages++;
    }

    /* Cleanup message history */
    for (int i = 0; i < num_messages; i++) {
        if (messages[i].role != GEMMA3_ROLE_SYSTEM) {
            free((char *)messages[i].content);
        }
    }

    return 0;
}

/* ============================================================================
 * Main
 * ========================================================================== */

int main(int argc, char **argv) {
    cli_config config;

    if (!parse_args(argc, argv, &config)) {
        print_usage(argv[0]);
        return 1;
    }

    /* Set up signal handler */
    signal(SIGINT, signal_handler);

    /* Load model */
    gemma3_ctx *ctx = gemma3_load_dir_ex(config.model_dir, config.context_size);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to load model: %s\n", gemma3_get_error());
        return 1;
    }

    if (config.verbose) {
        const gemma3_config *model_config = gemma3_get_config(ctx);
        printf("Model configuration:\n");
        printf("  Vocab size: %d\n", model_config->vocab_size);
        printf("  Hidden size: %d\n", model_config->hidden_size);
        printf("  Layers: %d\n", model_config->num_layers);
        printf("  Heads: %d (KV: %d)\n", model_config->num_heads, model_config->num_kv_heads);
        printf("  Head dim: %d\n", model_config->head_dim);
        printf("  Context: %d\n", model_config->max_context);
        printf("\n");
    }

    int result;
    if (config.tokenize_mode) {
        result = run_tokenize_mode(ctx, &config);
    } else if (config.detokenize_mode) {
        result = run_detokenize_mode(ctx, &config);
    } else if (config.logits_mode) {
        result = run_logits_mode(ctx, &config);
    } else if (config.interactive) {
        result = run_interactive(ctx, &config);
    } else {
        result = run_single_prompt(ctx, &config);
    }

    gemma3_free(ctx);
    return result;
}
