/*
 * main.c - CLI interface for Gemma 3 inference
 *
 * Usage:
 *   ./gemma3 -p "Your prompt here"
 *   ./gemma3 -i                          # interactive chat
 *   echo "Summarize this" | ./gemma3     # prompt from stdin
 */

#include "gemma3.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* ============================================================================
 * Configuration
 * ========================================================================== */

typedef struct {
    const char *model_dir;
    const char *prompt;
    const char *prompt_file;
    const char *system_prompt;
    int interactive;
    int max_tokens;
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    int seed;
    int context_size;
    int num_threads;
    int use_gpu;
    int verbose;
    int quiet;
    int show_stats;
    int color;
    int greedy;
    int verbose_tokens;
    int tokenize_mode;      /* --tokenize: print token IDs for prompt */
    int detokenize_mode;    /* --detokenize: decode token IDs */
    int logits_mode;        /* --logits: show top logits for single forward */
} cli_config;

static cli_config default_cli_config(void) {
    const char *env_model = getenv("GEMMA3_MODEL");
    return (cli_config){
        .model_dir = (env_model && *env_model) ? env_model : "gemma-3-4b-it",
        .system_prompt = "You are a helpful assistant.",
        .max_tokens = 512,
        .temperature = 0.7f,
        .top_k = 50,
        .top_p = 0.9f,
        .min_p = 0.0f,
        .seed = -1,
        .context_size = 8192,
        .num_threads = 0,
        .use_gpu = 1,
        .color = -1,  /* auto */
    };
}

/* ============================================================================
 * Terminal styling
 * ========================================================================== */

static int g_color = 0;

static const char *sty(const char *code) {
    return g_color ? code : "";
}
#define DIM    sty("\033[2m")
#define BOLD   sty("\033[1m")
#define CYAN   sty("\033[36m")
#define GREEN  sty("\033[32m")
#define YELLOW sty("\033[33m")
#define RESET  sty("\033[0m")

/* ============================================================================
 * Signal Handling
 * ========================================================================== */

static volatile sig_atomic_t g_interrupted = 0;
static gemma3_ctx *g_ctx = NULL;

static void signal_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
    if (g_ctx) gemma3_abort(g_ctx);
}

static void install_signal_handler(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  /* no SA_RESTART: Ctrl+C interrupts a blocking read */
    sigaction(SIGINT, &sa, NULL);
}

/* ============================================================================
 * Streaming Callback
 * ========================================================================== */

typedef struct {
    gemma3_tokenizer *tok;
    int at_line_start;
    int printed;
} stream_state;

static int stream_callback(int token_id, const char *token_str, void *user_data) {
    (void)token_str;
    stream_state *st = (stream_state *)user_data;
    if (g_interrupted) return 1;

    char buf[256];
    int n = gemma3_token_to_bytes(st->tok, token_id, buf, sizeof(buf));
    if (n <= 0) return 0;

    /* Skip leading whitespace at the very start of a reply */
    int off = 0;
    if (!st->printed) {
        while (off < n && (buf[off] == ' ' || buf[off] == '\n')) off++;
        if (off == n) return 0;
    }
    fwrite(buf + off, 1, (size_t)(n - off), stdout);
    fflush(stdout);
    st->printed = 1;
    st->at_line_start = buf[n - 1] == '\n';
    return 0;
}

/* ============================================================================
 * Help and Usage
 * ========================================================================== */

static void print_usage(FILE *out, const char *prog) {
    fprintf(out,
        "gemma3.c %s - Gemma 3 4B inference in pure C\n"
        "\n"
        "Usage: %s [options]\n"
        "\n"
        "Input:\n"
        "  -p, --prompt <text>     Prompt text (use '-' to read from stdin)\n"
        "  -f, --file <path>       Read the prompt from a file\n"
        "  -i, --interactive       Interactive multi-turn chat\n"
        "  -s, --system <text>     System prompt (default: \"You are a helpful assistant.\")\n"
        "                          Without -p/-f/-i, a piped stdin is used as the prompt.\n"
        "\n"
        "Model and runtime:\n"
        "  -m, --model <path>      Model directory (default: $GEMMA3_MODEL or ./gemma-3-4b-it)\n"
        "  -c, --context <n>       Context size in tokens (default: 8192)\n"
        "      --threads <n>       CPU threads (default: $GEMMA3_THREADS or all cores)\n"
        "      --cpu               Use the CPU even in a Metal (make mps) build\n"
        "\n"
        "Generation:\n"
        "  -n, --max-tokens <n>    Max tokens to generate (default: 512)\n"
        "  -t, --temperature <f>   Sampling temperature (default: 0.7, 0 = greedy)\n"
        "  -k, --top-k <n>         Top-k sampling (default: 50, 0 = disabled)\n"
        "      --top-p <f>         Top-p (nucleus) sampling (default: 0.9)\n"
        "      --min-p <f>         Min-p sampling (default: 0 = disabled)\n"
        "      --seed <n>          Random seed (default: -1 = random)\n"
        "      --greedy            Deterministic greedy decoding\n"
        "\n"
        "Output:\n"
        "      --stats             Print timing stats (TTFT, prefill and decode tokens/s)\n"
        "  -q, --quiet             Don't print the model loading line\n"
        "  -v, --verbose           Verbose loading output and model configuration\n"
        "      --no-color          Disable colored output (also honours NO_COLOR)\n"
        "  -h, --help              Show this help\n"
        "      --version           Show version\n"
        "\n"
        "Debugging:\n"
        "      --tokenize          Print the token IDs of the prompt\n"
        "      --detokenize        Decode comma-separated token IDs given as the prompt\n"
        "      --logits            Show the top-20 next-token logits for the prompt\n"
        "      --verbose-tokens    Print each sampled token ID to stderr\n"
        "\n"
        "Examples:\n"
        "  %s -p \"Explain quantum computing simply.\"\n"
        "  %s -i -s \"You are a pirate.\"\n"
        "  cat notes.txt | %s -s \"Summarize the user's text in 3 bullet points.\"\n"
        "  %s -p \"Write a haiku\" --stats\n",
        gemma3_version(), prog, prog, prog, prog, prog);
}

/* ============================================================================
 * Argument Parsing
 * ========================================================================== */

static int parse_int(const char *opt, const char *s, int min, int max, int *out) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end || v < min || v > max) {
        fprintf(stderr, "Error: %s expects an integer in [%d, %d], got '%s'\n", opt, min, max, s);
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int parse_float(const char *opt, const char *s, float min, float max, float *out) {
    char *end;
    errno = 0;
    float v = strtof(s, &end);
    if (errno || end == s || *end || v < min || v > max) {
        fprintf(stderr, "Error: %s expects a number in [%g, %g], got '%s'\n", opt, min, max, s);
        return 0;
    }
    *out = v;
    return 1;
}

/* Returns 1 to continue, 0 on error, -1 to exit successfully (help/version) */
static int parse_args(int argc, char **argv, cli_config *config) {
    *config = default_cli_config();

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
#define NEED_VALUE() \
        if (i + 1 >= argc) { fprintf(stderr, "Error: %s requires an argument\n", arg); return 0; } \
        const char *val = argv[++i];

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            NEED_VALUE(); config->model_dir = val;
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            NEED_VALUE(); config->prompt = val;
        } else if (!strcmp(arg, "-f") || !strcmp(arg, "--file")) {
            NEED_VALUE(); config->prompt_file = val;
        } else if (!strcmp(arg, "-s") || !strcmp(arg, "--system")) {
            NEED_VALUE(); config->system_prompt = val;
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--max-tokens")) {
            NEED_VALUE(); if (!parse_int(arg, val, 1, 1 << 20, &config->max_tokens)) return 0;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--temperature")) {
            NEED_VALUE(); if (!parse_float(arg, val, 0.0f, 100.0f, &config->temperature)) return 0;
        } else if (!strcmp(arg, "-k") || !strcmp(arg, "--top-k")) {
            NEED_VALUE(); if (!parse_int(arg, val, 0, 1 << 20, &config->top_k)) return 0;
        } else if (!strcmp(arg, "--top-p")) {
            NEED_VALUE(); if (!parse_float(arg, val, 0.0f, 1.0f, &config->top_p)) return 0;
        } else if (!strcmp(arg, "--min-p")) {
            NEED_VALUE(); if (!parse_float(arg, val, 0.0f, 1.0f, &config->min_p)) return 0;
        } else if (!strcmp(arg, "--seed")) {
            NEED_VALUE(); if (!parse_int(arg, val, -1, 2147483647, &config->seed)) return 0;
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--context")) {
            NEED_VALUE(); if (!parse_int(arg, val, 16, GEMMA3_MAX_CONTEXT, &config->context_size)) return 0;
        } else if (!strcmp(arg, "--threads")) {
            NEED_VALUE(); if (!parse_int(arg, val, 1, 256, &config->num_threads)) return 0;
        } else if (!strcmp(arg, "--cpu")) {
            config->use_gpu = 0;
        } else if (!strcmp(arg, "--stats")) {
            config->show_stats = 1;
        } else if (!strcmp(arg, "-q") || !strcmp(arg, "--quiet")) {
            config->quiet = 1;
        } else if (!strcmp(arg, "-v") || !strcmp(arg, "--verbose")) {
            config->verbose = 1;
        } else if (!strcmp(arg, "--no-color")) {
            config->color = 0;
        } else if (!strcmp(arg, "--greedy")) {
            config->greedy = 1;
        } else if (!strcmp(arg, "--verbose-tokens")) {
            config->verbose_tokens = 1;
        } else if (!strcmp(arg, "--tokenize")) {
            config->tokenize_mode = 1;
        } else if (!strcmp(arg, "--detokenize")) {
            config->detokenize_mode = 1;
        } else if (!strcmp(arg, "--logits")) {
            config->logits_mode = 1;
        } else if (!strcmp(arg, "-i") || !strcmp(arg, "--interactive")) {
            config->interactive = 1;
        } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            print_usage(stdout, argv[0]);
            return -1;
        } else if (!strcmp(arg, "--version")) {
            printf("gemma3.c %s\n", gemma3_version());
            return -1;
        } else {
            fprintf(stderr, "Error: Unknown option '%s' (see --help)\n", arg);
            return 0;
        }
#undef NEED_VALUE
    }
    return 1;
}

/* Read all of a stream into a NUL-terminated buffer */
static char *read_stream(FILE *f) {
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    size_t n;
    while ((n = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += n;
        if (cap - len - 1 == 0) {
            char *nb = (char *)realloc(buf, cap * 2);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
            cap *= 2;
        }
    }
    buf[len] = '\0';
    return buf;
}

/* Resolve where the prompt comes from (-p, -p -, -f, or piped stdin) */
static int resolve_prompt(cli_config *config, char **owned) {
    *owned = NULL;
    if (config->prompt_file) {
        FILE *f = fopen(config->prompt_file, "rb");
        if (!f) {
            fprintf(stderr, "Error: cannot open prompt file '%s': %s\n",
                    config->prompt_file, strerror(errno));
            return 0;
        }
        *owned = read_stream(f);
        fclose(f);
    } else if (config->prompt && !strcmp(config->prompt, "-")) {
        *owned = read_stream(stdin);
    } else if (!config->prompt && !config->interactive && !isatty(STDIN_FILENO)) {
        *owned = read_stream(stdin);
    } else {
        return 1;
    }
    if (!*owned) {
        fprintf(stderr, "Error: failed to read prompt\n");
        return 0;
    }
    config->prompt = *owned;
    return 1;
}

static int check_model_dir(const char *dir) {
    struct stat st;
    if (stat(dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Error: model directory '%s' not found.\n", dir);
        fprintf(stderr, "  Download it with:  python download_model.py\n");
        fprintf(stderr, "  or point to it with -m <path> / GEMMA3_MODEL=<path>\n");
        return 0;
    }
    return 1;
}

/* ============================================================================
 * Statistics
 * ========================================================================== */

static void print_stats(gemma3_ctx *ctx, int interactive) {
    const gemma3_stats *s = gemma3_get_stats(ctx);
    if (!s) return;
    int new_tokens = s->prompt_tokens - s->reused_tokens;
    fprintf(stderr, "%s%s", interactive ? "" : "\n", DIM);
    fprintf(stderr, "[ prompt %d tok", s->prompt_tokens);
    if (s->reused_tokens > 0) fprintf(stderr, " (%d cached)", s->reused_tokens);
    if (new_tokens > 0 && s->prefill_tok_per_s > 0) {
        fprintf(stderr, " @ %.1f tok/s", s->prefill_tok_per_s);
    }
    fprintf(stderr, " | first token %.2f s", s->ttft_ms / 1000.0);
    fprintf(stderr, " | generated %d tok", s->generated_tokens);
    if (s->decode_tok_per_s > 0) fprintf(stderr, " @ %.1f tok/s", s->decode_tok_per_s);
    fprintf(stderr, " ]%s\n", RESET);
}

static gemma3_gen_params make_params(const cli_config *config) {
    gemma3_gen_params params = gemma3_default_params();
    params.max_tokens = config->max_tokens;
    params.temperature = config->temperature;
    params.top_k = config->top_k;
    params.top_p = config->top_p;
    params.min_p = config->min_p;
    params.seed = config->seed;
    params.greedy = config->greedy;
    params.verbose_tokens = config->verbose_tokens;
    return params;
}

/* ============================================================================
 * Debug Modes
 * ========================================================================== */

static int run_tokenize_mode(gemma3_ctx *ctx, const cli_config *config) {
    gemma3_tokenizer *tok = gemma3_get_tokenizer(ctx);
    int max_tokens = (int)strlen(config->prompt) + 8;
    int *tokens = (int *)malloc((size_t)max_tokens * sizeof(int));
    if (!tokens) return 1;

    int n_tokens = gemma3_tokenize(tok, config->prompt, tokens, max_tokens, 1, 0);
    if (n_tokens < 0) {
        fprintf(stderr, "Error: Tokenization failed (code %d)\n", n_tokens);
        free(tokens);
        return 1;
    }

    printf("Input: \"%s\"\n", config->prompt);
    printf("Token count: %d\n", n_tokens);
    printf("Token IDs: [");
    for (int i = 0; i < n_tokens; i++) printf("%s%d", i ? ", " : "", tokens[i]);
    printf("]\n\nToken breakdown:\n");
    for (int i = 0; i < n_tokens; i++) {
        const char *piece = gemma3_decode_token(tok, tokens[i]);
        printf("  %4d: %6d -> '%s'\n", i, tokens[i], piece ? piece : "(null)");
    }
    free(tokens);
    return 0;
}

static int run_detokenize_mode(gemma3_ctx *ctx, const cli_config *config) {
    gemma3_tokenizer *tok = gemma3_get_tokenizer(ctx);
    int max_tokens = (int)strlen(config->prompt) + 1;
    int *tokens = (int *)malloc((size_t)max_tokens * sizeof(int));
    if (!tokens) return 1;

    int n_tokens = 0;
    const char *p = config->prompt;
    while (*p && n_tokens < max_tokens) {
        while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n' || *p == '[' || *p == ']') p++;
        if (!*p) break;
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

    char *text = gemma3_detokenize(tok, tokens, n_tokens);
    if (!text) {
        fprintf(stderr, "Error: Detokenization failed\n");
        free(tokens);
        return 1;
    }
    printf("Token IDs: [");
    for (int i = 0; i < n_tokens; i++) printf("%s%d", i ? ", " : "", tokens[i]);
    printf("]\nDecoded text: \"%s\"\n", text);
    free(text);
    free(tokens);
    return 0;
}

static int run_logits_mode(gemma3_ctx *ctx, const cli_config *config) {
    gemma3_tokenizer *tok = gemma3_get_tokenizer(ctx);
    int vocab_size = gemma3_get_config(ctx)->vocab_size;
    int max_tokens = config->context_size;
    int *tokens = (int *)malloc((size_t)max_tokens * sizeof(int));
    float *logits = (float *)malloc((size_t)vocab_size * sizeof(float));
    if (!tokens || !logits) {
        free(tokens);
        free(logits);
        return 1;
    }

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
    for (int i = 0; i < n_tokens; i++) printf("%s%d", i ? ", " : "", tokens[i]);
    printf("]\n\n");

    gemma3_reset_cache(ctx);
    int err = gemma3_forward_batch(ctx, tokens, n_tokens, 0, logits);
    if (err != 0) {
        fprintf(stderr, "Error: Forward pass failed (code %d)\n", err);
        free(tokens);
        free(logits);
        return 1;
    }

    typedef struct { int id; float logit; } token_logit;
    token_logit top[20];
    for (int i = 0; i < 20; i++) { top[i].id = -1; top[i].logit = -1e30f; }
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] <= top[19].logit) continue;
        int j = 19;
        while (j > 0 && logits[i] > top[j - 1].logit) { top[j] = top[j - 1]; j--; }
        top[j].id = i;
        top[j].logit = logits[i];
    }

    printf("Top-20 next token predictions:\n");
    printf("%-6s  %-10s  %-10s  %s\n", "Rank", "Token ID", "Logit", "Token");
    printf("------  ----------  ----------  --------\n");
    for (int i = 0; i < 20 && top[i].id >= 0; i++) {
        const char *piece = gemma3_decode_token(tok, top[i].id);
        printf("%-6d  %-10d  %10.4f  '%s'\n", i + 1, top[i].id, top[i].logit,
               piece ? piece : "(null)");
    }

    free(tokens);
    free(logits);
    return 0;
}

/* ============================================================================
 * Single Prompt Mode
 * ========================================================================== */

static int run_single_prompt(gemma3_ctx *ctx, const cli_config *config) {
    gemma3_gen_params params = make_params(config);

    gemma3_message messages[2];
    int num_messages = 0;
    if (config->system_prompt && config->system_prompt[0]) {
        messages[num_messages++] = (gemma3_message){ GEMMA3_ROLE_SYSTEM, config->system_prompt };
    }
    messages[num_messages++] = (gemma3_message){ GEMMA3_ROLE_USER, config->prompt };

    stream_state st = { gemma3_get_tokenizer(ctx), 1, 0 };
    g_interrupted = 0;
    char *response = gemma3_chat(ctx, messages, num_messages, &params, stream_callback, &st);
    printf("\n");
    fflush(stdout);

    if (!response) {
        fprintf(stderr, "Error: %s\n", gemma3_get_error());
        return 1;
    }
    if (config->show_stats) print_stats(ctx, 0);
    free(response);
    return 0;
}

/* ============================================================================
 * Interactive Chat Mode
 * ========================================================================== */

typedef struct {
    gemma3_message *items;
    int count;
    int cap;
} history;

static int history_push(history *h, gemma3_role role, const char *text) {
    if (h->count == h->cap) {
        int ncap = h->cap ? h->cap * 2 : 16;
        gemma3_message *n = (gemma3_message *)realloc(h->items, (size_t)ncap * sizeof(*n));
        if (!n) return 0;
        h->items = n;
        h->cap = ncap;
    }
    char *copy = strdup(text);
    if (!copy) return 0;
    h->items[h->count++] = (gemma3_message){ role, copy };
    return 1;
}

static void history_clear(history *h, int keep) {
    for (int i = keep; i < h->count; i++) free((void *)h->items[i].content);
    h->count = keep;
}

/* Drop the oldest user/model exchange (after the optional system message) */
static int history_drop_oldest(history *h, int first) {
    if (h->count - first < 3) return 0;  /* keep at least the latest user turn */
    free((void *)h->items[first].content);
    free((void *)h->items[first + 1].content);
    memmove(h->items + first, h->items + first + 2,
            (size_t)(h->count - first - 2) * sizeof(gemma3_message));
    h->count -= 2;
    return 1;
}

/* Number of tokens the conversation occupies when formatted */
static int history_tokens(gemma3_ctx *ctx, const history *h) {
    gemma3_tokenizer *tok = gemma3_get_tokenizer(ctx);
    char *text = gemma3_format_chat(tok, h->items, h->count);
    if (!text) return -1;
    size_t len = strlen(text);
    int *buf = (int *)malloc((len + 2) * sizeof(int));
    int n = buf ? gemma3_tokenize(tok, text, buf, (int)len + 2, 0, 0) : -1;
    free(buf);
    free(text);
    return n;
}

/* Read one (possibly multi-line) input. Lines ending in '\' continue.
 * Returns 1 on input, 0 on EOF, -1 on Ctrl+C. */
static int read_input(char **out) {
    size_t cap = 1024, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return 0;
    buf[0] = '\0';
    for (;;) {
        char line[4096];
        if (!fgets(line, sizeof(line), stdin)) {
            if (g_interrupted || errno == EINTR) {
                clearerr(stdin);
                free(buf);
                return -1;
            }
            if (len > 0) break;
            free(buf);
            return 0;
        }
        size_t ll = strlen(line);
        int continued = 0;
        if (ll > 0 && line[ll - 1] == '\n') {
            line[--ll] = '\0';
            if (ll > 0 && line[ll - 1] == '\\') {
                line[--ll] = '\n';
                line[++ll] = '\0';
                continued = 1;
            }
        } else if (!feof(stdin)) {
            continued = 1;  /* very long line: keep reading */
        }
        if (len + ll + 1 > cap) {
            while (len + ll + 1 > cap) cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return 0; }
            buf = nb;
        }
        memcpy(buf + len, line, ll + 1);
        len += ll;
        if (!continued) break;
        printf("%s... %s", DIM, RESET);
        fflush(stdout);
    }
    *out = buf;
    return 1;
}

static void print_chat_help(void) {
    printf("%sCommands:%s\n", BOLD, RESET);
    printf("  /clear           Start a new conversation\n");
    printf("  /system <text>   Set the system prompt (and start over)\n");
    printf("  /stats           Toggle timing stats after each reply\n");
    printf("  /help            Show this help\n");
    printf("  /exit, /quit     Leave (or press Ctrl+D)\n");
    printf("  End a line with \\ to continue typing on the next line.\n");
    printf("  Ctrl+C stops the current reply.\n\n");
}

static int run_interactive(gemma3_ctx *ctx, cli_config *config) {
    gemma3_gen_params params = make_params(config);
    history h = { NULL, 0, 0 };
    char *system_owned = NULL;
    int first = 0;

    if (config->system_prompt && config->system_prompt[0]) {
        if (!history_push(&h, GEMMA3_ROLE_SYSTEM, config->system_prompt)) return 1;
        first = 1;
    }

    printf("%sChat with Gemma 3 4B%s %s(%s, context %d) - /help for commands, Ctrl+D to exit%s\n",
           BOLD, RESET, DIM, gemma3_backend_name(ctx), gemma3_get_config(ctx)->max_context, RESET);
    if (first) printf("%ssystem: %s%s\n", DIM, config->system_prompt, RESET);
    printf("\n");

    for (;;) {
        printf("%s%s>%s ", BOLD, GREEN, RESET);
        fflush(stdout);
        g_interrupted = 0;

        char *input = NULL;
        int r = read_input(&input);
        if (r == 0) { printf("\n"); break; }
        if (r < 0) { printf("\n%s(Ctrl+D or /exit to quit)%s\n", DIM, RESET); continue; }

        /* Trim surrounding whitespace */
        char *in = input;
        while (*in == ' ' || *in == '\t' || *in == '\n') in++;
        size_t len = strlen(in);
        while (len > 0 && (in[len - 1] == ' ' || in[len - 1] == '\t' || in[len - 1] == '\n')) in[--len] = '\0';
        if (len == 0) { free(input); continue; }

        if (!strcmp(in, "/exit") || !strcmp(in, "/quit") || !strcmp(in, "exit") || !strcmp(in, "quit")) {
            free(input);
            break;
        }
        if (!strcmp(in, "/clear") || !strcmp(in, "clear")) {
            history_clear(&h, first);
            gemma3_reset_cache(ctx);
            printf("%s(conversation cleared)%s\n\n", DIM, RESET);
            free(input);
            continue;
        }
        if (!strcmp(in, "/help") || !strcmp(in, "/?")) {
            print_chat_help();
            free(input);
            continue;
        }
        if (!strcmp(in, "/stats")) {
            config->show_stats = !config->show_stats;
            printf("%s(stats %s)%s\n\n", DIM, config->show_stats ? "on" : "off", RESET);
            free(input);
            continue;
        }
        if (!strncmp(in, "/system", 7) && (in[7] == ' ' || in[7] == '\0')) {
            const char *text = in + 7;
            while (*text == ' ') text++;
            history_clear(&h, 0);
            first = 0;
            free(system_owned);
            system_owned = NULL;
            if (*text) {
                system_owned = strdup(text);
                history_push(&h, GEMMA3_ROLE_SYSTEM, text);
                first = 1;
            }
            gemma3_reset_cache(ctx);
            printf("%s(system prompt %s; conversation cleared)%s\n\n", DIM, *text ? "set" : "removed", RESET);
            free(input);
            continue;
        }
        if (in[0] == '/') {
            printf("%sUnknown command %s - try /help%s\n\n", YELLOW, in, RESET);
            free(input);
            continue;
        }

        if (!history_push(&h, GEMMA3_ROLE_USER, in)) { free(input); break; }
        free(input);

        /* Keep the conversation within the context window: drop the oldest
         * exchanges so the prompt plus a full reply still fit. */
        int budget = gemma3_get_config(ctx)->max_context - params.max_tokens;
        if (budget < gemma3_get_config(ctx)->max_context / 2) budget = gemma3_get_config(ctx)->max_context / 2;
        int dropped = 0;
        while (history_tokens(ctx, &h) > budget && history_drop_oldest(&h, first)) dropped++;
        if (dropped) {
            printf("%s(context full: forgot the %d oldest exchange%s)%s\n", DIM, dropped,
                   dropped > 1 ? "s" : "", RESET);
        }

        printf("%s", CYAN);
        fflush(stdout);
        stream_state st = { gemma3_get_tokenizer(ctx), 1, 0 };
        g_interrupted = 0;
        char *response = gemma3_chat(ctx, h.items, h.count, &params, stream_callback, &st);
        printf("%s\n", RESET);
        if (g_interrupted) printf("%s(interrupted)%s\n", DIM, RESET);

        if (!response) {
            if (!g_interrupted) fprintf(stderr, "%sError: %s%s\n", YELLOW, gemma3_get_error(), RESET);
            free((void *)h.items[--h.count].content);
            printf("\n");
            continue;
        }

        history_push(&h, GEMMA3_ROLE_MODEL, response);
        free(response);
        if (config->show_stats) print_stats(ctx, 1);
        printf("\n");
    }

    history_clear(&h, 0);
    free(h.items);
    free(system_owned);
    return 0;
}

/* ============================================================================
 * Main
 * ========================================================================== */

int main(int argc, char **argv) {
    cli_config config;
    int pr = parse_args(argc, argv, &config);
    if (pr < 0) return 0;
    if (pr == 0) return 2;

    g_color = config.color >= 0 ? config.color
            : (isatty(STDOUT_FILENO) && !getenv("NO_COLOR"));

    char *owned_prompt = NULL;
    if (!resolve_prompt(&config, &owned_prompt)) return 1;

    int debug_mode = config.tokenize_mode || config.detokenize_mode || config.logits_mode;
    if (debug_mode && !config.prompt) {
        fprintf(stderr, "Error: --tokenize/--detokenize/--logits need a prompt (-p, -f or stdin)\n");
        return 2;
    }
    if (!debug_mode && !config.prompt && !config.interactive) {
        print_usage(stderr, argv[0]);
        fprintf(stderr, "\nError: give a prompt with -p, -f or stdin, or use -i for chat\n");
        return 2;
    }
    if (!check_model_dir(config.model_dir)) return 1;

    install_signal_handler();

    gemma3_load_options opts = gemma3_default_load_options();
    opts.max_context = config.context_size;
    opts.num_threads = config.num_threads;
    opts.use_gpu = config.use_gpu;
    opts.verbose = config.verbose;

    gemma3_ctx *ctx = gemma3_load_dir_opts(config.model_dir, &opts);
    if (!ctx) {
        fprintf(stderr, "Error: failed to load model: %s\n", gemma3_get_error());
        free(owned_prompt);
        return 1;
    }
    g_ctx = ctx;

    if (!config.quiet && !debug_mode) {
        fprintf(stderr, "%sgemma3.c %s | %s | context %d | loaded in %.2f s%s\n", DIM,
                gemma3_version(), gemma3_backend_name(ctx), gemma3_get_config(ctx)->max_context,
                gemma3_get_stats(ctx)->load_ms / 1000.0, RESET);
    }

    if (config.verbose) {
        const gemma3_config *mc = gemma3_get_config(ctx);
        fprintf(stderr, "Model configuration:\n");
        fprintf(stderr, "  Vocab size: %d\n", mc->vocab_size);
        fprintf(stderr, "  Hidden size: %d\n", mc->hidden_size);
        fprintf(stderr, "  Layers: %d\n", mc->num_layers);
        fprintf(stderr, "  Heads: %d (KV: %d)\n", mc->num_heads, mc->num_kv_heads);
        fprintf(stderr, "  Head dim: %d\n", mc->head_dim);
        fprintf(stderr, "  Context: %d\n", mc->max_context);
        fprintf(stderr, "  RoPE scaling (global): %.1f\n\n", mc->rope_scale_global);
    }

    int result;
    if (config.tokenize_mode) result = run_tokenize_mode(ctx, &config);
    else if (config.detokenize_mode) result = run_detokenize_mode(ctx, &config);
    else if (config.logits_mode) result = run_logits_mode(ctx, &config);
    else if (config.interactive) result = run_interactive(ctx, &config);
    else result = run_single_prompt(ctx, &config);

    g_ctx = NULL;
    gemma3_free(ctx);
    free(owned_prompt);
    return result;
}
