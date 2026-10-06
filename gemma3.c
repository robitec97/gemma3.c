/*
 * gemma3.c - Main library implementation
 *
 * Ties together model loading, tokenization, and generation.
 * Implements the public API defined in gemma3.h
 */

#include "gemma3_internal.h"
#include "gemma3_kernels.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <time.h>

/* ============================================================================
 * Version
 * ========================================================================== */

#define GEMMA3_VERSION "0.2.0"

const char *gemma3_version(void) {
    return GEMMA3_VERSION;
}

/* ============================================================================
 * Error Handling
 * ========================================================================== */

static _Thread_local char g_error_msg[512] = {0};

static void set_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_msg, sizeof(g_error_msg), fmt, args);
    va_end(args);
}

const char *gemma3_get_error(void) {
    return g_error_msg;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

/* ============================================================================
 * Context Structure
 * ========================================================================== */

struct gemma3_ctx {
    gemma3_config config;
    st_context *safetensors;
    gemma3_weights_t *weights;
    gemma3_tokenizer *tokenizer;
    gemma3_transformer *transformer;
    gemma3_sampler *sampler;
    float *logits_buf;      /* [vocab_size] */
    int max_context;

    /* Tokens whose K/V are in the cache, in position order. Lets the next
     * generation skip the prefix it shares with the previous one. */
    int *cached_tokens;     /* [max_context] */
    int n_cached;

    uint64_t rng_state;
    volatile int abort_flag;
    gemma3_stats stats;
    char backend_name[64];
};

/* ============================================================================
 * Default Configuration
 * ========================================================================== */

static gemma3_config default_config(void) {
    return (gemma3_config){
        .vocab_size = GEMMA3_VOCAB_SIZE,
        .hidden_size = GEMMA3_HIDDEN_SIZE,
        .intermediate_size = GEMMA3_INTERMEDIATE_SIZE,
        .num_layers = GEMMA3_NUM_LAYERS,
        .num_heads = GEMMA3_NUM_HEADS,
        .num_kv_heads = GEMMA3_NUM_KV_HEADS,
        .head_dim = GEMMA3_HEAD_DIM,
        .max_context = GEMMA3_DEFAULT_CONTEXT,
        .sliding_window = GEMMA3_SLIDING_WINDOW,
        .rmsnorm_eps = GEMMA3_RMSNORM_EPS,
        .rope_theta_local = GEMMA3_ROPE_THETA_LOCAL,
        .rope_theta_global = GEMMA3_ROPE_THETA_GLOBAL,
        .rope_scale_global = GEMMA3_ROPE_SCALE_GLOBAL,
    };
}

/* Read the few config.json values that matter at runtime. The file is optional;
 * missing keys keep the Gemma 3 4B defaults. */
static void read_config_json(const char *model_dir, gemma3_config *cfg) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    char buf[65536];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';

    /* "rope_scaling": {"factor": 8.0, "rope_type": "linear"}  (or null) */
    const char *rs = strstr(buf, "\"rope_scaling\"");
    if (rs) {
        const char *colon = strchr(rs, ':');
        while (colon && (*++colon == ' ' || *colon == '\t' || *colon == '\n' || *colon == '\r')) {}
        if (colon && strncmp(colon, "null", 4) == 0) {
            cfg->rope_scale_global = 1.0f;
        } else if (colon && *colon == '{') {
            const char *close = strchr(colon, '}');
            const char *fac = strstr(colon, "\"factor\"");
            if (fac && close && fac < close && (fac = strchr(fac, ':'))) {
                float v = strtof(fac + 1, NULL);
                if (v > 0.0f) cfg->rope_scale_global = v;
            }
        }
    }
}

gemma3_gen_params gemma3_default_params(void) {
    return (gemma3_gen_params){
        .max_tokens = 512,
        .temperature = 0.7f,
        .top_k = 50,
        .top_p = 0.9f,
        .seed = -1,
        .stop_on_eos = 1,
        .greedy = 0,
        .verbose_tokens = 0,
        .min_p = 0.0f,
    };
}

gemma3_load_options gemma3_default_load_options(void) {
    return (gemma3_load_options){
        .max_context = GEMMA3_DEFAULT_CONTEXT,
        .num_threads = 0,
        .use_gpu = 1,
        .verbose = 0,
    };
}

/* ============================================================================
 * Model Loading
 * ========================================================================== */

gemma3_ctx *gemma3_load_dir_opts(const char *model_dir, const gemma3_load_options *opts_in) {
    gemma3_load_options opts = opts_in ? *opts_in : gemma3_default_load_options();
    if (!model_dir) {
        set_error("Invalid model directory");
        return NULL;
    }
    if (opts.max_context <= 0) opts.max_context = GEMMA3_DEFAULT_CONTEXT;
    if (opts.max_context > GEMMA3_MAX_CONTEXT) opts.max_context = GEMMA3_MAX_CONTEXT;

    double t0 = now_ms();
    gemma3_ctx *ctx = (gemma3_ctx *)calloc(1, sizeof(gemma3_ctx));
    if (!ctx) {
        set_error("Out of memory");
        return NULL;
    }

    ctx->config = default_config();
    read_config_json(model_dir, &ctx->config);
    ctx->max_context = opts.max_context;
    ctx->config.max_context = ctx->max_context;

    if (opts.verbose) fprintf(stderr, "Loading model from %s...\n", model_dir);
    ctx->safetensors = st_load(model_dir);
    if (!ctx->safetensors) {
        set_error("No readable .safetensors files in %s", model_dir);
        gemma3_free(ctx);
        return NULL;
    }

    ctx->weights = gemma3_load_weights(ctx->safetensors);
    if (!ctx->weights) {
        set_error("Failed to load weights (see messages above)");
        gemma3_free(ctx);
        return NULL;
    }

    char tokenizer_path[1024];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.model", model_dir);
    if (opts.verbose) fprintf(stderr, "Loading tokenizer from %s...\n", tokenizer_path);
    ctx->tokenizer = gemma3_tokenizer_load(tokenizer_path);
    if (!ctx->tokenizer) {
        set_error("Failed to load tokenizer from %s", tokenizer_path);
        gemma3_free(ctx);
        return NULL;
    }

    if (opts.verbose) {
        fprintf(stderr, "Initializing transformer (max context: %d)...\n", ctx->max_context);
    }
    /* The Metal backend honours GEMMA3_NO_METAL; map use_gpu=0 onto it. */
    if (!opts.use_gpu) setenv("GEMMA3_NO_METAL", "1", 1);
    ctx->transformer = gemma3_transformer_create(ctx->weights, &ctx->config,
                                                 ctx->max_context, opts.num_threads);
    if (!ctx->transformer) {
        set_error("Failed to create transformer (out of memory for context %d?)",
                  ctx->max_context);
        gemma3_free(ctx);
        return NULL;
    }
    gemma3_transformer_set_abort_flag(ctx->transformer, &ctx->abort_flag);

    ctx->sampler = gemma3_sampler_create(ctx->config.vocab_size);
    ctx->logits_buf = (float *)malloc((size_t)ctx->config.vocab_size * sizeof(float));
    ctx->cached_tokens = (int *)malloc((size_t)ctx->max_context * sizeof(int));
    if (!ctx->sampler || !ctx->logits_buf || !ctx->cached_tokens) {
        set_error("Failed to allocate output buffers");
        gemma3_free(ctx);
        return NULL;
    }

    const char *backend = gemma3_transformer_backend(ctx->transformer);
    if (strcmp(backend, "cpu") == 0) {
        snprintf(ctx->backend_name, sizeof(ctx->backend_name), "cpu (%s, %d threads)",
                 gemma3_simd_name(), gemma3_transformer_num_threads(ctx->transformer));
    } else {
        snprintf(ctx->backend_name, sizeof(ctx->backend_name), "%s", backend);
    }

    ctx->rng_state = 0x853C49E6748FEA9BULL;
    ctx->stats.load_ms = now_ms() - t0;
    if (opts.verbose) fprintf(stderr, "Model loaded in %.0f ms\n", ctx->stats.load_ms);
    return ctx;
}

gemma3_ctx *gemma3_load_dir_ex(const char *model_dir, int max_context) {
    gemma3_load_options opts = gemma3_default_load_options();
    opts.max_context = max_context;
    opts.verbose = 1;
    return gemma3_load_dir_opts(model_dir, &opts);
}

gemma3_ctx *gemma3_load_dir(const char *model_dir) {
    return gemma3_load_dir_ex(model_dir, GEMMA3_DEFAULT_CONTEXT);
}

void gemma3_free(gemma3_ctx *ctx) {
    if (!ctx) return;
    free(ctx->logits_buf);
    free(ctx->cached_tokens);
    gemma3_sampler_free(ctx->sampler);
    gemma3_transformer_destroy(ctx->transformer);
    gemma3_tokenizer_free(ctx->tokenizer);
    gemma3_free_weights(ctx->weights);
    st_free(ctx->safetensors);
    free(ctx);
}

const gemma3_config *gemma3_get_config(const gemma3_ctx *ctx) {
    return ctx ? &ctx->config : NULL;
}

gemma3_tokenizer *gemma3_get_tokenizer(gemma3_ctx *ctx) {
    return ctx ? ctx->tokenizer : NULL;
}

const char *gemma3_backend_name(const gemma3_ctx *ctx) {
    return ctx ? ctx->backend_name : "";
}

const gemma3_stats *gemma3_get_stats(const gemma3_ctx *ctx) {
    return ctx ? &ctx->stats : NULL;
}

void gemma3_abort(gemma3_ctx *ctx) {
    if (ctx) ctx->abort_flag = 1;
}

/* ============================================================================
 * KV Cache Management
 * ========================================================================== */

void gemma3_reset_cache(gemma3_ctx *ctx) {
    if (ctx && ctx->transformer) {
        gemma3_transformer_reset(ctx->transformer);
        ctx->n_cached = 0;
    }
}

int gemma3_get_cache_position(gemma3_ctx *ctx) {
    return (ctx && ctx->transformer) ? gemma3_transformer_get_pos(ctx->transformer) : 0;
}

/* Record tokens written to the cache at [start, start + n) */
static void track_cached(gemma3_ctx *ctx, const int *tokens, int n, int start) {
    if (start > ctx->n_cached || start + n > ctx->max_context) {
        ctx->n_cached = 0;  /* gap: we no longer know what is cached */
        return;
    }
    memcpy(ctx->cached_tokens + start, tokens, (size_t)n * sizeof(int));
    ctx->n_cached = start + n;
}

/* ============================================================================
 * Forward Pass
 * ========================================================================== */

int gemma3_forward(gemma3_ctx *ctx, int token_id, int pos, float *logits) {
    if (!ctx || !logits) return GEMMA3_ERR_INVALID_ARG;
    int ret = gemma3_transformer_forward_token(ctx->transformer, token_id, pos, logits);
    if (ret == 0) track_cached(ctx, &token_id, 1, pos);
    else ctx->n_cached = 0;
    return ret;
}

int gemma3_forward_batch(gemma3_ctx *ctx, const int *tokens, int num_tokens,
                         int start_pos, float *logits) {
    if (!ctx || !tokens || !logits || num_tokens <= 0) {
        return GEMMA3_ERR_INVALID_ARG;
    }
    int ret = gemma3_transformer_prefill_tokens(ctx->transformer, tokens, num_tokens,
                                                start_pos, logits);
    if (ret == 0) track_cached(ctx, tokens, num_tokens, start_pos);
    else ctx->n_cached = 0;
    return ret;
}

/* ============================================================================
 * Text Generation
 * ========================================================================== */

static int sample_next(gemma3_ctx *ctx, const gemma3_gen_params *p) {
    float temp = p->greedy ? 0.0f : p->temperature;
    return gemma3_sample_logits(ctx->sampler, ctx->logits_buf, ctx->config.vocab_size,
                                temp, p->top_k, p->top_p, p->min_p, &ctx->rng_state);
}

char *gemma3_generate_tokens(gemma3_ctx *ctx, const int *tokens, int num_tokens,
                             gemma3_gen_params *params,
                             gemma3_token_callback callback, void *user_data) {
    if (!ctx || !tokens || num_tokens <= 0) {
        set_error("Invalid arguments");
        return NULL;
    }
    if (num_tokens >= ctx->max_context) {
        set_error("Prompt is %d tokens but the context size is %d (use a larger -c / max_context)",
                  num_tokens, ctx->max_context);
        return NULL;
    }

    gemma3_gen_params p = params ? *params : gemma3_default_params();
    ctx->abort_flag = 0;
    memset(&ctx->stats.prompt_tokens, 0,
           sizeof(gemma3_stats) - offsetof(gemma3_stats, prompt_tokens));
    ctx->stats.prompt_tokens = num_tokens;

    if (p.seed < 0) {
        ctx->rng_state = (uint64_t)time(NULL) ^ ((uint64_t)(uintptr_t)ctx << 16) ^
                         (uint64_t)(now_ms() * 1000.0);
    } else {
        ctx->rng_state = (uint64_t)p.seed * 0x9E3779B97F4A7C15ULL + 1;
    }

    /* Reuse the longest cached prefix; always recompute at least the last
     * prompt token so we have its logits. */
    int reuse = 0;
    while (reuse < ctx->n_cached && reuse < num_tokens && ctx->cached_tokens[reuse] == tokens[reuse]) {
        reuse++;
    }
    if (reuse >= num_tokens) reuse = num_tokens - 1;
    if (reuse > 0 && !gemma3_transformer_can_rewind(ctx->transformer, ctx->n_cached, reuse)) {
        reuse = 0;
    }
    if (reuse == 0) gemma3_reset_cache(ctx);
    ctx->stats.reused_tokens = reuse;

    double t_start = now_ms();
    int ret = gemma3_transformer_prefill_tokens(ctx->transformer, tokens + reuse,
                                                num_tokens - reuse, reuse, ctx->logits_buf);
    if (ret != 0) {
        ctx->n_cached = 0;
        set_error(ret == GEMMA3_ERR_ABORTED ? "Generation aborted" : "Prefill failed (error %d)", ret);
        return NULL;
    }
    track_cached(ctx, tokens + reuse, num_tokens - reuse, reuse);
    double t_prefill = now_ms();
    ctx->stats.prefill_ms = t_prefill - t_start;
    if (num_tokens - reuse > 0 && ctx->stats.prefill_ms > 0) {
        ctx->stats.prefill_tok_per_s = (num_tokens - reuse) / (ctx->stats.prefill_ms / 1000.0);
    }

    int pos = num_tokens;
    int max_gen = p.max_tokens > 0 ? p.max_tokens : 0;
    if (max_gen > ctx->max_context - num_tokens) max_gen = ctx->max_context - num_tokens;

    int *gen_tokens = (int *)malloc((size_t)(max_gen > 0 ? max_gen : 1) * sizeof(int));
    if (!gen_tokens) {
        set_error("Out of memory");
        return NULL;
    }
    int n_gen = 0;

    int eos_id = gemma3_eos_token(ctx->tokenizer);
    int end_turn_id = gemma3_end_turn_token(ctx->tokenizer);
    double t_first = 0.0;

    while (n_gen < max_gen && !ctx->abort_flag) {
        int next_token = sample_next(ctx, &p);
        if (n_gen == 0) {
            t_first = now_ms();
            ctx->stats.ttft_ms = t_first - t_start;
        }

        if (p.verbose_tokens) {
            const char *tok_str = gemma3_decode_token(ctx->tokenizer, next_token);
            fprintf(stderr, "[DEBUG] pos=%d token=%d '%s'\n", pos, next_token, tok_str ? tok_str : "");
        }

        /* Gemma 3 IT ends its turn with <end_of_turn> */
        if (p.stop_on_eos && (next_token == eos_id || next_token == end_turn_id)) {
            break;
        }

        gen_tokens[n_gen++] = next_token;

        if (callback) {
            const char *token_str = gemma3_decode_token(ctx->tokenizer, next_token);
            if (callback(next_token, token_str ? token_str : "", user_data)) break;
        }

        if (n_gen >= max_gen || pos >= ctx->max_context) {
            if (pos >= ctx->max_context) set_error("Context full");
            break;
        }

        if (gemma3_transformer_forward_token(ctx->transformer, next_token, pos,
                                             ctx->logits_buf) != 0) {
            set_error("Forward pass failed");
            ctx->n_cached = 0;
            break;
        }
        track_cached(ctx, &next_token, 1, pos);
        pos++;
    }

    double t_end = now_ms();
    ctx->stats.generated_tokens = n_gen;
    if (n_gen > 0) {
        ctx->stats.decode_ms = t_end - t_first;
        if (n_gen > 1 && ctx->stats.decode_ms > 0) {
            ctx->stats.decode_tok_per_s = (n_gen - 1) / (ctx->stats.decode_ms / 1000.0);
        }
    }

    char *output = n_gen > 0 ? gemma3_detokenize(ctx->tokenizer, gen_tokens, n_gen)
                             : (char *)calloc(1, 1);
    free(gen_tokens);
    return output;
}

/* Tokenize text, making sure it starts with exactly one BOS token. */
static int *tokenize_with_bos(gemma3_ctx *ctx, const char *text, int *out_n) {
    int max_tokens = ctx->max_context;
    size_t len = strlen(text);
    /* A token is at least one byte, so this bound is always enough */
    if ((size_t)max_tokens < len + 2) max_tokens = (int)(len + 2 < (size_t)INT32_MAX ? len + 2 : INT32_MAX);
    int *tokens = (int *)malloc((size_t)max_tokens * sizeof(int));
    if (!tokens) {
        set_error("Out of memory");
        return NULL;
    }
    int bos = gemma3_bos_token(ctx->tokenizer);
    int n = gemma3_tokenize(ctx->tokenizer, text, tokens + 1, max_tokens - 1, 0, 0);
    if (n < 0) {
        set_error("Tokenization failed");
        free(tokens);
        return NULL;
    }
    if (n > 0 && tokens[1] == bos) {
        memmove(tokens, tokens + 1, (size_t)n * sizeof(int));
    } else {
        tokens[0] = bos;
        n++;
    }
    *out_n = n;
    return tokens;
}

char *gemma3_generate(gemma3_ctx *ctx, const char *prompt,
                      gemma3_gen_params *params,
                      gemma3_token_callback callback, void *user_data) {
    if (!ctx || !prompt) {
        set_error("Invalid arguments");
        return NULL;
    }
    int num_tokens = 0;
    int *tokens = tokenize_with_bos(ctx, prompt, &num_tokens);
    if (!tokens) return NULL;
    char *output = gemma3_generate_tokens(ctx, tokens, num_tokens, params, callback, user_data);
    free(tokens);
    return output;
}

/* ============================================================================
 * Chat Interface
 * ========================================================================== */

char *gemma3_chat(gemma3_ctx *ctx, const gemma3_message *messages, int num_msgs,
                  gemma3_gen_params *params,
                  gemma3_token_callback callback, void *user_data) {
    if (!ctx || !messages || num_msgs <= 0) {
        set_error("Invalid arguments");
        return NULL;
    }

    char *formatted = gemma3_format_chat(ctx->tokenizer, messages, num_msgs);
    if (!formatted) {
        set_error("Failed to format chat messages");
        return NULL;
    }

    /* The template starts with <bos>, which the tokenizer maps to the BOS id */
    char *response = gemma3_generate(ctx, formatted, params, callback, user_data);
    free(formatted);
    return response;
}
