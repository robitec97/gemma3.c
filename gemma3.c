/*
 * gemma3.c - Main library implementation
 *
 * Ties together model loading, tokenization, and generation.
 * Implements the public API defined in gemma3.h
 */

#include "gemma3.h"
#include "gemma3_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

/* ============================================================================
 * Version
 * ========================================================================== */

#define GEMMA3_VERSION "0.1.0"

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

/* ============================================================================
 * Forward Declarations (from other translation units)
 * ========================================================================== */

/* From gemma3_safetensors.c */
typedef struct st_context st_context;
st_context *st_load(const char *model_dir);
void st_free(st_context *ctx);
void st_print_info(st_context *ctx);

/* gemma3_weights_t is defined in gemma3_safetensors.c */
typedef struct gemma3_weights_t gemma3_weights_t;
gemma3_weights_t *gemma3_load_weights(st_context *st);
void gemma3_free_weights(gemma3_weights_t *w);

/* From gemma3_tokenizer.c */
gemma3_tokenizer *gemma3_tokenizer_load(const char *path);
void gemma3_tokenizer_free(gemma3_tokenizer *tok);

/* From gemma3_transformer.c */
typedef struct gemma3_transformer gemma3_transformer;
gemma3_transformer *gemma3_transformer_create(gemma3_weights_t *weights,
                                               const gemma3_config *cfg,
                                               int max_context);
void gemma3_transformer_destroy(gemma3_transformer *t);
int gemma3_transformer_forward_token(gemma3_transformer *t, int token_id,
                                      int pos, float *logits);
int gemma3_transformer_prefill_tokens(gemma3_transformer *t, const int *tokens,
                                       int num_tokens, int start_pos, float *logits);
void gemma3_transformer_reset(gemma3_transformer *t);
int gemma3_transformer_get_pos(gemma3_transformer *t);

/* ============================================================================
 * Context Structure
 * ========================================================================== */

struct gemma3_ctx {
    gemma3_config config;
    st_context *safetensors;
    gemma3_weights_t *weights;
    gemma3_tokenizer *tokenizer;
    gemma3_transformer *transformer;
    float *logits_buf;  /* [vocab_size] */
    float *probs_buf;   /* [vocab_size] */
    int max_context;
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
    };
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
    };
}

/* ============================================================================
 * Model Loading
 * ========================================================================== */

gemma3_ctx *gemma3_load_dir_ex(const char *model_dir, int max_context) {
    if (!model_dir) {
        set_error("Invalid model directory");
        return NULL;
    }

    gemma3_ctx *ctx = (gemma3_ctx *)calloc(1, sizeof(gemma3_ctx));
    if (!ctx) {
        set_error("Out of memory");
        return NULL;
    }

    ctx->config = default_config();
    ctx->max_context = max_context > 0 ? max_context : GEMMA3_DEFAULT_CONTEXT;
    ctx->config.max_context = ctx->max_context;

    /* Load SafeTensors weights */
    fprintf(stderr, "Loading model from %s...\n", model_dir);
    ctx->safetensors = st_load(model_dir);
    if (!ctx->safetensors) {
        set_error("Failed to load SafeTensors from %s", model_dir);
        free(ctx);
        return NULL;
    }

    /* Load weights into memory */
    fprintf(stderr, "Loading weights...\n");
    ctx->weights = gemma3_load_weights(ctx->safetensors);
    if (!ctx->weights) {
        set_error("Failed to load weights");
        st_free(ctx->safetensors);
        free(ctx);
        return NULL;
    }

    /* Load tokenizer */
    char tokenizer_path[1024];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.model", model_dir);

    fprintf(stderr, "Loading tokenizer from %s...\n", tokenizer_path);
    ctx->tokenizer = gemma3_tokenizer_load(tokenizer_path);
    if (!ctx->tokenizer) {
        set_error("Failed to load tokenizer from %s", tokenizer_path);
        gemma3_free_weights(ctx->weights);
        st_free(ctx->safetensors);
        free(ctx);
        return NULL;
    }

    /* Create transformer */
    fprintf(stderr, "Initializing transformer (max context: %d)...\n", ctx->max_context);
    ctx->transformer = gemma3_transformer_create(ctx->weights, &ctx->config, ctx->max_context);
    if (!ctx->transformer) {
        set_error("Failed to create transformer");
        gemma3_tokenizer_free(ctx->tokenizer);
        gemma3_free_weights(ctx->weights);
        st_free(ctx->safetensors);
        free(ctx);
        return NULL;
    }

    /* Allocate output buffers */
    ctx->logits_buf = (float *)malloc(ctx->config.vocab_size * sizeof(float));
    ctx->probs_buf = (float *)malloc(ctx->config.vocab_size * sizeof(float));
    if (!ctx->logits_buf || !ctx->probs_buf) {
        set_error("Failed to allocate output buffers");
        gemma3_transformer_destroy(ctx->transformer);
        gemma3_tokenizer_free(ctx->tokenizer);
        gemma3_free_weights(ctx->weights);
        st_free(ctx->safetensors);
        free(ctx->logits_buf);
        free(ctx->probs_buf);
        free(ctx);
        return NULL;
    }

    fprintf(stderr, "Model loaded successfully!\n");
    return ctx;
}

gemma3_ctx *gemma3_load_dir(const char *model_dir) {
    return gemma3_load_dir_ex(model_dir, GEMMA3_DEFAULT_CONTEXT);
}

void gemma3_free(gemma3_ctx *ctx) {
    if (!ctx) return;

    free(ctx->logits_buf);
    free(ctx->probs_buf);
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

/* ============================================================================
 * KV Cache Management
 * ========================================================================== */

void gemma3_reset_cache(gemma3_ctx *ctx) {
    if (ctx && ctx->transformer) {
        gemma3_transformer_reset(ctx->transformer);
    }
}

int gemma3_get_cache_position(gemma3_ctx *ctx) {
    return (ctx && ctx->transformer) ? gemma3_transformer_get_pos(ctx->transformer) : 0;
}

/* ============================================================================
 * Forward Pass
 * ========================================================================== */

int gemma3_forward(gemma3_ctx *ctx, int token_id, int pos, float *logits) {
    if (!ctx || !logits) return GEMMA3_ERR_INVALID_ARG;

    return gemma3_transformer_forward_token(ctx->transformer, token_id, pos, logits);
}

int gemma3_forward_batch(gemma3_ctx *ctx, const int *tokens, int num_tokens,
                         int start_pos, float *logits) {
    if (!ctx || !tokens || !logits || num_tokens <= 0) {
        return GEMMA3_ERR_INVALID_ARG;
    }

    return gemma3_transformer_prefill_tokens(ctx->transformer, tokens, num_tokens,
                                             start_pos, logits);
}

/* ============================================================================
 * Sampling
 * ========================================================================== */

static int sample_token(gemma3_ctx *ctx, float *logits, const gemma3_gen_params *params) {
    int vocab_size = ctx->config.vocab_size;

    /* Copy logits to working buffer */
    memcpy(ctx->probs_buf, logits, vocab_size * sizeof(float));

    /* Greedy sampling: explicit flag or temperature == 0 */
    if (params->greedy || params->temperature == 0.0f) {
        return gemma3_argmax(ctx->probs_buf, vocab_size);
    }

    /* Apply temperature */
    if (params->temperature != 1.0f) {
        gemma3_apply_temperature(ctx->probs_buf, vocab_size, params->temperature);
    }

    /* Apply top-k filtering */
    if (params->top_k > 0 && params->top_k < vocab_size) {
        gemma3_topk_filter(ctx->probs_buf, vocab_size, params->top_k);
    }

    /* Apply top-p filtering */
    if (params->top_p > 0.0f && params->top_p < 1.0f) {
        gemma3_topp_filter(ctx->probs_buf, vocab_size, params->top_p);
    }

    /* Convert to probabilities */
    gemma3_softmax_inplace(ctx->probs_buf, vocab_size);

    /* Sample */
    return gemma3_sample(ctx->probs_buf, vocab_size);
}

/* ============================================================================
 * Text Generation
 * ========================================================================== */

char *gemma3_generate_tokens(gemma3_ctx *ctx, const int *tokens, int num_tokens,
                             gemma3_gen_params *params,
                             gemma3_token_callback callback, void *user_data) {
    if (!ctx || !tokens || num_tokens <= 0) {
        set_error("Invalid arguments");
        return NULL;
    }

    gemma3_gen_params p = params ? *params : gemma3_default_params();

    /* Initialize random seed */
    if (p.seed < 0) {
        gemma3_set_seed((uint64_t)time(NULL));
    } else {
        gemma3_set_seed((uint64_t)p.seed);
    }

    /* Reset KV cache */
    gemma3_reset_cache(ctx);

    /* Prefill with input tokens */
    if (gemma3_forward_batch(ctx, tokens, num_tokens, 0, ctx->logits_buf) != 0) {
        set_error("Prefill failed");
        return NULL;
    }

    int pos = num_tokens;

    /* Output buffer for generated tokens */
    int max_gen = p.max_tokens;
    int *gen_tokens = (int *)malloc(max_gen * sizeof(int));
    if (!gen_tokens) {
        set_error("Out of memory");
        return NULL;
    }
    int n_gen = 0;

    int eos_id = gemma3_eos_token(ctx->tokenizer);
    int end_turn_id = gemma3_end_turn_token(ctx->tokenizer);

    /* Generation loop */
    while (n_gen < max_gen) {
        /* Sample next token */
        int next_token = sample_token(ctx, ctx->logits_buf, &p);

        /* Verbose token debug output */
        if (p.verbose_tokens) {
            const char *tok_str = gemma3_decode_token(ctx->tokenizer, next_token);
            fprintf(stderr, "[DEBUG] pos=%d token=%d '%s'\n", pos, next_token, tok_str ? tok_str : "");
        }

        /* Check for EOS or end-of-turn (Gemma 3 IT uses <end_of_turn>) */
        if (p.stop_on_eos && (next_token == eos_id || next_token == end_turn_id)) {
            break;
        }

        /* Store token */
        gen_tokens[n_gen++] = next_token;

        /* Callback for streaming */
        if (callback) {
            const char *token_str = gemma3_decode_token(ctx->tokenizer, next_token);
            int stop = callback(next_token, token_str ? token_str : "", user_data);
            if (stop) break;
        }

        /* Check context limit */
        if (pos >= ctx->max_context - 1) {
            set_error("Context overflow");
            break;
        }

        /* Forward pass for next token */
        if (gemma3_forward(ctx, next_token, pos, ctx->logits_buf) != 0) {
            set_error("Forward pass failed");
            break;
        }
        pos++;
    }

    /* Decode generated tokens to text */
    char *output = gemma3_detokenize(ctx->tokenizer, gen_tokens, n_gen);
    free(gen_tokens);

    return output;
}

char *gemma3_generate(gemma3_ctx *ctx, const char *prompt,
                      gemma3_gen_params *params,
                      gemma3_token_callback callback, void *user_data) {
    if (!ctx || !prompt) {
        set_error("Invalid arguments");
        return NULL;
    }

    /* Tokenize prompt */
    int max_tokens = ctx->max_context;
    int *tokens = (int *)malloc(max_tokens * sizeof(int));
    if (!tokens) {
        set_error("Out of memory");
        return NULL;
    }

    int num_tokens = gemma3_tokenize(ctx->tokenizer, prompt, tokens, max_tokens, 1, 0);
    if (num_tokens < 0) {
        set_error("Tokenization failed");
        free(tokens);
        return NULL;
    }

    /* Generate */
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

    /* Format messages with chat template */
    char *formatted = gemma3_format_chat(ctx->tokenizer, messages, num_msgs);
    if (!formatted) {
        set_error("Failed to format chat messages");
        return NULL;
    }

    /* Generate response */
    char *response = gemma3_generate(ctx, formatted, params, callback, user_data);
    free(formatted);

    return response;
}
