/*
 * gemma3.h - Public API for Gemma 3 4B inference in pure C
 *
 * This library provides zero-dependency inference for Google's Gemma 3 4B IT model.
 * It loads weights directly from SafeTensors format and performs text generation.
 *
 * Example usage:
 *     gemma3_ctx *ctx = gemma3_load_dir("./gemma-3-4b-it");
 *     if (!ctx) {
 *         fprintf(stderr, "Failed to load model\n");
 *         return 1;
 *     }
 *
 *     gemma3_gen_params params = gemma3_default_params();
 *     char *response = gemma3_generate(ctx, "Hello, world!", &params, NULL, NULL);
 *     printf("%s\n", response);
 *     free(response);
 *
 *     gemma3_free(ctx);
 */

#ifndef GEMMA3_H
#define GEMMA3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Model Configuration Constants
 * ========================================================================== */

#define GEMMA3_VOCAB_SIZE       262208
#define GEMMA3_HIDDEN_SIZE      2560
#define GEMMA3_INTERMEDIATE_SIZE 10240
#define GEMMA3_NUM_LAYERS       34
#define GEMMA3_NUM_HEADS        8
#define GEMMA3_NUM_KV_HEADS     4
#define GEMMA3_HEAD_DIM         256
#define GEMMA3_MAX_CONTEXT      131072  /* 128K tokens */
#define GEMMA3_SLIDING_WINDOW   1024
#define GEMMA3_LOCAL_RATIO      5       /* 5 local : 1 global */
#define GEMMA3_RMSNORM_EPS      1e-6f
#define GEMMA3_ROPE_THETA_LOCAL  10000.0f
#define GEMMA3_ROPE_THETA_GLOBAL 1000000.0f

/* Default context size for memory allocation */
#define GEMMA3_DEFAULT_CONTEXT  8192

/* ============================================================================
 * Error Codes
 * ========================================================================== */

typedef enum {
    GEMMA3_OK = 0,
    GEMMA3_ERR_INVALID_ARG = -1,
    GEMMA3_ERR_FILE_NOT_FOUND = -2,
    GEMMA3_ERR_INVALID_FORMAT = -3,
    GEMMA3_ERR_OUT_OF_MEMORY = -4,
    GEMMA3_ERR_MMAP_FAILED = -5,
    GEMMA3_ERR_TOKENIZER_FAILED = -6,
    GEMMA3_ERR_GENERATION_FAILED = -7,
    GEMMA3_ERR_CONTEXT_OVERFLOW = -8,
} gemma3_error;

/* ============================================================================
 * Forward Declarations
 * ========================================================================== */

typedef struct gemma3_ctx gemma3_ctx;
typedef struct gemma3_tokenizer gemma3_tokenizer;
typedef struct gemma3_weights gemma3_weights;
typedef struct gemma3_kv_cache gemma3_kv_cache;

/* ============================================================================
 * Model Configuration
 * ========================================================================== */

typedef struct {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int max_context;
    int sliding_window;
    float rmsnorm_eps;
    float rope_theta_local;
    float rope_theta_global;
} gemma3_config;

/* ============================================================================
 * Generation Parameters
 * ========================================================================== */

typedef struct {
    int max_tokens;         /* Maximum tokens to generate */
    float temperature;      /* Sampling temperature (0 = greedy) */
    int top_k;              /* Top-k sampling (0 = disabled) */
    float top_p;            /* Top-p (nucleus) sampling (1.0 = disabled) */
    int seed;               /* Random seed (-1 for random) */
    int stop_on_eos;        /* Stop when EOS token generated */
    int greedy;             /* Force greedy decoding (overrides temperature) */
    int verbose_tokens;     /* Print token IDs during generation */
} gemma3_gen_params;

/**
 * Get default generation parameters
 * Default: max_tokens=512, temperature=0.7, top_k=50, top_p=0.9
 */
gemma3_gen_params gemma3_default_params(void);

/* ============================================================================
 * Token Callback
 * ========================================================================== */

/**
 * Callback function called for each generated token
 * @param token_id  The token ID that was generated
 * @param token_str The decoded string for this token (may be partial UTF-8)
 * @param user_data User-provided context pointer
 * @return 0 to continue generation, non-zero to stop early
 */
typedef int (*gemma3_token_callback)(int token_id, const char *token_str,
                                      void *user_data);

/* ============================================================================
 * Model Loading and Context
 * ========================================================================== */

/**
 * Load Gemma 3 model from a HuggingFace model directory
 * @param model_dir Path to directory containing:
 *                  - model.safetensors or model-00001-of-00002.safetensors, etc.
 *                  - tokenizer.model (SentencePiece)
 *                  - config.json (optional, uses defaults if missing)
 * @return Context pointer on success, NULL on failure
 */
gemma3_ctx *gemma3_load_dir(const char *model_dir);

/**
 * Load model with custom configuration
 * @param model_dir Path to model directory
 * @param max_context Maximum context length to support (affects memory usage)
 * @return Context pointer on success, NULL on failure
 */
gemma3_ctx *gemma3_load_dir_ex(const char *model_dir, int max_context);

/**
 * Free all resources associated with a context
 */
void gemma3_free(gemma3_ctx *ctx);

/**
 * Get the last error message (thread-local)
 */
const char *gemma3_get_error(void);

/**
 * Get the model configuration
 */
const gemma3_config *gemma3_get_config(const gemma3_ctx *ctx);

/**
 * Get the tokenizer from a context
 */
gemma3_tokenizer *gemma3_get_tokenizer(gemma3_ctx *ctx);

/* ============================================================================
 * Tokenization
 * ========================================================================== */

/**
 * Encode text to token IDs
 * @param tok       Tokenizer from gemma3_get_tokenizer()
 * @param text      Input text (UTF-8)
 * @param tokens    Output array for token IDs
 * @param max_tokens Maximum number of tokens to output
 * @param add_bos   Add beginning-of-sequence token
 * @param add_eos   Add end-of-sequence token
 * @return Number of tokens written, or negative error code
 */
int gemma3_tokenize(gemma3_tokenizer *tok, const char *text,
                    int *tokens, int max_tokens, int add_bos, int add_eos);

/**
 * Decode token IDs to text
 * @param tok       Tokenizer from gemma3_get_tokenizer()
 * @param tokens    Array of token IDs
 * @param num_tokens Number of tokens
 * @return Decoded string (caller must free), or NULL on error
 */
char *gemma3_detokenize(gemma3_tokenizer *tok, const int *tokens, int num_tokens);

/**
 * Decode a single token ID to text
 * @param tok       Tokenizer
 * @param token_id  Token ID to decode
 * @return Token string (pointer to internal storage, do not free), or NULL
 */
const char *gemma3_decode_token(gemma3_tokenizer *tok, int token_id);

/**
 * Get special token IDs
 */
int gemma3_bos_token(gemma3_tokenizer *tok);
int gemma3_eos_token(gemma3_tokenizer *tok);
int gemma3_pad_token(gemma3_tokenizer *tok);
int gemma3_end_turn_token(gemma3_tokenizer *tok);
int gemma3_start_turn_token(gemma3_tokenizer *tok);

/* ============================================================================
 * Text Generation
 * ========================================================================== */

/**
 * Generate text from a prompt
 * @param ctx       Model context
 * @param prompt    Input prompt (raw text, will be tokenized)
 * @param params    Generation parameters (NULL for defaults)
 * @param callback  Optional callback for streaming output
 * @param user_data User data passed to callback
 * @return Generated text (caller must free), or NULL on error
 */
char *gemma3_generate(gemma3_ctx *ctx, const char *prompt,
                      gemma3_gen_params *params,
                      gemma3_token_callback callback, void *user_data);

/**
 * Generate text with pre-tokenized input
 * @param ctx         Model context
 * @param tokens      Input token IDs
 * @param num_tokens  Number of input tokens
 * @param params      Generation parameters
 * @param callback    Optional callback for streaming
 * @param user_data   User data for callback
 * @return Generated text (caller must free), or NULL on error
 */
char *gemma3_generate_tokens(gemma3_ctx *ctx, const int *tokens, int num_tokens,
                             gemma3_gen_params *params,
                             gemma3_token_callback callback, void *user_data);

/* ============================================================================
 * Chat Interface
 * ========================================================================== */

/**
 * Role for chat messages
 */
typedef enum {
    GEMMA3_ROLE_USER,
    GEMMA3_ROLE_MODEL,
    GEMMA3_ROLE_SYSTEM,
} gemma3_role;

/**
 * Chat message structure
 */
typedef struct {
    gemma3_role role;
    const char *content;
} gemma3_message;

/**
 * Generate chat completion using Gemma 3 chat template
 * @param ctx       Model context
 * @param messages  Array of chat messages
 * @param num_msgs  Number of messages
 * @param params    Generation parameters
 * @param callback  Optional callback for streaming
 * @param user_data User data for callback
 * @return Generated response (caller must free), or NULL on error
 */
char *gemma3_chat(gemma3_ctx *ctx, const gemma3_message *messages, int num_msgs,
                  gemma3_gen_params *params,
                  gemma3_token_callback callback, void *user_data);

/**
 * Format messages with Gemma 3 chat template
 * @param tok       Tokenizer
 * @param messages  Array of chat messages
 * @param num_msgs  Number of messages
 * @return Formatted prompt string (caller must free), or NULL on error
 */
char *gemma3_format_chat(gemma3_tokenizer *tok, const gemma3_message *messages,
                         int num_msgs);

/* ============================================================================
 * KV Cache Management
 * ========================================================================== */

/**
 * Reset the KV cache (start fresh generation)
 */
void gemma3_reset_cache(gemma3_ctx *ctx);

/**
 * Get current cache position (number of tokens processed)
 */
int gemma3_get_cache_position(gemma3_ctx *ctx);

/* ============================================================================
 * Low-Level Forward Pass (Advanced Use)
 * ========================================================================== */

/**
 * Run forward pass for a single token
 * @param ctx       Model context
 * @param token_id  Input token ID
 * @param pos       Position in sequence
 * @param logits    Output logits array [vocab_size] (must be pre-allocated)
 * @return 0 on success, negative error code on failure
 */
int gemma3_forward(gemma3_ctx *ctx, int token_id, int pos, float *logits);

/**
 * Run forward pass for multiple tokens (prefill)
 * @param ctx         Model context
 * @param tokens      Input token IDs
 * @param num_tokens  Number of input tokens
 * @param start_pos   Starting position in sequence
 * @param logits      Output logits for last token [vocab_size]
 * @return 0 on success, negative error code on failure
 */
int gemma3_forward_batch(gemma3_ctx *ctx, const int *tokens, int num_tokens,
                         int start_pos, float *logits);

/* ============================================================================
 * Utility Functions
 * ========================================================================== */

/**
 * Get library version string
 */
const char *gemma3_version(void);

/**
 * Check if a layer uses global attention
 * @param layer_idx Layer index (0-based)
 * @return 1 if global attention, 0 if local (sliding window)
 */
static inline int gemma3_is_global_layer(int layer_idx) {
    // Global every 6th layer: layers 5, 11, 17, 23, 29 (0-indexed)
    return ((layer_idx + 1) % 6 == 0);
}

/**
 * Get RoPE theta for a layer
 * @param layer_idx Layer index
 * @return theta value (10K for local, 1M for global)
 */
static inline float gemma3_layer_rope_theta(int layer_idx) {
    return gemma3_is_global_layer(layer_idx) ?
           GEMMA3_ROPE_THETA_GLOBAL : GEMMA3_ROPE_THETA_LOCAL;
}

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_H */
