/*
 * gemma3_internal.h - Internal declarations shared between translation units
 *
 * Not part of the public API. Holds the weight layout, the safetensors /
 * tokenizer / transformer entry points used by gemma3.c, and constants that
 * the CPU and Metal backends must agree on.
 */

#ifndef GEMMA3_INTERNAL_H
#define GEMMA3_INTERNAL_H

#include "gemma3.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Shared constants
 * ========================================================================== */

/* Maximum number of tokens processed together during batched prefill. */
#define GEMMA3_PREFILL_CHUNK 128

/* Local (sliding-window) layers keep their KV cache in a ring buffer of
 * sliding_window + GEMMA3_LOCAL_RING_EXTRA slots. The extra slots let a whole
 * prefill chunk write its K/V before attention runs without overwriting
 * entries that earlier tokens in the same chunk still need, and they allow
 * the cache to be rewound by up to GEMMA3_LOCAL_RING_EXTRA tokens (used to
 * reuse the KV cache across chat turns). */
#define GEMMA3_LOCAL_RING_EXTRA GEMMA3_PREFILL_CHUNK

static inline int gemma3_local_ring_size(int sliding_window) {
    return sliding_window + GEMMA3_LOCAL_RING_EXTRA;
}

/* ============================================================================
 * Weights (BF16 pointers into memory-mapped safetensors files)
 * ========================================================================== */

#define GEMMA3_MAX_MAPPED_REGIONS 16

/* A memory-mapped file the weight pointers point into. The Metal backend
 * wraps these regions directly so weights are never copied. */
typedef struct {
    const void *base;
    size_t size;
} gemma3_mapped_region;

typedef struct gemma3_weights_t {
    const uint16_t *embed_tokens;              /* [vocab_size, hidden_size] */
    struct {
        const uint16_t *input_layernorm;       /* [hidden_size] */
        const uint16_t *q_proj;                /* [num_heads * head_dim, hidden_size] */
        const uint16_t *k_proj;                /* [num_kv_heads * head_dim, hidden_size] */
        const uint16_t *v_proj;                /* [num_kv_heads * head_dim, hidden_size] */
        const uint16_t *o_proj;                /* [hidden_size, num_heads * head_dim] */
        const uint16_t *q_norm;                /* [head_dim] */
        const uint16_t *k_norm;                /* [head_dim] */
        const uint16_t *post_attention_layernorm;   /* [hidden_size] */
        const uint16_t *gate_proj;             /* [intermediate_size, hidden_size] */
        const uint16_t *up_proj;               /* [intermediate_size, hidden_size] */
        const uint16_t *down_proj;             /* [hidden_size, intermediate_size] */
        const uint16_t *pre_feedforward_layernorm;  /* [hidden_size] */
        const uint16_t *post_feedforward_layernorm; /* [hidden_size] */
    } layers[GEMMA3_NUM_LAYERS];
    const uint16_t *norm;                      /* [hidden_size] */

    gemma3_mapped_region regions[GEMMA3_MAX_MAPPED_REGIONS];
    int num_regions;
} gemma3_weights_t;

/* ============================================================================
 * SafeTensors (gemma3_safetensors.c)
 * ========================================================================== */

typedef struct st_context st_context;

st_context *st_load(const char *model_dir);
void st_free(st_context *ctx);
gemma3_weights_t *gemma3_load_weights(st_context *st);
void gemma3_free_weights(gemma3_weights_t *w);

/* ============================================================================
 * Tokenizer (gemma3_tokenizer.c)
 * ========================================================================== */

gemma3_tokenizer *gemma3_tokenizer_load(const char *path);
void gemma3_tokenizer_free(gemma3_tokenizer *tok);

/* ============================================================================
 * Transformer (gemma3_transformer.c)
 * ========================================================================== */

typedef struct gemma3_transformer gemma3_transformer;

gemma3_transformer *gemma3_transformer_create(gemma3_weights_t *weights,
                                               const gemma3_config *cfg,
                                               int max_context, int num_threads);
void gemma3_transformer_destroy(gemma3_transformer *t);
int gemma3_transformer_forward_token(gemma3_transformer *t, int token_id,
                                      int pos, float *logits);
int gemma3_transformer_prefill_tokens(gemma3_transformer *t, const int *tokens,
                                       int num_tokens, int start_pos, float *logits);
void gemma3_transformer_reset(gemma3_transformer *t);
int gemma3_transformer_get_pos(gemma3_transformer *t);
const char *gemma3_transformer_backend(const gemma3_transformer *t);
int gemma3_transformer_num_threads(const gemma3_transformer *t);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_INTERNAL_H */
