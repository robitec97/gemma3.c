/*
 * gemma3_metal.h - Metal GPU backend C API for Gemma 3 inference
 *
 * Provides GPU-accelerated forward pass using custom Metal compute shaders.
 * Enabled via compile-time flag USE_MPS.
 */

#ifndef GEMMA3_METAL_H
#define GEMMA3_METAL_H

#ifdef USE_MPS

#include "gemma3.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque Metal context (defined in gemma3_metal.m) */
typedef struct gemma3_metal_context gemma3_metal_context;

/* Check if Metal GPU is available on this system */
int gemma3_metal_available(void);

/* Initialize Metal context. Returns NULL if Metal not available (CPU fallback). */
gemma3_metal_context *gemma3_metal_init(const gemma3_config *cfg, int max_context);

/* Free Metal context and all GPU resources */
void gemma3_metal_free(gemma3_metal_context *ctx);

/* Upload model weights to GPU (zero-copy when possible).
 * weights is a gemma3_weights_t* (void* to avoid header dependency). */
int gemma3_metal_upload_weights(gemma3_metal_context *ctx, const void *weights);

/* Upload precomputed RoPE cos/sin tables to GPU */
int gemma3_metal_upload_rope(gemma3_metal_context *ctx,
                              const float *rope_local, const float *rope_global,
                              int max_context, int head_dim);

/* Forward pass for a single token (entire transformer on GPU).
 * If compute_logits is 0, skips final norm + vocab projection. */
int gemma3_metal_forward_token(gemma3_metal_context *ctx, int token_id, int pos,
                                float *logits, int compute_logits);

/* Prefill multiple tokens sequentially on GPU */
int gemma3_metal_prefill(gemma3_metal_context *ctx, const int *tokens, int num_tokens,
                          int start_pos, float *logits);

/* Reset KV cache on GPU */
void gemma3_metal_reset_cache(gemma3_metal_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* USE_MPS */
#endif /* GEMMA3_METAL_H */
