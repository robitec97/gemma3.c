/*
 * gemma3_metal.h - Metal GPU backend C API for Gemma 3 inference
 *
 * Provides GPU-accelerated forward pass using custom Metal compute shaders.
 * Enabled via compile-time flag USE_MPS.
 *
 * Environment variables:
 *   GEMMA3_METAL_KV=f16       store the KV cache in half precision (half the
 *                             memory and attention bandwidth; default f32)
 *   GEMMA3_METAL_DEBUG=1      print device, KV cache and weight-mapping info
 *   GEMMA3_METAL_PROFILE=1    print GPU time per command buffer
 *   GEMMA3_METAL_NO_GEMM=1    disable batched prefill (one token at a time)
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

/* Make the model weights visible to the GPU. The mmap'd safetensors regions
 * in gemma3_weights_t.regions are wrapped without copying; tensors outside a
 * region (or misaligned) are copied. weights is a gemma3_weights_t*. */
int gemma3_metal_upload_weights(gemma3_metal_context *ctx, const void *weights);

/* Upload precomputed RoPE cos/sin tables to GPU */
int gemma3_metal_upload_rope(gemma3_metal_context *ctx,
                              const float *rope_local, const float *rope_global,
                              int max_context, int head_dim);

/* Forward pass for a single token (entire transformer on GPU).
 * If compute_logits is 0 or logits is NULL, skips the vocab projection.
 * pos may be earlier than the current position (cache rewind) as long as the
 * rewind is within GEMMA3_LOCAL_RING_EXTRA tokens or nothing has wrapped. */
int gemma3_metal_forward_token(gemma3_metal_context *ctx, int token_id, int pos,
                                float *logits, int compute_logits);

/* Prefill tokens at positions start_pos.. in chunks of GEMMA3_PREFILL_CHUNK
 * using batched GEMM kernels. logits (may be NULL) receives the last token's
 * logits. Same rewind rules as gemma3_metal_forward_token. */
int gemma3_metal_prefill(gemma3_metal_context *ctx, const int *tokens, int num_tokens,
                          int start_pos, float *logits);

/* Reset the KV cache (O(1): only positions written for the current
 * sequence are ever read) */
void gemma3_metal_reset_cache(gemma3_metal_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* USE_MPS */
#endif /* GEMMA3_METAL_H */
