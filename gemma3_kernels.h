/*
 * gemma3_kernels.h - CPU compute kernel declarations for Gemma 3 inference
 *
 * Pure C implementation of matrix operations, normalization, activations,
 * and positional encoding for transformer inference.
 */

#ifndef GEMMA3_KERNELS_H
#define GEMMA3_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Basic Tensor Operations
 * ========================================================================== */

/**
 * Matrix multiplication: C = A @ B
 * A: [M, K], B: [K, N], C: [M, N]
 * All matrices in row-major order
 */
void gemma3_matmul(float *C, const float *A, const float *B,
                   int M, int K, int N);

/**
 * Matrix-vector multiplication: y = A @ x
 * A: [M, K], x: [K], y: [M]
 */
void gemma3_matvec(float *y, const float *A, const float *x, int M, int K);

/**
 * Batched matrix-vector multiplication
 * A: [batch, M, K], x: [batch, K], y: [batch, M]
 */
void gemma3_matvec_batched(float *y, const float *A, const float *x,
                           int batch, int M, int K);

/**
 * Element-wise vector addition: y = a + b
 */
void gemma3_vec_add(float *y, const float *a, const float *b, int n);

/**
 * Element-wise vector multiplication: y = a * b
 */
void gemma3_vec_mul(float *y, const float *a, const float *b, int n);

/**
 * Scale vector: y = x * scale
 */
void gemma3_vec_scale(float *y, const float *x, float scale, int n);

/**
 * Copy vector: dst = src
 */
void gemma3_vec_copy(float *dst, const float *src, int n);

/**
 * Zero vector: x = 0
 */
void gemma3_vec_zero(float *x, int n);

/* ============================================================================
 * Normalization
 * ========================================================================== */

/**
 * RMS Normalization: y = x * rsqrt(mean(x^2) + eps) * weight
 * x: [n], weight: [n], y: [n]
 * eps: typically 1e-6 for Gemma 3
 */
void gemma3_rmsnorm(float *y, const float *x, const float *weight,
                    int n, float eps);

/**
 * RMS Normalization in-place: x = x * rsqrt(mean(x^2) + eps) * weight
 */
void gemma3_rmsnorm_inplace(float *x, const float *weight, int n, float eps);

/* ============================================================================
 * Activation Functions
 * ========================================================================== */

/**
 * GELU activation (tanh approximation): y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * This is the approximation used by Gemma 3
 */
void gemma3_gelu_tanh(float *y, const float *x, int n);

/**
 * GELU activation in-place
 */
void gemma3_gelu_tanh_inplace(float *x, int n);

/**
 * SiLU (Swish) activation: y = x * sigmoid(x)
 */
void gemma3_silu(float *y, const float *x, int n);

/**
 * SiLU activation in-place
 */
void gemma3_silu_inplace(float *x, int n);

/**
 * Softmax: y[i] = exp(x[i]) / sum(exp(x))
 * Numerically stable implementation
 */
void gemma3_softmax(float *y, const float *x, int n);

/**
 * Softmax in-place
 */
void gemma3_softmax_inplace(float *x, int n);

/* ============================================================================
 * Positional Encoding (RoPE)
 * ========================================================================== */

/**
 * Apply Rotary Position Embedding to Q and K tensors
 * q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
 * pos: position index
 * head_dim: dimension of each head (must be even)
 * theta: base frequency (10000 for local, 1000000 for global)
 */
void gemma3_rope(float *q, float *k, int n_heads, int n_kv_heads,
                 int head_dim, int pos, float theta);

/**
 * Apply RoPE to a single vector
 * x: [head_dim], must be even length
 */
void gemma3_rope_single(float *x, int head_dim, int pos, float theta);

/**
 * Precompute RoPE frequencies for given positions
 * freqs: output [max_pos, head_dim/2, 2] (cos, sin pairs)
 */
void gemma3_rope_precompute(float *freqs, int max_pos, int head_dim, float theta);

/**
 * Apply precomputed RoPE frequencies
 */
void gemma3_rope_apply_precomputed(float *x, const float *freqs,
                                    int head_dim, int pos);

/* ============================================================================
 * Attention Operations
 * ========================================================================== */

/**
 * Scaled dot-product attention for a single query
 * q: [head_dim]
 * k_cache: [seq_len, head_dim]
 * v_cache: [seq_len, head_dim]
 * output: [head_dim]
 * scale: typically 1/sqrt(head_dim)
 * mask: attention mask [seq_len], NULL for no mask
 */
void gemma3_attention_single(float *output, const float *q,
                             const float *k_cache, const float *v_cache,
                             int seq_len, int head_dim, float scale,
                             const float *mask);

/**
 * Grouped Query Attention
 * Handles multiple query heads sharing KV heads
 * q: [n_heads, head_dim]
 * k_cache: [seq_len, n_kv_heads, head_dim] - interleaved by position
 * v_cache: [seq_len, n_kv_heads, head_dim] - interleaved by position
 * output: [n_heads, head_dim]
 * n_heads must be divisible by n_kv_heads
 */
void gemma3_gqa(float *output, const float *q,
                const float *k_cache, const float *v_cache,
                int n_heads, int n_kv_heads, int seq_len, int head_dim,
                float scale, const float *mask);

/**
 * Sliding window attention mask generation
 * mask: output [query_pos + 1]
 * query_pos: current position
 * window_size: sliding window size (1024 for Gemma 3 local)
 */
void gemma3_sliding_window_mask(float *mask, int query_pos, int window_size);

/**
 * Causal attention mask (for global attention)
 * mask: output [seq_len]
 * Sets positions > query_pos to -inf
 */
void gemma3_causal_mask(float *mask, int seq_len, int query_pos);

/* ============================================================================
 * BF16 Kernel Operations (for memory-efficient inference)
 * ========================================================================== */

/**
 * Matrix-vector multiplication with BF16 matrix: y = A @ x
 * A: [M, K] in BF16, x: [K] in F32, y: [M] in F32
 * Converts BF16 to F32 on-the-fly during computation
 */
void gemma3_matvec_bf16(float *y, const uint16_t *A, const float *x, int M, int K);

/**
 * RMS Normalization with BF16 weights: y = x * rsqrt(mean(x^2) + eps) * weight
 * x: [n] in F32, weight: [n] in BF16, y: [n] in F32
 */
void gemma3_rmsnorm_bf16(float *y, const float *x, const uint16_t *weight,
                         int n, float eps);

/**
 * RMS Normalization in-place with BF16 weights
 */
void gemma3_rmsnorm_bf16_inplace(float *x, const uint16_t *weight, int n, float eps);

/**
 * Embedding lookup from BF16 table
 * embed: [vocab_size, hidden_size] in BF16
 * output: [hidden_size] in F32
 */
void gemma3_embed_bf16(float *output, const uint16_t *embed, int token_id, int hidden_size);

/* ============================================================================
 * Data Type Conversions
 * ========================================================================== */

/**
 * Convert BF16 to F32
 * bf16: input array of uint16_t representing BF16 values
 * f32: output array of floats
 * n: number of elements
 */
void gemma3_bf16_to_f32(float *f32, const uint16_t *bf16, int n);

/**
 * Convert F32 to BF16 (for debugging/testing)
 */
void gemma3_f32_to_bf16(uint16_t *bf16, const float *f32, int n);

/**
 * Convert a single BF16 value to F32
 */
static inline float gemma3_bf16_to_f32_single(uint16_t bf16) {
    uint32_t bits = ((uint32_t)bf16) << 16;
    float result;
    // Use memcpy for type-punning to avoid strict aliasing issues
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

/**
 * Convert a single F32 value to BF16 (truncation, not rounding)
 */
static inline uint16_t gemma3_f32_to_bf16_single(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

/* ============================================================================
 * Sampling Operations
 * ========================================================================== */

/**
 * Apply temperature scaling to logits (in-place)
 * logits: [vocab_size]
 * temperature: scaling factor (> 0)
 */
void gemma3_apply_temperature(float *logits, int vocab_size, float temperature);

/**
 * Top-k filtering: keep only top k logits, set rest to -inf
 * logits: [vocab_size], modified in-place
 * k: number of top logits to keep
 */
void gemma3_topk_filter(float *logits, int vocab_size, int k);

/**
 * Top-p (nucleus) filtering: keep smallest set with cumulative prob >= p
 * logits: [vocab_size], modified in-place
 * p: probability threshold (0-1)
 */
void gemma3_topp_filter(float *logits, int vocab_size, float p);

/**
 * Sample token from probability distribution
 * probs: [vocab_size], must sum to 1
 * Returns sampled token index
 */
int gemma3_sample(const float *probs, int vocab_size);

/**
 * Argmax: return index of maximum value
 */
int gemma3_argmax(const float *x, int n);

/* ============================================================================
 * Utility Functions
 * ========================================================================== */

/**
 * Compute sum of vector elements
 */
float gemma3_vec_sum(const float *x, int n);

/**
 * Compute maximum value in vector
 */
float gemma3_vec_max(const float *x, int n);

/**
 * Compute dot product of two vectors
 */
float gemma3_dot(const float *a, const float *b, int n);

/**
 * Initialize random seed for sampling
 */
void gemma3_set_seed(uint64_t seed);

/**
 * Generate random float in [0, 1)
 */
float gemma3_random(void);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_KERNELS_H */
