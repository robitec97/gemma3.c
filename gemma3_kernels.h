/*
 * gemma3_kernels.h - CPU compute kernel declarations for Gemma 3 inference
 *
 * Pure C implementation of matrix operations, normalization, activations,
 * positional encoding, attention and sampling. Hot loops have NEON (arm64)
 * and AVX2+FMA (x86-64) paths with a portable scalar fallback.
 */

#ifndef GEMMA3_KERNELS_H
#define GEMMA3_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#include "gemma3_threads.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Name of the SIMD code path compiled in: "neon", "avx2" or "scalar". */
const char *gemma3_simd_name(void);

/* ============================================================================
 * Basic Tensor Operations (F32)
 * ========================================================================== */

/* C = A @ B with A: [M, K], B: [K, N], C: [M, N], row-major */
void gemma3_matmul(float *C, const float *A, const float *B, int M, int K, int N);

/* y = A @ x with A: [M, K] */
void gemma3_matvec(float *y, const float *A, const float *x, int M, int K);

/* A: [batch, M, K], x: [batch, K], y: [batch, M] */
void gemma3_matvec_batched(float *y, const float *A, const float *x,
                           int batch, int M, int K);

void gemma3_vec_add(float *y, const float *a, const float *b, int n);
void gemma3_vec_mul(float *y, const float *a, const float *b, int n);
void gemma3_vec_scale(float *y, const float *x, float scale, int n);
void gemma3_vec_copy(float *dst, const float *src, int n);
void gemma3_vec_zero(float *x, int n);

/* ============================================================================
 * Normalization
 * ========================================================================== */

/* y = x * rsqrt(mean(x^2) + eps) * weight */
void gemma3_rmsnorm(float *y, const float *x, const float *weight, int n, float eps);
void gemma3_rmsnorm_inplace(float *x, const float *weight, int n, float eps);

/* Gemma variant with BF16 weights: y = x * rsqrt(mean(x^2) + eps) * (1 + weight).
 * y may alias x. */
void gemma3_rmsnorm_bf16(float *y, const float *x, const uint16_t *weight, int n, float eps);
void gemma3_rmsnorm_bf16_inplace(float *x, const uint16_t *weight, int n, float eps);

/* ============================================================================
 * Activation Functions
 * ========================================================================== */

/* GELU, tanh approximation (as used by Gemma 3) */
void gemma3_gelu_tanh(float *y, const float *x, int n);
void gemma3_gelu_tanh_inplace(float *x, int n);

/* gate = gelu_tanh(gate) * up  (the Gemma 3 gated MLP activation) */
void gemma3_gelu_tanh_mul(float *gate, const float *up, int n);

void gemma3_silu(float *y, const float *x, int n);
void gemma3_silu_inplace(float *x, int n);

/* Numerically stable softmax */
void gemma3_softmax(float *y, const float *x, int n);
void gemma3_softmax_inplace(float *x, int n);

/* ============================================================================
 * Positional Encoding (RoPE, rotate-half layout)
 * ========================================================================== */

void gemma3_rope(float *q, float *k, int n_heads, int n_kv_heads,
                 int head_dim, int pos, float theta);
void gemma3_rope_single(float *x, int head_dim, int pos, float theta);

/* freqs: output [max_pos, head_dim/2, 2] (cos, sin pairs).
 * scaling: linear position scaling (1.0 = none, 8.0 for Gemma 3 4B global layers) */
void gemma3_rope_precompute(float *freqs, int max_pos, int head_dim, float theta,
                            float scaling);
void gemma3_rope_apply_precomputed(float *x, const float *freqs, int head_dim, int pos);

/* ============================================================================
 * Attention
 * ========================================================================== */

/* Single-head attention over seq_len contiguous K/V rows of length head_dim. */
void gemma3_attention_single(float *output, const float *q,
                             const float *k_cache, const float *v_cache,
                             int seq_len, int head_dim, float scale,
                             const float *mask);

/* Grouped-query attention over a cache laid out as [seq_len, n_kv_heads, head_dim]. */
void gemma3_gqa(float *output, const float *q,
                const float *k_cache, const float *v_cache,
                int n_heads, int n_kv_heads, int seq_len, int head_dim,
                float scale, const float *mask, float *scores_buf);

/* Attention for one query head over cache positions [pos_start, pos_end].
 * Position p lives in row (ring > 0 ? p % ring : p) of k_cache/v_cache;
 * rows are kv_stride floats apart. scores must hold pos_end - pos_start + 1
 * floats. */
void gemma3_attention_head(float *out, const float *q,
                           const float *k_cache, const float *v_cache,
                           int kv_stride, int pos_start, int pos_end, int ring,
                           int head_dim, float scale, float *scores);

void gemma3_sliding_window_mask(float *mask, int query_pos, int window_size);
void gemma3_causal_mask(float *mask, int seq_len, int query_pos);

/* ============================================================================
 * BF16 Weight Kernels
 * ========================================================================== */

/* y = A @ x, A: [M, K] BF16, x: [K] F32. scratch is unused (kept for API
 * compatibility) and may be NULL. */
void gemma3_matvec_bf16(float *y, const uint16_t *A, const float *x, int M, int K,
                        float *scratch);

/* Same, multi-threaded over rows. pool may be NULL. */
void gemma3_matvec_bf16_mt(float *y, const uint16_t *A, const float *x, int M, int K,
                           float *scratch, gemma3_thread_pool *pool);

/* Rows [row_start, row_end) of y = A @ x. */
void gemma3_matvec_bf16_range(float *y, const uint16_t *A, const float *x,
                              int row_start, int row_end, int K);

/* Batched projection: Y[n, m] = sum_k X[n, k] * W[m, k]
 *   X: [N, K] F32 (row stride K), W: [M, K] BF16, Y: [N, M] F32 (row stride ldy)
 * Computes only rows m in [row_start, row_end). Weights are read once per
 * register tile of tokens, so cost is compute-bound rather than
 * bandwidth-bound for N > 1. */
void gemma3_matmul_bf16_range(float *Y, int ldy, const float *X, int N,
                              const uint16_t *W, int K, int row_start, int row_end);

/* Full batched projection, multi-threaded over weight rows. pool may be NULL. */
void gemma3_matmul_bf16_mt(float *Y, const float *X, int N, const uint16_t *W,
                           int M, int K, gemma3_thread_pool *pool);

/* Embedding lookup from a BF16 table [vocab, hidden] */
void gemma3_embed_bf16(float *output, const uint16_t *embed, int token_id, int hidden_size);

/* ============================================================================
 * Data Type Conversions
 * ========================================================================== */

void gemma3_bf16_to_f32(float *f32, const uint16_t *bf16, int n);
void gemma3_f32_to_bf16(uint16_t *bf16, const float *f32, int n);

static inline float gemma3_bf16_to_f32_single(uint16_t bf16) {
    uint32_t bits = ((uint32_t)bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

/* Truncating conversion (for testing) */
static inline uint16_t gemma3_f32_to_bf16_single(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

/* ============================================================================
 * Sampling
 * ========================================================================== */

/* Scratch space for gemma3_sample_logits (allocate once, reuse per token). */
typedef struct gemma3_sampler gemma3_sampler;

gemma3_sampler *gemma3_sampler_create(int vocab_size);
void gemma3_sampler_free(gemma3_sampler *s);

/* Pick the next token from raw logits.
 * temperature <= 0 (or non-finite) means greedy. top_k <= 0 disables top-k,
 * top_p >= 1 disables top-p (top_p <= 0 keeps only the best token), min_p <= 0
 * disables min-p. NaN logits are never sampled. Only the surviving candidates are sorted and
 * softmaxed, so the cost is O(vocab) for the scan plus O(k log k).
 * rng_state is advanced (xorshift64*); pass a pointer to a per-context state. */
int gemma3_sample_logits(gemma3_sampler *s, const float *logits, int vocab_size,
                         float temperature, int top_k, float top_p, float min_p,
                         uint64_t *rng_state);

/* Legacy filtering helpers (operate on the full logits array) */
void gemma3_apply_temperature(float *logits, int vocab_size, float temperature);
void gemma3_topk_filter(float *logits, int vocab_size, int k);
void gemma3_topp_filter(float *logits, int vocab_size, float p);
int gemma3_sample(const float *probs, int vocab_size);
int gemma3_argmax(const float *x, int n);

/* ============================================================================
 * Utility Functions
 * ========================================================================== */

float gemma3_vec_sum(const float *x, int n);
float gemma3_vec_max(const float *x, int n);
float gemma3_dot(const float *a, const float *b, int n);

/* Global RNG used by gemma3_sample() */
void gemma3_set_seed(uint64_t seed);
float gemma3_random(void);

/* Uniform float in [0, 1) from a caller-owned xorshift64* state */
float gemma3_random_r(uint64_t *state);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_KERNELS_H */
