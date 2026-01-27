/*
 * gemma3_kernels.c - CPU compute kernel implementations for Gemma 3 inference
 *
 * Pure C implementation optimized for clarity and correctness.
 * Can be extended with SIMD optimizations.
 */

#include "gemma3_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* Random state for sampling */
static uint64_t g_rng_state = 12345678901234567ULL;

/* ============================================================================
 * Basic Tensor Operations
 * ========================================================================== */

void gemma3_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    // C = A * B: A is [M,K], B is [K,N], C is [M,N], all row-major
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#else
    // C[i,j] = sum_k A[i,k] * B[k,j]
    // A: [M, K], B: [K, N], C: [M, N]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#endif
}

void gemma3_matvec(float *y, const float *A, const float *x, int M, int K) {
#ifdef USE_BLAS
    // y = A * x: A is [M, K] row-major, x is [K], y is [M]
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, K, 1.0f, A, K, x, 1, 0.0f, y, 1);
#else
    // y[i] = sum_k A[i,k] * x[k]
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const float *row = A + i * K;
        for (int k = 0; k < K; k++) {
            sum += row[k] * x[k];
        }
        y[i] = sum;
    }
#endif
}

void gemma3_matvec_batched(float *y, const float *A, const float *x,
                           int batch, int M, int K) {
    for (int b = 0; b < batch; b++) {
        gemma3_matvec(y + b * M, A + b * M * K, x + b * K, M, K);
    }
}

void gemma3_vec_add(float *y, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}

void gemma3_vec_mul(float *y, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}

void gemma3_vec_scale(float *y, const float *x, float scale, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * scale;
    }
}

void gemma3_vec_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

void gemma3_vec_zero(float *x, int n) {
    memset(x, 0, n * sizeof(float));
}

/* ============================================================================
 * Normalization
 * ========================================================================== */

void gemma3_rmsnorm(float *y, const float *x, const float *weight,
                    int n, float eps) {
    // Compute mean of squares
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / n + eps;
    float rsqrt_ss = 1.0f / sqrtf(ss);

    // Normalize and scale
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * rsqrt_ss * weight[i];
    }
}

void gemma3_rmsnorm_inplace(float *x, const float *weight, int n, float eps) {
    // Compute mean of squares
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / n + eps;
    float rsqrt_ss = 1.0f / sqrtf(ss);

    // Normalize and scale in-place
    for (int i = 0; i < n; i++) {
        x[i] = x[i] * rsqrt_ss * weight[i];
    }
}

/* ============================================================================
 * BF16 Kernel Operations
 * ========================================================================== */

/* Helper: Convert BF16 to F32 inline */
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = ((uint32_t)bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

#ifdef __AVX2__
/* AVX2: Fused BF16-to-F32 conversion + dot product.
 * Processes 8 BF16 elements at a time: load 8x uint16 -> zero-extend to 32-bit
 * -> shift left 16 -> reinterpret as float -> FMA with x vector. */
static float avx2_bf16_dot(const uint16_t *a_bf16, const float *x, int K) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int k = 0;
    /* Process 16 elements per iteration (2x8 for better ILP) */
    for (; k + 15 < K; k += 16) {
        /* Load 8 BF16 values, zero-extend to 32-bit, shift left 16 to get F32 */
        __m128i bf16_lo = _mm_loadu_si128((const __m128i *)(a_bf16 + k));
        __m256i i32_lo = _mm256_cvtepu16_epi32(bf16_lo);
        __m256i f32_bits_lo = _mm256_slli_epi32(i32_lo, 16);
        __m256 a_lo = _mm256_castsi256_ps(f32_bits_lo);
        __m256 x_lo = _mm256_loadu_ps(x + k);
        acc0 = _mm256_fmadd_ps(a_lo, x_lo, acc0);

        __m128i bf16_hi = _mm_loadu_si128((const __m128i *)(a_bf16 + k + 8));
        __m256i i32_hi = _mm256_cvtepu16_epi32(bf16_hi);
        __m256i f32_bits_hi = _mm256_slli_epi32(i32_hi, 16);
        __m256 a_hi = _mm256_castsi256_ps(f32_bits_hi);
        __m256 x_hi = _mm256_loadu_ps(x + k + 8);
        acc1 = _mm256_fmadd_ps(a_hi, x_hi, acc1);
    }
    /* Process remaining 8-element chunks */
    for (; k + 7 < K; k += 8) {
        __m128i bf16_v = _mm_loadu_si128((const __m128i *)(a_bf16 + k));
        __m256i i32_v = _mm256_cvtepu16_epi32(bf16_v);
        __m256i f32_bits = _mm256_slli_epi32(i32_v, 16);
        __m256 a_v = _mm256_castsi256_ps(f32_bits);
        __m256 x_v = _mm256_loadu_ps(x + k);
        acc0 = _mm256_fmadd_ps(a_v, x_v, acc0);
    }

    /* Horizontal sum of acc0 + acc1 */
    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 hi128 = _mm256_extractf128_ps(acc0, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc0);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    float result = _mm_cvtss_f32(sum1);

    /* Scalar tail */
    for (; k < K; k++) {
        result += bf16_to_f32(a_bf16[k]) * x[k];
    }
    return result;
}
#endif /* __AVX2__ */

void gemma3_matvec_bf16(float *y, const uint16_t *A, const float *x, int M, int K,
                        float *scratch) {
#ifdef USE_BLAS
    // Use pre-allocated scratch buffer for BF16->F32 conversion (avoids malloc per call)
    float *row_f32 = scratch;
    int need_free = 0;
    if (!row_f32) {
        row_f32 = (float *)malloc(K * sizeof(float));
        need_free = 1;
    }
    if (row_f32) {
        for (int i = 0; i < M; i++) {
            const uint16_t *row = A + i * K;
            // Convert BF16 row to F32 (stays hot in L1 cache)
            for (int k = 0; k < K; k++) {
                row_f32[k] = bf16_to_f32(row[k]);
            }
            y[i] = cblas_sdot(K, row_f32, 1, x, 1);
        }
        if (need_free) free(row_f32);
    } else {
        // Fallback to scalar loop if malloc fails
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            const uint16_t *row = A + i * K;
            for (int k = 0; k < K; k++) {
                sum += bf16_to_f32(row[k]) * x[k];
            }
            y[i] = sum;
        }
    }
#else
    (void)scratch;
#ifdef __AVX2__
    // AVX2 path: fused BF16->F32 + FMA dot product
    for (int i = 0; i < M; i++) {
        y[i] = avx2_bf16_dot(A + i * K, x, K);
    }
#else
    // Scalar fallback: y[i] = sum_k A[i,k] * x[k], where A is in BF16
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const uint16_t *row = A + i * K;
        for (int k = 0; k < K; k++) {
            sum += bf16_to_f32(row[k]) * x[k];
        }
        y[i] = sum;
    }
#endif /* __AVX2__ */
#endif /* USE_BLAS */
}

#ifdef USE_THREADS
/* Thread task argument for parallel matvec */
typedef struct {
    float *y;
    const uint16_t *A;
    const float *x;
    int M;
    int K;
} matvec_bf16_task_t;

static void matvec_bf16_worker(void *arg, int thread_idx, int num_threads) {
    matvec_bf16_task_t *t = (matvec_bf16_task_t *)arg;
    int rows_per_thread = (t->M + num_threads - 1) / num_threads;
    int start = thread_idx * rows_per_thread;
    int end = start + rows_per_thread;
    if (end > t->M) end = t->M;

    for (int i = start; i < end; i++) {
#ifdef __AVX2__
        t->y[i] = avx2_bf16_dot(t->A + i * t->K, t->x, t->K);
#else
        const uint16_t *row = t->A + i * t->K;
        float sum = 0.0f;
        for (int k = 0; k < t->K; k++) {
            sum += bf16_to_f32(row[k]) * t->x[k];
        }
        t->y[i] = sum;
#endif
    }
}

void gemma3_matvec_bf16_mt(float *y, const uint16_t *A, const float *x, int M, int K,
                           float *scratch, gemma3_thread_pool *pool) {
    if (!pool || gemma3_thread_pool_size(pool) <= 1) {
        gemma3_matvec_bf16(y, A, x, M, K, scratch);
        return;
    }
    matvec_bf16_task_t task = { y, A, x, M, K };
    gemma3_thread_pool_run(pool, matvec_bf16_worker, &task);
}
#endif /* USE_THREADS */

void gemma3_rmsnorm_bf16(float *y, const float *x, const uint16_t *weight,
                         int n, float eps) {
    // Compute mean of squares
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / n + eps;
    float rsqrt_ss = 1.0f / sqrtf(ss);

    // Normalize and scale (weight is BF16)
    // Gemma uses (1.0 + weight) formula
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * rsqrt_ss * (1.0f + bf16_to_f32(weight[i]));
    }
}

void gemma3_rmsnorm_bf16_inplace(float *x, const uint16_t *weight, int n, float eps) {
    // Compute mean of squares
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / n + eps;
    float rsqrt_ss = 1.0f / sqrtf(ss);

    // Normalize and scale in-place (weight is BF16)
    // Gemma uses (1.0 + weight) formula
    for (int i = 0; i < n; i++) {
        x[i] = x[i] * rsqrt_ss * (1.0f + bf16_to_f32(weight[i]));
    }
}

void gemma3_embed_bf16(float *output, const uint16_t *embed, int token_id, int hidden_size) {
    // Copy embedding row, converting from BF16 to F32
    const uint16_t *row = embed + token_id * hidden_size;
    for (int i = 0; i < hidden_size; i++) {
        output[i] = bf16_to_f32(row[i]);
    }
}

/* ============================================================================
 * Activation Functions
 * ========================================================================== */

void gemma3_gelu_tanh(float *y, const float *x, int n) {
    // GELU with tanh approximation:
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    const float coeff = 0.044715f;

    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float x3 = xi * xi * xi;
        float inner = sqrt_2_over_pi * (xi + coeff * x3);
        y[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

void gemma3_gelu_tanh_inplace(float *x, int n) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float x3 = xi * xi * xi;
        float inner = sqrt_2_over_pi * (xi + coeff * x3);
        x[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

void gemma3_silu(float *y, const float *x, int n) {
    // SiLU (Swish): silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        y[i] = xi / (1.0f + expf(-xi));
    }
}

void gemma3_silu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        x[i] = xi / (1.0f + expf(-xi));
    }
}

void gemma3_softmax(float *y, const float *x, int n) {
    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        y[i] = expf(x[i] - max_val);
        sum += y[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        y[i] *= inv_sum;
    }
}

void gemma3_softmax_inplace(float *x, int n) {
    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

/* ============================================================================
 * Positional Encoding (RoPE)
 * ========================================================================== */

void gemma3_rope_single(float *x, int head_dim, int pos, float theta) {
    // Apply rotary position embedding to a single vector
    // x is modified in-place
    // head_dim must be even

    int half_dim = head_dim / 2;
    for (int i = 0; i < half_dim; i++) {
        // Compute frequency for this dimension
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        // Apply rotation
        float x0 = x[i];
        float x1 = x[i + half_dim];
        x[i] = x0 * cos_val - x1 * sin_val;
        x[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

void gemma3_rope(float *q, float *k, int n_heads, int n_kv_heads,
                 int head_dim, int pos, float theta) {
    // Apply RoPE to all query heads
    for (int h = 0; h < n_heads; h++) {
        gemma3_rope_single(q + h * head_dim, head_dim, pos, theta);
    }

    // Apply RoPE to all key heads
    for (int h = 0; h < n_kv_heads; h++) {
        gemma3_rope_single(k + h * head_dim, head_dim, pos, theta);
    }
}

void gemma3_rope_precompute(float *freqs, int max_pos, int head_dim, float theta) {
    // Precompute cos/sin values for all positions
    // freqs: [max_pos, head_dim/2, 2] where last dim is (cos, sin)
    int half_dim = head_dim / 2;

    for (int pos = 0; pos < max_pos; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            freqs[(pos * half_dim + i) * 2] = cosf(angle);
            freqs[(pos * half_dim + i) * 2 + 1] = sinf(angle);
        }
    }
}

void gemma3_rope_apply_precomputed(float *x, const float *freqs,
                                    int head_dim, int pos) {
    int half_dim = head_dim / 2;
    const float *pos_freqs = freqs + pos * half_dim * 2;

    for (int i = 0; i < half_dim; i++) {
        float cos_val = pos_freqs[i * 2];
        float sin_val = pos_freqs[i * 2 + 1];

        float x0 = x[i];
        float x1 = x[i + half_dim];
        x[i] = x0 * cos_val - x1 * sin_val;
        x[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

/* ============================================================================
 * Attention Operations
 * ========================================================================== */

void gemma3_attention_single(float *output, const float *q,
                             const float *k_cache, const float *v_cache,
                             int seq_len, int head_dim, float scale,
                             const float *mask) {
    // Allocate temporary storage for attention scores
    float *scores = (float *)malloc(seq_len * sizeof(float));
    if (!scores) return;

    // Compute attention scores: scores[i] = q . k[i] * scale
    for (int i = 0; i < seq_len; i++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q[d] * k_cache[i * head_dim + d];
        }
        scores[i] = score * scale;

        // Apply mask if provided
        if (mask) {
            scores[i] += mask[i];
        }
    }

    // Softmax
    gemma3_softmax_inplace(scores, seq_len);

    // Compute weighted sum of values
    gemma3_vec_zero(output, head_dim);
    for (int i = 0; i < seq_len; i++) {
        float w = scores[i];
        for (int d = 0; d < head_dim; d++) {
            output[d] += w * v_cache[i * head_dim + d];
        }
    }

    free(scores);
}

void gemma3_gqa(float *output, const float *q,
                const float *k_cache, const float *v_cache,
                int n_heads, int n_kv_heads, int seq_len, int head_dim,
                float scale, const float *mask, float *scores_buf) {
    // Grouped Query Attention
    // n_heads query heads share n_kv_heads KV heads
    //
    // IMPORTANT: k_cache and v_cache have layout [seq_len, n_kv_heads, head_dim]
    // (interleaved by sequence position, not by head)

    int heads_per_group = n_heads / n_kv_heads;
    int kv_stride = n_kv_heads * head_dim;  // Stride between sequence positions

    // Use pre-allocated scores buffer to avoid malloc/free per token
    float *scores = scores_buf;
    int need_free = 0;
    if (!scores) {
        scores = (float *)malloc(seq_len * sizeof(float));
        need_free = 1;
    }
    if (!scores) return;

    for (int h = 0; h < n_heads; h++) {
        // Determine which KV head this query head uses
        int kv_head = h / heads_per_group;

        const float *q_head = q + h * head_dim;
        float *out_head = output + h * head_dim;

        // Compute attention scores: scores[i] = q . k[i] * scale
        // k_cache layout: [seq_pos][kv_head][head_dim]
        // So k at position i, kv_head h is at: k_cache[i * kv_stride + kv_head * head_dim]
        for (int i = 0; i < seq_len; i++) {
            const float *k_pos = k_cache + i * kv_stride + kv_head * head_dim;
#ifdef USE_BLAS
            float score = cblas_sdot(head_dim, q_head, 1, k_pos, 1);
#else
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_pos[d];
            }
#endif
            scores[i] = score * scale;

            // Apply mask if provided
            if (mask) {
                scores[i] += mask[i];
            }
        }

        // Softmax
        gemma3_softmax_inplace(scores, seq_len);

        // Compute weighted sum of values
        // v_cache layout: [seq_pos][kv_head][head_dim]
        gemma3_vec_zero(out_head, head_dim);
        for (int i = 0; i < seq_len; i++) {
            const float *v_pos = v_cache + i * kv_stride + kv_head * head_dim;
            float w = scores[i];
#ifdef USE_BLAS
            cblas_saxpy(head_dim, w, v_pos, 1, out_head, 1);
#else
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += w * v_pos[d];
            }
#endif
        }
    }

    if (need_free) free(scores);
}

void gemma3_sliding_window_mask(float *mask, int query_pos, int window_size) {
    // Create mask for sliding window attention
    // Positions outside the window get -inf
    int start = (query_pos >= window_size) ? query_pos - window_size + 1 : 0;

    for (int i = 0; i <= query_pos; i++) {
        if (i >= start) {
            mask[i] = 0.0f;  // Within window
        } else {
            mask[i] = -INFINITY;  // Outside window
        }
    }
}

void gemma3_causal_mask(float *mask, int seq_len, int query_pos) {
    // Create causal mask: can only attend to positions <= query_pos
    for (int i = 0; i < seq_len; i++) {
        if (i <= query_pos) {
            mask[i] = 0.0f;
        } else {
            mask[i] = -INFINITY;
        }
    }
}

/* ============================================================================
 * Data Type Conversions
 * ========================================================================== */

void gemma3_bf16_to_f32(float *f32, const uint16_t *bf16, int n) {
    for (int i = 0; i < n; i++) {
        f32[i] = gemma3_bf16_to_f32_single(bf16[i]);
    }
}

void gemma3_f32_to_bf16(uint16_t *bf16, const float *f32, int n) {
    for (int i = 0; i < n; i++) {
        bf16[i] = gemma3_f32_to_bf16_single(f32[i]);
    }
}

/* ============================================================================
 * Sampling Operations
 * ========================================================================== */

void gemma3_apply_temperature(float *logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f) return;  // Avoid division by zero

    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }
}

// Comparison function for qsort (descending order)
typedef struct {
    float value;
    int index;
} IndexedFloat;

static int compare_indexed_float_desc(const void *a, const void *b) {
    float va = ((const IndexedFloat *)a)->value;
    float vb = ((const IndexedFloat *)b)->value;
    if (va > vb) return -1;
    if (va < vb) return 1;
    return 0;
}

void gemma3_topk_filter(float *logits, int vocab_size, int k) {
    if (k <= 0 || k >= vocab_size) return;

    /* Min-heap of size k to find the k-th largest value in O(n log k).
     * The heap root is the smallest of the top-k values (the threshold). */
    float *heap = (float *)malloc(k * sizeof(float));
    if (!heap) return;

    /* Initialize heap with first k elements */
    for (int i = 0; i < k; i++) {
        heap[i] = logits[i];
    }

    /* Build min-heap (heapify) */
    for (int i = k / 2 - 1; i >= 0; i--) {
        int pos = i;
        while (1) {
            int smallest = pos;
            int left = 2 * pos + 1;
            int right = 2 * pos + 2;
            if (left < k && heap[left] < heap[smallest]) smallest = left;
            if (right < k && heap[right] < heap[smallest]) smallest = right;
            if (smallest == pos) break;
            float tmp = heap[pos]; heap[pos] = heap[smallest]; heap[smallest] = tmp;
            pos = smallest;
        }
    }

    /* Process remaining elements: if larger than heap root, replace and sift down */
    for (int i = k; i < vocab_size; i++) {
        if (logits[i] > heap[0]) {
            heap[0] = logits[i];
            int pos = 0;
            while (1) {
                int smallest = pos;
                int left = 2 * pos + 1;
                int right = 2 * pos + 2;
                if (left < k && heap[left] < heap[smallest]) smallest = left;
                if (right < k && heap[right] < heap[smallest]) smallest = right;
                if (smallest == pos) break;
                float tmp = heap[pos]; heap[pos] = heap[smallest]; heap[smallest] = tmp;
                pos = smallest;
            }
        }
    }

    /* heap[0] is now the k-th largest value */
    float threshold = heap[0];
    free(heap);

    /* Set logits below threshold to -inf */
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] < threshold) {
            logits[i] = -INFINITY;
        }
    }
}

void gemma3_topp_filter(float *logits, int vocab_size, float p) {
    if (p <= 0.0f || p >= 1.0f) return;

    // Create indexed array
    IndexedFloat *indexed = (IndexedFloat *)malloc(vocab_size * sizeof(IndexedFloat));
    if (!indexed) return;

    // First apply softmax to get probabilities
    float *probs = (float *)malloc(vocab_size * sizeof(float));
    if (!probs) {
        free(indexed);
        return;
    }

    gemma3_softmax(probs, logits, vocab_size);

    for (int i = 0; i < vocab_size; i++) {
        indexed[i].value = probs[i];
        indexed[i].index = i;
    }

    // Sort by probability descending
    qsort(indexed, vocab_size, sizeof(IndexedFloat), compare_indexed_float_desc);

    // Find cutoff point
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += indexed[i].value;
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }

    // Create set of indices to keep
    int *keep = (int *)calloc(vocab_size, sizeof(int));
    if (!keep) {
        free(indexed);
        free(probs);
        return;
    }

    for (int i = 0; i < cutoff; i++) {
        keep[indexed[i].index] = 1;
    }

    // Set non-kept logits to -inf
    for (int i = 0; i < vocab_size; i++) {
        if (!keep[i]) {
            logits[i] = -INFINITY;
        }
    }

    free(indexed);
    free(probs);
    free(keep);
}

int gemma3_sample(const float *probs, int vocab_size) {
    float r = gemma3_random();
    float cumsum = 0.0f;

    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
            return i;
        }
    }

    // Fallback to last token (shouldn't happen with proper probs)
    return vocab_size - 1;
}

int gemma3_argmax(const float *x, int n) {
    int max_idx = 0;
    float max_val = x[0];

    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }

    return max_idx;
}

/* ============================================================================
 * Utility Functions
 * ========================================================================== */

float gemma3_vec_sum(const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

float gemma3_vec_max(const float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    return max_val;
}

float gemma3_dot(const float *a, const float *b, int n) {
#ifdef USE_BLAS
    return cblas_sdot(n, a, 1, b, 1);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

void gemma3_set_seed(uint64_t seed) {
    g_rng_state = seed;
}

float gemma3_random(void) {
    // xorshift64* algorithm
    g_rng_state ^= g_rng_state >> 12;
    g_rng_state ^= g_rng_state << 25;
    g_rng_state ^= g_rng_state >> 27;
    uint64_t result = g_rng_state * 0x2545F4914F6CDD1DULL;

    // Convert to float in [0, 1)
    return (float)(result >> 11) * (1.0f / 9007199254740992.0f);
}
