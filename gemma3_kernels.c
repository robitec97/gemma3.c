/*
 * gemma3_kernels.c - CPU compute kernel implementations for Gemma 3 inference
 *
 * The forward pass is dominated by BF16-weight projections. Decoding a single
 * token is memory-bandwidth bound (every weight is read once), while prefill
 * is compute bound once several tokens share each weight load. The kernels
 * below therefore come in two shapes:
 *   - matvec: 4 weight rows per iteration, wide loads, multiple accumulators
 *   - matmul: register-tiled (rows x tokens) micro-kernel for batched prefill
 * Both have NEON (arm64) and AVX2+FMA (x86-64) implementations and a portable
 * scalar fallback. BF16 -> F32 conversion is a 16-bit left shift, done in
 * registers on the fly.
 */

#include "gemma3_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define G3_NEON 1
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define G3_AVX2 1
#endif

const char *gemma3_simd_name(void) {
#if defined(G3_NEON)
    return "neon";
#elif defined(G3_AVX2)
    return "avx2";
#else
    return "scalar";
#endif
}

/* Random state for the legacy global sampler API */
static uint64_t g_rng_state = 12345678901234567ULL;

static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = ((uint32_t)bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

/* ============================================================================
 * SIMD helpers
 * ========================================================================== */

#if defined(G3_NEON)

static inline float32x4_t bf16x4_to_f32(uint16x4_t v) {
    return vreinterpretq_f32_u32(vshll_n_u16(v, 16));
}
static inline float32x4_t bf16x8_lo(uint16x8_t v) {
    return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16));
}
static inline float32x4_t bf16x8_hi(uint16x8_t v) {
    return vreinterpretq_f32_u32(vshll_high_n_u16(v, 16));
}

/* expf for 4 lanes (Cephes polynomial, ~2 ulp on the clamped range) */
static inline float32x4_t exp_f32x4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.3f));
    float32x4_t fn = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
    float32x4_t r = vfmsq_f32(x, fn, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, fn, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    p = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), p, vmulq_f32(r, r));
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fn), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

#elif defined(G3_AVX2)

static inline __m256 bf16x8_to_f32(__m128i v) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16));
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline __m256 exp_f32x8(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
    __m256 fn = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fn, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(fn, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fn), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

#endif

/* dot(a, b) over n floats */
static inline float dot_f32(const float *a, const float *b, int n) {
    int i = 0;
#if defined(G3_NEON)
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i),      vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
#elif defined(G3_AVX2)
    __m256 s0 = _mm256_setzero_ps(), s1 = s0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    float sum = hsum256(_mm256_add_ps(s0, s1));
#else
    float sum = 0.0f;
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* y += alpha * x */
static inline void axpy_f32(float *y, float alpha, const float *x, int n) {
    int i = 0;
#if defined(G3_NEON)
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i,     vfmaq_f32(vld1q_f32(y + i),     va, vld1q_f32(x + i)));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), va, vld1q_f32(x + i + 4)));
    }
#elif defined(G3_AVX2)
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
#endif
    for (; i < n; i++) y[i] += alpha * x[i];
}

static inline float sumsq_f32(const float *x, int n) {
    return dot_f32(x, x, n);
}

/* ============================================================================
 * Basic Tensor Operations
 * ========================================================================== */

void gemma3_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        float *c = C + (size_t)i * N;
        memset(c, 0, (size_t)N * sizeof(float));
        for (int k = 0; k < K; k++) {
            axpy_f32(c, A[(size_t)i * K + k], B + (size_t)k * N, N);
        }
    }
}

void gemma3_matvec(float *y, const float *A, const float *x, int M, int K) {
    for (int i = 0; i < M; i++) {
        y[i] = dot_f32(A + (size_t)i * K, x, K);
    }
}

void gemma3_matvec_batched(float *y, const float *A, const float *x,
                           int batch, int M, int K) {
    for (int b = 0; b < batch; b++) {
        gemma3_matvec(y + (size_t)b * M, A + (size_t)b * M * K, x + (size_t)b * K, M, K);
    }
}

void gemma3_vec_add(float *y, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) y[i] = a[i] + b[i];
}

void gemma3_vec_mul(float *y, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) y[i] = a[i] * b[i];
}

void gemma3_vec_scale(float *y, const float *x, float scale, int n) {
    for (int i = 0; i < n; i++) y[i] = x[i] * scale;
}

void gemma3_vec_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

void gemma3_vec_zero(float *x, int n) {
    memset(x, 0, (size_t)n * sizeof(float));
}

/* ============================================================================
 * Normalization
 * ========================================================================== */

void gemma3_rmsnorm(float *y, const float *x, const float *weight, int n, float eps) {
    float rs = 1.0f / sqrtf(sumsq_f32(x, n) / (float)n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * rs * weight[i];
}

void gemma3_rmsnorm_inplace(float *x, const float *weight, int n, float eps) {
    gemma3_rmsnorm(x, x, weight, n, eps);
}

void gemma3_rmsnorm_bf16(float *y, const float *x, const uint16_t *weight, int n, float eps) {
    float rs = 1.0f / sqrtf(sumsq_f32(x, n) / (float)n + eps);
    int i = 0;
#if defined(G3_NEON)
    float32x4_t vrs = vdupq_n_f32(rs), one = vdupq_n_f32(1.0f);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t w = vld1q_u16(weight + i);
        float32x4_t w0 = vaddq_f32(one, bf16x8_lo(w));
        float32x4_t w1 = vaddq_f32(one, bf16x8_hi(w));
        vst1q_f32(y + i,     vmulq_f32(vmulq_f32(vld1q_f32(x + i),     vrs), w0));
        vst1q_f32(y + i + 4, vmulq_f32(vmulq_f32(vld1q_f32(x + i + 4), vrs), w1));
    }
#endif
    for (; i < n; i++) y[i] = x[i] * rs * (1.0f + bf16_to_f32(weight[i]));
}

void gemma3_rmsnorm_bf16_inplace(float *x, const uint16_t *weight, int n, float eps) {
    gemma3_rmsnorm_bf16(x, x, weight, n, eps);
}

/* ============================================================================
 * BF16 matrix-vector product (decode path, bandwidth bound)
 * ========================================================================== */

static inline float dot_bf16_scalar(const uint16_t *a, const float *x, int K) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        s0 += bf16_to_f32(a[k])     * x[k];
        s1 += bf16_to_f32(a[k + 1]) * x[k + 1];
        s2 += bf16_to_f32(a[k + 2]) * x[k + 2];
        s3 += bf16_to_f32(a[k + 3]) * x[k + 3];
    }
    for (; k < K; k++) s0 += bf16_to_f32(a[k]) * x[k];
    return (s0 + s1) + (s2 + s3);
}

void gemma3_matvec_bf16_range(float *y, const uint16_t *A, const float *x,
                              int row_start, int row_end, int K) {
    int r = row_start;
#if defined(G3_NEON)
    for (; r + 4 <= row_end; r += 4) {
        const uint16_t *a0 = A + (size_t)r * K;
        const uint16_t *a1 = a0 + K, *a2 = a1 + K, *a3 = a2 + K;
        float32x4_t s0a = vdupq_n_f32(0.0f), s0b = s0a, s1a = s0a, s1b = s0a;
        float32x4_t s2a = s0a, s2b = s0a, s3a = s0a, s3b = s0a;
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            float32x4_t xl = vld1q_f32(x + k), xh = vld1q_f32(x + k + 4);
            uint16x8_t w0 = vld1q_u16(a0 + k), w1 = vld1q_u16(a1 + k);
            uint16x8_t w2 = vld1q_u16(a2 + k), w3 = vld1q_u16(a3 + k);
            s0a = vfmaq_f32(s0a, bf16x8_lo(w0), xl); s0b = vfmaq_f32(s0b, bf16x8_hi(w0), xh);
            s1a = vfmaq_f32(s1a, bf16x8_lo(w1), xl); s1b = vfmaq_f32(s1b, bf16x8_hi(w1), xh);
            s2a = vfmaq_f32(s2a, bf16x8_lo(w2), xl); s2b = vfmaq_f32(s2b, bf16x8_hi(w2), xh);
            s3a = vfmaq_f32(s3a, bf16x8_lo(w3), xl); s3b = vfmaq_f32(s3b, bf16x8_hi(w3), xh);
        }
        float r0 = vaddvq_f32(vaddq_f32(s0a, s0b));
        float r1 = vaddvq_f32(vaddq_f32(s1a, s1b));
        float r2 = vaddvq_f32(vaddq_f32(s2a, s2b));
        float r3 = vaddvq_f32(vaddq_f32(s3a, s3b));
        for (; k < K; k++) {
            r0 += bf16_to_f32(a0[k]) * x[k];
            r1 += bf16_to_f32(a1[k]) * x[k];
            r2 += bf16_to_f32(a2[k]) * x[k];
            r3 += bf16_to_f32(a3[k]) * x[k];
        }
        y[r] = r0; y[r + 1] = r1; y[r + 2] = r2; y[r + 3] = r3;
    }
#elif defined(G3_AVX2)
    for (; r + 4 <= row_end; r += 4) {
        const uint16_t *a0 = A + (size_t)r * K;
        const uint16_t *a1 = a0 + K, *a2 = a1 + K, *a3 = a2 + K;
        __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            __m256 xv = _mm256_loadu_ps(x + k);
            s0 = _mm256_fmadd_ps(bf16x8_to_f32(_mm_loadu_si128((const __m128i *)(a0 + k))), xv, s0);
            s1 = _mm256_fmadd_ps(bf16x8_to_f32(_mm_loadu_si128((const __m128i *)(a1 + k))), xv, s1);
            s2 = _mm256_fmadd_ps(bf16x8_to_f32(_mm_loadu_si128((const __m128i *)(a2 + k))), xv, s2);
            s3 = _mm256_fmadd_ps(bf16x8_to_f32(_mm_loadu_si128((const __m128i *)(a3 + k))), xv, s3);
        }
        float r0 = hsum256(s0), r1 = hsum256(s1), r2 = hsum256(s2), r3 = hsum256(s3);
        for (; k < K; k++) {
            r0 += bf16_to_f32(a0[k]) * x[k];
            r1 += bf16_to_f32(a1[k]) * x[k];
            r2 += bf16_to_f32(a2[k]) * x[k];
            r3 += bf16_to_f32(a3[k]) * x[k];
        }
        y[r] = r0; y[r + 1] = r1; y[r + 2] = r2; y[r + 3] = r3;
    }
#endif
    for (; r < row_end; r++) {
        y[r] = dot_bf16_scalar(A + (size_t)r * K, x, K);
    }
}

void gemma3_matvec_bf16(float *y, const uint16_t *A, const float *x, int M, int K,
                        float *scratch) {
    (void)scratch;
    gemma3_matvec_bf16_range(y, A, x, 0, M, K);
}

typedef struct {
    float *y;
    const uint16_t *A;
    const float *x;
    int K;
} matvec_task;

static void matvec_range_fn(void *arg, int start, int end) {
    matvec_task *t = (matvec_task *)arg;
    gemma3_matvec_bf16_range(t->y, t->A, t->x, start, end, t->K);
}

/* Rows per scheduling chunk: a multiple of 4 that gives each thread several
 * chunks so faster cores can pick up more of the work. */
static int row_chunk(int M, gemma3_thread_pool *pool) {
    int nt = gemma3_thread_pool_size(pool);
    int chunk = M / (nt * 8);
    if (chunk < 16) chunk = 16;
    chunk = (chunk + 3) & ~3;
    return chunk;
}

void gemma3_matvec_bf16_mt(float *y, const uint16_t *A, const float *x, int M, int K,
                           float *scratch, gemma3_thread_pool *pool) {
    (void)scratch;
    matvec_task t = { y, A, x, K };
    gemma3_parallel_for(pool, M, row_chunk(M, pool), matvec_range_fn, &t);
}

/* ============================================================================
 * BF16 batched projection (prefill path, compute bound)
 * ========================================================================== */

void gemma3_matmul_bf16_range(float *Y, int ldy, const float *X, int N,
                              const uint16_t *W, int K, int row_start, int row_end) {
    int m = row_start;
#if defined(G3_NEON)
    /* 4 weight rows x 4 tokens register tile: 16 accumulators */
    for (; m + 4 <= row_end; m += 4) {
        const uint16_t *w0 = W + (size_t)m * K;
        const uint16_t *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
        int n = 0;
        for (; n + 4 <= N; n += 4) {
            const float *x0 = X + (size_t)n * K;
            const float *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
            float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00, c03 = c00;
            float32x4_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;
            float32x4_t c20 = c00, c21 = c00, c22 = c00, c23 = c00;
            float32x4_t c30 = c00, c31 = c00, c32 = c00, c33 = c00;
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t a0 = bf16x4_to_f32(vld1_u16(w0 + k));
                float32x4_t a1 = bf16x4_to_f32(vld1_u16(w1 + k));
                float32x4_t a2 = bf16x4_to_f32(vld1_u16(w2 + k));
                float32x4_t a3 = bf16x4_to_f32(vld1_u16(w3 + k));
                float32x4_t b;
                b = vld1q_f32(x0 + k);
                c00 = vfmaq_f32(c00, a0, b); c10 = vfmaq_f32(c10, a1, b);
                c20 = vfmaq_f32(c20, a2, b); c30 = vfmaq_f32(c30, a3, b);
                b = vld1q_f32(x1 + k);
                c01 = vfmaq_f32(c01, a0, b); c11 = vfmaq_f32(c11, a1, b);
                c21 = vfmaq_f32(c21, a2, b); c31 = vfmaq_f32(c31, a3, b);
                b = vld1q_f32(x2 + k);
                c02 = vfmaq_f32(c02, a0, b); c12 = vfmaq_f32(c12, a1, b);
                c22 = vfmaq_f32(c22, a2, b); c32 = vfmaq_f32(c32, a3, b);
                b = vld1q_f32(x3 + k);
                c03 = vfmaq_f32(c03, a0, b); c13 = vfmaq_f32(c13, a1, b);
                c23 = vfmaq_f32(c23, a2, b); c33 = vfmaq_f32(c33, a3, b);
            }
            float r[4][4] = {
                { vaddvq_f32(c00), vaddvq_f32(c01), vaddvq_f32(c02), vaddvq_f32(c03) },
                { vaddvq_f32(c10), vaddvq_f32(c11), vaddvq_f32(c12), vaddvq_f32(c13) },
                { vaddvq_f32(c20), vaddvq_f32(c21), vaddvq_f32(c22), vaddvq_f32(c23) },
                { vaddvq_f32(c30), vaddvq_f32(c31), vaddvq_f32(c32), vaddvq_f32(c33) },
            };
            for (; k < K; k++) {
                float a[4] = { bf16_to_f32(w0[k]), bf16_to_f32(w1[k]),
                               bf16_to_f32(w2[k]), bf16_to_f32(w3[k]) };
                float b[4] = { x0[k], x1[k], x2[k], x3[k] };
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++) r[i][j] += a[i] * b[j];
            }
            for (int j = 0; j < 4; j++) {
                float *yrow = Y + (size_t)(n + j) * ldy + m;
                yrow[0] = r[0][j]; yrow[1] = r[1][j]; yrow[2] = r[2][j]; yrow[3] = r[3][j];
            }
        }
        /* leftover tokens: 4-row dot products per token */
        for (; n < N; n++) {
            float tmp[4];
            const float *xn = X + (size_t)n * K;
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t b = vld1q_f32(xn + k);
                s0 = vfmaq_f32(s0, bf16x4_to_f32(vld1_u16(w0 + k)), b);
                s1 = vfmaq_f32(s1, bf16x4_to_f32(vld1_u16(w1 + k)), b);
                s2 = vfmaq_f32(s2, bf16x4_to_f32(vld1_u16(w2 + k)), b);
                s3 = vfmaq_f32(s3, bf16x4_to_f32(vld1_u16(w3 + k)), b);
            }
            tmp[0] = vaddvq_f32(s0); tmp[1] = vaddvq_f32(s1);
            tmp[2] = vaddvq_f32(s2); tmp[3] = vaddvq_f32(s3);
            for (; k < K; k++) {
                tmp[0] += bf16_to_f32(w0[k]) * xn[k];
                tmp[1] += bf16_to_f32(w1[k]) * xn[k];
                tmp[2] += bf16_to_f32(w2[k]) * xn[k];
                tmp[3] += bf16_to_f32(w3[k]) * xn[k];
            }
            float *yrow = Y + (size_t)n * ldy + m;
            yrow[0] = tmp[0]; yrow[1] = tmp[1]; yrow[2] = tmp[2]; yrow[3] = tmp[3];
        }
    }
#elif defined(G3_AVX2)
    /* 2 weight rows x 4 tokens register tile (AVX2 has 16 vector registers) */
    for (; m + 2 <= row_end; m += 2) {
        const uint16_t *w0 = W + (size_t)m * K;
        const uint16_t *w1 = w0 + K;
        int n = 0;
        for (; n + 4 <= N; n += 4) {
            const float *x0 = X + (size_t)n * K;
            const float *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
            __m256 c00 = _mm256_setzero_ps(), c01 = c00, c02 = c00, c03 = c00;
            __m256 c10 = c00, c11 = c00, c12 = c00, c13 = c00;
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 a0 = bf16x8_to_f32(_mm_loadu_si128((const __m128i *)(w0 + k)));
                __m256 a1 = bf16x8_to_f32(_mm_loadu_si128((const __m128i *)(w1 + k)));
                __m256 b;
                b = _mm256_loadu_ps(x0 + k);
                c00 = _mm256_fmadd_ps(a0, b, c00); c10 = _mm256_fmadd_ps(a1, b, c10);
                b = _mm256_loadu_ps(x1 + k);
                c01 = _mm256_fmadd_ps(a0, b, c01); c11 = _mm256_fmadd_ps(a1, b, c11);
                b = _mm256_loadu_ps(x2 + k);
                c02 = _mm256_fmadd_ps(a0, b, c02); c12 = _mm256_fmadd_ps(a1, b, c12);
                b = _mm256_loadu_ps(x3 + k);
                c03 = _mm256_fmadd_ps(a0, b, c03); c13 = _mm256_fmadd_ps(a1, b, c13);
            }
            float r[2][4] = {
                { hsum256(c00), hsum256(c01), hsum256(c02), hsum256(c03) },
                { hsum256(c10), hsum256(c11), hsum256(c12), hsum256(c13) },
            };
            for (; k < K; k++) {
                float a0 = bf16_to_f32(w0[k]), a1 = bf16_to_f32(w1[k]);
                r[0][0] += a0 * x0[k]; r[0][1] += a0 * x1[k]; r[0][2] += a0 * x2[k]; r[0][3] += a0 * x3[k];
                r[1][0] += a1 * x0[k]; r[1][1] += a1 * x1[k]; r[1][2] += a1 * x2[k]; r[1][3] += a1 * x3[k];
            }
            for (int j = 0; j < 4; j++) {
                float *yrow = Y + (size_t)(n + j) * ldy + m;
                yrow[0] = r[0][j]; yrow[1] = r[1][j];
            }
        }
        for (; n < N; n++) {
            const float *xn = X + (size_t)n * K;
            Y[(size_t)n * ldy + m]     = dot_bf16_scalar(w0, xn, K);
            Y[(size_t)n * ldy + m + 1] = dot_bf16_scalar(w1, xn, K);
        }
    }
#endif
    /* Remaining rows (and the whole range on scalar builds) */
    for (; m < row_end; m++) {
        const uint16_t *w = W + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            Y[(size_t)n * ldy + m] = dot_bf16_scalar(w, X + (size_t)n * K, K);
        }
    }
}

typedef struct {
    float *Y;
    const float *X;
    int N;
    const uint16_t *W;
    int M;
    int K;
} matmul_task;

static void matmul_range_fn(void *arg, int start, int end) {
    matmul_task *t = (matmul_task *)arg;
    /* Process tokens in blocks so the X rows a tile touches stay in cache */
    const int NB = 32;
    for (int n0 = 0; n0 < t->N; n0 += NB) {
        int nb = t->N - n0 < NB ? t->N - n0 : NB;
        gemma3_matmul_bf16_range(t->Y + (size_t)n0 * t->M, t->M, t->X + (size_t)n0 * t->K,
                                 nb, t->W, t->K, start, end);
    }
}

void gemma3_matmul_bf16_mt(float *Y, const float *X, int N, const uint16_t *W,
                           int M, int K, gemma3_thread_pool *pool) {
    if (N == 1) {
        gemma3_matvec_bf16_mt(Y, W, X, M, K, NULL, pool);
        return;
    }
    matmul_task t = { Y, X, N, W, M, K };
    gemma3_parallel_for(pool, M, row_chunk(M, pool), matmul_range_fn, &t);
}

void gemma3_embed_bf16(float *output, const uint16_t *embed, int token_id, int hidden_size) {
    gemma3_bf16_to_f32(output, embed + (size_t)token_id * hidden_size, hidden_size);
}

/* ============================================================================
 * Activation Functions
 * ========================================================================== */

/* gelu_tanh(x) = 0.5 x (1 + tanh(u)) = x / (1 + exp(-2u)),
 * u = sqrt(2/pi) (x + 0.044715 x^3) */
static inline float gelu_scalar(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float u = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(u));
}

void gemma3_gelu_tanh_mul(float *gate, const float *up, int n) {
    int i = 0;
#if defined(G3_NEON)
    const float32x4_t c0 = vdupq_n_f32(-2.0f * 0.7978845608028654f);
    const float32x4_t c1 = vdupq_n_f32(0.044715f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(gate + i);
        float32x4_t x3 = vmulq_f32(vmulq_f32(x, x), x);
        float32x4_t t = vmulq_f32(c0, vfmaq_f32(x, c1, x3));
        float32x4_t g = vdivq_f32(x, vaddq_f32(one, exp_f32x4(t)));
        vst1q_f32(gate + i, up ? vmulq_f32(g, vld1q_f32(up + i)) : g);
    }
#elif defined(G3_AVX2)
    const __m256 c0 = _mm256_set1_ps(-2.0f * 0.7978845608028654f);
    const __m256 c1 = _mm256_set1_ps(0.044715f);
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(gate + i);
        __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        __m256 t = _mm256_mul_ps(c0, _mm256_fmadd_ps(c1, x3, x));
        __m256 g = _mm256_div_ps(x, _mm256_add_ps(one, exp_f32x8(t)));
        _mm256_storeu_ps(gate + i, up ? _mm256_mul_ps(g, _mm256_loadu_ps(up + i)) : g);
    }
#endif
    for (; i < n; i++) {
        float g = gelu_scalar(gate[i]);
        gate[i] = up ? g * up[i] : g;
    }
}

void gemma3_gelu_tanh(float *y, const float *x, int n) {
    if (y != x) memcpy(y, x, (size_t)n * sizeof(float));
    gemma3_gelu_tanh_mul(y, NULL, n);
}

void gemma3_gelu_tanh_inplace(float *x, int n) {
    gemma3_gelu_tanh_mul(x, NULL, n);
}

void gemma3_silu(float *y, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        y[i] = xi / (1.0f + expf(-xi));
    }
}

void gemma3_silu_inplace(float *x, int n) {
    gemma3_silu(x, x, n);
}

/* x[i] = exp(x[i] - max); returns the sum */
static float exp_shift_sum(float *x, int n, float max_val) {
    int i = 0;
    float sum = 0.0f;
#if defined(G3_NEON)
    float32x4_t vm = vdupq_n_f32(max_val), vs = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t e = exp_f32x4(vsubq_f32(vld1q_f32(x + i), vm));
        vst1q_f32(x + i, e);
        vs = vaddq_f32(vs, e);
    }
    sum = vaddvq_f32(vs);
#elif defined(G3_AVX2)
    __m256 vm = _mm256_set1_ps(max_val), vs = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 e = exp_f32x8(_mm256_sub_ps(_mm256_loadu_ps(x + i), vm));
        _mm256_storeu_ps(x + i, e);
        vs = _mm256_add_ps(vs, e);
    }
    sum = hsum256(vs);
#endif
    for (; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    return sum;
}

void gemma3_softmax_inplace(float *x, int n) {
    if (n <= 0) return;
    float max_val = gemma3_vec_max(x, n);
    if (max_val == -INFINITY) {
        float u = 1.0f / (float)n;
        for (int i = 0; i < n; i++) x[i] = u;
        return;
    }
    float sum = exp_shift_sum(x, n, max_val);
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

void gemma3_softmax(float *y, const float *x, int n) {
    if (y != x) memcpy(y, x, (size_t)n * sizeof(float));
    gemma3_softmax_inplace(y, n);
}

/* ============================================================================
 * Positional Encoding (RoPE)
 * ========================================================================== */

void gemma3_rope_single(float *x, int head_dim, int pos, float theta) {
    int half_dim = head_dim / 2;
    for (int i = 0; i < half_dim; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        float x0 = x[i], x1 = x[i + half_dim];
        x[i] = x0 * c - x1 * s;
        x[i + half_dim] = x0 * s + x1 * c;
    }
}

void gemma3_rope(float *q, float *k, int n_heads, int n_kv_heads,
                 int head_dim, int pos, float theta) {
    for (int h = 0; h < n_heads; h++) gemma3_rope_single(q + h * head_dim, head_dim, pos, theta);
    for (int h = 0; h < n_kv_heads; h++) gemma3_rope_single(k + h * head_dim, head_dim, pos, theta);
}

void gemma3_rope_precompute(float *freqs, int max_pos, int head_dim, float theta,
                            float scaling) {
    int half_dim = head_dim / 2;
    if (scaling <= 0.0f) scaling = 1.0f;
    for (int i = 0; i < half_dim; i++) {
        double inv_freq = 1.0 / pow((double)theta, (double)(2 * i) / (double)head_dim);
        inv_freq /= (double)scaling;
        for (int pos = 0; pos < max_pos; pos++) {
            double angle = (double)pos * inv_freq;
            freqs[((size_t)pos * half_dim + i) * 2] = (float)cos(angle);
            freqs[((size_t)pos * half_dim + i) * 2 + 1] = (float)sin(angle);
        }
    }
}

void gemma3_rope_apply_precomputed(float *x, const float *freqs, int head_dim, int pos) {
    int half_dim = head_dim / 2;
    const float *pf = freqs + (size_t)pos * half_dim * 2;
    for (int i = 0; i < half_dim; i++) {
        float c = pf[i * 2], s = pf[i * 2 + 1];
        float x0 = x[i], x1 = x[i + half_dim];
        x[i] = x0 * c - x1 * s;
        x[i + half_dim] = x0 * s + x1 * c;
    }
}

/* ============================================================================
 * Attention
 * ========================================================================== */

void gemma3_attention_head(float *out, const float *q,
                           const float *k_cache, const float *v_cache,
                           int kv_stride, int pos_start, int pos_end, int ring,
                           int head_dim, float scale, float *scores) {
    int n = pos_end - pos_start + 1;
    if (n <= 0) {
        memset(out, 0, (size_t)head_dim * sizeof(float));
        return;
    }

    int row = ring > 0 ? pos_start % ring : pos_start;
    float max_val = -INFINITY;
    for (int i = 0; i < n; i++) {
        float s = dot_f32(q, k_cache + (size_t)row * kv_stride, head_dim) * scale;
        scores[i] = s;
        if (s > max_val) max_val = s;
        if (++row == ring) row = 0;
    }

    float inv_sum = 1.0f / exp_shift_sum(scores, n, max_val);

    memset(out, 0, (size_t)head_dim * sizeof(float));
    row = ring > 0 ? pos_start % ring : pos_start;
    for (int i = 0; i < n; i++) {
        axpy_f32(out, scores[i] * inv_sum, v_cache + (size_t)row * kv_stride, head_dim);
        if (++row == ring) row = 0;
    }
}

void gemma3_attention_single(float *output, const float *q,
                             const float *k_cache, const float *v_cache,
                             int seq_len, int head_dim, float scale,
                             const float *mask) {
    float *scores = (float *)malloc((size_t)seq_len * sizeof(float));
    if (!scores) return;
    for (int i = 0; i < seq_len; i++) {
        scores[i] = dot_f32(q, k_cache + (size_t)i * head_dim, head_dim) * scale;
        if (mask) scores[i] += mask[i];
    }
    gemma3_softmax_inplace(scores, seq_len);
    memset(output, 0, (size_t)head_dim * sizeof(float));
    for (int i = 0; i < seq_len; i++) {
        axpy_f32(output, scores[i], v_cache + (size_t)i * head_dim, head_dim);
    }
    free(scores);
}

void gemma3_gqa(float *output, const float *q,
                const float *k_cache, const float *v_cache,
                int n_heads, int n_kv_heads, int seq_len, int head_dim,
                float scale, const float *mask, float *scores_buf) {
    int heads_per_group = n_heads / n_kv_heads;
    int kv_stride = n_kv_heads * head_dim;
    float *scores = scores_buf ? scores_buf : (float *)malloc((size_t)seq_len * sizeof(float));
    if (!scores) return;

    for (int h = 0; h < n_heads; h++) {
        int kv_head = h / heads_per_group;
        const float *qh = q + (size_t)h * head_dim;
        float *oh = output + (size_t)h * head_dim;
        for (int i = 0; i < seq_len; i++) {
            scores[i] = dot_f32(qh, k_cache + (size_t)i * kv_stride + kv_head * head_dim,
                                head_dim) * scale;
            if (mask) scores[i] += mask[i];
        }
        gemma3_softmax_inplace(scores, seq_len);
        memset(oh, 0, (size_t)head_dim * sizeof(float));
        for (int i = 0; i < seq_len; i++) {
            axpy_f32(oh, scores[i], v_cache + (size_t)i * kv_stride + kv_head * head_dim, head_dim);
        }
    }
    if (!scores_buf) free(scores);
}

void gemma3_sliding_window_mask(float *mask, int query_pos, int window_size) {
    int start = (query_pos >= window_size) ? query_pos - window_size + 1 : 0;
    for (int i = 0; i <= query_pos; i++) {
        mask[i] = (i >= start) ? 0.0f : -INFINITY;
    }
}

void gemma3_causal_mask(float *mask, int seq_len, int query_pos) {
    for (int i = 0; i < seq_len; i++) {
        mask[i] = (i <= query_pos) ? 0.0f : -INFINITY;
    }
}

/* ============================================================================
 * Data Type Conversions
 * ========================================================================== */

void gemma3_bf16_to_f32(float *f32, const uint16_t *bf16, int n) {
    int i = 0;
#if defined(G3_NEON)
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vld1q_u16(bf16 + i);
        vst1q_f32(f32 + i, bf16x8_lo(v));
        vst1q_f32(f32 + i + 4, bf16x8_hi(v));
    }
#endif
    for (; i < n; i++) f32[i] = bf16_to_f32(bf16[i]);
}

void gemma3_f32_to_bf16(uint16_t *bf16, const float *f32, int n) {
    for (int i = 0; i < n; i++) bf16[i] = gemma3_f32_to_bf16_single(f32[i]);
}

/* ============================================================================
 * Sampling
 * ========================================================================== */

typedef struct {
    float value;
    int index;
} IndexedFloat;

struct gemma3_sampler {
    int vocab_size;
    float *val;            /* heap of candidate logits */
    int *idx;              /* candidate token ids */
    IndexedFloat *cand;    /* sorted candidates */
};

gemma3_sampler *gemma3_sampler_create(int vocab_size) {
    gemma3_sampler *s = (gemma3_sampler *)calloc(1, sizeof(gemma3_sampler));
    if (!s) return NULL;
    s->vocab_size = vocab_size;
    s->val = (float *)malloc((size_t)vocab_size * sizeof(float));
    s->idx = (int *)malloc((size_t)vocab_size * sizeof(int));
    s->cand = (IndexedFloat *)malloc((size_t)vocab_size * sizeof(IndexedFloat));
    if (!s->val || !s->idx || !s->cand) {
        gemma3_sampler_free(s);
        return NULL;
    }
    return s;
}

void gemma3_sampler_free(gemma3_sampler *s) {
    if (!s) return;
    free(s->val);
    free(s->idx);
    free(s->cand);
    free(s);
}

float gemma3_random_r(uint64_t *state) {
    /* xorshift64* */
    uint64_t x = *state ? *state : 0x9E3779B97F4A7C15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    uint64_t r = x * 0x2545F4914F6CDD1DULL;
    return (float)(r >> 40) * (1.0f / 16777216.0f);  /* 24 random bits -> [0, 1) */
}

/* Min-heap sift-down on (val, idx) pairs keyed by val */
static void heap_sift_down(float *val, int *idx, int n, int pos) {
    for (;;) {
        int l = 2 * pos + 1, r = l + 1, smallest = pos;
        if (l < n && val[l] < val[smallest]) smallest = l;
        if (r < n && val[r] < val[smallest]) smallest = r;
        if (smallest == pos) return;
        float tv = val[pos]; val[pos] = val[smallest]; val[smallest] = tv;
        int ti = idx[pos]; idx[pos] = idx[smallest]; idx[smallest] = ti;
        pos = smallest;
    }
}

static int compare_indexed_float_desc(const void *a, const void *b) {
    float va = ((const IndexedFloat *)a)->value;
    float vb = ((const IndexedFloat *)b)->value;
    if (va > vb) return -1;
    if (va < vb) return 1;
    return ((const IndexedFloat *)a)->index - ((const IndexedFloat *)b)->index;
}

int gemma3_sample_logits(gemma3_sampler *s, const float *logits, int vocab_size,
                         float temperature, int top_k, float top_p, float min_p,
                         uint64_t *rng_state) {
    if (temperature <= 0.0f || top_k == 1 || !s || vocab_size > s->vocab_size) {
        return gemma3_argmax(logits, vocab_size);
    }

    float *val = s->val;
    int *idx = s->idx;
    IndexedFloat *cand = s->cand;
    int n;

    if (top_k > 0 && top_k < vocab_size) {
        /* Keep the top_k largest logits with a min-heap: O(V log k) */
        n = top_k;
        for (int i = 0; i < n; i++) { val[i] = logits[i]; idx[i] = i; }
        for (int i = n / 2 - 1; i >= 0; i--) heap_sift_down(val, idx, n, i);
        for (int i = n; i < vocab_size; i++) {
            if (logits[i] > val[0]) {
                val[0] = logits[i];
                idx[0] = i;
                heap_sift_down(val, idx, n, 0);
            }
        }
        for (int i = 0; i < n; i++) { cand[i].value = val[i]; cand[i].index = idx[i]; }
    } else {
        n = vocab_size;
        for (int i = 0; i < n; i++) { cand[i].value = logits[i]; cand[i].index = i; }
    }

    /* Sort candidates by logit, descending (k is small in the common case) */
    qsort(cand, (size_t)n, sizeof(IndexedFloat), compare_indexed_float_desc);

    /* Softmax with temperature over the candidates */
    float inv_t = 1.0f / temperature;
    float top = cand[0].value;
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        float p = expf((cand[i].value - top) * inv_t);
        cand[i].value = p;
        total += p;
    }

    /* Nucleus (top-p): smallest prefix whose probability mass reaches top_p */
    int keep = n;
    if (top_p > 0.0f && top_p < 1.0f) {
        float target = top_p * total, cum = 0.0f;
        for (int i = 0; i < n; i++) {
            cum += cand[i].value;
            if (cum >= target) { keep = i + 1; break; }
        }
    }

    /* min-p: drop tokens less likely than min_p * p(best) */
    if (min_p > 0.0f && min_p < 1.0f) {
        float thresh = min_p * cand[0].value;
        int k = 1;
        while (k < keep && cand[k].value >= thresh) k++;
        keep = k;
    }

    float kept = 0.0f;
    for (int i = 0; i < keep; i++) kept += cand[i].value;

    float r = gemma3_random_r(rng_state) * kept;
    int chosen = cand[keep - 1].index;
    float cum = 0.0f;
    for (int i = 0; i < keep; i++) {
        cum += cand[i].value;
        if (r < cum) { chosen = cand[i].index; break; }
    }
    return chosen;
}

void gemma3_apply_temperature(float *logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f) return;
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) logits[i] *= inv_temp;
}

void gemma3_topk_filter(float *logits, int vocab_size, int k) {
    if (k <= 0 || k >= vocab_size) return;
    float *heap = (float *)malloc((size_t)k * sizeof(float));
    int *hidx = (int *)malloc((size_t)k * sizeof(int));
    if (!heap || !hidx) { free(heap); free(hidx); return; }
    for (int i = 0; i < k; i++) { heap[i] = logits[i]; hidx[i] = i; }
    for (int i = k / 2 - 1; i >= 0; i--) heap_sift_down(heap, hidx, k, i);
    for (int i = k; i < vocab_size; i++) {
        if (logits[i] > heap[0]) {
            heap[0] = logits[i];
            hidx[0] = i;
            heap_sift_down(heap, hidx, k, 0);
        }
    }
    float threshold = heap[0];
    free(heap);
    free(hidx);
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] < threshold) logits[i] = -INFINITY;
    }
}

void gemma3_topp_filter(float *logits, int vocab_size, float p) {
    if (p <= 0.0f || p >= 1.0f) return;
    IndexedFloat *indexed = (IndexedFloat *)malloc((size_t)vocab_size * sizeof(IndexedFloat));
    float *probs = (float *)malloc((size_t)vocab_size * sizeof(float));
    if (!indexed || !probs) { free(indexed); free(probs); return; }

    gemma3_softmax(probs, logits, vocab_size);
    int n = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (probs[i] > 0.0f) { indexed[n].value = probs[i]; indexed[n].index = i; n++; }
    }
    qsort(indexed, (size_t)n, sizeof(IndexedFloat), compare_indexed_float_desc);

    float cumsum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cumsum += indexed[i].value;
        if (cumsum > p) { cutoff = i + 1; break; }
    }
    /* Everything outside the nucleus becomes -inf */
    for (int i = 0; i < vocab_size; i++) probs[i] = -INFINITY;
    for (int i = 0; i < cutoff; i++) probs[indexed[i].index] = logits[indexed[i].index];
    memcpy(logits, probs, (size_t)vocab_size * sizeof(float));

    free(indexed);
    free(probs);
}

int gemma3_sample(const float *probs, int vocab_size) {
    float r = gemma3_random();
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    return vocab_size - 1;
}

int gemma3_argmax(const float *x, int n) {
    int max_idx = 0;
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) { max_val = x[i]; max_idx = i; }
    }
    return max_idx;
}

/* ============================================================================
 * Utility Functions
 * ========================================================================== */

float gemma3_vec_sum(const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum;
}

float gemma3_vec_max(const float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    return max_val;
}

float gemma3_dot(const float *a, const float *b, int n) {
    return dot_f32(a, b, n);
}

void gemma3_set_seed(uint64_t seed) {
    g_rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

float gemma3_random(void) {
    return gemma3_random_r(&g_rng_state);
}
