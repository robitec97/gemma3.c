/*
 * test_kernels.c - Unit tests for gemma3.c compute kernels (no model needed)
 *
 * Checks the SIMD kernels against straightforward double-precision
 * references, plus the sampler and the thread pool.
 *
 * Build & run:  make test
 */

#include "../gemma3_kernels.h"
#include "../gemma3_threads.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    g_checks++; \
    if (!(cond)) { \
        g_failures++; \
        printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } \
} while (0)

static uint64_t rng = 0x1234567887654321ULL;
static float frand(void) {  /* uniform in [-1, 1) */
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / 8388608.0)) - 1.0f;
}

static uint16_t to_bf16(float f) {
    return gemma3_f32_to_bf16_single(f);
}

static double bf16d(uint16_t b) {
    return (double)gemma3_bf16_to_f32_single(b);
}

static double rel_err(double got, double want, double scale) {
    return fabs(got - want) / (fabs(want) + scale);
}

/* ------------------------------------------------------------------------ */

static void test_matvec_bf16(gemma3_thread_pool *pool) {
    printf("matvec_bf16 (%s)\n", gemma3_simd_name());
    const int shapes[][2] = { {4, 8}, {7, 37}, {16, 2560}, {33, 10240}, {130, 2560}, {5, 3} };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int M = shapes[s][0], K = shapes[s][1];
        uint16_t *A = malloc((size_t)M * K * sizeof(uint16_t));
        float *x = malloc((size_t)K * sizeof(float));
        float *y = malloc((size_t)M * sizeof(float));
        float *y2 = malloc((size_t)M * sizeof(float));
        for (int i = 0; i < M * K; i++) A[i] = to_bf16(frand());
        for (int i = 0; i < K; i++) x[i] = frand();

        gemma3_matvec_bf16(y, A, x, M, K, NULL);
        gemma3_matvec_bf16_mt(y2, A, x, M, K, NULL, pool);
        double worst = 0, worst_mt = 0;
        for (int m = 0; m < M; m++) {
            double ref = 0;
            for (int k = 0; k < K; k++) ref += bf16d(A[(size_t)m * K + k]) * x[k];
            double e = rel_err(y[m], ref, sqrt((double)K) * 1e-2);
            double e2 = rel_err(y2[m], ref, sqrt((double)K) * 1e-2);
            if (e > worst) worst = e;
            if (e2 > worst_mt) worst_mt = e2;
        }
        CHECK(worst < 1e-4, "matvec M=%d K=%d rel err %.3g", M, K, worst);
        CHECK(worst_mt < 1e-4, "matvec_mt M=%d K=%d rel err %.3g", M, K, worst_mt);
        free(A); free(x); free(y); free(y2);
    }
}

static void test_matmul_bf16(gemma3_thread_pool *pool) {
    printf("matmul_bf16\n");
    const int shapes[][3] = { /* N, M, K */
        {1, 8, 16}, {2, 9, 33}, {4, 16, 64}, {5, 7, 2560}, {9, 40, 257}, {33, 64, 512}, {128, 36, 100}
    };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int N = shapes[s][0], M = shapes[s][1], K = shapes[s][2];
        uint16_t *W = malloc((size_t)M * K * sizeof(uint16_t));
        float *X = malloc((size_t)N * K * sizeof(float));
        float *Y = malloc((size_t)N * M * sizeof(float));
        float *Y2 = malloc((size_t)N * M * sizeof(float));
        for (int i = 0; i < M * K; i++) W[i] = to_bf16(frand());
        for (int i = 0; i < N * K; i++) X[i] = frand();

        gemma3_matmul_bf16_range(Y, M, X, N, W, K, 0, M);
        gemma3_matmul_bf16_mt(Y2, X, N, W, M, K, pool);
        double worst = 0, worst_mt = 0;
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                double ref = 0;
                for (int k = 0; k < K; k++) ref += bf16d(W[(size_t)m * K + k]) * X[(size_t)n * K + k];
                double e = rel_err(Y[(size_t)n * M + m], ref, sqrt((double)K) * 1e-2);
                double e2 = rel_err(Y2[(size_t)n * M + m], ref, sqrt((double)K) * 1e-2);
                if (e > worst) worst = e;
                if (e2 > worst_mt) worst_mt = e2;
            }
        }
        CHECK(worst < 1e-4, "matmul N=%d M=%d K=%d rel err %.3g", N, M, K, worst);
        CHECK(worst_mt < 1e-4, "matmul_mt N=%d M=%d K=%d rel err %.3g", N, M, K, worst_mt);
        free(W); free(X); free(Y); free(Y2);
    }
}

static void test_rmsnorm(void) {
    printf("rmsnorm_bf16\n");
    int n = 2563;
    float *x = malloc(n * sizeof(float)), *y = malloc(n * sizeof(float));
    uint16_t *w = malloc(n * sizeof(uint16_t));
    for (int i = 0; i < n; i++) { x[i] = 3.0f * frand(); w[i] = to_bf16(frand()); }
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    double rs = 1.0 / sqrt(ss / n + 1e-6);
    gemma3_rmsnorm_bf16(y, x, w, n, 1e-6f);
    double worst = 0;
    for (int i = 0; i < n; i++) {
        double ref = x[i] * rs * (1.0 + bf16d(w[i]));
        double e = fabs(y[i] - ref);
        if (e > worst) worst = e;
    }
    CHECK(worst < 1e-5, "rmsnorm abs err %.3g", worst);
    /* in-place */
    gemma3_rmsnorm_bf16(x, x, w, n, 1e-6f);
    CHECK(memcmp(x, y, n * sizeof(float)) == 0, "in-place rmsnorm differs");
    free(x); free(y); free(w);
}

static void test_gelu(void) {
    printf("gelu_tanh_mul\n");
    int n = 4099;
    float *g = malloc(n * sizeof(float)), *u = malloc(n * sizeof(float));
    float *g0 = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        g0[i] = g[i] = 12.0f * frand();   /* covers saturation on both sides */
        u[i] = frand();
    }
    g[0] = g0[0] = -100.0f; g[1] = g0[1] = 100.0f; g[2] = g0[2] = 0.0f;
    gemma3_gelu_tanh_mul(g, u, n);
    double worst = 0;
    for (int i = 0; i < n; i++) {
        double x = g0[i];
        /* x * sigmoid(2u) == 0.5 x (1 + tanh(u)) without cancellation */
        double ref = x / (1.0 + exp(-2.0 * 0.7978845608028654 * (x + 0.044715 * x * x * x))) * u[i];
        double e = fabs(g[i] - ref) / (fabs(ref) + 1e-6);
        if (e > worst) worst = e;
        CHECK(isfinite(g[i]), "gelu non-finite at x=%g", x);
    }
    CHECK(worst < 2e-5, "gelu rel err %.3g", worst);
    free(g); free(u); free(g0);
}

static void test_softmax(void) {
    printf("softmax\n");
    int n = 1001;
    float *x = malloc(n * sizeof(float)), *y = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = 20.0f * frand();
    double mx = -1e30, sum = 0;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    for (int i = 0; i < n; i++) sum += exp(x[i] - mx);
    gemma3_softmax(y, x, n);
    double worst = 0, total = 0;
    for (int i = 0; i < n; i++) {
        double ref = exp(x[i] - mx) / sum;
        double e = fabs(y[i] - ref) / (ref + 1e-12);
        if (ref > 1e-30 && e > worst) worst = e;
        total += y[i];
    }
    CHECK(worst < 1e-5, "softmax rel err %.3g", worst);
    CHECK(fabs(total - 1.0) < 1e-5, "softmax sums to %.8f", total);
    free(x); free(y);
}

static void test_attention_ring(void) {
    printf("attention_head (linear + ring buffer)\n");
    const int hd = 64, kv_stride = 128, ring = 40;
    const int lo = 70, hi = 100;  /* positions; ring rows = pos % 40 */
    float *k = malloc((size_t)ring * kv_stride * sizeof(float));
    float *v = malloc((size_t)ring * kv_stride * sizeof(float));
    float *kl = malloc((size_t)(hi + 1) * kv_stride * sizeof(float));
    float *vl = malloc((size_t)(hi + 1) * kv_stride * sizeof(float));
    float q[64], out[64], out_lin[64], scores[256];
    for (int i = 0; i < hd; i++) q[i] = frand();
    for (int p = 0; p <= hi; p++) {
        for (int d = 0; d < kv_stride; d++) {
            float kv = frand(), vv = frand();
            kl[(size_t)p * kv_stride + d] = kv;
            vl[(size_t)p * kv_stride + d] = vv;
            k[(size_t)(p % ring) * kv_stride + d] = kv;
            v[(size_t)(p % ring) * kv_stride + d] = vv;
        }
    }
    float scale = 0.125f;
    gemma3_attention_head(out, q, k, v, kv_stride, lo, hi, ring, hd, scale, scores);
    gemma3_attention_head(out_lin, q, kl, vl, kv_stride, lo, hi, 0, hd, scale, scores);

    /* naive reference over positions lo..hi */
    double s[64], mx = -1e30, sum = 0;
    for (int p = lo; p <= hi; p++) {
        double d = 0;
        for (int i = 0; i < hd; i++) d += (double)q[i] * kl[(size_t)p * kv_stride + i];
        s[p - lo] = d * scale;
        if (s[p - lo] > mx) mx = s[p - lo];
    }
    for (int p = lo; p <= hi; p++) { s[p - lo] = exp(s[p - lo] - mx); sum += s[p - lo]; }
    double worst = 0, worst_lin = 0;
    for (int i = 0; i < hd; i++) {
        double ref = 0;
        for (int p = lo; p <= hi; p++) ref += s[p - lo] / sum * vl[(size_t)p * kv_stride + i];
        if (fabs(out[i] - ref) > worst) worst = fabs(out[i] - ref);
        if (fabs(out_lin[i] - ref) > worst_lin) worst_lin = fabs(out_lin[i] - ref);
    }
    CHECK(worst < 1e-5, "ring attention abs err %.3g", worst);
    CHECK(worst_lin < 1e-5, "linear attention abs err %.3g", worst_lin);
    free(k); free(v); free(kl); free(vl);
}

static void test_rope(void) {
    printf("rope_precompute (linear scaling)\n");
    int hd = 256, maxp = 64;
    float *f1 = malloc((size_t)maxp * hd * sizeof(float));
    float *f8 = malloc((size_t)maxp * hd * sizeof(float));
    gemma3_rope_precompute(f1, maxp, hd, 1e6f, 1.0f);
    gemma3_rope_precompute(f8, maxp, hd, 1e6f, 8.0f);
    /* scaling by 8 means position 8p behaves like position p unscaled */
    double worst = 0;
    for (int p = 0; p < maxp / 8; p++) {
        for (int i = 0; i < hd; i++) {
            double e = fabs(f8[(size_t)(8 * p) * hd + i] - f1[(size_t)p * hd + i]);
            if (e > worst) worst = e;
        }
    }
    CHECK(worst < 1e-6, "scaled rope mismatch %.3g", worst);
    /* rotation preserves the norm */
    float x[256];
    double n0 = 0, n1 = 0;
    for (int i = 0; i < hd; i++) { x[i] = frand(); n0 += (double)x[i] * x[i]; }
    gemma3_rope_apply_precomputed(x, f1, hd, 37);
    for (int i = 0; i < hd; i++) n1 += (double)x[i] * x[i];
    CHECK(fabs(n0 - n1) / n0 < 1e-5, "rope changed the norm (%.6f -> %.6f)", n0, n1);
    free(f1); free(f8);
}

static void test_sampler(void) {
    printf("sampler\n");
    const int V = 262208;
    float *logits = malloc((size_t)V * sizeof(float));
    for (int i = 0; i < V; i++) logits[i] = 4.0f * frand();
    logits[123457] = 30.0f;  /* clear winner */
    gemma3_sampler *s = gemma3_sampler_create(V);
    uint64_t st = 42;

    CHECK(gemma3_sample_logits(s, logits, V, 0.0f, 50, 0.9f, 0.0f, &st) == 123457, "greedy != argmax");
    CHECK(gemma3_sample_logits(s, logits, V, 1.0f, 1, 1.0f, 0.0f, &st) == 123457, "top_k=1 != argmax");
    CHECK(gemma3_sample_logits(s, logits, V, 0.7f, 50, 0.01f, 0.0f, &st) == 123457, "tiny top_p != argmax");

    /* every sample must come from the top-k set */
    logits[123457] = 2.0f;
    int k = 20, bad = 0;
    float *sorted = malloc((size_t)V * sizeof(float));
    memcpy(sorted, logits, (size_t)V * sizeof(float));
    float kth = -1e30f;
    for (int r = 0; r < k; r++) {           /* k-th largest by repeated max */
        int best = 0;
        for (int i = 1; i < V; i++) if (sorted[i] > sorted[best]) best = i;
        kth = sorted[best];
        sorted[best] = -1e30f;
    }
    for (int t = 0; t < 2000; t++) {
        int id = gemma3_sample_logits(s, logits, V, 1.5f, k, 1.0f, 0.0f, &st);
        if (logits[id] < kth) bad++;
    }
    CHECK(bad == 0, "%d samples outside top-k", bad);

    /* distribution: two tokens with p = 1/3 and 2/3 */
    float two[2] = { 0.0f, logf(2.0f) };
    gemma3_sampler *s2 = gemma3_sampler_create(2);
    int c1 = 0, n = 30000;
    for (int t = 0; t < n; t++) c1 += gemma3_sample_logits(s2, two, 2, 1.0f, 0, 1.0f, 0.0f, &st) == 1;
    double f = (double)c1 / n;
    CHECK(fabs(f - 2.0 / 3.0) < 0.015, "sampled frequency %.4f, expected 0.6667", f);

    /* min_p = 0.6 removes the 1/3 token */
    int c0 = 0;
    for (int t = 0; t < 2000; t++) c0 += gemma3_sample_logits(s2, two, 2, 1.0f, 0, 1.0f, 0.6f, &st) == 0;
    CHECK(c0 == 0, "min_p kept a token below threshold (%d times)", c0);

    gemma3_sampler_free(s2);
    gemma3_sampler_free(s);
    free(sorted);
    free(logits);
}

typedef struct { int *hits; } pfor_arg;
static void pfor_fn(void *arg, int start, int end) {
    pfor_arg *a = (pfor_arg *)arg;
    for (int i = start; i < end; i++) a->hits[i]++;
}

static void noop_task(void *arg, int idx, int nt) { (void)arg; (void)idx; (void)nt; }

static void test_thread_pool(gemma3_thread_pool *pool) {
    printf("thread pool (%d threads)\n", gemma3_thread_pool_size(pool));
    int n = 100003;
    int *hits = calloc(n, sizeof(int));
    pfor_arg a = { hits };
    for (int rep = 0; rep < 50; rep++) gemma3_parallel_for(pool, n, 97, pfor_fn, &a);
    int bad = 0;
    for (int i = 0; i < n; i++) if (hits[i] != 50) bad++;
    CHECK(bad == 0, "%d items not visited exactly 50 times", bad);
    free(hits);

    struct timespec t0, t1;
    int reps = 20000;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < reps; i++) gemma3_thread_pool_run(pool, noop_task, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us = ((t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3) / reps;
    printf("  dispatch latency: %.2f us per parallel job\n", us);
    CHECK(us < 1000.0, "dispatch latency too high: %.1f us", us);
}

int main(void) {
    gemma3_thread_pool *pool = gemma3_thread_pool_create(4);
    if (!pool) {
        printf("failed to create thread pool\n");
        return 1;
    }
    test_matvec_bf16(pool);
    test_matmul_bf16(pool);
    test_rmsnorm();
    test_gelu();
    test_softmax();
    test_attention_ring();
    test_rope();
    test_sampler();
    test_thread_pool(pool);
    gemma3_thread_pool_destroy(pool);

    printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures ? 1 : 0;
}
