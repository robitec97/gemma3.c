/*
 * bench_kernels.c - Micro-benchmarks for gemma3.c CPU kernels (no model needed)
 *
 * Uses random weights with the exact Gemma 3 4B shapes and reports:
 *   - BF16 matvec bandwidth (GB/s) per projection, 1 thread and all threads
 *   - BF16 batched GEMM throughput (GFLOP/s) as used by prompt processing
 *   - thread-pool dispatch latency
 *   - sampler cost per token (new candidate-based sampler vs full-vocab filtering)
 *
 * Usage: ./gemma3-bench-kernels [--threads N] [--json out.json]
 */

#include "../gemma3.h"
#include "../gemma3_kernels.h"
#include "../gemma3_threads.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t rng = 88172645463325252ULL;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / 8388608.0)) - 1.0f;
}

typedef struct { const char *name; int M, K; } shape;

static FILE *g_json = NULL;
static int g_json_first = 1;

static void json_item(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void json_item(const char *fmt, ...) {
    if (!g_json) return;
    fprintf(g_json, "%s\n    ", g_json_first ? "" : ",");
    g_json_first = 0;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_json, fmt, ap);
    va_end(ap);
}

/* Run `call` repeatedly for at least min_time seconds; store seconds per call in out */
#define TIME_IT(out, min_time, call) do { \
    int _reps = 0; double _t0 = now_s(), _el; \
    do { call; _reps++; _el = now_s() - _t0; } while (_el < (min_time)); \
    (out) = _el / _reps; \
} while (0)

int main(int argc, char **argv) {
    int threads = 0;
    const char *json_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
        else { fprintf(stderr, "Usage: %s [--threads N] [--json out.json]\n", argv[0]); return 1; }
    }

    gemma3_thread_pool *pool = gemma3_thread_pool_create(threads);
    int nt = gemma3_thread_pool_size(pool);
    if (json_path) {
        g_json = fopen(json_path, "w");
        if (g_json) fprintf(g_json, "{\n  \"simd\": \"%s\",\n  \"threads\": %d,\n  \"results\": [",
                            gemma3_simd_name(), nt);
    }

    printf("gemma3.c kernel benchmarks | simd: %s | threads: %d\n\n", gemma3_simd_name(), nt);

    /* ---- matvec (decode) ---- */
    const shape shapes[] = {
        { "q_proj   2048x2560",   2048,  2560 },
        { "gate/up 10240x2560",  10240,  2560 },
        { "down     2560x10240",  2560, 10240 },
        { "lm_head 262144x2560", 262144, 2560 },
    };
    size_t max_elems = (size_t)262144 * 2560;
    uint16_t *W = malloc(max_elems * sizeof(uint16_t));
    float *x = malloc(262144 * sizeof(float));
    float *y = malloc(262144 * sizeof(float));
    if (!W || !x || !y) { fprintf(stderr, "OOM\n"); return 1; }
    for (size_t i = 0; i < max_elems; i++) W[i] = gemma3_f32_to_bf16_single(frand());
    for (int i = 0; i < 262144; i++) x[i] = frand();

    printf("BF16 matvec (decode path)          1 thread      %2d threads\n", nt);
    printf("-------------------------------  ------------  ------------\n");
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int M = shapes[s].M, K = shapes[s].K;
        double bytes = (double)M * K * 2.0;
        double t1, tn;
        TIME_IT(t1, 0.3, gemma3_matvec_bf16(y, W, x, M, K, NULL));
        TIME_IT(tn, 0.3, gemma3_matvec_bf16_mt(y, W, x, M, K, NULL, pool));
        printf("%-31s  %7.1f GB/s  %7.1f GB/s\n", shapes[s].name, bytes / t1 / 1e9, bytes / tn / 1e9);
        json_item("{\"kernel\": \"matvec_bf16\", \"shape\": \"%dx%d\", \"gbps_1t\": %.2f, \"gbps_mt\": %.2f}",
                  M, K, bytes / t1 / 1e9, bytes / tn / 1e9);
    }

    /* ---- matmul (prefill) ---- */
    printf("\nBF16 GEMM (prefill path, N tokens)  1 thread      %2d threads\n", nt);
    printf("-------------------------------  ------------  ------------\n");
    int Ns[] = { 8, 32, 128 };
    int M = 10240, K = 2560;
    float *X = malloc((size_t)128 * K * sizeof(float));
    float *Y = malloc((size_t)128 * M * sizeof(float));
    for (int i = 0; i < 128 * K; i++) X[i] = frand();
    for (size_t i = 0; i < sizeof(Ns) / sizeof(Ns[0]); i++) {
        int N = Ns[i];
        double flops = 2.0 * N * M * K;
        double t1, tn;
        TIME_IT(t1, 0.5, gemma3_matmul_bf16_range(Y, M, X, N, W, K, 0, M));
        TIME_IT(tn, 0.5, gemma3_matmul_bf16_mt(Y, X, N, W, M, K, pool));
        char name[64];
        snprintf(name, sizeof(name), "%dx%d, N=%d", M, K, N);
        printf("%-31s  %6.1f GFLOP/s %6.1f GFLOP/s\n", name, flops / t1 / 1e9, flops / tn / 1e9);
        json_item("{\"kernel\": \"matmul_bf16\", \"shape\": \"%dx%d\", \"n\": %d, \"gflops_1t\": %.2f, \"gflops_mt\": %.2f}",
                  M, K, N, flops / t1 / 1e9, flops / tn / 1e9);
    }

    /* ---- thread pool ---- */
    printf("\nThread pool\n-----------\n");
    {
        int reps = 0;
        double t0 = now_s();
        while (now_s() - t0 < 0.3) {
            for (int i = 0; i < 100; i++) gemma3_matvec_bf16_mt(y, W, x, 4 * nt, 64, NULL, pool);
            reps += 100;
        }
        double us = (now_s() - t0) / reps * 1e6;
        printf("tiny parallel job round-trip      %6.2f us\n", us);
        json_item("{\"kernel\": \"pool_dispatch\", \"us\": %.3f}", us);
    }

    /* ---- sampler ---- */
    printf("\nSampling (vocab 262208, T=0.7, top-k 50, top-p 0.9)\n");
    printf("---------------------------------------------------\n");
    {
        const int V = 262208;
        float *logits = malloc((size_t)V * sizeof(float));
        float *work = malloc((size_t)V * sizeof(float));
        for (int i = 0; i < V; i++) logits[i] = 8.0f * frand();
        gemma3_sampler *s = gemma3_sampler_create(V);
        uint64_t st = 1;
        volatile int sink = 0;
        double t_new, t_old;
        TIME_IT(t_new, 0.5, sink += gemma3_sample_logits(s, logits, V, 0.7f, 50, 0.9f, 0.0f, &st));
        TIME_IT(t_old, 0.5, {
            memcpy(work, logits, (size_t)V * sizeof(float));
            gemma3_apply_temperature(work, V, 0.7f);
            gemma3_topk_filter(work, V, 50);
            gemma3_topp_filter(work, V, 0.9f);
            gemma3_softmax_inplace(work, V);
            sink += gemma3_sample(work, V);
        });
        printf("full-vocab filter + sort (old)    %7.3f ms/token\n", t_old * 1e3);
        printf("top-k heap + candidates (new)     %7.3f ms/token  (%.0fx faster)\n",
               t_new * 1e3, t_old / t_new);
        json_item("{\"kernel\": \"sampler\", \"old_ms\": %.4f, \"new_ms\": %.4f}", t_old * 1e3, t_new * 1e3);
        gemma3_sampler_free(s);
        free(logits);
        free(work);
        (void)sink;
    }

    if (g_json) {
        fprintf(g_json, "\n  ]\n}\n");
        fclose(g_json);
    }
    free(W); free(x); free(y); free(X); free(Y);
    gemma3_thread_pool_destroy(pool);
    return 0;
}
