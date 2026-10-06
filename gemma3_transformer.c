/*
 * gemma3_transformer.c - Transformer forward pass implementation
 *
 * Implements the Gemma 3 transformer architecture:
 * - Grouped Query Attention (GQA) with 8 Q heads and 4 KV heads
 * - Hybrid local/global attention (5:1 ratio)
 * - GELU-gated MLP
 * - RMSNorm with additional pre/post feedforward norms
 * - RoPE with layer-specific theta (and linear scaling on global layers)
 *
 * Prefill and decode share one code path: tokens are processed in chunks of
 * up to GEMMA3_PREFILL_CHUNK, so every projection is a batched BF16 GEMM that
 * reads each weight once per chunk. Decoding is simply a chunk of one token.
 */

#include "gemma3_internal.h"
#include "gemma3_kernels.h"
#include "gemma3_threads.h"
#ifdef USE_MPS
#include "gemma3_metal.h"
#endif
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
/* Below this many tokens the native kernels beat convert-then-sgemm */
#define GEMMA3_BLAS_MIN_TOKENS 32
#endif
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * KV Cache
 * ========================================================================== */

/* KV cache for a single layer. Position p is stored in row
 * (ring > 0 ? p % ring : p). Global layers keep every position; local
 * layers keep a ring of sliding_window + GEMMA3_LOCAL_RING_EXTRA rows. */
typedef struct {
    float *k;   /* [rows, num_kv_heads * head_dim] */
    float *v;   /* [rows, num_kv_heads * head_dim] */
    int rows;
    int ring;   /* 0 = linear (global layer) */
} layer_kv_cache;

struct gemma3_kv_cache {
    layer_kv_cache layers[GEMMA3_NUM_LAYERS];
    int max_seq;
    int current_pos;  /* number of positions currently cached */
};

static void kv_cache_free(gemma3_kv_cache *cache) {
    if (!cache) return;
    for (int l = 0; l < GEMMA3_NUM_LAYERS; l++) {
        free(cache->layers[l].k);
        free(cache->layers[l].v);
    }
    free(cache);
}

static gemma3_kv_cache *kv_cache_alloc(const gemma3_config *cfg, int max_seq) {
    gemma3_kv_cache *cache = (gemma3_kv_cache *)calloc(1, sizeof(gemma3_kv_cache));
    if (!cache) return NULL;
    cache->max_seq = max_seq;

    size_t kv_size = (size_t)cfg->num_kv_heads * cfg->head_dim;
    int ring_rows = gemma3_local_ring_size(cfg->sliding_window);

    for (int l = 0; l < cfg->num_layers; l++) {
        layer_kv_cache *lc = &cache->layers[l];
        if (gemma3_is_global_layer(l) || ring_rows >= max_seq) {
            lc->rows = max_seq;
            lc->ring = 0;
        } else {
            lc->rows = ring_rows;
            lc->ring = ring_rows;
        }
        lc->k = (float *)malloc((size_t)lc->rows * kv_size * sizeof(float));
        lc->v = (float *)malloc((size_t)lc->rows * kv_size * sizeof(float));
        if (!lc->k || !lc->v) {
            kv_cache_free(cache);
            return NULL;
        }
    }
    return cache;
}

/* ============================================================================
 * Activation Buffers (sized for one prefill chunk)
 * ========================================================================== */

typedef struct {
    int cap;        /* tokens per chunk */
    float *x;       /* [cap, hidden] residual stream */
    float *xn;      /* [cap, hidden] normalized input */
    float *q;       /* [cap, q_size] */
    float *k;       /* [cap, kv_size] */
    float *v;       /* [cap, kv_size] */
    float *attn;    /* [cap, q_size] attention output */
    float *proj;    /* [cap, hidden] projection output */
    float *gate;    /* [cap, intermediate] */
    float *up;      /* [cap, intermediate] */
    float *scores;  /* [num_threads, max_context] attention scores */
    int score_stride;
} activation_buffers;

static void free_buffers(activation_buffers *b) {
    if (!b) return;
    free(b->x); free(b->xn); free(b->q); free(b->k); free(b->v);
    free(b->attn); free(b->proj); free(b->gate); free(b->up); free(b->scores);
    free(b);
}

static activation_buffers *alloc_buffers(const gemma3_config *cfg, int cap,
                                         int max_context, int num_threads) {
    activation_buffers *b = (activation_buffers *)calloc(1, sizeof(activation_buffers));
    if (!b) return NULL;
    size_t hs = cfg->hidden_size, is = cfg->intermediate_size;
    size_t q_size = (size_t)cfg->num_heads * cfg->head_dim;
    size_t kv_size = (size_t)cfg->num_kv_heads * cfg->head_dim;
    b->cap = cap;
    b->x = (float *)malloc(cap * hs * sizeof(float));
    b->xn = (float *)malloc(cap * hs * sizeof(float));
    b->q = (float *)malloc(cap * q_size * sizeof(float));
    b->k = (float *)malloc(cap * kv_size * sizeof(float));
    b->v = (float *)malloc(cap * kv_size * sizeof(float));
    b->attn = (float *)malloc(cap * q_size * sizeof(float));
    b->proj = (float *)malloc(cap * hs * sizeof(float));
    b->gate = (float *)malloc(cap * is * sizeof(float));
    b->up = (float *)malloc(cap * is * sizeof(float));
    b->score_stride = max_context;
    b->scores = (float *)malloc((size_t)num_threads * max_context * sizeof(float));
    if (!b->x || !b->xn || !b->q || !b->k || !b->v || !b->attn || !b->proj ||
        !b->gate || !b->up || !b->scores) {
        free_buffers(b);
        return NULL;
    }
    return b;
}

/* ============================================================================
 * Transformer Context
 * ========================================================================== */

struct gemma3_transformer {
    gemma3_weights_t *weights;
    gemma3_kv_cache *cache;
    activation_buffers *buffers;
    gemma3_config config;
    int max_context;
    float *rope_freqs_local;   /* [max_context, head_dim/2, 2] cos/sin, theta=10K */
    float *rope_freqs_global;  /* [max_context, head_dim/2, 2] cos/sin, theta=1M, scaled */
    gemma3_thread_pool *pool;
    float *blas_scratch;       /* F32 copy of one weight matrix (BLAS builds) */
    const volatile int *abort_flag;
#ifdef USE_MPS
    gemma3_metal_context *metal_ctx;
#endif
};

/* ============================================================================
 * Parallel helpers
 * ========================================================================== */

/* Several independent projections that share the same input, run as one
 * parallel job (Q/K/V and gate/up). Rows of the outputs are concatenated
 * into a single index space for scheduling. */
typedef struct {
    int count;
    const uint16_t *W[3];
    float *Y[3];
    int M[3];
    int K;
    const float *X;
    int N;
} multi_proj_task;

static void multi_proj_fn(void *arg, int start, int end) {
    multi_proj_task *t = (multi_proj_task *)arg;
    int base = 0;
    for (int i = 0; i < t->count && start < end; i++) {
        int lo = start - base, hi = end - base;
        if (lo < t->M[i] && hi > 0) {
            if (lo < 0) lo = 0;
            if (hi > t->M[i]) hi = t->M[i];
            if (t->N == 1) {
                gemma3_matvec_bf16_range(t->Y[i], t->W[i], t->X, lo, hi, t->K);
            } else {
                const int NB = 32;
                for (int n0 = 0; n0 < t->N; n0 += NB) {
                    int nb = t->N - n0 < NB ? t->N - n0 : NB;
                    gemma3_matmul_bf16_range(t->Y[i] + (size_t)n0 * t->M[i], t->M[i],
                                             t->X + (size_t)n0 * t->K, nb,
                                             t->W[i], t->K, lo, hi);
                }
            }
        }
        base += t->M[i];
    }
}

#ifdef USE_BLAS
typedef struct {
    float *dst;
    const uint16_t *src;
} cvt_task;

static void cvt_fn(void *arg, int start, int end) {
    cvt_task *c = (cvt_task *)arg;
    gemma3_bf16_to_f32(c->dst + start, c->src + start, end - start);
}
#endif

static void run_projections(gemma3_thread_pool *pool, multi_proj_task *t,
                            float *blas_scratch) {
#ifdef USE_BLAS
    if (blas_scratch && t->N >= GEMMA3_BLAS_MIN_TOKENS) {
        for (int i = 0; i < t->count; i++) {
            cvt_task c = { blas_scratch, t->W[i] };
            gemma3_parallel_for(pool, t->M[i] * t->K, 1 << 16, cvt_fn, &c);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        t->N, t->M[i], t->K, 1.0f, t->X, t->K,
                        blas_scratch, t->K, 0.0f, t->Y[i], t->M[i]);
        }
        return;
    }
#else
    (void)blas_scratch;
#endif
    int total = 0;
    for (int i = 0; i < t->count; i++) total += t->M[i];
    int nt = gemma3_thread_pool_size(pool);
    /* Chunks are multiples of 16 rows so they never straddle two matrices
     * (all Gemma projection sizes are multiples of 16) and keep the 4-row
     * kernels fully busy. */
    int chunk = total / (nt * 8);
    if (chunk < 16) chunk = 16;
    chunk = (chunk + 15) & ~15;
    gemma3_parallel_for(pool, total, chunk, multi_proj_fn, t);
}

static void project(gemma3_thread_pool *pool, float *Y, const float *X, int N,
                    const uint16_t *W, int M, int K, float *blas_scratch) {
    multi_proj_task t = { 1, { W, NULL, NULL }, { Y, NULL, NULL }, { M, 0, 0 }, K, X, N };
    run_projections(pool, &t, blas_scratch);
}

/* Per-token RMSNorm over rows of a [N, n] matrix (optionally adding the
 * result into a residual stream). */
typedef struct {
    float *dst;            /* output rows (may equal src) */
    const float *src;
    float *residual;       /* if non-NULL: residual += normalized */
    const uint16_t *weight;
    int n;
    float eps;
} rows_norm_task;

static void rows_norm_fn(void *arg, int start, int end) {
    rows_norm_task *t = (rows_norm_task *)arg;
    for (int i = start; i < end; i++) {
        float *d = t->dst + (size_t)i * t->n;
        gemma3_rmsnorm_bf16(d, t->src + (size_t)i * t->n, t->weight, t->n, t->eps);
        if (t->residual) {
            float *r = t->residual + (size_t)i * t->n;
            for (int j = 0; j < t->n; j++) r[j] += d[j];
        }
    }
}

static void rows_norm(gemma3_thread_pool *pool, float *dst, const float *src, float *residual,
                      const uint16_t *weight, int N, int n, float eps) {
    rows_norm_task t = { dst, src, residual, weight, n, eps };
    gemma3_parallel_for(N > 1 ? pool : NULL, N, 1, rows_norm_fn, &t);
}

typedef struct {
    float *gate;
    const float *up;
} gelu_task;

static void gelu_fn(void *arg, int start, int end) {
    gelu_task *t = (gelu_task *)arg;
    gemma3_gelu_tanh_mul(t->gate + start, t->up + start, end - start);
}

/* Attention for every (token, head) pair of a chunk */
typedef struct {
    const gemma3_config *cfg;
    const layer_kv_cache *lc;
    const float *q;       /* [N, q_size] */
    float *out;           /* [N, q_size] */
    float *scores;        /* [num_threads, score_stride] */
    int score_stride;
    int N;
    int start_pos;
    int is_global;
    atomic_int next;
} attn_task;

static void attn_worker(void *arg, int thread_idx, int num_threads) {
    (void)num_threads;
    attn_task *t = (attn_task *)arg;
    const gemma3_config *cfg = t->cfg;
    int hd = cfg->head_dim;
    int nh = cfg->num_heads;
    int heads_per_kv = cfg->num_heads / cfg->num_kv_heads;
    int kv_stride = cfg->num_kv_heads * hd;
    int q_size = nh * hd;
    float scale = 1.0f / sqrtf((float)hd);  /* query_pre_attn_scalar = head_dim */
    float *scores = t->scores + (size_t)thread_idx * t->score_stride;
    int total = t->N * nh;

    for (;;) {
        int item = atomic_fetch_add_explicit(&t->next, 1, memory_order_relaxed);
        if (item >= total) break;
        int i = item / nh, h = item % nh;
        int pos = t->start_pos + i;
        int lo = t->is_global ? 0 : pos - cfg->sliding_window + 1;
        if (lo < 0) lo = 0;
        int kv_head = h / heads_per_kv;
        gemma3_attention_head(t->out + (size_t)i * q_size + h * hd,
                              t->q + (size_t)i * q_size + h * hd,
                              t->lc->k + kv_head * hd, t->lc->v + kv_head * hd,
                              kv_stride, lo, pos, t->lc->ring, hd, scale, scores);
    }
}

/* ============================================================================
 * Forward pass for one chunk of tokens (CPU)
 * ========================================================================== */

static void forward_chunk(gemma3_transformer *t, const int *tokens, int N,
                          int start_pos, float *logits) {
    const gemma3_config *cfg = &t->config;
    const gemma3_weights_t *w = t->weights;
    activation_buffers *b = t->buffers;
    gemma3_thread_pool *pool = t->pool;

    const int hs = cfg->hidden_size;
    const int is = cfg->intermediate_size;
    const int hd = cfg->head_dim;
    const int nh = cfg->num_heads;
    const int nkv = cfg->num_kv_heads;
    const int q_size = nh * hd;
    const int kv_size = nkv * hd;
    const float eps = cfg->rmsnorm_eps;

    /* Embedding lookup, scaled by sqrt(hidden_size) */
    const float embed_scale = sqrtf((float)hs);
    for (int i = 0; i < N; i++) {
        float *xi = b->x + (size_t)i * hs;
        gemma3_embed_bf16(xi, w->embed_tokens, tokens[i], hs);
        for (int j = 0; j < hs; j++) xi[j] *= embed_scale;
    }

    for (int l = 0; l < cfg->num_layers; l++) {
        const int is_global = gemma3_is_global_layer(l);
        const float *rope = is_global ? t->rope_freqs_global : t->rope_freqs_local;
        layer_kv_cache *lc = &t->cache->layers[l];

        /* ---- Attention block ---- */
        rows_norm(pool, b->xn, b->x, NULL, w->layers[l].input_layernorm, N, hs, eps);

        multi_proj_task qkv = {
            3,
            { w->layers[l].q_proj, w->layers[l].k_proj, w->layers[l].v_proj },
            { b->q, b->k, b->v },
            { q_size, kv_size, kv_size },
            hs, b->xn, N
        };
        run_projections(pool, &qkv, t->blas_scratch);

        /* QK-norm, RoPE, and append K/V to the cache for every token */
        for (int i = 0; i < N; i++) {
            int pos = start_pos + i;
            float *qi = b->q + (size_t)i * q_size;
            float *ki = b->k + (size_t)i * kv_size;
            for (int h = 0; h < nh; h++) {
                gemma3_rmsnorm_bf16(qi + h * hd, qi + h * hd, w->layers[l].q_norm, hd, eps);
                gemma3_rope_apply_precomputed(qi + h * hd, rope, hd, pos);
            }
            for (int h = 0; h < nkv; h++) {
                gemma3_rmsnorm_bf16(ki + h * hd, ki + h * hd, w->layers[l].k_norm, hd, eps);
                gemma3_rope_apply_precomputed(ki + h * hd, rope, hd, pos);
            }
            size_t row = (size_t)(lc->ring > 0 ? pos % lc->ring : pos);
            memcpy(lc->k + row * kv_size, ki, (size_t)kv_size * sizeof(float));
            memcpy(lc->v + row * kv_size, b->v + (size_t)i * kv_size, (size_t)kv_size * sizeof(float));
        }

        attn_task at;
        at.cfg = cfg;
        at.lc = lc;
        at.q = b->q;
        at.out = b->attn;
        at.scores = b->scores;
        at.score_stride = b->score_stride;
        at.N = N;
        at.start_pos = start_pos;
        at.is_global = is_global;
        atomic_init(&at.next, 0);
        /* Short contexts are cheaper to run inline than to dispatch */
        int span = start_pos + N;
        if (!is_global && span > cfg->sliding_window) span = cfg->sliding_window;
        if ((long)N * nh * span >= 4096) {
            gemma3_thread_pool_run(pool, attn_worker, &at);
        } else {
            attn_worker(&at, 0, 1);
        }

        project(pool, b->proj, b->attn, N, w->layers[l].o_proj, hs, q_size, t->blas_scratch);

        /* x += post_attention_norm(proj) */
        rows_norm(pool, b->proj, b->proj, b->x, w->layers[l].post_attention_layernorm, N, hs, eps);

        /* ---- MLP block ---- */
        rows_norm(pool, b->xn, b->x, NULL, w->layers[l].pre_feedforward_layernorm, N, hs, eps);

        multi_proj_task gu = {
            2,
            { w->layers[l].gate_proj, w->layers[l].up_proj, NULL },
            { b->gate, b->up, NULL },
            { is, is, 0 },
            hs, b->xn, N
        };
        run_projections(pool, &gu, t->blas_scratch);

        gelu_task gt = { b->gate, b->up };
        gemma3_parallel_for(N > 1 ? pool : NULL, N * is, 4096, gelu_fn, &gt);

        project(pool, b->proj, b->gate, N, w->layers[l].down_proj, hs, is, t->blas_scratch);

        /* x += post_feedforward_norm(mlp_out) */
        rows_norm(pool, b->proj, b->proj, b->x, w->layers[l].post_feedforward_layernorm, N, hs, eps);
    }

    if (logits) {
        /* Final norm + tied-embedding output projection for the last token */
        float *last = b->x + (size_t)(N - 1) * hs;
        gemma3_rmsnorm_bf16(b->xn, last, w->norm, hs, eps);
        gemma3_matvec_bf16_mt(logits, w->embed_tokens, b->xn, cfg->vocab_size, hs, NULL, pool);
    }
}

/* ============================================================================
 * Internal API (see gemma3_internal.h)
 * ========================================================================== */

gemma3_transformer *gemma3_transformer_create(
    gemma3_weights_t *weights,
    const gemma3_config *cfg,
    int max_context,
    int num_threads
) {
    gemma3_transformer *t = (gemma3_transformer *)calloc(1, sizeof(gemma3_transformer));
    if (!t) return NULL;

    t->weights = weights;
    t->config = *cfg;
    t->max_context = max_context;

    t->pool = gemma3_thread_pool_create(num_threads);
    t->cache = kv_cache_alloc(cfg, max_context);
    int nthreads = gemma3_thread_pool_size(t->pool);
    t->buffers = alloc_buffers(cfg, GEMMA3_PREFILL_CHUNK, max_context, nthreads);

    size_t rope_table_size = (size_t)max_context * (cfg->head_dim / 2) * 2;
    t->rope_freqs_local = (float *)malloc(rope_table_size * sizeof(float));
    t->rope_freqs_global = (float *)malloc(rope_table_size * sizeof(float));

#ifdef USE_BLAS
    size_t max_w = (size_t)cfg->intermediate_size * cfg->hidden_size;
    size_t qw = (size_t)cfg->num_heads * cfg->head_dim * cfg->hidden_size;
    if (qw > max_w) max_w = qw;
    t->blas_scratch = (float *)malloc(max_w * sizeof(float));
    if (!t->blas_scratch) {
        gemma3_transformer_destroy(t);
        return NULL;
    }
#endif

    if (!t->pool || !t->cache || !t->buffers || !t->rope_freqs_local || !t->rope_freqs_global) {
        gemma3_transformer_destroy(t);
        return NULL;
    }

    gemma3_rope_precompute(t->rope_freqs_local, max_context, cfg->head_dim,
                           cfg->rope_theta_local, 1.0f);
    gemma3_rope_precompute(t->rope_freqs_global, max_context, cfg->head_dim,
                           cfg->rope_theta_global, cfg->rope_scale_global);

#ifdef USE_MPS
    if (!getenv("GEMMA3_NO_METAL")) {
        t->metal_ctx = gemma3_metal_init(cfg, max_context);
        if (t->metal_ctx) {
            if (gemma3_metal_upload_weights(t->metal_ctx, t->weights) != 0 ||
                gemma3_metal_upload_rope(t->metal_ctx, t->rope_freqs_local,
                                         t->rope_freqs_global, max_context,
                                         cfg->head_dim) != 0) {
                fprintf(stderr, "Metal: weight/rope upload failed, falling back to CPU\n");
                gemma3_metal_free(t->metal_ctx);
                t->metal_ctx = NULL;
            }
        } else {
            fprintf(stderr, "Metal GPU not available, using CPU\n");
        }
    }
#endif

    return t;
}

void gemma3_transformer_destroy(gemma3_transformer *t) {
    if (!t) return;
#ifdef USE_MPS
    if (t->metal_ctx) gemma3_metal_free(t->metal_ctx);
#endif
    gemma3_thread_pool_destroy(t->pool);
    kv_cache_free(t->cache);
    free_buffers(t->buffers);
    free(t->rope_freqs_local);
    free(t->rope_freqs_global);
    free(t->blas_scratch);
    free(t);
}

void gemma3_transformer_set_abort_flag(gemma3_transformer *t, const volatile int *flag) {
    if (t) t->abort_flag = flag;
}

int gemma3_transformer_forward_token(gemma3_transformer *t, int token_id, int pos,
                                     float *logits) {
    if (!t || pos < 0 || pos >= t->max_context) return GEMMA3_ERR_CONTEXT_OVERFLOW;
    if (token_id < 0 || token_id >= t->config.vocab_size) return GEMMA3_ERR_INVALID_ARG;
#ifdef USE_MPS
    if (t->metal_ctx) {
        int ret = gemma3_metal_forward_token(t->metal_ctx, token_id, pos, logits, logits != NULL);
        if (ret == 0) t->cache->current_pos = pos + 1;
        return ret;
    }
#endif
    forward_chunk(t, &token_id, 1, pos, logits);
    t->cache->current_pos = pos + 1;
    return 0;
}

int gemma3_transformer_prefill_tokens(gemma3_transformer *t, const int *tokens,
                                      int num_tokens, int start_pos, float *logits) {
    if (!t || !tokens || num_tokens <= 0 || start_pos < 0) return GEMMA3_ERR_INVALID_ARG;
    if (start_pos + num_tokens > t->max_context) return GEMMA3_ERR_CONTEXT_OVERFLOW;
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= t->config.vocab_size) return GEMMA3_ERR_INVALID_ARG;
    }

#ifdef USE_MPS
    if (t->metal_ctx) {
        /* Feed the Metal backend in slices so long prompts can be interrupted */
        for (int done = 0; done < num_tokens; ) {
            if (t->abort_flag && *t->abort_flag) return GEMMA3_ERR_ABORTED;
            int n = num_tokens - done;
            if (n > GEMMA3_PREFILL_CHUNK * 4) n = GEMMA3_PREFILL_CHUNK * 4;
            int last = (done + n == num_tokens);
            int ret = gemma3_metal_prefill(t->metal_ctx, tokens + done, n, start_pos + done,
                                           last ? logits : NULL);
            if (ret != 0) return ret;
            done += n;
            t->cache->current_pos = start_pos + done;
        }
        return 0;
    }
#endif

    for (int done = 0; done < num_tokens; ) {
        if (t->abort_flag && *t->abort_flag) return GEMMA3_ERR_ABORTED;
        int n = num_tokens - done;
        if (n > GEMMA3_PREFILL_CHUNK) n = GEMMA3_PREFILL_CHUNK;
        int last = (done + n == num_tokens);
        forward_chunk(t, tokens + done, n, start_pos + done, last ? logits : NULL);
        done += n;
        t->cache->current_pos = start_pos + done;
    }
    return 0;
}

void gemma3_transformer_reset(gemma3_transformer *t) {
    if (!t) return;
    if (t->cache) t->cache->current_pos = 0;
#ifdef USE_MPS
    if (t->metal_ctx) gemma3_metal_reset_cache(t->metal_ctx);
#endif
}

int gemma3_transformer_get_pos(gemma3_transformer *t) {
    return t && t->cache ? t->cache->current_pos : 0;
}

int gemma3_transformer_can_rewind(const gemma3_transformer *t, int from_pos, int to_pos) {
    if (!t || to_pos < 0 || to_pos > from_pos) return 0;
    /* Safe if no local ring has wrapped yet, or if the rewind is short enough
     * that every row a future query needs is still intact. */
    int ring = gemma3_local_ring_size(t->config.sliding_window);
    return from_pos <= ring || from_pos - to_pos <= GEMMA3_LOCAL_RING_EXTRA + 1;
}

const char *gemma3_transformer_backend(const gemma3_transformer *t) {
#ifdef USE_MPS
    if (t && t->metal_ctx) return "metal";
#endif
    (void)t;
    return "cpu";
}

int gemma3_transformer_num_threads(const gemma3_transformer *t) {
    return t ? gemma3_thread_pool_size(t->pool) : 1;
}
