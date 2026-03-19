/*
 * gemma3_metal.m - Metal GPU backend for Gemma 3 inference
 *
 * Custom Metal Shading Language compute kernels for the full transformer
 * forward pass. MSL source is embedded as a C string and compiled at runtime.
 */

#ifdef USE_MPS

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "gemma3_metal.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* ============================================================================
 * Weights type (must match gemma3_transformer.c definition)
 * ========================================================================== */

typedef struct {
    const uint16_t *embed_tokens;
    struct {
        const uint16_t *input_layernorm;
        const uint16_t *q_proj;
        const uint16_t *k_proj;
        const uint16_t *v_proj;
        const uint16_t *o_proj;
        const uint16_t *q_norm;
        const uint16_t *k_norm;
        const uint16_t *post_attention_layernorm;
        const uint16_t *gate_proj;
        const uint16_t *up_proj;
        const uint16_t *down_proj;
        const uint16_t *pre_feedforward_layernorm;
        const uint16_t *post_feedforward_layernorm;
    } layers[GEMMA3_NUM_LAYERS];
    const uint16_t *norm;
} gemma3_weights_t;

/* ============================================================================
 * GPU parameter structs (must match MSL definitions exactly)
 * ========================================================================== */

typedef struct { uint32_t M; uint32_t K; } MetalMatvecParams;
typedef struct { uint32_t N; float eps; uint32_t stride; uint32_t _pad; } MetalNormParams;
typedef struct { uint32_t token_id; uint32_t hidden_size; } MetalEmbedParams;
typedef struct { uint32_t N; float scale_val; } MetalVecParams;
typedef struct { uint32_t head_dim; uint32_t pos; uint32_t num_heads; uint32_t _pad; } MetalRopeParams;
typedef struct { uint32_t kv_size; uint32_t cache_pos; } MetalCacheParams;
typedef struct {
    uint32_t num_heads; uint32_t num_kv_heads; uint32_t head_dim;
    uint32_t seq_len; float scale; uint32_t max_seq;
} MetalAttnParams;

/* ============================================================================
 * Embedded Metal Shading Language source
 * ========================================================================== */

static const char *metalShaderSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"// BF16 to F32 conversion\n"
"inline float bf16_to_f32(ushort val) {\n"
"    return as_type<float>(uint(val) << 16);\n"
"}\n"
"\n"
"// --- Parameter structs ---\n"
"struct MatvecParams { uint M; uint K; };\n"
"struct NormParams { uint N; float eps; uint stride; uint _pad; };\n"
"struct EmbedParams { uint token_id; uint hidden_size; };\n"
"struct VecParams { uint N; float scale_val; };\n"
"struct RopeParams { uint head_dim; uint pos; uint num_heads; uint _pad; };\n"
"struct CacheParams { uint kv_size; uint cache_pos; };\n"
"struct AttnParams {\n"
"    uint num_heads; uint num_kv_heads; uint head_dim;\n"
"    uint seq_len; float scale; uint max_seq;\n"
"};\n"
"\n"
"// --- matvec_bf16 ---\n"
"// y[gid] = dot(A[gid,:], x).  Dispatch: [M] threadgroups, [256] threads.\n"
"kernel void matvec_bf16(\n"
"    device const ushort *A [[buffer(0)]],\n"
"    device const float  *x [[buffer(1)]],\n"
"    device float        *y [[buffer(2)]],\n"
"    constant MatvecParams &p [[buffer(3)]],\n"
"    uint gid [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    if (gid >= p.M) return;\n"
"    device const ushort *row = A + gid * p.K;\n"
"    float sum = 0.0f;\n"
"    for (uint i = tid; i < p.K; i += tg_size) {\n"
"        sum += bf16_to_f32(row[i]) * x[i];\n"
"    }\n"
"    sum = simd_sum(sum);\n"
"    threadgroup float shared[8];\n"
"    uint simd_lane = tid % 32;\n"
"    uint simd_id   = tid / 32;\n"
"    uint n_simd    = (tg_size + 31) / 32;\n"
"    if (simd_lane == 0) shared[simd_id] = sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        for (uint i = 0; i < n_simd; i++) total += shared[i];\n"
"        y[gid] = total;\n"
"    }\n"
"}\n"
"\n"
"// --- rmsnorm_bf16 ---\n"
"// Per-subvector: y = x * rsqrt(mean(x^2)+eps) * (1+weight)\n"
"// Dispatch: [num_subvecs] threadgroups, [256] threads.\n"
"// gid selects which subvector (stride apart).\n"
"kernel void rmsnorm_bf16(\n"
"    device const float  *input  [[buffer(0)]],\n"
"    device float        *output [[buffer(1)]],\n"
"    device const ushort *weight [[buffer(2)]],\n"
"    constant NormParams &p [[buffer(3)]],\n"
"    uint gid [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    uint base = gid * p.stride;\n"
"    device const float *x = input + base;\n"
"    device float       *y = output + base;\n"
"    float local_ss = 0.0f;\n"
"    for (uint i = tid; i < p.N; i += tg_size) {\n"
"        float v = x[i]; local_ss += v * v;\n"
"    }\n"
"    local_ss = simd_sum(local_ss);\n"
"    threadgroup float shared[8];\n"
"    threadgroup float tg_rsqrt;\n"
"    uint simd_lane = tid % 32;\n"
"    uint simd_id   = tid / 32;\n"
"    uint n_simd    = (tg_size + 31) / 32;\n"
"    if (simd_lane == 0) shared[simd_id] = local_ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        for (uint i = 0; i < n_simd; i++) total += shared[i];\n"
"        tg_rsqrt = rsqrt(total / float(p.N) + p.eps);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float rs = tg_rsqrt;\n"
"    for (uint i = tid; i < p.N; i += tg_size) {\n"
"        y[i] = x[i] * rs * (1.0f + bf16_to_f32(weight[i]));\n"
"    }\n"
"}\n"
"\n"
"// --- embed_bf16 ---\n"
"kernel void embed_bf16(\n"
"    device const ushort *embed [[buffer(0)]],\n"
"    device float        *out   [[buffer(1)]],\n"
"    constant EmbedParams &p [[buffer(2)]],\n"
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    if (tid >= p.hidden_size) return;\n"
"    out[tid] = bf16_to_f32(embed[p.token_id * p.hidden_size + tid]);\n"
"}\n"
"\n"
"// --- gelu_tanh (in-place) ---\n"
"kernel void gelu_tanh(\n"
"    device float *x [[buffer(0)]],\n"
"    constant uint &N [[buffer(1)]],\n"
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    if (tid >= N) return;\n"
"    float v = x[tid];\n"
"    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);\n"
"    // Clamp inner to avoid NaN from Metal fast-math tanh on extreme inputs\n"
"    inner = clamp(inner, -16.0f, 16.0f);\n"
"    x[tid] = 0.5f * v * (1.0f + tanh(inner));\n"
"}\n"
"\n"
"// --- vec_mul (a *= b) ---\n"
"kernel void vec_mul(\n"
"    device float       *a [[buffer(0)]],\n"
"    device const float *b [[buffer(1)]],\n"
"    constant uint &N [[buffer(2)]],\n"
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    if (tid >= N) return;\n"
"    a[tid] *= b[tid];\n"
"}\n"
"\n"
"// --- vec_add (y = a + b) ---\n"
"kernel void vec_add(\n"
"    device float       *y [[buffer(0)]],\n"
"    device const float *a [[buffer(1)]],\n"
"    device const float *b [[buffer(2)]],\n"
"    constant uint &N [[buffer(3)]],\n"
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    if (tid >= N) return;\n"
"    y[tid] = a[tid] + b[tid];\n"
"}\n"
"\n"
"// --- vec_scale (x *= s) ---\n"
"kernel void vec_scale(\n"
"    device float *x [[buffer(0)]],\n"
"    constant VecParams &p [[buffer(1)]],\n"
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    if (tid >= p.N) return;\n"
"    x[tid] *= p.scale_val;\n"
"}\n"
"\n"
"// --- vec_copy (dst = src) ---\n"
"kernel void vec_copy(\n"
"    device const float *src [[buffer(0)]],\n"
"    device float       *dst [[buffer(1)]],\n"
"    constant uint &N [[buffer(2)]],\n"
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    if (tid >= N) return;\n"
"    dst[tid] = src[tid];\n"
"}\n"
"\n"
"// --- rope_apply ---\n"
"// Applies precomputed RoPE to multi-head vector.\n"
"// Each thread handles one (head, dim_pair).\n"
"kernel void rope_apply(\n"
"    device float       *x     [[buffer(0)]],\n"
"    device const float *freqs [[buffer(1)]],\n"
"    constant RopeParams &p [[buffer(2)]],\n"
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    uint half_dim = p.head_dim / 2;\n"
"    uint total = p.num_heads * half_dim;\n"
"    if (tid >= total) return;\n"
"    uint head = tid / half_dim;\n"
"    uint i    = tid % half_dim;\n"
"    device float *h = x + head * p.head_dim;\n"
"    device const float *pf = freqs + p.pos * half_dim * 2;\n"
"    float c = pf[i * 2];\n"
"    float s = pf[i * 2 + 1];\n"
"    float x0 = h[i];\n"
"    float x1 = h[i + half_dim];\n"
"    h[i]            = x0 * c - x1 * s;\n"
"    h[i + half_dim] = x0 * s + x1 * c;\n"
"}\n"
"\n"
"// --- cache_kv ---\n"
"kernel void cache_kv(\n"
"    device const float *k_in    [[buffer(0)]],\n"
"    device const float *v_in    [[buffer(1)]],\n"
"    device float       *k_cache [[buffer(2)]],\n"
"    device float       *v_cache [[buffer(3)]],\n"
"    constant CacheParams &p [[buffer(4)]],\n"
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    if (tid >= p.kv_size) return;\n"
"    uint off = p.cache_pos * p.kv_size;\n"
"    k_cache[off + tid] = k_in[tid];\n"
"    v_cache[off + tid] = v_in[tid];\n"
"}\n"
"\n"
"// --- gqa_attention ---\n"
"// 1 threadgroup per query head, [256] threads.\n"
"// Scores stored in device memory (scores_buf).\n"
"kernel void gqa_attention(\n"
"    device const float *q          [[buffer(0)]],\n"
"    device const float *k_cache    [[buffer(1)]],\n"
"    device const float *v_cache    [[buffer(2)]],\n"
"    device float       *output     [[buffer(3)]],\n"
"    device float       *scores_buf [[buffer(4)]],\n"
"    device const float *mask       [[buffer(5)]],\n"
"    constant AttnParams &p [[buffer(6)]],\n"
"    uint gid [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    if (gid >= p.num_heads) return;\n"
"    uint heads_per_group = p.num_heads / p.num_kv_heads;\n"
"    uint kv_head   = gid / heads_per_group;\n"
"    uint kv_stride = p.num_kv_heads * p.head_dim;\n"
"    uint seq_len   = p.seq_len;\n"
"    uint head_dim  = p.head_dim;\n"
"    float scale    = p.scale;\n"
"\n"
"    device const float *q_head = q + gid * head_dim;\n"
"    device float *scores = scores_buf + gid * p.max_seq;\n"
"    device float *out_head = output + gid * head_dim;\n"
"\n"
"    // Phase 1: scores + max\n"
"    float local_max = -MAXFLOAT;\n"
"    for (uint i = tid; i < seq_len; i += tg_size) {\n"
"        device const float *k_pos = k_cache + i * kv_stride + kv_head * head_dim;\n"
"        float score = 0.0f;\n"
"        for (uint d = 0; d < head_dim; d++) score += q_head[d] * k_pos[d];\n"
"        score = score * scale + mask[i];\n"
"        scores[i] = score;\n"
"        local_max = max(local_max, score);\n"
"    }\n"
"    local_max = simd_max(local_max);\n"
"    threadgroup float shared[8];\n"
"    uint simd_lane = tid % 32;\n"
"    uint simd_id   = tid / 32;\n"
"    uint n_simd    = (tg_size + 31) / 32;\n"
"    if (simd_lane == 0) shared[simd_id] = local_max;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        float gm = shared[0];\n"
"        for (uint i = 1; i < n_simd; i++) gm = max(gm, shared[i]);\n"
"        shared[0] = gm;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float global_max = shared[0];\n"
"\n"
"    // Phase 2: exp + sum\n"
"    float local_sum = 0.0f;\n"
"    for (uint i = tid; i < seq_len; i += tg_size) {\n"
"        float v = exp(scores[i] - global_max);\n"
"        scores[i] = v;\n"
"        local_sum += v;\n"
"    }\n"
"    local_sum = simd_sum(local_sum);\n"
"    if (simd_lane == 0) shared[simd_id] = local_sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        float gs = 0.0f;\n"
"        for (uint i = 0; i < n_simd; i++) gs += shared[i];\n"
"        shared[0] = gs;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float inv_sum = 1.0f / shared[0];\n"
"\n"
"    // Ensure all scores are visible\n"
"    threadgroup_barrier(mem_flags::mem_device);\n"
"\n"
"    // Phase 3: weighted sum of values (threads iterate over head_dim)\n"
"    for (uint d = tid; d < head_dim; d += tg_size) {\n"
"        float acc = 0.0f;\n"
"        for (uint i = 0; i < seq_len; i++) {\n"
"            device const float *v_pos = v_cache + i * kv_stride + kv_head * head_dim;\n"
"            acc += scores[i] * v_pos[d];\n"
"        }\n"
"        out_head[d] = acc * inv_sum;\n"
"    }\n"
"}\n";

/* ============================================================================
 * Metal Context Structure
 * ========================================================================== */

struct gemma3_metal_context {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    /* Pipeline states */
    id<MTLComputePipelineState> pso_matvec_bf16;
    id<MTLComputePipelineState> pso_rmsnorm_bf16;
    id<MTLComputePipelineState> pso_embed_bf16;
    id<MTLComputePipelineState> pso_gelu_tanh;
    id<MTLComputePipelineState> pso_vec_mul;
    id<MTLComputePipelineState> pso_vec_add;
    id<MTLComputePipelineState> pso_vec_scale;
    id<MTLComputePipelineState> pso_vec_copy;
    id<MTLComputePipelineState> pso_rope_apply;
    id<MTLComputePipelineState> pso_cache_kv;
    id<MTLComputePipelineState> pso_gqa_attention;

    /* Weight buffers */
    id<MTLBuffer> buf_embed_tokens;
    id<MTLBuffer> buf_norm;
    struct {
        id<MTLBuffer> input_layernorm;
        id<MTLBuffer> q_proj;
        id<MTLBuffer> k_proj;
        id<MTLBuffer> v_proj;
        id<MTLBuffer> o_proj;
        id<MTLBuffer> q_norm;
        id<MTLBuffer> k_norm;
        id<MTLBuffer> post_attention_layernorm;
        id<MTLBuffer> gate_proj;
        id<MTLBuffer> up_proj;
        id<MTLBuffer> down_proj;
        id<MTLBuffer> pre_feedforward_layernorm;
        id<MTLBuffer> post_feedforward_layernorm;
    } layer_bufs[GEMMA3_NUM_LAYERS];

    /* Activation buffers */
    id<MTLBuffer> buf_x;
    id<MTLBuffer> buf_x_norm;
    id<MTLBuffer> buf_q;
    id<MTLBuffer> buf_k;
    id<MTLBuffer> buf_v;
    id<MTLBuffer> buf_attn_out;
    id<MTLBuffer> buf_proj_out;
    id<MTLBuffer> buf_mlp_gate;
    id<MTLBuffer> buf_mlp_up;
    id<MTLBuffer> buf_mlp_out;
    id<MTLBuffer> buf_logits;
    id<MTLBuffer> buf_mask;
    id<MTLBuffer> buf_attn_scores;

    /* KV cache */
    struct { id<MTLBuffer> k; id<MTLBuffer> v; int max_seq; } kv_cache[GEMMA3_NUM_LAYERS];
    int current_pos;

    /* RoPE tables */
    id<MTLBuffer> buf_rope_local;
    id<MTLBuffer> buf_rope_global;

    /* Config */
    gemma3_config config;
    int max_context;
};

/* ============================================================================
 * Helpers
 * ========================================================================== */

static id<MTLBuffer> wrap_or_copy(id<MTLDevice> dev, const void *ptr, size_t size) {
    if (!ptr || size == 0) return nil;
    size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0) {
        size_t aligned = (size + page - 1) & ~(page - 1);
        id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:(void *)ptr
                                                   length:aligned
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        if (buf) return buf;
    }
    return [dev newBufferWithBytes:ptr length:size options:MTLResourceStorageModeShared];
}

static id<MTLComputePipelineState> make_pso(id<MTLDevice> dev, id<MTLLibrary> lib,
                                             const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        fprintf(stderr, "Metal: function '%s' not found in library\n", name);
        return nil;
    }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Metal: pipeline '%s' failed: %s\n", name,
                err.localizedDescription.UTF8String);
    }
    return pso;
}

/* ============================================================================
 * Public API: availability / init / free
 * ========================================================================== */

int gemma3_metal_available(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil;
}

gemma3_metal_context *gemma3_metal_init(const gemma3_config *cfg, int max_context) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return NULL;

        gemma3_metal_context *ctx = calloc(1, sizeof(gemma3_metal_context));
        if (!ctx) return NULL;

        ctx->device = dev;
        ctx->config = *cfg;
        ctx->max_context = max_context;

        ctx->queue = [dev newCommandQueue];
        if (!ctx->queue) { free(ctx); return NULL; }

        /* Compile MSL */
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:metalShaderSource];
        ctx->library = [dev newLibraryWithSource:src options:nil error:&err];
        if (!ctx->library) {
            fprintf(stderr, "Metal: shader compile failed: %s\n",
                    err.localizedDescription.UTF8String);
            free(ctx);
            return NULL;
        }

        /* Create pipeline states */
        ctx->pso_matvec_bf16   = make_pso(dev, ctx->library, "matvec_bf16");
        ctx->pso_rmsnorm_bf16  = make_pso(dev, ctx->library, "rmsnorm_bf16");
        ctx->pso_embed_bf16    = make_pso(dev, ctx->library, "embed_bf16");
        ctx->pso_gelu_tanh     = make_pso(dev, ctx->library, "gelu_tanh");
        ctx->pso_vec_mul       = make_pso(dev, ctx->library, "vec_mul");
        ctx->pso_vec_add       = make_pso(dev, ctx->library, "vec_add");
        ctx->pso_vec_scale     = make_pso(dev, ctx->library, "vec_scale");
        ctx->pso_vec_copy      = make_pso(dev, ctx->library, "vec_copy");
        ctx->pso_rope_apply    = make_pso(dev, ctx->library, "rope_apply");
        ctx->pso_cache_kv      = make_pso(dev, ctx->library, "cache_kv");
        ctx->pso_gqa_attention = make_pso(dev, ctx->library, "gqa_attention");

        if (!ctx->pso_matvec_bf16 || !ctx->pso_rmsnorm_bf16 || !ctx->pso_embed_bf16 ||
            !ctx->pso_gelu_tanh || !ctx->pso_vec_mul || !ctx->pso_vec_add ||
            !ctx->pso_vec_scale || !ctx->pso_vec_copy || !ctx->pso_rope_apply ||
            !ctx->pso_cache_kv || !ctx->pso_gqa_attention) {
            free(ctx);
            return NULL;
        }

        /* Allocate activation buffers */
        int hs = cfg->hidden_size;
        int is = cfg->intermediate_size;
        int q_size  = cfg->num_heads * cfg->head_dim;
        int kv_size = cfg->num_kv_heads * cfg->head_dim;

        #define ALLOC_F32(name, count) \
            ctx->name = [dev newBufferWithLength:(size_t)(count)*sizeof(float) \
                                        options:MTLResourceStorageModeShared]; \
            if (!ctx->name) { free(ctx); return NULL; }

        ALLOC_F32(buf_x,           hs)
        ALLOC_F32(buf_x_norm,      hs)
        ALLOC_F32(buf_q,           q_size)
        ALLOC_F32(buf_k,           kv_size)
        ALLOC_F32(buf_v,           kv_size)
        ALLOC_F32(buf_attn_out,    q_size)
        ALLOC_F32(buf_proj_out,    hs)
        ALLOC_F32(buf_mlp_gate,    is)
        ALLOC_F32(buf_mlp_up,      is)
        ALLOC_F32(buf_mlp_out,     hs)
        ALLOC_F32(buf_logits,      cfg->vocab_size)
        ALLOC_F32(buf_mask,        max_context)
        ALLOC_F32(buf_attn_scores, cfg->num_heads * max_context)
        #undef ALLOC_F32

        /* Zero the mask buffer (all-zeros causal mask for single-token generation) */
        memset(ctx->buf_mask.contents, 0, (size_t)max_context * sizeof(float));

        /* Allocate KV cache */
        for (int l = 0; l < cfg->num_layers; l++) {
            int layer_max = gemma3_is_global_layer(l) ? max_context : cfg->sliding_window;
            size_t cache_bytes = (size_t)layer_max * kv_size * sizeof(float);
            ctx->kv_cache[l].k = [dev newBufferWithLength:cache_bytes
                                                  options:MTLResourceStorageModeShared];
            ctx->kv_cache[l].v = [dev newBufferWithLength:cache_bytes
                                                  options:MTLResourceStorageModeShared];
            ctx->kv_cache[l].max_seq = layer_max;
            if (!ctx->kv_cache[l].k || !ctx->kv_cache[l].v) { free(ctx); return NULL; }
        }

        ctx->current_pos = 0;
        return ctx;
    }
}

void gemma3_metal_free(gemma3_metal_context *ctx) {
    if (!ctx) return;
    /* ARC handles all Objective-C object releases */
    free(ctx);
}

/* ============================================================================
 * Weight and RoPE upload
 * ========================================================================== */

int gemma3_metal_upload_weights(gemma3_metal_context *ctx, const void *weights_ptr) {
    const gemma3_weights_t *w = (const gemma3_weights_t *)weights_ptr;
    id<MTLDevice> dev = ctx->device;
    const gemma3_config *cfg = &ctx->config;

    int hs = cfg->hidden_size;
    int is = cfg->intermediate_size;
    int q_size  = cfg->num_heads * cfg->head_dim;
    int kv_size = cfg->num_kv_heads * cfg->head_dim;
    int hd = cfg->head_dim;

    /* Embedding + final norm */
    ctx->buf_embed_tokens = wrap_or_copy(dev, w->embed_tokens,
                                          (size_t)cfg->vocab_size * hs * sizeof(uint16_t));
    ctx->buf_norm = wrap_or_copy(dev, w->norm, (size_t)hs * sizeof(uint16_t));
    if (!ctx->buf_embed_tokens || !ctx->buf_norm) return -1;

    /* Per-layer weights */
    for (int l = 0; l < cfg->num_layers; l++) {
        #define WRAP(field, rows, cols) \
            ctx->layer_bufs[l].field = wrap_or_copy(dev, w->layers[l].field, \
                (size_t)(rows) * (cols) * sizeof(uint16_t)); \
            if (!ctx->layer_bufs[l].field && w->layers[l].field) return -1;

        WRAP(input_layernorm,           hs, 1)
        WRAP(q_proj,                    q_size, hs)
        WRAP(k_proj,                    kv_size, hs)
        WRAP(v_proj,                    kv_size, hs)
        WRAP(o_proj,                    hs, q_size)
        WRAP(q_norm,                    hd, 1)
        WRAP(k_norm,                    hd, 1)
        WRAP(post_attention_layernorm,  hs, 1)
        WRAP(gate_proj,                 is, hs)
        WRAP(up_proj,                   is, hs)
        WRAP(down_proj,                 hs, is)
        WRAP(pre_feedforward_layernorm, hs, 1)
        WRAP(post_feedforward_layernorm,hs, 1)
        #undef WRAP
    }

    return 0;
}

int gemma3_metal_upload_rope(gemma3_metal_context *ctx,
                              const float *rope_local, const float *rope_global,
                              int max_context, int head_dim) {
    size_t bytes = (size_t)max_context * (head_dim / 2) * 2 * sizeof(float);
    ctx->buf_rope_local = [ctx->device newBufferWithBytes:rope_local
                                                   length:bytes
                                                  options:MTLResourceStorageModeShared];
    ctx->buf_rope_global = [ctx->device newBufferWithBytes:rope_global
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared];
    return (ctx->buf_rope_local && ctx->buf_rope_global) ? 0 : -1;
}

/* ============================================================================
 * Forward pass
 * ========================================================================== */

int gemma3_metal_forward_token(gemma3_metal_context *ctx, int token_id, int pos,
                                float *logits, int compute_logits) {
    @autoreleasepool {
        const gemma3_config *cfg = &ctx->config;
        int hs = cfg->hidden_size;
        int is = cfg->intermediate_size;
        int q_size  = cfg->num_heads * cfg->head_dim;
        int kv_size = cfg->num_kv_heads * cfg->head_dim;
        int hd = cfg->head_dim;
        int half_dim = hd / 2;

        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        /* === Embedding lookup === */
        {
            MetalEmbedParams p = { (uint32_t)token_id, (uint32_t)hs };
            [enc setComputePipelineState:ctx->pso_embed_bf16];
            [enc setBuffer:ctx->buf_embed_tokens offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_x offset:0 atIndex:1];
            [enc setBytes:&p length:sizeof(p) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(hs, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(MIN(256, hs), 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        /* === Scale embedding by sqrt(hidden_size) === */
        {
            MetalVecParams p = { (uint32_t)hs, sqrtf((float)hs) };
            [enc setComputePipelineState:ctx->pso_vec_scale];
            [enc setBuffer:ctx->buf_x offset:0 atIndex:0];
            [enc setBytes:&p length:sizeof(p) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(hs, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        /* === Layer loop === */
        for (int l = 0; l < cfg->num_layers; l++) {
            int is_global = gemma3_is_global_layer(l);
            id<MTLBuffer> rope_buf = is_global ? ctx->buf_rope_global : ctx->buf_rope_local;

            /* -- Pre-attention RMSNorm -- */
            {
                MetalNormParams p = { (uint32_t)hs, cfg->rmsnorm_eps, (uint32_t)hs, 0 };
                [enc setComputePipelineState:ctx->pso_rmsnorm_bf16];
                [enc setBuffer:ctx->buf_x offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->layer_bufs[l].input_layernorm offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- QKV projections (independent, no barrier between) -- */
            {
                MetalMatvecParams mp = { (uint32_t)q_size, (uint32_t)hs };
                [enc setComputePipelineState:ctx->pso_matvec_bf16];
                [enc setBuffer:ctx->layer_bufs[l].q_proj offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_q offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(q_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            {
                MetalMatvecParams mp = { (uint32_t)kv_size, (uint32_t)hs };
                [enc setComputePipelineState:ctx->pso_matvec_bf16];
                [enc setBuffer:ctx->layer_bufs[l].k_proj offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_k offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(kv_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setBuffer:ctx->layer_bufs[l].v_proj offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_v offset:0 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(kv_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- QK normalization (per-head) -- */
            if (ctx->layer_bufs[l].q_norm && ctx->layer_bufs[l].k_norm) {
                MetalNormParams np = { (uint32_t)hd, cfg->rmsnorm_eps, (uint32_t)hd, 0 };
                [enc setComputePipelineState:ctx->pso_rmsnorm_bf16];
                /* Q heads */
                [enc setBuffer:ctx->buf_q offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_q offset:0 atIndex:1];
                [enc setBuffer:ctx->layer_bufs[l].q_norm offset:0 atIndex:2];
                [enc setBytes:&np length:sizeof(np) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(cfg->num_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                /* K heads */
                [enc setBuffer:ctx->buf_k offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_k offset:0 atIndex:1];
                [enc setBuffer:ctx->layer_bufs[l].k_norm offset:0 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(cfg->num_kv_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            /* -- RoPE -- */
            {
                MetalRopeParams rp = { (uint32_t)hd, (uint32_t)pos,
                                        (uint32_t)cfg->num_heads, 0 };
                [enc setComputePipelineState:ctx->pso_rope_apply];
                [enc setBuffer:ctx->buf_q offset:0 atIndex:0];
                [enc setBuffer:rope_buf offset:0 atIndex:1];
                [enc setBytes:&rp length:sizeof(rp) atIndex:2];
                uint32_t nthreads_q = cfg->num_heads * half_dim;
                [enc dispatchThreads:MTLSizeMake(nthreads_q, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(MIN(256, nthreads_q), 1, 1)];

                rp.num_heads = (uint32_t)cfg->num_kv_heads;
                [enc setBuffer:ctx->buf_k offset:0 atIndex:0];
                [enc setBytes:&rp length:sizeof(rp) atIndex:2];
                uint32_t nthreads_k = cfg->num_kv_heads * half_dim;
                [enc dispatchThreads:MTLSizeMake(nthreads_k, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(MIN(256, nthreads_k), 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- Cache KV -- */
            {
                int cache_pos = is_global ? pos : (pos % cfg->sliding_window);
                MetalCacheParams cp = { (uint32_t)kv_size, (uint32_t)cache_pos };
                [enc setComputePipelineState:ctx->pso_cache_kv];
                [enc setBuffer:ctx->buf_k offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_v offset:0 atIndex:1];
                [enc setBuffer:ctx->kv_cache[l].k offset:0 atIndex:2];
                [enc setBuffer:ctx->kv_cache[l].v offset:0 atIndex:3];
                [enc setBytes:&cp length:sizeof(cp) atIndex:4];
                [enc dispatchThreads:MTLSizeMake(kv_size, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(MIN(256, kv_size), 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- GQA Attention -- */
            {
                int seq_len = is_global ? (pos + 1)
                    : ((pos < cfg->sliding_window) ? (pos + 1) : cfg->sliding_window);
                MetalAttnParams ap = {
                    (uint32_t)cfg->num_heads, (uint32_t)cfg->num_kv_heads,
                    (uint32_t)hd, (uint32_t)seq_len,
                    1.0f / sqrtf((float)hd), (uint32_t)ctx->max_context
                };
                [enc setComputePipelineState:ctx->pso_gqa_attention];
                [enc setBuffer:ctx->buf_q offset:0 atIndex:0];
                [enc setBuffer:ctx->kv_cache[l].k offset:0 atIndex:1];
                [enc setBuffer:ctx->kv_cache[l].v offset:0 atIndex:2];
                [enc setBuffer:ctx->buf_attn_out offset:0 atIndex:3];
                [enc setBuffer:ctx->buf_attn_scores offset:0 atIndex:4];
                [enc setBuffer:ctx->buf_mask offset:0 atIndex:5];
                [enc setBytes:&ap length:sizeof(ap) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(cfg->num_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- Output projection -- */
            {
                MetalMatvecParams mp = { (uint32_t)hs, (uint32_t)q_size };
                [enc setComputePipelineState:ctx->pso_matvec_bf16];
                [enc setBuffer:ctx->layer_bufs[l].o_proj offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_attn_out offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_proj_out offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(hs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- Post-attention RMSNorm (in-place) -- */
            if (ctx->layer_bufs[l].post_attention_layernorm) {
                MetalNormParams np = { (uint32_t)hs, cfg->rmsnorm_eps, (uint32_t)hs, 0 };
                [enc setComputePipelineState:ctx->pso_rmsnorm_bf16];
                [enc setBuffer:ctx->buf_proj_out offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_proj_out offset:0 atIndex:1];
                [enc setBuffer:ctx->layer_bufs[l].post_attention_layernorm offset:0 atIndex:2];
                [enc setBytes:&np length:sizeof(np) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            /* -- Residual: x += proj_out -- */
            {
                uint32_t n = (uint32_t)hs;
                [enc setComputePipelineState:ctx->pso_vec_add];
                [enc setBuffer:ctx->buf_x offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_x offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_proj_out offset:0 atIndex:2];
                [enc setBytes:&n length:sizeof(n) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(hs, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- Pre-feedforward RMSNorm -- */
            if (ctx->layer_bufs[l].pre_feedforward_layernorm) {
                MetalNormParams np = { (uint32_t)hs, cfg->rmsnorm_eps, (uint32_t)hs, 0 };
                [enc setComputePipelineState:ctx->pso_rmsnorm_bf16];
                [enc setBuffer:ctx->buf_x offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->layer_bufs[l].pre_feedforward_layernorm offset:0 atIndex:2];
                [enc setBytes:&np length:sizeof(np) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                uint32_t n = (uint32_t)hs;
                [enc setComputePipelineState:ctx->pso_vec_copy];
                [enc setBuffer:ctx->buf_x offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_x_norm offset:0 atIndex:1];
                [enc setBytes:&n length:sizeof(n) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(hs, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- Gate + Up projections (independent) -- */
            {
                MetalMatvecParams mp = { (uint32_t)is, (uint32_t)hs };
                [enc setComputePipelineState:ctx->pso_matvec_bf16];
                [enc setBuffer:ctx->layer_bufs[l].gate_proj offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_mlp_gate offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(is, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setBuffer:ctx->layer_bufs[l].up_proj offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_mlp_up offset:0 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(is, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- GELU -- */
            {
                uint32_t n = (uint32_t)is;
                [enc setComputePipelineState:ctx->pso_gelu_tanh];
                [enc setBuffer:ctx->buf_mlp_gate offset:0 atIndex:0];
                [enc setBytes:&n length:sizeof(n) atIndex:1];
                [enc dispatchThreads:MTLSizeMake(is, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];


            /* -- vec_mul: gate *= up -- */
            {
                uint32_t n = (uint32_t)is;
                [enc setComputePipelineState:ctx->pso_vec_mul];
                [enc setBuffer:ctx->buf_mlp_gate offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_mlp_up offset:0 atIndex:1];
                [enc setBytes:&n length:sizeof(n) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(is, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- Down projection -- */
            {
                MetalMatvecParams mp = { (uint32_t)hs, (uint32_t)is };
                [enc setComputePipelineState:ctx->pso_matvec_bf16];
                [enc setBuffer:ctx->layer_bufs[l].down_proj offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_mlp_gate offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_mlp_out offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(hs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* -- Post-feedforward RMSNorm (in-place) -- */
            if (ctx->layer_bufs[l].post_feedforward_layernorm) {
                MetalNormParams np = { (uint32_t)hs, cfg->rmsnorm_eps, (uint32_t)hs, 0 };
                [enc setComputePipelineState:ctx->pso_rmsnorm_bf16];
                [enc setBuffer:ctx->buf_mlp_out offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_mlp_out offset:0 atIndex:1];
                [enc setBuffer:ctx->layer_bufs[l].post_feedforward_layernorm offset:0 atIndex:2];
                [enc setBytes:&np length:sizeof(np) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            /* -- Residual: x += mlp_out -- */
            {
                uint32_t n = (uint32_t)hs;
                [enc setComputePipelineState:ctx->pso_vec_add];
                [enc setBuffer:ctx->buf_x offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_x offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_mlp_out offset:0 atIndex:2];
                [enc setBytes:&n length:sizeof(n) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(hs, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        } /* end layer loop */

        if (compute_logits) {
            /* === Final RMSNorm === */
            {
                MetalNormParams np = { (uint32_t)hs, cfg->rmsnorm_eps, (uint32_t)hs, 0 };
                [enc setComputePipelineState:ctx->pso_rmsnorm_bf16];
                [enc setBuffer:ctx->buf_x offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_norm offset:0 atIndex:2];
                [enc setBytes:&np length:sizeof(np) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* === Logits projection (tied embeddings) === */
            {
                MetalMatvecParams mp = { (uint32_t)cfg->vocab_size, (uint32_t)hs };
                [enc setComputePipelineState:ctx->pso_matvec_bf16];
                [enc setBuffer:ctx->buf_embed_tokens offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_logits offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(cfg->vocab_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        /* Check for GPU errors */
        if (cmdBuf.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Metal: GPU error: %s\n",
                    cmdBuf.error.localizedDescription.UTF8String);
            return -1;
        }

        /* Copy logits back to CPU */
        if (compute_logits && logits) {
            memcpy(logits, ctx->buf_logits.contents,
                   (size_t)cfg->vocab_size * sizeof(float));
        }

        ctx->current_pos = pos + 1;
        return 0;
    }
}

/* ============================================================================
 * Prefill and cache reset
 * ========================================================================== */

int gemma3_metal_prefill(gemma3_metal_context *ctx, const int *tokens, int num_tokens,
                          int start_pos, float *logits) {
    for (int i = 0; i < num_tokens; i++) {
        int p = start_pos + i;
        int is_last = (i == num_tokens - 1);
        int ret = gemma3_metal_forward_token(ctx, tokens[i], p, logits, is_last);
        if (ret != 0) return ret;
    }
    ctx->current_pos = start_pos + num_tokens;
    return 0;
}

void gemma3_metal_reset_cache(gemma3_metal_context *ctx) {
    if (!ctx) return;
    ctx->current_pos = 0;
    /* Zero all KV cache buffers */
    for (int l = 0; l < ctx->config.num_layers; l++) {
        memset(ctx->kv_cache[l].k.contents, 0, ctx->kv_cache[l].k.length);
        memset(ctx->kv_cache[l].v.contents, 0, ctx->kv_cache[l].v.length);
    }
}

#endif /* USE_MPS */
