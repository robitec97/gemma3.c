/*
 * gemma3_metal.m - Metal GPU backend for Gemma 3 inference
 *
 * Custom Metal Shading Language compute kernels for the full transformer
 * forward pass. MSL source is embedded as a C string and compiled at runtime.
 *
 * Design:
 *   - Weights are never copied: the mmap'd safetensors files are wrapped with
 *     newBufferWithBytesNoCopy and each tensor is bound by offset.
 *   - Decode (one token) uses bandwidth-oriented matrix-vector kernels that
 *     process several rows per simdgroup, plus fused kernels for QKV,
 *     gate/up/GELU, QK-norm/RoPE/KV-write and norm/residual/norm.
 *   - Prefill processes up to GEMMA3_PREFILL_CHUNK tokens at a time with a
 *     simdgroup-matrix GEMM, so each weight row is read once per chunk.
 *   - Attention is position-aware: local layers use a KV ring buffer of
 *     gemma3_local_ring_size() slots, so the cache can be rewound and a whole
 *     chunk can write its K/V before attention runs.
 */

#ifdef USE_MPS

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "gemma3_metal.h"
#include "gemma3_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* ============================================================================
 * Kernel tiling constants (must match the MSL prelude generated below)
 * ========================================================================== */

#define MV_NSG    4    /* simdgroups per matvec threadgroup */
#define MV_R      4    /* rows per simdgroup (plain / QKV matvec) */
#define GU_R      2    /* rows per simdgroup (fused gate/up matvec) */
#define NORM_TG   256  /* threads per normalization threadgroup */
#define ATTN_NSG  4    /* simdgroups per attention threadgroup */
#define ATTN_SPLIT_LEN 128  /* keys per split for decode attention */
#define ATTN_MAX_SPLITS 64
#define GEMM_BM   64
#define GEMM_BN   32

/* ============================================================================
 * GPU parameter structs (must match the MSL definitions exactly)
 * ========================================================================== */

typedef struct { uint32_t M; uint32_t K; } MetalMatvecArgs;
typedef struct { uint32_t q_rows; uint32_t kv_rows; uint32_t K; } MetalQKVArgs;
typedef struct { float eps; float scale; } MetalNormArgs;
typedef struct {
    uint32_t pos0; uint32_t is_global; uint32_t ring; float eps;
} MetalRopeArgs;
typedef struct {
    uint32_t pos0; uint32_t is_global; uint32_t window; uint32_t ring;
    uint32_t n_splits; uint32_t split_len; float scale; uint32_t n_tokens;
} MetalAttnArgs;
typedef struct { uint32_t M; uint32_t K; } MetalGemmArgs;
typedef struct { uint32_t n; } MetalCountArgs;

/* ============================================================================
 * Embedded Metal Shading Language source
 *
 * A prelude with model dimensions (#define HIDDEN, HD, ...) is prepended at
 * runtime so loops over fixed sizes are fully unrolled.
 * ========================================================================== */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
static const char *metalShaderSource =
"#include <metal_stdlib>\n"
"#include <metal_simdgroup_matrix>\n"
"using namespace metal;\n"
"\n"
"inline float bf16lo(uint v) { return as_type<float>(v << 16); }\n"
"inline float bf16hi(uint v) { return as_type<float>(v & 0xFFFF0000u); }\n"
"inline float bf16f(ushort v) { return as_type<float>(uint(v) << 16); }\n"
"inline float4 bf16x4(uint2 w) {\n"
"    return float4(bf16lo(w.x), bf16hi(w.x), bf16lo(w.y), bf16hi(w.y));\n"
"}\n"
"\n"
"inline float gelu_tanh(float v) {\n"
"    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);\n"
"    inner = clamp(inner, -15.0f, 15.0f);  // fast-math tanh misbehaves on huge inputs\n"
"    return 0.5f * v * (1.0f + tanh(inner));\n"
"}\n"
"\n"
"struct MatvecArgs { uint M; uint K; };\n"
"struct QKVArgs { uint q_rows; uint kv_rows; uint K; };\n"
"struct NormArgs { float eps; float scale; };\n"
"struct RopeArgs { uint pos0; uint is_global; uint ring; float eps; };\n"
"struct AttnArgs {\n"
"    uint pos0; uint is_global; uint window; uint ring;\n"
"    uint n_splits; uint split_len; float scale; uint n_tokens;\n"
"};\n"
"struct GemmArgs { uint M; uint K; };\n"
"struct CountArgs { uint n; };\n"
"\n"
"// ---------------------------------------------------------------------------\n"
"// Matrix-vector: y[M] = W[M,K] (bf16) * x[K].  W rows must be 8-byte aligned,\n"
"// K a multiple of 4. Each simdgroup computes R rows; lanes stride over K.\n"
"// ---------------------------------------------------------------------------\n"
"template <uint R>\n"
"inline void mv_accum(device const uint2 *W, device const float4 *x, uint K4,\n"
"                     uint row0, uint M, uint lane, thread float *sum) {\n"
"    device const uint2 *w[R];\n"
"    for (uint r = 0; r < R; r++) {\n"
"        sum[r] = 0.0f;\n"
"        w[r] = W + min(row0 + r, M - 1) * K4;\n"
"    }\n"
"    uint i = lane;\n"
"    for (; i + 32 < K4; i += 64) {\n"
"        float4 x0 = x[i];\n"
"        float4 x1 = x[i + 32];\n"
"        for (uint r = 0; r < R; r++) {\n"
"            sum[r] += dot(bf16x4(w[r][i]), x0) + dot(bf16x4(w[r][i + 32]), x1);\n"
"        }\n"
"    }\n"
"    for (; i < K4; i += 32) {\n"
"        float4 x0 = x[i];\n"
"        for (uint r = 0; r < R; r++) sum[r] += dot(bf16x4(w[r][i]), x0);\n"
"    }\n"
"}\n"
"\n"
"kernel void matvec_bf16(\n"
"    device const uint2  *W [[buffer(0)]],\n"
"    device const float4 *x [[buffer(1)]],\n"
"    device float        *y [[buffer(2)]],\n"
"    constant MatvecArgs &a [[buffer(3)]],\n"
"    uint tg   [[threadgroup_position_in_grid]],\n"
"    uint sg   [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]\n"
") {\n"
"    uint row0 = (tg * MV_NSG + sg) * MV_R;\n"
"    if (row0 >= a.M) return;\n"
"    float sum[MV_R];\n"
"    mv_accum<MV_R>(W, x, a.K / 4, row0, a.M, lane, sum);\n"
"    for (uint r = 0; r < MV_R; r++) {\n"
"        float s = simd_sum(sum[r]);\n"
"        if (lane == 0 && row0 + r < a.M) y[row0 + r] = s;\n"
"    }\n"
"}\n"
"\n"
"// Q, K and V projections in one dispatch (rows of the three matrices concatenated).\n"
"kernel void matvec_qkv(\n"
"    device const uint2  *Wq [[buffer(0)]],\n"
"    device const uint2  *Wk [[buffer(1)]],\n"
"    device const uint2  *Wv [[buffer(2)]],\n"
"    device const float4 *x  [[buffer(3)]],\n"
"    device float        *q  [[buffer(4)]],\n"
"    device float        *k  [[buffer(5)]],\n"
"    device float        *v  [[buffer(6)]],\n"
"    constant QKVArgs    &a  [[buffer(7)]],\n"
"    uint tg   [[threadgroup_position_in_grid]],\n"
"    uint sg   [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]\n"
") {\n"
"    uint row = (tg * MV_NSG + sg) * MV_R;\n"
"    device const uint2 *W;\n"
"    device float *y;\n"
"    uint M;\n"
"    if (row < a.q_rows) {\n"
"        W = Wq; y = q; M = a.q_rows;\n"
"    } else if (row < a.q_rows + a.kv_rows) {\n"
"        W = Wk; y = k; M = a.kv_rows; row -= a.q_rows;\n"
"    } else {\n"
"        row -= a.q_rows + a.kv_rows;\n"
"        if (row >= a.kv_rows) return;\n"
"        W = Wv; y = v; M = a.kv_rows;\n"
"    }\n"
"    float sum[MV_R];\n"
"    mv_accum<MV_R>(W, x, a.K / 4, row, M, lane, sum);\n"
"    for (uint r = 0; r < MV_R; r++) {\n"
"        float s = simd_sum(sum[r]);\n"
"        if (lane == 0 && row + r < M) y[row + r] = s;\n"
"    }\n"
"}\n"
"\n"
"// y[i] = gelu(Wg[i].x) * (Wu[i].x)\n"
"kernel void matvec_gateup(\n"
"    device const uint2  *Wg [[buffer(0)]],\n"
"    device const uint2  *Wu [[buffer(1)]],\n"
"    device const float4 *x  [[buffer(2)]],\n"
"    device float        *y  [[buffer(3)]],\n"
"    constant MatvecArgs &a  [[buffer(4)]],\n"
"    uint tg   [[threadgroup_position_in_grid]],\n"
"    uint sg   [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]\n"
") {\n"
"    uint row0 = (tg * MV_NSG + sg) * GU_R;\n"
"    if (row0 >= a.M) return;\n"
"    const uint K4 = a.K / 4;\n"
"    device const uint2 *wg[GU_R];\n"
"    device const uint2 *wu[GU_R];\n"
"    float g[GU_R], u[GU_R];\n"
"    for (uint r = 0; r < GU_R; r++) {\n"
"        uint row = min(row0 + r, a.M - 1);\n"
"        wg[r] = Wg + row * K4;\n"
"        wu[r] = Wu + row * K4;\n"
"        g[r] = 0.0f; u[r] = 0.0f;\n"
"    }\n"
"    uint i = lane;\n"
"    for (; i + 32 < K4; i += 64) {\n"
"        float4 x0 = x[i];\n"
"        float4 x1 = x[i + 32];\n"
"        for (uint r = 0; r < GU_R; r++) {\n"
"            g[r] += dot(bf16x4(wg[r][i]), x0) + dot(bf16x4(wg[r][i + 32]), x1);\n"
"            u[r] += dot(bf16x4(wu[r][i]), x0) + dot(bf16x4(wu[r][i + 32]), x1);\n"
"        }\n"
"    }\n"
"    for (; i < K4; i += 32) {\n"
"        float4 x0 = x[i];\n"
"        for (uint r = 0; r < GU_R; r++) {\n"
"            g[r] += dot(bf16x4(wg[r][i]), x0);\n"
"            u[r] += dot(bf16x4(wu[r][i]), x0);\n"
"        }\n"
"    }\n"
"    for (uint r = 0; r < GU_R; r++) {\n"
"        float gs = simd_sum(g[r]);\n"
"        float us = simd_sum(u[r]);\n"
"        if (lane == 0 && row0 + r < a.M) y[row0 + r] = gelu_tanh(gs) * us;\n"
"    }\n"
"}\n"
"\n"
"// ---------------------------------------------------------------------------\n"
"// GEMM for prefill: Y[N,M] = X[N,K] * W[M,K]^T, W in bf16 (converted while\n"
"// staging into threadgroup memory). Tile 64(M) x 32(N) x 32(K), 4 simdgroups\n"
"// each owning a 32x16 block of 8x8 accumulators. N is padded to 32.\n"
"// ---------------------------------------------------------------------------\n"
"#define BM 64\n"
"#define BN 32\n"
"#define BK 32\n"
"inline void gemm_tile(device const uint2 *W, device const float4 *X, device float *Y,\n"
"                      uint M, uint K, uint m0, uint n0, uint tid, uint sg,\n"
"                      threadgroup float *sW, threadgroup float *sX) {\n"
"    const uint K4 = K / 4;\n"
"    // Staging: each thread converts 16 bf16 of W and copies 8 floats of X per K step.\n"
"    device const uint2 *wp = W + (m0 + tid / 2) * K4 + (tid % 2) * 4;\n"
"    threadgroup float4 *sw = (threadgroup float4 *)(sW + (tid / 2) * BK + (tid % 2) * 16);\n"
"    device const float4 *xp = X + (n0 + tid / 4) * K4 + (tid % 4) * 2;\n"
"    threadgroup float4 *sx = (threadgroup float4 *)(sX + (tid / 4) * BK + (tid % 4) * 8);\n"
"    const uint sm = (sg / 2) * 32;\n"
"    const uint sn = (sg % 2) * 16;\n"
"    simdgroup_float8x8 acc[4][2];\n"
"    for (uint i = 0; i < 4; i++)\n"
"        for (uint j = 0; j < 2; j++)\n"
"            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n"
"    for (uint k = 0; k < K4; k += BK / 4) {\n"
"        uint2 w0 = wp[k], w1 = wp[k + 1], w2 = wp[k + 2], w3 = wp[k + 3];\n"
"        float4 x0 = xp[k], x1 = xp[k + 1];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        sw[0] = bf16x4(w0); sw[1] = bf16x4(w1); sw[2] = bf16x4(w2); sw[3] = bf16x4(w3);\n"
"        sx[0] = x0; sx[1] = x1;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint kk = 0; kk < BK; kk += 8) {\n"
"            simdgroup_float8x8 A[4];\n"
"            simdgroup_float8x8 B[2];\n"
"            for (uint i = 0; i < 4; i++)\n"
"                simdgroup_load(A[i], sW + (sm + i * 8) * BK + kk, BK);\n"
"            for (uint j = 0; j < 2; j++)\n"
"                simdgroup_load(B[j], sX + (sn + j * 8) * BK + kk, BK, ulong2(0, 0), true);\n"
"            for (uint i = 0; i < 4; i++)\n"
"                for (uint j = 0; j < 2; j++)\n"
"                    simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);\n"
"        }\n"
"    }\n"
"    for (uint i = 0; i < 4; i++)\n"
"        for (uint j = 0; j < 2; j++)\n"
"            simdgroup_store(acc[i][j], Y + (n0 + sn + j * 8) * M + m0 + sm + i * 8,\n"
"                            M, ulong2(0, 0), true);\n"
"}\n"
"\n"
"kernel void gemm_bf16(\n"
"    device const uint2  *W [[buffer(0)]],\n"
"    device const float4 *X [[buffer(1)]],\n"
"    device float        *Y [[buffer(2)]],\n"
"    constant GemmArgs   &a [[buffer(3)]],\n"
"    uint2 tg  [[threadgroup_position_in_grid]],\n"
"    uint tid  [[thread_index_in_threadgroup]],\n"
"    uint sg   [[simdgroup_index_in_threadgroup]]\n"
") {\n"
"    threadgroup float sW[BM * BK];\n"
"    threadgroup float sX[BN * BK];\n"
"    gemm_tile(W, X, Y, a.M, a.K, tg.x * BM, tg.y * BN, tid, sg, sW, sX);\n"
"}\n"
"\n"
"// Q, K and V projections in one dispatch: the grid covers the rows of all\n"
"// three matrices, so the narrow K/V projections still fill the GPU.\n"
"kernel void gemm_qkv(\n"
"    device const uint2  *Wq [[buffer(0)]],\n"
"    device const uint2  *Wk [[buffer(1)]],\n"
"    device const uint2  *Wv [[buffer(2)]],\n"
"    device const float4 *X  [[buffer(3)]],\n"
"    device float        *Q  [[buffer(4)]],\n"
"    device float        *K  [[buffer(5)]],\n"
"    device float        *V  [[buffer(6)]],\n"
"    constant QKVArgs    &a  [[buffer(7)]],\n"
"    uint2 tg  [[threadgroup_position_in_grid]],\n"
"    uint tid  [[thread_index_in_threadgroup]],\n"
"    uint sg   [[simdgroup_index_in_threadgroup]]\n"
") {\n"
"    threadgroup float sW[BM * BK];\n"
"    threadgroup float sX[BN * BK];\n"
"    uint m0 = tg.x * BM;\n"
"    if (m0 < a.q_rows) {\n"
"        gemm_tile(Wq, X, Q, a.q_rows, a.K, m0, tg.y * BN, tid, sg, sW, sX);\n"
"    } else if (m0 < a.q_rows + a.kv_rows) {\n"
"        gemm_tile(Wk, X, K, a.kv_rows, a.K, m0 - a.q_rows, tg.y * BN, tid, sg, sW, sX);\n"
"    } else {\n"
"        gemm_tile(Wv, X, V, a.kv_rows, a.K, m0 - a.q_rows - a.kv_rows, tg.y * BN, tid, sg, sW, sX);\n"
"    }\n"
"}\n"
"\n"
"// g[i] = gelu(g[i]) * u[i]\n"
"kernel void gelu_mul(\n"
"    device float       *g [[buffer(0)]],\n"
"    device const float *u [[buffer(1)]],\n"
"    constant CountArgs &a [[buffer(2)]],\n"
"    uint i [[thread_position_in_grid]]\n"
") {\n"
"    if (i >= a.n) return;\n"
"    g[i] = gelu_tanh(g[i]) * u[i];\n"
"}\n"
"\n"
"// ---------------------------------------------------------------------------\n"
"// Normalization (one threadgroup of NORM_TG threads per token row)\n"
"// ---------------------------------------------------------------------------\n"
"#define NORM_NPT ((HIDDEN + NORM_TG - 1) / NORM_TG)\n"
"inline float tg_sum(float v, threadgroup float *sh, uint sg, uint lane) {\n"
"    v = simd_sum(v);\n"
"    if (lane == 0) sh[sg] = v;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float t = 0.0f;\n"
"    for (uint i = 0; i < NORM_TG / 32; i++) t += sh[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    return t;\n"
"}\n"
"\n"
"// X[t] = embed[token[t]] * scale;  Xn[t] = rmsnorm(X[t]) * (1 + w)\n"
"kernel void embed_norm(\n"
"    device const ushort *E      [[buffer(0)]],\n"
"    device float        *X      [[buffer(1)]],\n"
"    device float        *Xn     [[buffer(2)]],\n"
"    device const ushort *w      [[buffer(3)]],\n"
"    constant NormArgs   &a      [[buffer(4)]],\n"
"    constant int        *tokens [[buffer(5)]],\n"
"    uint t    [[threadgroup_position_in_grid]],\n"
"    uint tid  [[thread_index_in_threadgroup]],\n"
"    uint sg   [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]\n"
") {\n"
"    threadgroup float sh[NORM_TG / 32];\n"
"    device const ushort *row = E + uint(tokens[t]) * HIDDEN;\n"
"    device float *xo = X + t * HIDDEN;\n"
"    device float *xn = Xn + t * HIDDEN;\n"
"    float vals[NORM_NPT];\n"
"    float ss = 0.0f;\n"
"    for (uint k = 0; k < NORM_NPT; k++) {\n"
"        uint i = tid + k * NORM_TG;\n"
"        float v = (i < HIDDEN) ? bf16f(row[i]) * a.scale : 0.0f;\n"
"        vals[k] = v;\n"
"        ss += v * v;\n"
"        if (i < HIDDEN) xo[i] = v;\n"
"    }\n"
"    float rs = rsqrt(tg_sum(ss, sh, sg, lane) / float(HIDDEN) + a.eps);\n"
"    for (uint k = 0; k < NORM_NPT; k++) {\n"
"        uint i = tid + k * NORM_TG;\n"
"        if (i < HIDDEN) xn[i] = vals[k] * rs * (1.0f + bf16f(w[i]));\n"
"    }\n"
"}\n"
"\n"
"// X[t] += rmsnorm(P[t]) * (1 + w1);  Xn[t] = rmsnorm(X[t]) * (1 + w2)\n"
"kernel void add_norm(\n"
"    device const float  *P  [[buffer(0)]],\n"
"    device float        *X  [[buffer(1)]],\n"
"    device float        *Xn [[buffer(2)]],\n"
"    device const ushort *w1 [[buffer(3)]],\n"
"    device const ushort *w2 [[buffer(4)]],\n"
"    constant NormArgs   &a  [[buffer(5)]],\n"
"    uint t    [[threadgroup_position_in_grid]],\n"
"    uint tid  [[thread_index_in_threadgroup]],\n"
"    uint sg   [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]\n"
") {\n"
"    threadgroup float sh[NORM_TG / 32];\n"
"    device const float *p = P + t * HIDDEN;\n"
"    device float *x = X + t * HIDDEN;\n"
"    device float *xn = Xn + t * HIDDEN;\n"
"    float vals[NORM_NPT];\n"
"    float ss = 0.0f;\n"
"    for (uint k = 0; k < NORM_NPT; k++) {\n"
"        uint i = tid + k * NORM_TG;\n"
"        float v = (i < HIDDEN) ? p[i] : 0.0f;\n"
"        vals[k] = v;\n"
"        ss += v * v;\n"
"    }\n"
"    float rs = rsqrt(tg_sum(ss, sh, sg, lane) / float(HIDDEN) + a.eps);\n"
"    ss = 0.0f;\n"
"    for (uint k = 0; k < NORM_NPT; k++) {\n"
"        uint i = tid + k * NORM_TG;\n"
"        if (i < HIDDEN) {\n"
"            float v = x[i] + vals[k] * rs * (1.0f + bf16f(w1[i]));\n"
"            x[i] = v;\n"
"            vals[k] = v;\n"
"            ss += v * v;\n"
"        }\n"
"    }\n"
"    rs = rsqrt(tg_sum(ss, sh, sg, lane) / float(HIDDEN) + a.eps);\n"
"    for (uint k = 0; k < NORM_NPT; k++) {\n"
"        uint i = tid + k * NORM_TG;\n"
"        if (i < HIDDEN) xn[i] = vals[k] * rs * (1.0f + bf16f(w2[i]));\n"
"    }\n"
"}\n"
"\n"
"// ---------------------------------------------------------------------------\n"
"// QK-norm + RoPE + KV-cache write. Grid: (NUM_HEADS + NUM_KV_HEADS, tokens),\n"
"// HD/2 threads; thread i owns the RoPE pair (i, i + HD/2).\n"
"// ---------------------------------------------------------------------------\n"
"kernel void qk_norm_rope_cache(\n"
"    device float        *Q    [[buffer(0)]],\n"
"    device const float  *K    [[buffer(1)]],\n"
"    device const float  *V    [[buffer(2)]],\n"
"    device KV_T         *Kc   [[buffer(3)]],\n"
"    device KV_T         *Vc   [[buffer(4)]],\n"
"    device const ushort *qw   [[buffer(5)]],\n"
"    device const ushort *kw   [[buffer(6)]],\n"
"    device const float2 *rope [[buffer(7)]],\n"
"    constant RopeArgs   &a    [[buffer(8)]],\n"
"    uint2 tg  [[threadgroup_position_in_grid]],\n"
"    uint tid  [[thread_index_in_threadgroup]],\n"
"    uint sg   [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]\n"
") {\n"
"    threadgroup float sh[HD / 64];\n"
"    const uint head = tg.x;\n"
"    const uint t = tg.y;\n"
"    const uint p = a.pos0 + t;\n"
"    const bool isq = head < NUM_HEADS;\n"
"    const uint kvh = head - NUM_HEADS;\n"
"    const uint i = tid;\n"
"    float x0, x1;\n"
"    if (isq) {\n"
"        device const float *src = Q + t * Q_SIZE + head * HD;\n"
"        x0 = src[i]; x1 = src[i + HD / 2];\n"
"    } else {\n"
"        device const float *src = K + t * KV_SIZE + kvh * HD;\n"
"        x0 = src[i]; x1 = src[i + HD / 2];\n"
"    }\n"
"    float ss = simd_sum(x0 * x0 + x1 * x1);\n"
"    if (lane == 0) sh[sg] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    ss = 0.0f;\n"
"    for (uint s = 0; s < HD / 64; s++) ss += sh[s];\n"
"    float rs = rsqrt(ss / float(HD) + a.eps);\n"
"    device const ushort *w = isq ? qw : kw;\n"
"    x0 *= rs * (1.0f + bf16f(w[i]));\n"
"    x1 *= rs * (1.0f + bf16f(w[i + HD / 2]));\n"
"    float2 cs = rope[p * (HD / 2) + i];\n"
"    float y0 = x0 * cs.x - x1 * cs.y;\n"
"    float y1 = x0 * cs.y + x1 * cs.x;\n"
"    if (isq) {\n"
"        device float *dst = Q + t * Q_SIZE + head * HD;\n"
"        dst[i] = y0; dst[i + HD / 2] = y1;\n"
"    } else {\n"
"        uint slot = a.is_global ? p : p % a.ring;\n"
"        device KV_T *kd = Kc + slot * KV_SIZE + kvh * HD;\n"
"        kd[i] = KV_T(y0); kd[i + HD / 2] = KV_T(y1);\n"
"        device const float *vs = V + t * KV_SIZE + kvh * HD;\n"
"        device KV_T *vd = Vc + slot * KV_SIZE + kvh * HD;\n"
"        vd[i] = KV_T(vs[i]); vd[i + HD / 2] = KV_T(vs[i + HD / 2]);\n"
"    }\n"
"}\n"
"\n"
"// ---------------------------------------------------------------------------\n"
"// Attention. Grid: (NUM_KV_HEADS, ceil(tokens / QB), splits), ATTN_NSG\n"
"// simdgroups. A threadgroup handles QB consecutive query tokens and the GQA\n"
"// query heads that share one KV head, so every key/value row it loads is\n"
"// reused for QB * GQA queries. Simdgroups take blocks of 32 keys: lane j\n"
"// scores key j against all queries, the block is folded into an online\n"
"// softmax with one simdgroup reduction, and V is accumulated with lanes over\n"
"// head dims. Simdgroups are merged at the end; with several splits, partial\n"
"// (max, sum, acc) are written out and merged by attn_reduce.\n"
"// ---------------------------------------------------------------------------\n"
"#define DPL (HD / 32)\n"
"#define MASKED (-3.0e38f)\n"
"inline uint attn_lo(uint p, constant AttnArgs &a) {\n"
"    return (a.is_global || p + 1 <= a.window) ? 0 : p + 1 - a.window;\n"
"}\n"
"\n"
"template <uint QB>\n"
"kernel void attention_t(\n"
"    device const float *Q    [[buffer(0)]],\n"
"    device const KV_T  *Kc   [[buffer(1)]],\n"
"    device const KV_T  *Vc   [[buffer(2)]],\n"
"    device float       *O    [[buffer(3)]],\n"
"    device float       *part [[buffer(4)]],\n"
"    constant AttnArgs  &a    [[buffer(5)]],\n"
"    uint3 tg  [[threadgroup_position_in_grid]],\n"
"    uint tid  [[thread_index_in_threadgroup]],\n"
"    uint sg   [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]\n"
") {\n"
"    constexpr uint NQ = QB * GQA;\n"
"    threadgroup float4 sq[NQ][HD / 4];\n"
"    threadgroup float sm[ATTN_NSG][NQ];\n"
"    threadgroup float sl[ATTN_NSG][NQ];\n"
"    threadgroup float sacc[NQ][HD];\n"
"\n"
"    const uint kvh = tg.x;\n"
"    const uint t0 = tg.y * QB;\n"
"    const uint split = tg.z;\n"
"    const uint nt = min(QB, a.n_tokens - t0);\n"
"    const uint p_last = a.pos0 + t0 + nt - 1;\n"
"    const uint s_lo = attn_lo(a.pos0 + t0, a) + split * a.split_len;\n"
"    const uint s_hi = min(p_last + 1, s_lo + a.split_len);\n"
"\n"
"    // Stage the (pre-scaled) queries in threadgroup memory.\n"
"    for (uint idx = tid; idx < NQ * HD / 4; idx += ATTN_NSG * 32) {\n"
"        uint qi = idx / (HD / 4);\n"
"        uint d4 = idx % (HD / 4);\n"
"        uint tq = qi / GQA;\n"
"        float4 v = float4(0.0f);\n"
"        if (tq < nt) {\n"
"            device const float4 *qp = (device const float4 *)\n"
"                (Q + (t0 + tq) * Q_SIZE + (kvh * GQA + qi % GQA) * HD);\n"
"            v = qp[d4] * a.scale;\n"
"        }\n"
"        sq[qi][d4] = v;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    // Per-query causal / sliding-window bounds\n"
"    uint q_lo[QB], q_hi[QB];\n"
"    for (uint tq = 0; tq < QB; tq++) {\n"
"        uint pq = a.pos0 + t0 + min(tq, nt - 1);\n"
"        q_lo[tq] = attn_lo(pq, a);\n"
"        q_hi[tq] = (tq < nt) ? pq : 0;  // inclusive; tq >= nt never matches below\n"
"    }\n"
"\n"
"    float m[NQ], l[NQ], acc[NQ][DPL];\n"
"    for (uint qi = 0; qi < NQ; qi++) {\n"
"        m[qi] = -1.0e30f; l[qi] = 0.0f;\n"
"        for (uint d = 0; d < DPL; d++) acc[qi][d] = 0.0f;\n"
"    }\n"
"\n"
"    for (uint b = s_lo + sg * 32; b < s_hi; b += ATTN_NSG * 32) {\n"
"        const uint j = b + lane;\n"
"        const bool in_range = j < s_hi;\n"
"        float s[NQ];\n"
"        for (uint qi = 0; qi < NQ; qi++) s[qi] = 0.0f;\n"
"        if (in_range) {\n"
"            uint slot = a.is_global ? j : j % a.ring;\n"
"            device const KV_T4 *kp = (device const KV_T4 *)(Kc + slot * KV_SIZE + kvh * HD);\n"
"            for (uint d4 = 0; d4 < HD / 4; d4++) {\n"
"                float4 k4 = float4(kp[d4]);\n"
"                for (uint qi = 0; qi < NQ; qi++) s[qi] += dot(k4, sq[qi][d4]);\n"
"            }\n"
"        }\n"
"        for (uint qi = 0; qi < NQ; qi++) {\n"
"            uint tq = qi / GQA;\n"
"            bool ok = in_range && tq < nt && j >= q_lo[tq] && j <= q_hi[tq];\n"
"            float sc = ok ? s[qi] : MASKED;\n"
"            float m_new = max(m[qi], simd_max(sc));\n"
"            float c = exp(m[qi] - m_new);\n"
"            float pj = exp(sc - m_new);\n"
"            l[qi] = l[qi] * c + simd_sum(pj);\n"
"            for (uint d = 0; d < DPL; d++) acc[qi][d] *= c;\n"
"            m[qi] = m_new;\n"
"            s[qi] = pj;\n"
"        }\n"
"        const uint nk = min(32u, s_hi - b);\n"
"        for (uint jj = 0; jj < nk; jj++) {\n"
"            uint jk = b + jj;\n"
"            uint slot = a.is_global ? jk : jk % a.ring;\n"
"            device const KV_T4 *vp = (device const KV_T4 *)(Vc + slot * KV_SIZE + kvh * HD + lane * DPL);\n"
"            float4 v0 = float4(vp[0]);\n"
"            float4 v1 = float4(vp[1]);\n"
"            for (uint qi = 0; qi < NQ; qi++) {\n"
"                float pj = simd_shuffle(s[qi], (ushort)jj);\n"
"                acc[qi][0] += pj * v0.x; acc[qi][1] += pj * v0.y;\n"
"                acc[qi][2] += pj * v0.z; acc[qi][3] += pj * v0.w;\n"
"                acc[qi][4] += pj * v1.x; acc[qi][5] += pj * v1.y;\n"
"                acc[qi][6] += pj * v1.z; acc[qi][7] += pj * v1.w;\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    // Merge simdgroups: rescale each to the common max, then sum in turn.\n"
"    if (lane == 0) {\n"
"        for (uint qi = 0; qi < NQ; qi++) { sm[sg][qi] = m[qi]; sl[sg][qi] = l[qi]; }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint qi = 0; qi < NQ; qi++) {\n"
"        float M = sm[0][qi];\n"
"        for (uint s2 = 1; s2 < ATTN_NSG; s2++) M = max(M, sm[s2][qi]);\n"
"        float f = exp(m[qi] - M);\n"
"        for (uint d = 0; d < DPL; d++) acc[qi][d] *= f;\n"
"    }\n"
"    for (uint s2 = 0; s2 < ATTN_NSG; s2++) {\n"
"        if (sg == s2) {\n"
"            for (uint qi = 0; qi < NQ; qi++)\n"
"                for (uint d = 0; d < DPL; d++) {\n"
"                    float prev = (s2 == 0) ? 0.0f : sacc[qi][lane * DPL + d];\n"
"                    sacc[qi][lane * DPL + d] = prev + acc[qi][d];\n"
"                }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    for (uint idx = tid; idx < NQ * HD; idx += ATTN_NSG * 32) {\n"
"        uint qi = idx / HD;\n"
"        uint d = idx % HD;\n"
"        uint tq = qi / GQA;\n"
"        if (tq >= nt) continue;\n"
"        float M = sm[0][qi];\n"
"        for (uint s2 = 1; s2 < ATTN_NSG; s2++) M = max(M, sm[s2][qi]);\n"
"        float L = 0.0f;\n"
"        for (uint s2 = 0; s2 < ATTN_NSG; s2++) L += exp(sm[s2][qi] - M) * sl[s2][qi];\n"
"        uint t = t0 + tq;\n"
"        uint h = kvh * GQA + qi % GQA;\n"
"        if (a.n_splits == 1) {\n"
"            O[t * Q_SIZE + h * HD + d] = sacc[qi][d] / L;\n"
"        } else {\n"
"            device float *pp = part + ((t * NUM_HEADS + h) * a.n_splits + split) * (HD + 2);\n"
"            pp[d] = sacc[qi][d];\n"
"            if (d == 0) { pp[HD] = M; pp[HD + 1] = L; }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// Decode attention (one query token). Lanes own DPL consecutive head dims,\n"
"// so each key/value row is read with coalesced vector loads; simdgroups\n"
"// stride over key positions with an online softmax. Grid: (NUM_KV_HEADS, 1,\n"
"// splits); partials are merged by attn_reduce when splits > 1.\n"
"kernel void attention_dec(\n"
"    device const float *Q    [[buffer(0)]],\n"
"    device const KV_T  *Kc   [[buffer(1)]],\n"
"    device const KV_T  *Vc   [[buffer(2)]],\n"
"    device float       *O    [[buffer(3)]],\n"
"    device float       *part [[buffer(4)]],\n"
"    constant AttnArgs  &a    [[buffer(5)]],\n"
"    uint3 tg  [[threadgroup_position_in_grid]],\n"
"    uint tid  [[thread_index_in_threadgroup]],\n"
"    uint sg   [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]\n"
") {\n"
"    const uint kvh = tg.x;\n"
"    const uint split = tg.z;\n"
"    const uint p = a.pos0;\n"
"    const uint s_lo = attn_lo(p, a) + split * a.split_len;\n"
"    const uint s_hi = min(p + 1, s_lo + a.split_len);\n"
"\n"
"    float4 q[GQA][2];\n"
"    for (uint g = 0; g < GQA; g++) {\n"
"        device const float4 *qp = (device const float4 *)(Q + (kvh * GQA + g) * HD + lane * DPL);\n"
"        q[g][0] = qp[0] * a.scale;\n"
"        q[g][1] = qp[1] * a.scale;\n"
"    }\n"
"    float m[GQA], l[GQA];\n"
"    float4 acc[GQA][2];\n"
"    for (uint g = 0; g < GQA; g++) {\n"
"        m[g] = -1.0e30f; l[g] = 0.0f;\n"
"        acc[g][0] = float4(0.0f); acc[g][1] = float4(0.0f);\n"
"    }\n"
"    for (uint j = s_lo + sg; j < s_hi; j += ATTN_NSG) {\n"
"        uint slot = a.is_global ? j : j % a.ring;\n"
"        device const KV_T4 *kp = (device const KV_T4 *)(Kc + slot * KV_SIZE + kvh * HD + lane * DPL);\n"
"        device const KV_T4 *vp = (device const KV_T4 *)(Vc + slot * KV_SIZE + kvh * HD + lane * DPL);\n"
"        float4 k0 = float4(kp[0]), k1 = float4(kp[1]);\n"
"        float4 v0 = float4(vp[0]), v1 = float4(vp[1]);\n"
"        for (uint g = 0; g < GQA; g++) {\n"
"            float sc = simd_sum(dot(q[g][0], k0) + dot(q[g][1], k1));\n"
"            float m_new = max(m[g], sc);\n"
"            float c = exp(m[g] - m_new);\n"
"            float e = exp(sc - m_new);\n"
"            l[g] = l[g] * c + e;\n"
"            acc[g][0] = acc[g][0] * c + e * v0;\n"
"            acc[g][1] = acc[g][1] * c + e * v1;\n"
"            m[g] = m_new;\n"
"        }\n"
"    }\n"
"\n"
"    threadgroup float sm[ATTN_NSG][GQA];\n"
"    threadgroup float sl[ATTN_NSG][GQA];\n"
"    threadgroup float4 sacc[ATTN_NSG][GQA][HD / 4];\n"
"    for (uint g = 0; g < GQA; g++) {\n"
"        if (lane == 0) { sm[sg][g] = m[g]; sl[sg][g] = l[g]; }\n"
"        sacc[sg][g][lane * 2] = acc[g][0];\n"
"        sacc[sg][g][lane * 2 + 1] = acc[g][1];\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint idx = tid; idx < GQA * HD / 4; idx += ATTN_NSG * 32) {\n"
"        uint g = idx / (HD / 4);\n"
"        uint d4 = idx % (HD / 4);\n"
"        float M = sm[0][g];\n"
"        for (uint s2 = 1; s2 < ATTN_NSG; s2++) M = max(M, sm[s2][g]);\n"
"        float L = 0.0f;\n"
"        float4 o = float4(0.0f);\n"
"        for (uint s2 = 0; s2 < ATTN_NSG; s2++) {\n"
"            float f = exp(sm[s2][g] - M);\n"
"            L += f * sl[s2][g];\n"
"            o += f * sacc[s2][g][d4];\n"
"        }\n"
"        uint h = kvh * GQA + g;\n"
"        if (a.n_splits == 1) {\n"
"            ((device float4 *)(O + h * HD))[d4] = o / L;\n"
"        } else {\n"
"            device float *pp = part + (h * a.n_splits + split) * (HD + 2);\n"
"            ((device float4 *)pp)[d4] = o;\n"
"            if (d4 == 0) { pp[HD] = M; pp[HD + 1] = L; }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"#define ATTN_ARGS \\\n"
"    device const float *Q [[buffer(0)]], device const KV_T *Kc [[buffer(1)]], \\\n"
"    device const KV_T *Vc [[buffer(2)]], device float *O [[buffer(3)]], \\\n"
"    device float *part [[buffer(4)]], constant AttnArgs &a [[buffer(5)]], \\\n"
"    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \\\n"
"    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n"
"template [[host_name(\"attention_q4\")]] kernel void attention_t<4>(ATTN_ARGS);\n"
"\n"
"// Merge split partials. Grid: (NUM_HEADS, tokens), HD threads.\n"
"kernel void attn_reduce(\n"
"    device const float *part [[buffer(0)]],\n"
"    device float       *O    [[buffer(1)]],\n"
"    constant AttnArgs  &a    [[buffer(2)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]],\n"
"    uint d   [[thread_index_in_threadgroup]]\n"
") {\n"
"    const uint h = tg.x;\n"
"    const uint t = tg.y;\n"
"    device const float *base = part + (t * NUM_HEADS + h) * a.n_splits * (HD + 2);\n"
"    float M = -1e30f;\n"
"    for (uint s = 0; s < a.n_splits; s++) M = max(M, base[s * (HD + 2) + HD]);\n"
"    float L = 0.0f, o = 0.0f;\n"
"    for (uint s = 0; s < a.n_splits; s++) {\n"
"        device const float *ps = base + s * (HD + 2);\n"
"        float f = exp(ps[HD] - M);\n"
"        L += f * ps[HD + 1];\n"
"        o += f * ps[d];\n"
"    }\n"
"    O[t * Q_SIZE + h * HD + d] = o / L;\n"
"}\n";
#pragma clang diagnostic pop

/* ============================================================================
 * Metal Context
 * ========================================================================== */

/* Holds the Objective-C objects so ARC manages their lifetime. */
@interface G3MetalObjects : NSObject {
@public
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    id<MTLComputePipelineState> p_matvec;
    id<MTLComputePipelineState> p_qkv;
    id<MTLComputePipelineState> p_gateup;
    id<MTLComputePipelineState> p_gemm;
    id<MTLComputePipelineState> p_gemm_qkv;
    id<MTLComputePipelineState> p_gelu_mul;
    id<MTLComputePipelineState> p_embed_norm;
    id<MTLComputePipelineState> p_add_norm;
    id<MTLComputePipelineState> p_rope;
    id<MTLComputePipelineState> p_attn;       /* decode: one query token */
    id<MTLComputePipelineState> p_attn_q4;    /* prefill: 4 query tokens per threadgroup */
    id<MTLComputePipelineState> p_attn_reduce;

    /* Weight buffers (wrapped mmap regions, or copies) */
    id<MTLBuffer> wbuf[512];
    int n_wbuf;

    /* Activations, sized for GEMMA3_PREFILL_CHUNK tokens */
    id<MTLBuffer> X, Xn, Q, K, V, A, P, G, U, logits, part;

    /* KV cache and RoPE tables */
    id<MTLBuffer> kc[GEMMA3_NUM_LAYERS];
    id<MTLBuffer> vc[GEMMA3_NUM_LAYERS];
    id<MTLBuffer> rope_local, rope_global;
}
@end

@implementation G3MetalObjects
@end

/* A weight tensor: buffer index + byte offset */
typedef struct { int buf; NSUInteger off; } wref;

struct gemma3_metal_context {
    CFTypeRef objs;            /* G3MetalObjects, retained */
    gemma3_config config;
    int max_context;
    int ring;                  /* local-layer KV ring size */
    int kv_bytes;              /* bytes per cached K/V element */
    int use_gemm;              /* batched prefill kernels available */
    int current_pos;

    wref embed, norm;
    struct {
        wref in_ln, q, k, v, o, qn, kn, post_attn, gate, up, down, pre_ff, post_ff;
    } L[GEMMA3_NUM_LAYERS];
};

static inline G3MetalObjects *OBJ(gemma3_metal_context *ctx) {
    return (__bridge G3MetalObjects *)ctx->objs;
}

/* ============================================================================
 * Helpers
 * ========================================================================== */

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

static int metal_debug(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("GEMMA3_METAL_DEBUG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v;
}

/* ============================================================================
 * Public API: availability / init / free
 * ========================================================================== */

int gemma3_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return dev != nil;
    }
}

gemma3_metal_context *gemma3_metal_init(const gemma3_config *cfg, int max_context) {
    @autoreleasepool {
        /* The kernels are specialized for Gemma 3 4B's head layout. */
        if (cfg->head_dim != 256 || cfg->num_heads % cfg->num_kv_heads != 0 ||
            cfg->num_heads / cfg->num_kv_heads > 4 || cfg->hidden_size % 4 != 0 ||
            cfg->intermediate_size % 4 != 0) {
            fprintf(stderr, "Metal: unsupported model dimensions, using CPU\n");
            return NULL;
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return NULL;

        G3MetalObjects *o = [[G3MetalObjects alloc] init];
        o->device = dev;
        o->queue = [dev newCommandQueue];
        if (!o->queue) return NULL;

        gemma3_metal_context *ctx = calloc(1, sizeof(gemma3_metal_context));
        if (!ctx) return NULL;
        ctx->config = *cfg;
        ctx->max_context = max_context;
        ctx->ring = gemma3_local_ring_size(cfg->sliding_window);
        if (ctx->ring > max_context) ctx->ring = max_context;

        /* KV cache precision: f32 by default, so results match the CPU path.
         * GEMMA3_METAL_KV=f16 halves the cache memory and attention bandwidth
         * (logits move by ~1e-2, which can flip near-tied greedy choices). */
        const char *kv_env = getenv("GEMMA3_METAL_KV");
        int kv_half = kv_env && (strcmp(kv_env, "f16") == 0 || strcmp(kv_env, "half") == 0);
        ctx->kv_bytes = kv_half ? 2 : 4;

        int q_size  = cfg->num_heads * cfg->head_dim;
        int kv_size = cfg->num_kv_heads * cfg->head_dim;

        /* Compile MSL with model dimensions baked in */
        char prelude[1024];
        snprintf(prelude, sizeof(prelude),
                 "#define HIDDEN %d\n#define HD %d\n#define NUM_HEADS %d\n"
                 "#define NUM_KV_HEADS %d\n#define GQA %d\n#define Q_SIZE %d\n"
                 "#define KV_SIZE %d\n#define KV_T %s\n#define KV_T4 %s4\n#define MV_NSG %d\n"
                 "#define MV_R %d\n#define GU_R %d\n#define NORM_TG %d\n"
                 "#define ATTN_NSG %d\n",
                 cfg->hidden_size, cfg->head_dim, cfg->num_heads, cfg->num_kv_heads,
                 cfg->num_heads / cfg->num_kv_heads, q_size, kv_size,
                 kv_half ? "half" : "float", kv_half ? "half" : "float",
                 MV_NSG, MV_R, GU_R, NORM_TG, ATTN_NSG);
        NSString *src = [NSString stringWithFormat:@"%s%s", prelude, metalShaderSource];

        NSError *err = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        o->library = [dev newLibraryWithSource:src options:opts error:&err];
        if (!o->library) {
            fprintf(stderr, "Metal: shader compile failed: %s\n",
                    err.localizedDescription.UTF8String);
            free(ctx);
            return NULL;
        }

        o->p_matvec      = make_pso(dev, o->library, "matvec_bf16");
        o->p_qkv         = make_pso(dev, o->library, "matvec_qkv");
        o->p_gateup      = make_pso(dev, o->library, "matvec_gateup");
        o->p_embed_norm  = make_pso(dev, o->library, "embed_norm");
        o->p_add_norm    = make_pso(dev, o->library, "add_norm");
        o->p_rope        = make_pso(dev, o->library, "qk_norm_rope_cache");
        o->p_attn        = make_pso(dev, o->library, "attention_dec");
        o->p_attn_q4     = make_pso(dev, o->library, "attention_q4");
        o->p_attn_reduce = make_pso(dev, o->library, "attn_reduce");
        if (!o->p_matvec || !o->p_qkv || !o->p_gateup || !o->p_embed_norm ||
            !o->p_add_norm || !o->p_rope || !o->p_attn || !o->p_attn_q4 ||
            !o->p_attn_reduce) {
            free(ctx);
            return NULL;
        }

        /* Batched prefill needs simdgroup matrices and 64-row-aligned weights;
         * without them prefill falls back to one token at a time. */
        o->p_gemm     = make_pso(dev, o->library, "gemm_bf16");
        o->p_gemm_qkv = make_pso(dev, o->library, "gemm_qkv");
        o->p_gelu_mul = make_pso(dev, o->library, "gelu_mul");
        ctx->use_gemm = o->p_gemm && o->p_gemm_qkv && o->p_gelu_mul &&
                        o->p_gemm.maxTotalThreadsPerThreadgroup >= 128 &&
                        q_size % GEMM_BM == 0 && kv_size % GEMM_BM == 0 &&
                        cfg->hidden_size % GEMM_BM == 0 &&
                        cfg->intermediate_size % GEMM_BM == 0 &&
                        cfg->hidden_size % 32 == 0 && q_size % 32 == 0 &&
                        cfg->intermediate_size % 32 == 0;
        if (getenv("GEMMA3_METAL_NO_GEMM")) ctx->use_gemm = 0;

        /* Activations for a full prefill chunk */
        size_t T = GEMMA3_PREFILL_CHUNK;
        int hs = cfg->hidden_size, is = cfg->intermediate_size;
        MTLResourceOptions shared = MTLResourceStorageModeShared;
#define ALLOC(name, count) \
        o->name = [dev newBufferWithLength:(size_t)(count) * sizeof(float) options:shared]; \
        if (!o->name) { free(ctx); return NULL; } \
        memset(o->name.contents, 0, o->name.length);
        ALLOC(X,  T * hs)
        ALLOC(Xn, T * hs)
        ALLOC(Q,  T * q_size)
        ALLOC(K,  T * kv_size)
        ALLOC(V,  T * kv_size)
        ALLOC(A,  T * q_size)
        ALLOC(P,  T * hs)
        ALLOC(G,  T * is)
        ALLOC(U,  T * is)
        ALLOC(logits, cfg->vocab_size)
        ALLOC(part, (size_t)cfg->num_heads * ATTN_MAX_SPLITS * (cfg->head_dim + 2))
#undef ALLOC

        /* KV cache: global layers hold the full context, local layers a ring */
        size_t kv_total = 0;
        for (int l = 0; l < cfg->num_layers; l++) {
            int slots = gemma3_is_global_layer(l) ? max_context : ctx->ring;
            size_t bytes = (size_t)slots * kv_size * ctx->kv_bytes;
            o->kc[l] = [dev newBufferWithLength:bytes options:shared];
            o->vc[l] = [dev newBufferWithLength:bytes options:shared];
            if (!o->kc[l] || !o->vc[l]) { free(ctx); return NULL; }
            kv_total += 2 * bytes;
        }
        if (metal_debug()) {
            fprintf(stderr, "Metal: %s, KV cache %.0f MB (%s), batched prefill %s\n",
                    dev.name.UTF8String, kv_total / 1e6, kv_half ? "f16" : "f32",
                    ctx->use_gemm ? "on" : "off");
        }

        ctx->objs = CFBridgingRetain(o);
        ctx->current_pos = 0;
        return ctx;
    }
}

void gemma3_metal_free(gemma3_metal_context *ctx) {
    if (!ctx) return;
    if (ctx->objs) CFRelease(ctx->objs);
    free(ctx);
}

/* ============================================================================
 * Weight and RoPE upload
 * ========================================================================== */

typedef struct {
    const uint8_t *ptr;
    size_t size;
    wref *ref;
} tensor_desc;

static int cmp_tensor_desc(const void *a, const void *b) {
    const uint8_t *pa = ((const tensor_desc *)a)->ptr;
    const uint8_t *pb = ((const tensor_desc *)b)->ptr;
    return (pa > pb) - (pa < pb);
}

static int add_wbuf(G3MetalObjects *o, id<MTLBuffer> buf) {
    if (!buf || o->n_wbuf >= (int)(sizeof(o->wbuf) / sizeof(o->wbuf[0]))) return -1;
    o->wbuf[o->n_wbuf] = buf;
    return o->n_wbuf++;
}

/* Copy one tensor into its own buffer (used only if wrapping is impossible). */
static int copy_tensor(G3MetalObjects *o, tensor_desc *t) {
    id<MTLBuffer> b = [o->device newBufferWithBytes:t->ptr length:t->size
                                            options:MTLResourceStorageModeShared];
    int idx = add_wbuf(o, b);
    if (idx < 0) return -1;
    t->ref->buf = idx;
    t->ref->off = 0;
    return 0;
}

int gemma3_metal_upload_weights(gemma3_metal_context *ctx, const void *weights_ptr) {
    @autoreleasepool {
    const gemma3_weights_t *w = (const gemma3_weights_t *)weights_ptr;
    G3MetalObjects *o = OBJ(ctx);
    const gemma3_config *cfg = &ctx->config;

    size_t hs = cfg->hidden_size, is = cfg->intermediate_size, hd = cfg->head_dim;
    size_t q_size = (size_t)cfg->num_heads * hd;
    size_t kv_size = (size_t)cfg->num_kv_heads * hd;
    const size_t B = sizeof(uint16_t);

    int max_t = 2 + 13 * cfg->num_layers;
    tensor_desc *td = calloc(max_t, sizeof(tensor_desc));
    if (!td) return -1;
    int nt = 0;
#define ADD(p, nelem, r) do { \
        if (!(p)) { free(td); return -1; } \
        td[nt].ptr = (const uint8_t *)(p); td[nt].size = (nelem) * B; td[nt].ref = (r); nt++; \
    } while (0)
    ADD(w->embed_tokens, (size_t)cfg->vocab_size * hs, &ctx->embed);
    ADD(w->norm, hs, &ctx->norm);
    for (int l = 0; l < cfg->num_layers; l++) {
        ADD(w->layers[l].input_layernorm,            hs,           &ctx->L[l].in_ln);
        ADD(w->layers[l].q_proj,                     q_size * hs,  &ctx->L[l].q);
        ADD(w->layers[l].k_proj,                     kv_size * hs, &ctx->L[l].k);
        ADD(w->layers[l].v_proj,                     kv_size * hs, &ctx->L[l].v);
        ADD(w->layers[l].o_proj,                     hs * q_size,  &ctx->L[l].o);
        ADD(w->layers[l].q_norm,                     hd,           &ctx->L[l].qn);
        ADD(w->layers[l].k_norm,                     hd,           &ctx->L[l].kn);
        ADD(w->layers[l].post_attention_layernorm,   hs,           &ctx->L[l].post_attn);
        ADD(w->layers[l].gate_proj,                  is * hs,      &ctx->L[l].gate);
        ADD(w->layers[l].up_proj,                    is * hs,      &ctx->L[l].up);
        ADD(w->layers[l].down_proj,                  hs * is,      &ctx->L[l].down);
        ADD(w->layers[l].pre_feedforward_layernorm,  hs,           &ctx->L[l].pre_ff);
        ADD(w->layers[l].post_feedforward_layernorm, hs,           &ctx->L[l].post_ff);
    }
#undef ADD
    qsort(td, nt, sizeof(tensor_desc), cmp_tensor_desc);

    const size_t page = (size_t)getpagesize();
    const size_t max_len = (size_t)o->device.maxBufferLength;
    size_t wrapped = 0, copied = 0;
    int *done = calloc(nt, sizeof(int));
    if (!done) { free(td); return -1; }

    /* Wrap the page-aligned span of each mapped file that covers the tensors
     * we use (the vision tower and other unused tensors stay unwired).
     * GEMMA3_METAL_COPY_WEIGHTS=1 forces private copies (debugging only). */
    int force_copy = getenv("GEMMA3_METAL_COPY_WEIGHTS") != NULL;
    for (int r = 0; r < w->num_regions && !force_copy; r++) {
        const uint8_t *rbase = (const uint8_t *)w->regions[r].base;
        const uint8_t *rend = rbase + w->regions[r].size;
        int i = 0;
        while (i < nt) {
            if (done[i] || td[i].ptr < rbase || td[i].ptr + td[i].size > rend ||
                ((uintptr_t)td[i].ptr & 7)) {
                i++;
                continue;
            }
            /* Greedily extend a buffer while it stays under maxBufferLength */
            const uint8_t *start = (const uint8_t *)((uintptr_t)td[i].ptr & ~(uintptr_t)(page - 1));
            const uint8_t *end = td[i].ptr + td[i].size;
            int j = i + 1;
            while (j < nt) {
                if (td[j].ptr < rbase || td[j].ptr + td[j].size > rend) break;
                const uint8_t *e2 = td[j].ptr + td[j].size;
                size_t len = ((uintptr_t)(e2 > end ? e2 : end) - (uintptr_t)start + page - 1) & ~(page - 1);
                if (len > max_len) break;
                if (e2 > end) end = e2;
                j++;
            }
            size_t len = ((uintptr_t)end - (uintptr_t)start + page - 1) & ~(page - 1);
            id<MTLBuffer> buf = nil;
            if (len <= max_len) {
                buf = [o->device newBufferWithBytesNoCopy:(void *)start
                                                   length:len
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
            }
            int idx = buf ? add_wbuf(o, buf) : -1;
            if (idx >= 0) {
                for (int k = i; k < j; k++) {
                    if ((uintptr_t)td[k].ptr & 7) continue;   /* misaligned: copy below */
                    td[k].ref->buf = idx;
                    td[k].ref->off = (NSUInteger)(td[k].ptr - start);
                    done[k] = 1;
                }
                wrapped += len;
                if (metal_debug()) {
                    fprintf(stderr, "Metal: wrapped region %d [%zu MB] for %d tensors (no copy)\n",
                            r, len >> 20, j - i);
                }
            }
            i = j;
        }
    }

    /* Anything not wrapped (unaligned, outside a mapped region, or wrapping
     * failed) is copied into its own buffer. */
    for (int i = 0; i < nt; i++) {
        if (done[i]) continue;
        if (copy_tensor(o, &td[i]) != 0) { free(done); free(td); return -1; }
        copied += td[i].size;
    }
    if (copied > 0 && metal_debug()) {
        fprintf(stderr, "Metal: copied %.1f MB of weights\n", copied / 1e6);
    }
    (void)wrapped;
    free(done);
    free(td);
    return 0;
    }
}

int gemma3_metal_upload_rope(gemma3_metal_context *ctx,
                              const float *rope_local, const float *rope_global,
                              int max_context, int head_dim) {
    G3MetalObjects *o = OBJ(ctx);
    size_t bytes = (size_t)max_context * (head_dim / 2) * 2 * sizeof(float);
    o->rope_local = [o->device newBufferWithBytes:rope_local length:bytes
                                          options:MTLResourceStorageModeShared];
    o->rope_global = [o->device newBufferWithBytes:rope_global length:bytes
                                           options:MTLResourceStorageModeShared];
    return (o->rope_local && o->rope_global) ? 0 : -1;
}

/* ============================================================================
 * Encoding helpers
 * ========================================================================== */

static inline void set_w(G3MetalObjects *o, id<MTLComputeCommandEncoder> enc,
                         wref r, NSUInteger index) {
    [enc setBuffer:o->wbuf[r.buf] offset:r.off atIndex:index];
}

static void enc_matvec(G3MetalObjects *o, id<MTLComputeCommandEncoder> enc, wref W,
                       id<MTLBuffer> x, NSUInteger x_off, id<MTLBuffer> y,
                       uint32_t M, uint32_t K) {
    MetalMatvecArgs a = { M, K };
    [enc setComputePipelineState:o->p_matvec];
    set_w(o, enc, W, 0);
    [enc setBuffer:x offset:x_off atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&a length:sizeof(a) atIndex:3];
    NSUInteger rows = MV_NSG * MV_R;
    [enc dispatchThreadgroups:MTLSizeMake((M + rows - 1) / rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MV_NSG * 32, 1, 1)];
}

static void enc_gemm(G3MetalObjects *o, id<MTLComputeCommandEncoder> enc, wref W,
                     id<MTLBuffer> X, id<MTLBuffer> Y, uint32_t M, uint32_t K, uint32_t n_pad) {
    MetalGemmArgs a = { M, K };
    [enc setComputePipelineState:o->p_gemm];
    set_w(o, enc, W, 0);
    [enc setBuffer:X offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&a length:sizeof(a) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(M / GEMM_BM, n_pad / GEMM_BN, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

/* Encode the transformer for n tokens at positions pos0..pos0+n-1.
 * n == 1 uses the matrix-vector kernels; n > 1 uses GEMM.
 * On return Xn holds the final-normed hidden states of all n tokens. */
static void encode_forward(gemma3_metal_context *ctx, id<MTLComputeCommandEncoder> enc,
                           const int *tokens, int n, int pos0) {
    G3MetalObjects *o = OBJ(ctx);
    const gemma3_config *cfg = &ctx->config;
    const uint32_t hs = cfg->hidden_size, is = cfg->intermediate_size;
    const uint32_t q_size = cfg->num_heads * cfg->head_dim;
    const uint32_t kv_size = cfg->num_kv_heads * cfg->head_dim;
    const uint32_t n_pad = (uint32_t)((n + GEMM_BN - 1) / GEMM_BN * GEMM_BN);
    const int batched = n > 1;

    /* Embedding * sqrt(hidden) and the first layer's input norm */
    {
        MetalNormArgs na = { cfg->rmsnorm_eps, sqrtf((float)hs) };
        [enc setComputePipelineState:o->p_embed_norm];
        set_w(o, enc, ctx->embed, 0);
        [enc setBuffer:o->X offset:0 atIndex:1];
        [enc setBuffer:o->Xn offset:0 atIndex:2];
        set_w(o, enc, ctx->L[0].in_ln, 3);
        [enc setBytes:&na length:sizeof(na) atIndex:4];
        [enc setBytes:tokens length:(NSUInteger)n * sizeof(int) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(NORM_TG, 1, 1)];
    }

    for (int l = 0; l < cfg->num_layers; l++) {
        const int is_global = gemma3_is_global_layer(l);

        /* -- Q, K, V projections -- */
        if (batched) {
            MetalQKVArgs qa = { q_size, kv_size, hs };
            [enc setComputePipelineState:o->p_gemm_qkv];
            set_w(o, enc, ctx->L[l].q, 0);
            set_w(o, enc, ctx->L[l].k, 1);
            set_w(o, enc, ctx->L[l].v, 2);
            [enc setBuffer:o->Xn offset:0 atIndex:3];
            [enc setBuffer:o->Q offset:0 atIndex:4];
            [enc setBuffer:o->K offset:0 atIndex:5];
            [enc setBuffer:o->V offset:0 atIndex:6];
            [enc setBytes:&qa length:sizeof(qa) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((q_size + 2 * kv_size) / GEMM_BM, n_pad / GEMM_BN, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            MetalQKVArgs qa = { q_size, kv_size, hs };
            [enc setComputePipelineState:o->p_qkv];
            set_w(o, enc, ctx->L[l].q, 0);
            set_w(o, enc, ctx->L[l].k, 1);
            set_w(o, enc, ctx->L[l].v, 2);
            [enc setBuffer:o->Xn offset:0 atIndex:3];
            [enc setBuffer:o->Q offset:0 atIndex:4];
            [enc setBuffer:o->K offset:0 atIndex:5];
            [enc setBuffer:o->V offset:0 atIndex:6];
            [enc setBytes:&qa length:sizeof(qa) atIndex:7];
            NSUInteger rows = MV_NSG * MV_R;
            NSUInteger total = q_size + 2 * kv_size;
            [enc dispatchThreadgroups:MTLSizeMake((total + rows - 1) / rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(MV_NSG * 32, 1, 1)];
        }

        /* -- QK-norm + RoPE + KV-cache write -- */
        {
            MetalRopeArgs ra = { (uint32_t)pos0, (uint32_t)is_global,
                                 (uint32_t)ctx->ring, cfg->rmsnorm_eps };
            [enc setComputePipelineState:o->p_rope];
            [enc setBuffer:o->Q offset:0 atIndex:0];
            [enc setBuffer:o->K offset:0 atIndex:1];
            [enc setBuffer:o->V offset:0 atIndex:2];
            [enc setBuffer:o->kc[l] offset:0 atIndex:3];
            [enc setBuffer:o->vc[l] offset:0 atIndex:4];
            set_w(o, enc, ctx->L[l].qn, 5);
            set_w(o, enc, ctx->L[l].kn, 6);
            [enc setBuffer:(is_global ? o->rope_global : o->rope_local) offset:0 atIndex:7];
            [enc setBytes:&ra length:sizeof(ra) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(cfg->num_heads + cfg->num_kv_heads, n, 1)
                threadsPerThreadgroup:MTLSizeMake(cfg->head_dim / 2, 1, 1)];
        }

        /* -- Attention -- */
        {
            int window = cfg->sliding_window;
            uint32_t n_splits = 1, split_len = 0x7fffffff;
            if (!batched) {
                int p = pos0;
                int lo = (is_global || p + 1 <= window) ? 0 : p + 1 - window;
                int n_keys = p - lo + 1;
                split_len = ATTN_SPLIT_LEN;
                n_splits = (uint32_t)((n_keys + split_len - 1) / split_len);
                if (n_splits > ATTN_MAX_SPLITS) {
                    split_len = (uint32_t)((n_keys + ATTN_MAX_SPLITS - 1) / ATTN_MAX_SPLITS);
                    n_splits = (uint32_t)((n_keys + split_len - 1) / split_len);
                }
            }
            MetalAttnArgs aa = { (uint32_t)pos0, (uint32_t)is_global, (uint32_t)window,
                                 (uint32_t)ctx->ring, n_splits, split_len,
                                 1.0f / sqrtf((float)cfg->head_dim), (uint32_t)n };
            const int qb = batched ? 4 : 1;
            [enc setComputePipelineState:(batched ? o->p_attn_q4 : o->p_attn)];
            [enc setBuffer:o->Q offset:0 atIndex:0];
            [enc setBuffer:o->kc[l] offset:0 atIndex:1];
            [enc setBuffer:o->vc[l] offset:0 atIndex:2];
            [enc setBuffer:o->A offset:0 atIndex:3];
            [enc setBuffer:o->part offset:0 atIndex:4];
            [enc setBytes:&aa length:sizeof(aa) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(cfg->num_kv_heads, (n + qb - 1) / qb, n_splits)
                threadsPerThreadgroup:MTLSizeMake(ATTN_NSG * 32, 1, 1)];
            if (n_splits > 1) {
                [enc setComputePipelineState:o->p_attn_reduce];
                [enc setBuffer:o->part offset:0 atIndex:0];
                [enc setBuffer:o->A offset:0 atIndex:1];
                [enc setBytes:&aa length:sizeof(aa) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(cfg->num_heads, n, 1)
                    threadsPerThreadgroup:MTLSizeMake(cfg->head_dim, 1, 1)];
            }
        }

        /* -- Output projection -- */
        if (batched) enc_gemm(o, enc, ctx->L[l].o, o->A, o->P, hs, q_size, n_pad);
        else         enc_matvec(o, enc, ctx->L[l].o, o->A, 0, o->P, hs, q_size);

        /* -- x += post_attn_norm(attn); xn = pre_ff_norm(x) -- */
        {
            MetalNormArgs na = { cfg->rmsnorm_eps, 1.0f };
            [enc setComputePipelineState:o->p_add_norm];
            [enc setBuffer:o->P offset:0 atIndex:0];
            [enc setBuffer:o->X offset:0 atIndex:1];
            [enc setBuffer:o->Xn offset:0 atIndex:2];
            set_w(o, enc, ctx->L[l].post_attn, 3);
            set_w(o, enc, ctx->L[l].pre_ff, 4);
            [enc setBytes:&na length:sizeof(na) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(NORM_TG, 1, 1)];
        }

        /* -- MLP: gelu(gate) * up, then down -- */
        if (batched) {
            enc_gemm(o, enc, ctx->L[l].gate, o->Xn, o->G, is, hs, n_pad);
            enc_gemm(o, enc, ctx->L[l].up, o->Xn, o->U, is, hs, n_pad);
            MetalCountArgs ca = { (uint32_t)n * is };
            [enc setComputePipelineState:o->p_gelu_mul];
            [enc setBuffer:o->G offset:0 atIndex:0];
            [enc setBuffer:o->U offset:0 atIndex:1];
            [enc setBytes:&ca length:sizeof(ca) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(ca.n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            enc_gemm(o, enc, ctx->L[l].down, o->G, o->P, hs, is, n_pad);
        } else {
            MetalMatvecArgs ma = { is, hs };
            [enc setComputePipelineState:o->p_gateup];
            set_w(o, enc, ctx->L[l].gate, 0);
            set_w(o, enc, ctx->L[l].up, 1);
            [enc setBuffer:o->Xn offset:0 atIndex:2];
            [enc setBuffer:o->G offset:0 atIndex:3];
            [enc setBytes:&ma length:sizeof(ma) atIndex:4];
            NSUInteger rows = MV_NSG * GU_R;
            [enc dispatchThreadgroups:MTLSizeMake((is + rows - 1) / rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(MV_NSG * 32, 1, 1)];
            enc_matvec(o, enc, ctx->L[l].down, o->G, 0, o->P, hs, is);
        }

        /* -- x += post_ff_norm(mlp); xn = next layer's input norm (or final norm) -- */
        {
            MetalNormArgs na = { cfg->rmsnorm_eps, 1.0f };
            wref next = (l + 1 < cfg->num_layers) ? ctx->L[l + 1].in_ln : ctx->norm;
            [enc setComputePipelineState:o->p_add_norm];
            [enc setBuffer:o->P offset:0 atIndex:0];
            [enc setBuffer:o->X offset:0 atIndex:1];
            [enc setBuffer:o->Xn offset:0 atIndex:2];
            set_w(o, enc, ctx->L[l].post_ff, 3);
            set_w(o, enc, next, 4);
            [enc setBytes:&na length:sizeof(na) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(NORM_TG, 1, 1)];
        }
    }
}

/* Logits for token row `row` of Xn (tied embeddings). */
static void encode_logits(gemma3_metal_context *ctx, id<MTLComputeCommandEncoder> enc, int row) {
    G3MetalObjects *o = OBJ(ctx);
    const gemma3_config *cfg = &ctx->config;
    enc_matvec(o, enc, ctx->embed, o->Xn,
               (NSUInteger)row * cfg->hidden_size * sizeof(float),
               o->logits, (uint32_t)cfg->vocab_size, (uint32_t)cfg->hidden_size);
}

static int metal_profile(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("GEMMA3_METAL_PROFILE");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v;
}

static void report_gpu_time(id<MTLCommandBuffer> cb, const char *what, int n) {
    if (!metal_profile()) return;
    double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    fprintf(stderr, "Metal: %s n=%d gpu %.2f ms\n", what, n, ms);
}

static int finish(gemma3_metal_context *ctx, id<MTLCommandBuffer> cb, float *logits) {
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "Metal: GPU error: %s\n", cb.error.localizedDescription.UTF8String);
        return -1;
    }
    if (logits) {
        memcpy(logits, OBJ(ctx)->logits.contents,
               (size_t)ctx->config.vocab_size * sizeof(float));
    }
    return 0;
}

/* ============================================================================
 * Forward pass
 * ========================================================================== */

int gemma3_metal_forward_token(gemma3_metal_context *ctx, int token_id, int pos,
                                float *logits, int compute_logits) {
    @autoreleasepool {
        if (pos < 0 || pos >= ctx->max_context) return -1;
        G3MetalObjects *o = OBJ(ctx);
        int want_logits = compute_logits && logits;
        CFAbsoluteTime t0 = metal_profile() ? CFAbsoluteTimeGetCurrent() : 0;
        id<MTLCommandBuffer> cb = [o->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode_forward(ctx, enc, &token_id, 1, pos);
        if (want_logits) encode_logits(ctx, enc, 0);
        [enc endEncoding];
        [cb commit];
        CFAbsoluteTime t1 = metal_profile() ? CFAbsoluteTimeGetCurrent() : 0;
        if (finish(ctx, cb, want_logits ? logits : NULL) != 0) return -1;
        if (metal_profile()) {
            fprintf(stderr, "Metal: decode encode %.2f ms, call total %.2f ms\n",
                    (t1 - t0) * 1e3, (CFAbsoluteTimeGetCurrent() - t0) * 1e3);
        }
        report_gpu_time(cb, "decode", 1);
        ctx->current_pos = pos + 1;
        return 0;
    }
}

/* ============================================================================
 * Prefill and cache reset
 * ========================================================================== */

int gemma3_metal_prefill(gemma3_metal_context *ctx, const int *tokens, int num_tokens,
                          int start_pos, float *logits) {
    if (num_tokens <= 0) return 0;
    if (start_pos < 0 || start_pos + num_tokens > ctx->max_context) return -1;

    if (!ctx->use_gemm) {
        /* Fallback: one token at a time */
        for (int i = 0; i < num_tokens; i++) {
            int is_last = (i == num_tokens - 1);
            int ret = gemma3_metal_forward_token(ctx, tokens[i], start_pos + i,
                                                 logits, is_last);
            if (ret != 0) return ret;
        }
        ctx->current_pos = start_pos + num_tokens;
        return 0;
    }

    @autoreleasepool {
        G3MetalObjects *o = OBJ(ctx);
        id<MTLCommandBuffer> last = nil;
        NSMutableArray *cbs = metal_profile() ? [NSMutableArray array] : nil;
        /* One command buffer per chunk. Command buffers on a queue execute in
         * order, so only the last one is waited on. */
        for (int done = 0; done < num_tokens; ) {
            int n = num_tokens - done;
            if (n > GEMMA3_PREFILL_CHUNK) n = GEMMA3_PREFILL_CHUNK;
            int is_last = (done + n == num_tokens);
            id<MTLCommandBuffer> cb = [o->queue commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode_forward(ctx, enc, tokens + done, n, start_pos + done);
            if (is_last && logits) encode_logits(ctx, enc, n - 1);
            [enc endEncoding];
            [cb commit];
            [cbs addObject:cb];
            last = cb;
            done += n;
        }
        if (finish(ctx, last, logits) != 0) return -1;
        for (id<MTLCommandBuffer> c in cbs) report_gpu_time(c, "prefill chunk", num_tokens);
        ctx->current_pos = start_pos + num_tokens;
        return 0;
    }
}

void gemma3_metal_reset_cache(gemma3_metal_context *ctx) {
    /* Attention only reads positions that have been written for the current
     * sequence, so resetting is just rewinding the position. */
    if (ctx) ctx->current_pos = 0;
}

#endif /* USE_MPS */
