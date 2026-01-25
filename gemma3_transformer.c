/*
 * gemma3_transformer.c - Transformer forward pass implementation
 *
 * Implements the Gemma 3 transformer architecture:
 * - Grouped Query Attention (GQA) with 8 Q heads and 4 KV heads
 * - Hybrid local/global attention (5:1 ratio)
 * - SwiGLU MLP
 * - RMSNorm with additional pre/post feedforward norms
 * - RoPE with layer-specific theta
 */

#include "gemma3.h"
#include "gemma3_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Internal Structures (shared with gemma3.c)
 * ========================================================================== */

/* Forward declarations from safetensors - BF16 weights */
typedef struct {
    const uint16_t *embed_tokens;
    struct {
        const uint16_t *input_layernorm;
        const uint16_t *q_proj;
        const uint16_t *k_proj;
        const uint16_t *v_proj;
        const uint16_t *o_proj;
        const uint16_t *q_norm;  /* QK normalization */
        const uint16_t *k_norm;  /* QK normalization */
        const uint16_t *post_attention_layernorm;
        const uint16_t *gate_proj;
        const uint16_t *up_proj;
        const uint16_t *down_proj;
        const uint16_t *pre_feedforward_layernorm;
        const uint16_t *post_feedforward_layernorm;
    } layers[GEMMA3_NUM_LAYERS];
    const uint16_t *norm;
} gemma3_weights_t;

/* KV Cache for a single layer */
typedef struct {
    float *k;  /* [max_seq, num_kv_heads, head_dim] */
    float *v;  /* [max_seq, num_kv_heads, head_dim] */
    int pos;   /* Current position (for ring buffer on local layers) */
} layer_kv_cache;

/* Full KV cache */
struct gemma3_kv_cache {
    layer_kv_cache layers[GEMMA3_NUM_LAYERS];
    int max_seq;
    int current_pos;  /* Global sequence position */
};

/* ============================================================================
 * Activation Buffers
 * ========================================================================== */

typedef struct {
    float *x;           /* [hidden_size] - current hidden state */
    float *x_norm;      /* [hidden_size] - normalized hidden state */
    float *q;           /* [num_heads * head_dim] - query */
    float *k;           /* [num_kv_heads * head_dim] - key */
    float *v;           /* [num_kv_heads * head_dim] - value */
    float *attn_out;    /* [num_heads * head_dim] - attention output */
    float *proj_out;    /* [hidden_size] - projection output */
    float *mlp_gate;    /* [intermediate_size] - MLP gate */
    float *mlp_up;      /* [intermediate_size] - MLP up */
    float *mlp_out;     /* [hidden_size] - MLP output */
    float *logits;      /* [vocab_size] - output logits */
    float *mask;        /* [max_seq] - attention mask */
} activation_buffers;

static activation_buffers *alloc_buffers(const gemma3_config *cfg) {
    activation_buffers *buf = (activation_buffers *)calloc(1, sizeof(activation_buffers));
    if (!buf) return NULL;

    buf->x = (float *)malloc(cfg->hidden_size * sizeof(float));
    buf->x_norm = (float *)malloc(cfg->hidden_size * sizeof(float));
    buf->q = (float *)malloc(cfg->num_heads * cfg->head_dim * sizeof(float));
    buf->k = (float *)malloc(cfg->num_kv_heads * cfg->head_dim * sizeof(float));
    buf->v = (float *)malloc(cfg->num_kv_heads * cfg->head_dim * sizeof(float));
    buf->attn_out = (float *)malloc(cfg->num_heads * cfg->head_dim * sizeof(float));
    buf->proj_out = (float *)malloc(cfg->hidden_size * sizeof(float));
    buf->mlp_gate = (float *)malloc(cfg->intermediate_size * sizeof(float));
    buf->mlp_up = (float *)malloc(cfg->intermediate_size * sizeof(float));
    buf->mlp_out = (float *)malloc(cfg->hidden_size * sizeof(float));
    buf->logits = (float *)malloc(cfg->vocab_size * sizeof(float));
    buf->mask = (float *)malloc(cfg->max_context * sizeof(float));

    if (!buf->x || !buf->x_norm || !buf->q || !buf->k || !buf->v ||
        !buf->attn_out || !buf->proj_out || !buf->mlp_gate || !buf->mlp_up ||
        !buf->mlp_out || !buf->logits || !buf->mask) {
        free(buf->x);
        free(buf->x_norm);
        free(buf->q);
        free(buf->k);
        free(buf->v);
        free(buf->attn_out);
        free(buf->proj_out);
        free(buf->mlp_gate);
        free(buf->mlp_up);
        free(buf->mlp_out);
        free(buf->logits);
        free(buf->mask);
        free(buf);
        return NULL;
    }

    return buf;
}

static void free_buffers(activation_buffers *buf) {
    if (!buf) return;
    free(buf->x);
    free(buf->x_norm);
    free(buf->q);
    free(buf->k);
    free(buf->v);
    free(buf->attn_out);
    free(buf->proj_out);
    free(buf->mlp_gate);
    free(buf->mlp_up);
    free(buf->mlp_out);
    free(buf->logits);
    free(buf->mask);
    free(buf);
}

/* ============================================================================
 * KV Cache Management
 * ========================================================================== */

gemma3_kv_cache *gemma3_kv_cache_alloc(const gemma3_config *cfg, int max_seq) {
    gemma3_kv_cache *cache = (gemma3_kv_cache *)calloc(1, sizeof(gemma3_kv_cache));
    if (!cache) return NULL;

    cache->max_seq = max_seq;
    cache->current_pos = 0;

    int kv_size = cfg->num_kv_heads * cfg->head_dim;

    for (int l = 0; l < cfg->num_layers; l++) {
        /* For local layers with sliding window, we only need window_size entries */
        int layer_max_seq;
        if (gemma3_is_global_layer(l)) {
            layer_max_seq = max_seq;  /* Global: full context */
        } else {
            layer_max_seq = cfg->sliding_window;  /* Local: ring buffer */
        }

        cache->layers[l].k = (float *)calloc(layer_max_seq * kv_size, sizeof(float));
        cache->layers[l].v = (float *)calloc(layer_max_seq * kv_size, sizeof(float));
        cache->layers[l].pos = 0;

        if (!cache->layers[l].k || !cache->layers[l].v) {
            /* Cleanup on failure */
            for (int j = 0; j <= l; j++) {
                free(cache->layers[j].k);
                free(cache->layers[j].v);
            }
            free(cache);
            return NULL;
        }
    }

    return cache;
}

void gemma3_kv_cache_free(gemma3_kv_cache *cache) {
    if (!cache) return;

    for (int l = 0; l < GEMMA3_NUM_LAYERS; l++) {
        free(cache->layers[l].k);
        free(cache->layers[l].v);
    }
    free(cache);
}

void gemma3_kv_cache_reset(gemma3_kv_cache *cache) {
    if (!cache) return;

    cache->current_pos = 0;
    for (int l = 0; l < GEMMA3_NUM_LAYERS; l++) {
        cache->layers[l].pos = 0;
    }
}

/* Add KV to cache for a layer */
static void cache_kv(layer_kv_cache *cache, const float *k, const float *v,
                     int kv_size, int is_global, int sliding_window, int pos) {
    int cache_pos;

    if (is_global) {
        /* Global layer: simple append */
        cache_pos = pos;
    } else {
        /* Local layer: ring buffer */
        cache_pos = pos % sliding_window;
    }

    memcpy(cache->k + cache_pos * kv_size, k, kv_size * sizeof(float));
    memcpy(cache->v + cache_pos * kv_size, v, kv_size * sizeof(float));
    cache->pos = pos + 1;
}

/* ============================================================================
 * Attention Implementation
 * ========================================================================== */

/* Compute attention for a single layer */
static void layer_attention(
    float *output,           /* [hidden_size] */
    const float *x,          /* [hidden_size] - input */
    const uint16_t *q_weight,   /* [num_heads * head_dim, hidden_size] BF16 */
    const uint16_t *k_weight,   /* [num_kv_heads * head_dim, hidden_size] BF16 */
    const uint16_t *v_weight,   /* [num_kv_heads * head_dim, hidden_size] BF16 */
    const uint16_t *o_weight,   /* [hidden_size, num_heads * head_dim] BF16 */
    const uint16_t *q_norm,     /* [head_dim] BF16 - QK normalization */
    const uint16_t *k_norm,     /* [head_dim] BF16 - QK normalization */
    layer_kv_cache *cache,
    float *q_buf,            /* [num_heads * head_dim] */
    float *k_buf,            /* [num_kv_heads * head_dim] */
    float *v_buf,            /* [num_kv_heads * head_dim] */
    float *attn_buf,         /* [num_heads * head_dim] */
    float *mask_buf,         /* [max_seq] */
    const gemma3_config *cfg,
    int layer_idx,
    int pos
) {
    int num_heads = cfg->num_heads;
    int num_kv_heads = cfg->num_kv_heads;
    int head_dim = cfg->head_dim;
    int hidden_size = cfg->hidden_size;

    int q_size = num_heads * head_dim;
    int kv_size = num_kv_heads * head_dim;

    /* Project Q, K, V (BF16 weights) */
    gemma3_matvec_bf16(q_buf, q_weight, x, q_size, hidden_size);
    gemma3_matvec_bf16(k_buf, k_weight, x, kv_size, hidden_size);
    gemma3_matvec_bf16(v_buf, v_weight, x, kv_size, hidden_size);

    /* Apply QK normalization (per-head RMSNorm with BF16 weights) */
    if (q_norm && k_norm) {
        for (int h = 0; h < num_heads; h++) {
            gemma3_rmsnorm_bf16(q_buf + h * head_dim, q_buf + h * head_dim,
                                q_norm, head_dim, cfg->rmsnorm_eps);
        }
        for (int h = 0; h < num_kv_heads; h++) {
            gemma3_rmsnorm_bf16(k_buf + h * head_dim, k_buf + h * head_dim,
                                k_norm, head_dim, cfg->rmsnorm_eps);
        }
    }

    /* Apply RoPE */
    int is_global = gemma3_is_global_layer(layer_idx);
    float theta = is_global ? cfg->rope_theta_global : cfg->rope_theta_local;
    gemma3_rope(q_buf, k_buf, num_heads, num_kv_heads, head_dim, pos, theta);

    /* Add K, V to cache */
    cache_kv(cache, k_buf, v_buf, kv_size, is_global, cfg->sliding_window, pos);

    /* Determine attention range */
    int seq_len;
    const float *k_cache, *v_cache;

    if (is_global) {
        /* Global attention: attend to all previous positions */
        seq_len = pos + 1;
        k_cache = cache->k;
        v_cache = cache->v;

        /* Causal mask */
        gemma3_causal_mask(mask_buf, seq_len, pos);
    } else {
        /* Local attention: sliding window */
        int window = cfg->sliding_window;
        int start_pos = (pos >= window) ? pos - window + 1 : 0;
        seq_len = pos - start_pos + 1;

        /* For ring buffer, we need to handle wraparound */
        /* Simplified: just use the cached entries that are valid */
        seq_len = (pos < window) ? pos + 1 : window;
        k_cache = cache->k;
        v_cache = cache->v;

        /* Sliding window mask */
        for (int i = 0; i < seq_len; i++) {
            mask_buf[i] = 0.0f;  /* All positions in window are valid */
        }
    }

    /* Compute scaled dot-product attention with GQA */
    float scale = 1.0f / sqrtf((float)head_dim);
    gemma3_gqa(attn_buf, q_buf, k_cache, v_cache,
               num_heads, num_kv_heads, seq_len, head_dim,
               scale, mask_buf);

    /* Output projection (BF16 weights) */
    gemma3_matvec_bf16(output, o_weight, attn_buf, hidden_size, q_size);
}

/* ============================================================================
 * MLP Implementation (SwiGLU)
 * ========================================================================== */

static void layer_mlp(
    float *output,            /* [hidden_size] */
    const float *x,           /* [hidden_size] */
    const uint16_t *gate_weight, /* [intermediate_size, hidden_size] BF16 */
    const uint16_t *up_weight,   /* [intermediate_size, hidden_size] BF16 */
    const uint16_t *down_weight, /* [hidden_size, intermediate_size] BF16 */
    float *gate_buf,          /* [intermediate_size] */
    float *up_buf,            /* [intermediate_size] */
    const gemma3_config *cfg,
    int layer_idx,
    int pos
) {
    int hidden_size = cfg->hidden_size;
    int intermediate_size = cfg->intermediate_size;

    /* Gate and up projections (BF16 weights) */
    gemma3_matvec_bf16(gate_buf, gate_weight, x, intermediate_size, hidden_size);
    gemma3_matvec_bf16(up_buf, up_weight, x, intermediate_size, hidden_size);

    /* SwiGLU: gate = SiLU(gate) * up */
    /* Gemma 3 uses GELU instead of SiLU for the gate */
    gemma3_gelu_tanh_inplace(gate_buf, intermediate_size);
    gemma3_vec_mul(gate_buf, gate_buf, up_buf, intermediate_size);

    /* Down projection (BF16 weights) */
    gemma3_matvec_bf16(output, down_weight, gate_buf, hidden_size, intermediate_size);

    (void)layer_idx;
    (void)pos;
}

/* ============================================================================
 * Full Forward Pass
 * ========================================================================== */

/* Forward pass for a single token */
int gemma3_transformer_forward(
    float *logits,            /* Output: [vocab_size] */
    int token_id,             /* Input token */
    int pos,                  /* Position in sequence */
    const gemma3_weights_t *weights,
    gemma3_kv_cache *cache,
    activation_buffers *buf,
    const gemma3_config *cfg
) {
    int hidden_size = cfg->hidden_size;
    int vocab_size = cfg->vocab_size;

    /* Token embedding lookup (BF16) */
    gemma3_embed_bf16(buf->x, weights->embed_tokens, token_id, hidden_size);
    const float *embed = buf->x;  /* buf->x now contains the F32 embedding */

    /* Gemma scales embeddings by sqrt(hidden_size) */
    float embed_scale = sqrtf((float)hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        buf->x[i] = embed[i] * embed_scale;
    }

    /* Process each layer */
    for (int l = 0; l < cfg->num_layers; l++) {
        const uint16_t *layer_weights_input_ln = weights->layers[l].input_layernorm;
        const uint16_t *layer_weights_q = weights->layers[l].q_proj;
        const uint16_t *layer_weights_k = weights->layers[l].k_proj;
        const uint16_t *layer_weights_v = weights->layers[l].v_proj;
        const uint16_t *layer_weights_o = weights->layers[l].o_proj;
        const uint16_t *layer_weights_q_norm = weights->layers[l].q_norm;
        const uint16_t *layer_weights_k_norm = weights->layers[l].k_norm;
        const uint16_t *layer_weights_post_attn_ln = weights->layers[l].post_attention_layernorm;
        const uint16_t *layer_weights_gate = weights->layers[l].gate_proj;
        const uint16_t *layer_weights_up = weights->layers[l].up_proj;
        const uint16_t *layer_weights_down = weights->layers[l].down_proj;
        const uint16_t *layer_weights_pre_ff_ln = weights->layers[l].pre_feedforward_layernorm;
        const uint16_t *layer_weights_post_ff_ln = weights->layers[l].post_feedforward_layernorm;

        /* === Self-Attention Block === */

        /* Pre-attention RMSNorm (BF16 weights) */
        gemma3_rmsnorm_bf16(buf->x_norm, buf->x, layer_weights_input_ln,
                            hidden_size, cfg->rmsnorm_eps);

        /* Attention */
        layer_attention(
            buf->proj_out,
            buf->x_norm,
            layer_weights_q, layer_weights_k, layer_weights_v, layer_weights_o,
            layer_weights_q_norm, layer_weights_k_norm,
            &cache->layers[l],
            buf->q, buf->k, buf->v, buf->attn_out, buf->mask,
            cfg, l, pos
        );

        /* Post-attention RMSNorm (Gemma 2/3 specific, BF16 weights with 1+weight) */
        if (layer_weights_post_attn_ln) {
            gemma3_rmsnorm_bf16_inplace(buf->proj_out, layer_weights_post_attn_ln,
                                        hidden_size, cfg->rmsnorm_eps);
        }

        /* Residual connection */
        gemma3_vec_add(buf->x, buf->x, buf->proj_out, hidden_size);

        /* === MLP Block === */

        /* Pre-feedforward RMSNorm (Gemma 3 specific, BF16 weights) */
        if (layer_weights_pre_ff_ln) {
            gemma3_rmsnorm_bf16(buf->x_norm, buf->x, layer_weights_pre_ff_ln,
                                hidden_size, cfg->rmsnorm_eps);
        } else {
            gemma3_vec_copy(buf->x_norm, buf->x, hidden_size);
        }

        /* MLP */
        layer_mlp(
            buf->mlp_out,
            buf->x_norm,
            layer_weights_gate, layer_weights_up, layer_weights_down,
            buf->mlp_gate, buf->mlp_up,
            cfg, l, pos
        );

        /* Post-feedforward RMSNorm (Gemma 2/3 specific, BF16 weights with 1+weight) */
        if (layer_weights_post_ff_ln) {
            gemma3_rmsnorm_bf16_inplace(buf->mlp_out, layer_weights_post_ff_ln,
                                        hidden_size, cfg->rmsnorm_eps);
        }

        /* Residual connection */
        gemma3_vec_add(buf->x, buf->x, buf->mlp_out, hidden_size);
    }

    /* Final RMSNorm (BF16 weights) */
    gemma3_rmsnorm_bf16(buf->x_norm, buf->x, weights->norm, hidden_size, cfg->rmsnorm_eps);

    /* Output projection (tied embeddings, BF16) */
    /* logits = x_norm @ embed_tokens.T */
    gemma3_matvec_bf16(logits, weights->embed_tokens, buf->x_norm, vocab_size, hidden_size);

    return 0;
}

/* Prefill: process multiple tokens at once */
int gemma3_transformer_prefill(
    float *logits,            /* Output: [vocab_size] for last token */
    const int *tokens,        /* Input tokens */
    int num_tokens,           /* Number of tokens */
    int start_pos,            /* Starting position */
    const gemma3_weights_t *weights,
    gemma3_kv_cache *cache,
    activation_buffers *buf,
    const gemma3_config *cfg
) {
    /* For simplicity, process tokens one at a time during prefill */
    /* A more optimized version would batch this */
    for (int i = 0; i < num_tokens; i++) {
        int pos = start_pos + i;
        int is_last = (i == num_tokens - 1);

        /* Only compute full logits for last token */
        if (is_last) {
            gemma3_transformer_forward(logits, tokens[i], pos, weights, cache, buf, cfg);
        } else {
            /* Compute forward but don't need logits (use temp buffer) */
            gemma3_transformer_forward(buf->logits, tokens[i], pos, weights, cache, buf, cfg);
        }
    }

    cache->current_pos = start_pos + num_tokens;
    return 0;
}

/* ============================================================================
 * Transformer Context (combines weights, cache, buffers)
 * ========================================================================== */

typedef struct gemma3_transformer {
    gemma3_weights_t *weights;
    gemma3_kv_cache *cache;
    activation_buffers *buffers;
    gemma3_config config;
} gemma3_transformer;

gemma3_transformer *gemma3_transformer_create(
    gemma3_weights_t *weights,
    const gemma3_config *cfg,
    int max_context
) {
    gemma3_transformer *t = (gemma3_transformer *)calloc(1, sizeof(gemma3_transformer));
    if (!t) return NULL;

    t->weights = weights;
    t->config = *cfg;

    t->cache = gemma3_kv_cache_alloc(cfg, max_context);
    if (!t->cache) {
        free(t);
        return NULL;
    }

    t->buffers = alloc_buffers(cfg);
    if (!t->buffers) {
        gemma3_kv_cache_free(t->cache);
        free(t);
        return NULL;
    }

    return t;
}

void gemma3_transformer_destroy(gemma3_transformer *t) {
    if (!t) return;
    gemma3_kv_cache_free(t->cache);
    free_buffers(t->buffers);
    free(t);
}

int gemma3_transformer_forward_token(
    gemma3_transformer *t,
    int token_id,
    int pos,
    float *logits
) {
    return gemma3_transformer_forward(
        logits, token_id, pos,
        t->weights, t->cache, t->buffers, &t->config
    );
}

int gemma3_transformer_prefill_tokens(
    gemma3_transformer *t,
    const int *tokens,
    int num_tokens,
    int start_pos,
    float *logits
) {
    return gemma3_transformer_prefill(
        logits, tokens, num_tokens, start_pos,
        t->weights, t->cache, t->buffers, &t->config
    );
}

void gemma3_transformer_reset(gemma3_transformer *t) {
    if (t && t->cache) {
        gemma3_kv_cache_reset(t->cache);
    }
}

int gemma3_transformer_get_pos(gemma3_transformer *t) {
    return t && t->cache ? t->cache->current_pos : 0;
}
