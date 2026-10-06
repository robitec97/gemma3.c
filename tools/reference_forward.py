#!/usr/bin/env python3
"""
reference_forward.py - Independent NumPy reference for the Gemma 3 4B text model.

Implements the forward pass the way Hugging Face transformers' modeling_gemma3
does (float32 math), without depending on torch or on any of the C code:

  * embeddings scaled by sqrt(hidden_size)
  * Gemma3RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w), computed in float32
  * q_norm / k_norm applied per head before RoPE
  * rotate_half RoPE; local layers use rope_local_base_freq (10k, no scaling),
    global layers ((l + 1) % sliding_window_pattern == 0) use rope_theta (1M)
    with the linear rope_scaling factor from config.json (8.0 for 4B)
  * sliding window: a query at position p attends to keys in (p - window, p]
  * GQA (8 query heads share 4 KV heads), scale = query_pre_attn_scalar ** -0.5
  * gelu(tanh) gated MLP, pre/post attention and pre/post feed-forward norms
  * final norm and tied lm_head (embed_tokens), no logit soft-capping

Weights are read straight from the safetensors files (np.fromfile, so they
stay in the OS page cache rather than this process) and converted from BF16
to float32 one layer at a time, so peak RSS stays around 1-2 GB plus
activations.

Usage:
  python tools/reference_forward.py -m gemma-3-4b-it --ids 2,9259,236764,1902
  python tools/reference_forward.py -m gemma-3-4b-it --text "The capital of France is"
  python tools/reference_forward.py -m gemma-3-4b-it --ids-file ids.txt --dump logits.npy

Options:
  --no-rope-scaling   ignore config.json rope_scaling (old gemma3.c behaviour)
  --hf-bf16-embed-scale
                      multiply embeddings by sqrt(hidden) rounded to bf16 (50.5),
                      as HF does when the model runs in bfloat16. The default
                      uses the exact float value (50.596), which is what HF
                      does in float32 and what gemma3.c does.
  --dtype float64     run the math in float64 (to measure float32 error)
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np


# ----------------------------------------------------------------------------
# SafeTensors loading
# ----------------------------------------------------------------------------

class SafeTensors:
    """Minimal multi-file safetensors reader.

    Tensors are read with np.fromfile rather than np.memmap so that weight
    pages live in the OS page cache instead of this process's resident set
    (keeps peak RSS around 1-2 GB even though every weight is touched)."""

    _DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}

    def __init__(self, model_dir):
        self.tensors = {}
        files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {model_dir}")
        for fname in files:
            path = os.path.join(model_dir, fname)
            with open(path, "rb") as f:
                (header_len,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(header_len))
            data_start = 8 + header_len
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                begin, end = info["data_offsets"]
                self.tensors[name] = (path, info["dtype"], tuple(info["shape"]),
                                      data_start + begin, end - begin)

    def __contains__(self, name):
        return name in self.tensors

    def shape(self, name):
        return self.tensors[name][2]

    def _read(self, name, rows=None):
        path, dtype, shape, offset, nbytes = self.tensors[name]
        np_dtype = np.dtype(self._DTYPES[dtype])
        row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        row_bytes = row_elems * np_dtype.itemsize
        with open(path, "rb") as f:
            if rows is None:
                f.seek(offset)
                return np.fromfile(f, dtype=np_dtype, count=nbytes // np_dtype.itemsize
                                   ).reshape(shape), dtype
            if isinstance(rows, slice):
                r0, r1, _ = rows.indices(shape[0])
                f.seek(offset + r0 * row_bytes)
                arr = np.fromfile(f, dtype=np_dtype, count=(r1 - r0) * row_elems)
                return arr.reshape((r1 - r0,) + tuple(shape[1:])), dtype
            out = np.empty((len(rows),) + tuple(shape[1:]), dtype=np_dtype)
            for j, r in enumerate(rows):
                f.seek(offset + int(r) * row_bytes)
                out[j] = np.fromfile(f, dtype=np_dtype, count=row_elems).reshape(shape[1:])
            return out, dtype

    def get(self, name, dtype=np.float32, rows=None):
        """Load a tensor (optionally a slice or list of rows) converted to `dtype`."""
        arr, st_dtype = self._read(name, rows)
        if st_dtype == "BF16":
            out = (arr.astype(np.uint32) << 16).view(np.float32)
        else:
            out = arr.astype(np.float32)
        return out.astype(dtype, copy=False)


def detect_prefix(st):
    for prefix in ("language_model.model.", "model.language_model.", "model.", ""):
        if prefix + "embed_tokens.weight" in st:
            return prefix
    raise KeyError("embed_tokens.weight not found - not a Gemma 3 checkpoint?")


def load_text_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


# ----------------------------------------------------------------------------
# Model math
# ----------------------------------------------------------------------------

def rms_norm(x, w, eps):
    """Gemma3RMSNorm: normalise over the last axis, scale by (1 + w)."""
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * (1.0 + w)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi).astype(x.dtype) *
                                    (x + 0.044715 * x * x * x)))


def rope_cos_sin(positions, head_dim, base, scaling, dtype):
    """cos/sin tables of shape [T, head_dim] as in HF (emb = cat(freqs, freqs))."""
    # HF computes inv_freq in float32 and divides by the factor for "linear".
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32)
                               / head_dim))
    inv_freq = (inv_freq / scaling).astype(np.float32)
    freqs = np.outer(positions.astype(np.float32), inv_freq)        # [T, D/2]
    emb = np.concatenate([freqs, freqs], axis=-1)                    # [T, D]
    return np.cos(emb).astype(dtype), np.sin(emb).astype(dtype)


def apply_rope(x, cos, sin):
    """x: [T, H, D]; cos/sin: [T, D]."""
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def attention_mask(T, window, dtype):
    """Additive mask [T, T]: causal, plus sliding window if `window` is set."""
    q = np.arange(T)[:, None]
    k = np.arange(T)[None, :]
    allowed = k <= q
    if window is not None:
        allowed &= k > q - window
    mask = np.zeros((T, T), dtype=dtype)
    mask[~allowed] = -np.inf
    return mask


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def forward(model_dir, ids, rope_scaling=True, bf16_embed_scale=False,
            dtype=np.float32, all_positions=False, verbose=True, sliding_window=True):
    """Run the text model on `ids`; return logits for the last position
    (or [T, vocab] if all_positions). sliding_window=False makes local layers
    attend to the full causal context (ablation)."""
    cfg = load_text_config(model_dir)
    st = SafeTensors(model_dir)
    pre = detect_prefix(st)

    hidden = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    window = cfg["sliding_window"]
    pattern = cfg.get("sliding_window_pattern", 6)
    local_base = cfg.get("rope_local_base_freq", 10000.0)
    global_base = cfg.get("rope_theta", 1000000.0)
    scaling_cfg = cfg.get("rope_scaling") or {}
    global_factor = float(scaling_cfg.get("factor", 1.0)) if rope_scaling else 1.0
    attn_scale = cfg.get("query_pre_attn_scalar", head_dim) ** -0.5

    ids = np.asarray(ids, dtype=np.int64)
    T = len(ids)
    positions = np.arange(T)
    t0 = time.time()

    # Embedding lookup (only the rows we need) and scaling
    x = st.get(pre + "embed_tokens.weight", dtype, rows=ids)
    embed_scale = np.float32(np.sqrt(hidden))
    if bf16_embed_scale:
        bits = np.array([embed_scale], dtype=np.float32).view(np.uint32)
        # round-to-nearest-even to bf16, as torch does for .to(bfloat16)
        bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
        embed_scale = bits.view(np.float32)[0]
    x = x * dtype(embed_scale)

    cos_l, sin_l = rope_cos_sin(positions, head_dim, local_base, 1.0, dtype)
    cos_g, sin_g = rope_cos_sin(positions, head_dim, global_base, global_factor, dtype)
    mask_local = attention_mask(T, window if sliding_window else None, dtype)
    mask_global = attention_mask(T, None, dtype)
    group = n_heads // n_kv

    for l in range(n_layers):
        p = f"{pre}layers.{l}."
        w = lambda n: st.get(p + n, dtype)  # noqa: E731
        is_global = (l + 1) % pattern == 0

        # --- self-attention ---
        h = rms_norm(x, w("input_layernorm.weight"), eps)
        q = (h @ w("self_attn.q_proj.weight").T).reshape(T, n_heads, head_dim)
        k = (h @ w("self_attn.k_proj.weight").T).reshape(T, n_kv, head_dim)
        v = (h @ w("self_attn.v_proj.weight").T).reshape(T, n_kv, head_dim)
        q = rms_norm(q, w("self_attn.q_norm.weight"), eps)
        k = rms_norm(k, w("self_attn.k_norm.weight"), eps)
        cos, sin = (cos_g, sin_g) if is_global else (cos_l, sin_l)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        mask = mask_global if is_global else mask_local
        out = np.empty((T, n_heads, head_dim), dtype=dtype)
        for hq in range(n_heads):
            kv = hq // group
            scores = (q[:, hq, :] @ k[:, kv, :].T) * dtype(attn_scale) + mask
            out[:, hq, :] = softmax(scores) @ v[:, kv, :]
        attn = out.reshape(T, n_heads * head_dim) @ w("self_attn.o_proj.weight").T
        attn = rms_norm(attn, w("post_attention_layernorm.weight"), eps)
        x = x + attn

        # --- MLP ---
        h = rms_norm(x, w("pre_feedforward_layernorm.weight"), eps)
        gate = h @ w("mlp.gate_proj.weight").T
        up = h @ w("mlp.up_proj.weight").T
        mlp = (gelu_tanh(gate) * up) @ w("mlp.down_proj.weight").T
        mlp = rms_norm(mlp, w("post_feedforward_layernorm.weight"), eps)
        x = x + mlp

        if verbose and (l % 8 == 7 or l == n_layers - 1):
            print(f"  [reference] layer {l + 1}/{n_layers} done ({time.time() - t0:.1f}s)",
                  file=sys.stderr)

    # --- final norm + tied lm_head, computed in row chunks of the embedding ---
    final = x if all_positions else x[-1:]
    final = rms_norm(final, st.get(pre + "norm.weight", dtype), eps)
    vocab = st.shape(pre + "embed_tokens.weight")[0]
    logits = np.empty((final.shape[0], vocab), dtype=np.float32)
    chunk = 16384
    for r0 in range(0, vocab, chunk):
        r1 = min(vocab, r0 + chunk)
        e = st.get(pre + "embed_tokens.weight", dtype, rows=slice(r0, r1))
        logits[:, r0:r1] = final @ e.T
    if verbose:
        print(f"  [reference] forward of {T} tokens took {time.time() - t0:.1f}s",
              file=sys.stderr)
    return logits if all_positions else logits[0]


def top_k(logits, k=20):
    idx = np.argpartition(-logits, k)[:k]
    idx = idx[np.argsort(-logits[idx], kind="stable")]
    return [(int(i), float(logits[i])) for i in idx]


def tokenize_hf(model_dir, text):
    """Tokenize with the HF `tokenizers` library (adds <bos>)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    return [2] + tok.encode(text, add_special_tokens=False).ids


def piece_names(model_dir, ids):
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        return [tok.id_to_token(i) for i in ids]
    except Exception:
        return ["?"] * len(ids)


def parse_ids(s):
    return [int(t) for t in s.replace("[", " ").replace("]", " ").replace(",", " ").split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-m", "--model", default="gemma-3-4b-it")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids", help="comma-separated token IDs (include <bos>=2 yourself)")
    src.add_argument("--ids-file", help="file containing token IDs")
    src.add_argument("--text", help="text to tokenize with tokenizer.json (adds <bos>)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--dump", help="save final-position logits to this .npy file")
    ap.add_argument("--all-positions", action="store_true",
                    help="with --dump, save logits for every position ([T, vocab])")
    ap.add_argument("--no-rope-scaling", action="store_true")
    ap.add_argument("--hf-bf16-embed-scale", action="store_true")
    ap.add_argument("--no-sliding-window", action="store_true",
                    help="ablation: local layers attend to the full context")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = ap.parse_args()

    if args.ids is not None:
        ids = parse_ids(args.ids)
    elif args.ids_file is not None:
        with open(args.ids_file) as f:
            ids = parse_ids(f.read())
    else:
        ids = tokenize_hf(args.model, args.text)

    logits = forward(args.model, ids,
                     rope_scaling=not args.no_rope_scaling,
                     bf16_embed_scale=args.hf_bf16_embed_scale,
                     dtype=np.float64 if args.dtype == "float64" else np.float32,
                     all_positions=args.all_positions and args.dump is not None,
                     sliding_window=not args.no_sliding_window)
    last = logits[-1] if logits.ndim == 2 else logits

    if args.dump:
        np.save(args.dump, logits)
        print(f"saved logits {logits.shape} to {args.dump}", file=sys.stderr)

    print(f"Token count: {len(ids)}")
    print(f"Token IDs: [{', '.join(str(i) for i in ids)}]\n")
    print(f"Top-{args.top} next token predictions:")
    print(f"{'Rank':<6}  {'Token ID':<10}  {'Logit':<10}  Token")
    top = top_k(last, args.top)
    names = piece_names(args.model, [i for i, _ in top])
    for r, ((i, v), name) in enumerate(zip(top, names), 1):
        print(f"{r:<6}  {i:<10}  {v:10.4f}  {name!r}")


if __name__ == "__main__":
    main()
