#!/usr/bin/env python3
"""
compare_logits.py - Check gemma3.c logits against the NumPy reference
(tools/reference_forward.py) or against another gemma3 binary.

For each prompt it runs `<bin> -m <model> --logits -p <text>`, parses the
token IDs and the top-20 table from stdout, then either runs the reference on
exactly the same token IDs or runs a second binary, and reports:

  * top-20 overlap   - how many of the binary's top-20 IDs are in the other top-20
  * max |diff|       - largest absolute logit difference over the binary's top-20 IDs
  * argmax match     - whether the most likely next token agrees

Usage:
  # binary vs reference on the built-in short prompts and the long (~1500 token) prompt
  python tools/compare_logits.py --bin ./gemma3 -m gemma-3-4b-it

  # only some prompts / custom prompt
  python tools/compare_logits.py --bin ./gemma3 --prompt "Hello there" --no-long

  # also report the reference without global RoPE scaling (old behaviour)
  python tools/compare_logits.py --bin ./gemma3 --rope-ablation

  # two binaries against each other (e.g. CPU vs Metal, before vs after)
  python tools/compare_logits.py --bin ./gemma3-cpu --bin2 ./gemma3-metal

Exit status is non-zero if any comparison fails the thresholds
(--min-overlap, --max-diff, argmax must match).
"""

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_forward as ref  # noqa: E402

SHORT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Water boils at a temperature of",
]

LONG_PARAGRAPH = (
    "The history of computing is a story of abstraction. Early machines were "
    "programmed by rewiring panels and flipping switches, and every program was "
    "tied to the physical layout of the hardware. Stored-program computers changed "
    "that: instructions became data, and data could be loaded, copied and modified "
    "like anything else. Assemblers turned mnemonic names into machine code, "
    "compilers turned structured languages into assembly, and operating systems "
    "turned the raw machine into a set of services that many programs could share. "
)


def long_prompt(target_tokens=1500):
    """A ~target_tokens prompt: a numbered list of repeated paragraphs (numbers
    keep the text from being perfectly periodic) ending in a question."""
    parts, i = [], 0
    # ~98 tokens per paragraph with this text
    while len(parts) * 98 < target_tokens - 40:
        i += 1
        parts.append(f"Section {i}. {LONG_PARAGRAPH}")
    return "".join(parts) + "\nQuestion: how many sections were there above? Answer:"


ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(-?(?:\d+\.\d+|nan|inf))\s")


def run_binary(binary, model, text, context=None, timeout=3600):
    cmd = [binary, "-m", model, "--logits", "-p", text]
    if context:
        cmd += ["-c", str(context)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    out = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} failed ({proc.returncode}):\n"
                           f"{proc.stderr.decode(errors='replace')[-2000:]}")
    m = re.search(r"Token IDs: \[([^\]]*)\]", out)
    if not m:
        raise RuntimeError(f"could not find token IDs in output of {binary}")
    ids = [int(x) for x in m.group(1).split(",") if x.strip()]
    table = out[out.index("Top-20"):] if "Top-20" in out else out
    rows = []
    for line in table.splitlines():
        r = ROW_RE.match(line)
        if r:
            rows.append((int(r.group(2)), float(r.group(3))))
    if not rows:
        raise RuntimeError(f"could not parse top-20 table from {binary}")
    return ids, rows, time.time() - t0


def compare_rows(rows_a, logits_b=None, rows_b=None):
    """Compare a binary's top-20 against full reference logits or another top-20."""
    ids_a = [i for i, _ in rows_a]
    if logits_b is not None:
        top_b = [i for i, _ in ref.top_k(logits_b, len(ids_a))]
        diffs = [abs(v - float(logits_b[i])) for i, v in rows_a]
        argmax_b = int(np.argmax(logits_b))
    else:
        top_b = [i for i, _ in rows_b]
        vb = dict(rows_b)
        diffs = [abs(v - vb[i]) for i, v in rows_a if i in vb]
        argmax_b = rows_b[0][0]
    overlap = len(set(ids_a) & set(top_b))
    return {
        "overlap": overlap,
        "n": len(ids_a),
        "max_diff": max(diffs) if diffs else float("nan"),
        "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "argmax_match": ids_a[0] == argmax_b,
        "argmax_a": ids_a[0],
        "argmax_b": argmax_b,
    }


def fmt_name(text, n_tokens):
    s = text.replace("\n", "\\n")
    s = s if len(s) <= 34 else s[:31] + "..."
    return f"{s} ({n_tokens} tok)"


def main():
    ap = argparse.ArgumentParser(description="Compare gemma3.c logits to a reference")
    ap.add_argument("--bin", required=True, help="gemma3 binary to test")
    ap.add_argument("--bin2", help="second binary: compare --bin against it instead of the reference")
    ap.add_argument("-m", "--model", default="gemma-3-4b-it")
    ap.add_argument("--prompt", action="append", help="custom prompt (repeatable)")
    ap.add_argument("--no-short", action="store_true", help="skip the built-in short prompts")
    ap.add_argument("--no-long", action="store_true", help="skip the ~1500-token prompt")
    ap.add_argument("--long-tokens", type=int, default=1500)
    ap.add_argument("--rope-ablation", action="store_true",
                    help="also compare against the reference without global RoPE scaling")
    ap.add_argument("--window-ablation", action="store_true",
                    help="also compare against the reference without the sliding window")
    ap.add_argument("--embed-ablation", action="store_true",
                    help="also compare against the reference with the bf16-rounded embed scale")
    ap.add_argument("--hf-bf16-embed-scale", action="store_true")
    ap.add_argument("--min-overlap", type=int, default=18)
    ap.add_argument("--max-diff", type=float, default=0.25)
    args = ap.parse_args()

    prompts = []
    if args.prompt:
        prompts += args.prompt
    if not args.no_short and not args.prompt:
        prompts += SHORT_PROMPTS
    if not args.no_long:
        prompts.append(long_prompt(args.long_tokens))

    header = f"{'prompt':<44} {'vs':<14} {'top20':>6} {'max|diff|':>10} {'mean|diff|':>11} {'argmax':>7}"
    results = []
    failed = False
    for text in prompts:
        ids, rows, t_bin = run_binary(args.bin, args.model, text,
                                      context=max(8192, len(text)))
        print(f"[{os.path.basename(args.bin)}] {len(ids)} tokens in {t_bin:.1f}s",
              file=sys.stderr)
        name = fmt_name(text, len(ids))
        n_before = len(results)
        if args.bin2:
            ids2, rows2, t2 = run_binary(args.bin2, args.model, text,
                                         context=max(8192, len(text)))
            if ids2 != ids:
                print(f"warning: binaries tokenized differently ({len(ids)} vs {len(ids2)} tokens); "
                      "logits are not comparable", file=sys.stderr)
            res = compare_rows(rows, rows_b=rows2)
            results.append((name, os.path.basename(args.bin2), res))
        else:
            base = dict(rope_scaling=True, sliding_window=True,
                        bf16_embed_scale=args.hf_bf16_embed_scale)
            variants = [("reference", base)]
            if args.rope_ablation:
                variants.append(("ref no-rope8", dict(base, rope_scaling=False)))
            if args.window_ablation and len(ids) > 1024:
                variants.append(("ref no-window", dict(base, sliding_window=False)))
            if args.embed_ablation:
                variants.append(("ref bf16-embed", dict(base, bf16_embed_scale=True)))
            for label, kw in variants:
                logits = ref.forward(args.model, ids, **kw)
                res = compare_rows(rows, logits_b=logits)
                results.append((name, label, res))
        for n, label, res in results[n_before:]:
            print(f"  {n} vs {label}: top20 {res['overlap']}/{res['n']}, "
                  f"max|diff| {res['max_diff']:.4f}, argmax "
                  f"{'match' if res['argmax_match'] else 'MISMATCH'}", file=sys.stderr)

    print()
    print(header)
    print("-" * len(header))
    for name, label, res in results:
        print(f"{name:<44} {label:<14} {res['overlap']:>3}/{res['n']:<2} "
              f"{res['max_diff']:>10.4f} {res['mean_diff']:>11.4f} "
              f"{'yes' if res['argmax_match'] else 'NO':>7}")
        # Only the primary comparison (reference with scaling, or bin2) gates the exit code
        if label in ("reference",) or args.bin2:
            if (res["overlap"] < args.min_overlap or res["max_diff"] > args.max_diff
                    or not res["argmax_match"]):
                failed = True
    print()
    print("PASS" if not failed else "FAIL",
          f"(thresholds: top20 overlap >= {args.min_overlap}, max|diff| <= {args.max_diff}, argmax match)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
