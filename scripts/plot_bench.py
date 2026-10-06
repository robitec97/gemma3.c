#!/usr/bin/env python3
"""Plot before/after benchmark results (bench_e2e JSON files) as SVG bar charts.

Each --pair names a configuration and gives the baseline and new result files:

  python scripts/plot_bench.py --out docs/images/speedup \
      --pair "CPU (10 threads)" bench/results/baseline-cpu.json bench/results/cpu.json \
      --pair "Metal GPU"        bench/results/baseline-metal.json bench/results/metal.json

Writes <out>-light.svg and <out>-dark.svg (for GitHub's light/dark themes).
Requires: pip install matplotlib
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984",
        "grid": "#e4e3df", "before": "#86b6ef", "after": "#1c5cab",
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "muted": "#8f8e86",
        "grid": "#2e2e2c", "before": "#184f95", "after": "#5598e7",
    },
}


def load(path):
    with open(path) as f:
        return json.load(f)


def prefill_at(res, n):
    for p in res["prefill"]:
        if p["tokens"] == n:
            return p["tok_per_s"]
    return None


def depth_at(res, d):
    for p in res.get("decode_at_depth", []):
        if p["depth"] == d:
            return p["tok_per_s"]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--pair", nargs=3, action="append", metavar=("LABEL", "BASELINE", "NEW"), required=True)
    ap.add_argument("--prompt-len", type=int, default=256, help="prompt length for the prefill panel")
    ap.add_argument("--depth", type=int, default=2048, help="context depth for the long-context panel (0 = off)")
    ap.add_argument("--title", default="Gemma 3 4B on Apple M4 (16 GB): original vs. this branch")
    ap.add_argument("--png", action="store_true", help="also write PNG previews")
    args = ap.parse_args()

    labels, pre_b, pre_a, dec_b, dec_a, dep_b, dep_a = [], [], [], [], [], [], []
    for label, base_path, new_path in args.pair:
        b, a = load(base_path), load(new_path)
        labels.append(label)
        pre_b.append(prefill_at(b, args.prompt_len) or 0.0)
        pre_a.append(prefill_at(a, args.prompt_len) or 0.0)
        dec_b.append(b["decode"]["tok_per_s"])
        dec_a.append(a["decode"]["tok_per_s"])
        dep_b.append(depth_at(b, args.depth))
        dep_a.append(depth_at(a, args.depth))

    panels = [
        (f"Prompt processing ({args.prompt_len} tokens)", pre_b, pre_a),
        ("Text generation", dec_b, dec_a),
    ]
    if args.depth and all(v is not None for v in dep_b + dep_a):
        panels.append((f"Generation at {args.depth // 1024}K context", dep_b, dep_a))

    for theme_name, th in THEMES.items():
        plt.rcParams.update({
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "svg.fonttype": "none",
        })
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 1.35 + 0.95 * len(labels)),
                                 facecolor=th["surface"])
        bar_h = 0.34
        for ax, (title, before, after) in zip(axes, panels):
            ax.set_facecolor(th["surface"])
            ys = list(range(len(labels)))[::-1]
            vmax = max(before + after) or 1.0
            for y, vb, va in zip(ys, before, after):
                ax.barh(y + bar_h / 2 + 0.02, vb, height=bar_h, color=th["before"], zorder=2)
                ax.barh(y - bar_h / 2 - 0.02, va, height=bar_h, color=th["after"], zorder=2)
                ax.text(vb + vmax * 0.015, y + bar_h / 2 + 0.02, f"{vb:.1f}", va="center",
                        ha="left", color=th["text2"], fontsize=10)
                speed = f"  ({va / vb:.1f}x)" if vb > 0 else ""
                ax.text(va + vmax * 0.015, y - bar_h / 2 - 0.02, f"{va:.1f}{speed}", va="center",
                        ha="left", color=th["text"], fontsize=10, fontweight="bold")
            ax.set_yticks(ys)
            ax.set_yticklabels(labels if ax is axes[0] else [""] * len(labels), color=th["text"])
            ax.set_xlim(0, vmax * 1.32)
            ax.set_xlabel("tokens / second", color=th["text2"], fontsize=10)
            ax.set_title(title, loc="left", color=th["text"], fontsize=12, fontweight="bold", pad=10)
            ax.tick_params(axis="x", colors=th["muted"], labelsize=9, length=0)
            ax.tick_params(axis="y", length=0, pad=8)
            ax.grid(axis="x", color=th["grid"], linewidth=0.8, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(th["grid"])

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=th["before"]),
            plt.Rectangle((0, 0), 1, 1, color=th["after"]),
        ]
        fig.legend(handles, ["original", "this branch"], loc="upper right", ncol=2, frameon=False,
                   labelcolor=th["text2"], fontsize=10, bbox_to_anchor=(0.99, 1.0))
        fig.suptitle(args.title, x=0.01, ha="left", color=th["text"], fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = f"{args.out}-{theme_name}.svg"
        fig.savefig(out, facecolor=th["surface"])
        if args.png:
            fig.savefig(out[:-4] + ".png", facecolor=th["surface"], dpi=110)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
