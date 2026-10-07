#!/usr/bin/env python3
"""Plot benchmark results (bench_e2e JSON files) as SVG bar charts.

Each --result names a build and gives its result file:

  python scripts/plot_bench.py --out docs/images/performance \
      --result "make mps (Metal GPU)" bench/results/mps.json \
      --result "make (CPU)"           bench/results/cpu.json

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
        "grid": "#e4e3df", "bar": "#1c5cab",
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "muted": "#8f8e86",
        "grid": "#2e2e2c", "bar": "#5598e7",
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
    ap.add_argument("--result", nargs=2, action="append", metavar=("LABEL", "FILE"), required=True)
    ap.add_argument("--prompt-len", type=int, default=256, help="prompt length for the prefill panel")
    ap.add_argument("--depth", type=int, default=2048, help="context depth for the long-context panel (0 = off)")
    ap.add_argument("--title", default="Gemma 3 4B IT on Apple M4 (16 GB), BF16 weights")
    ap.add_argument("--png", action="store_true", help="also write PNG previews")
    args = ap.parse_args()

    labels, pre, dec, dep = [], [], [], []
    for label, path in args.result:
        r = load(path)
        labels.append(label)
        pre.append(prefill_at(r, args.prompt_len))
        dec.append(r["decode"]["tok_per_s"])
        dep.append(depth_at(r, args.depth))

    panels = [
        (f"Prompt processing ({args.prompt_len} tokens)", pre),
        ("Text generation", dec),
    ]
    if args.depth and any(v is not None for v in dep):
        panels.append((f"Generation at {args.depth // 1024}K context", dep))

    for theme_name, th in THEMES.items():
        plt.rcParams.update({
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "svg.fonttype": "none",
        })
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 1.35 + 0.6 * len(labels)),
                                 facecolor=th["surface"])
        for ax, (title, values) in zip(axes, panels):
            ax.set_facecolor(th["surface"])
            ys = list(range(len(labels)))[::-1]
            vmax = max(v for v in values if v is not None) or 1.0
            for y, v in zip(ys, values):
                if v is None:
                    ax.text(vmax * 0.015, y, "not measured", va="center", ha="left",
                            color=th["muted"], fontsize=10)
                    continue
                ax.barh(y, v, height=0.55, color=th["bar"], zorder=2)
                ax.text(v + vmax * 0.015, y, f"{v:.1f}", va="center", ha="left",
                        color=th["text"], fontsize=10, fontweight="bold")
            ax.set_yticks(ys)
            ax.set_yticklabels(labels if ax is axes[0] else [""] * len(labels), color=th["text"])
            ax.set_xlim(0, vmax * 1.25)
            ax.set_xlabel("tokens / second", color=th["text2"], fontsize=10)
            ax.set_title(title, loc="left", color=th["text"], fontsize=12, fontweight="bold", pad=10)
            ax.tick_params(axis="x", colors=th["muted"], labelsize=9, length=0)
            ax.tick_params(axis="y", length=0, pad=8)
            ax.grid(axis="x", color=th["grid"], linewidth=0.8, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(th["grid"])

        fig.suptitle(args.title, x=0.01, ha="left", color=th["text"], fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out = f"{args.out}-{theme_name}.svg"
        fig.savefig(out, facecolor=th["surface"])
        if args.png:
            fig.savefig(out[:-4] + ".png", facecolor=th["surface"], dpi=110)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
