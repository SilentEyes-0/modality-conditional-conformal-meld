"""Build figures from the final test summaries.

Usage (from the project root):
    python -m src.report.make_figures --dir outputs_amendment --out report_out

Figure 6: coverage by pattern. Figure 7: set size by pattern. Figure 8: coverage against set size over alpha.
Figure 9: coverage under corruption (matched and mismatched calibration).
Reliability diagrams are not produced: the summary files hold no binned confidence data.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import load, PATTERN_ORDER, MAIN_METHODS, split_condition

COL = {"conditional": "#1f3864", "global_111": "#c0504d", "global_pooled": "#7f7f7f"}
LAB = {"conditional": "Conditional", "global_111": "Global-111", "global_pooled": "Global-pooled"}


def _clean_a(s, task, alpha, variant):
    return s[(s.design == "A") & (s.condition == "clean") & (s.task == task) & (s.variant == variant)
             & (s.alpha.round(4) == alpha) & (s.method.isin(MAIN_METHODS)) & (s.pattern.isin(PATTERN_ORDER))]


def fig_bars(s, out, col, name, ylabel, task, alpha, variant, target=None):
    x = _clean_a(s, task, alpha, variant)
    if x.empty:
        print("skip", name); return
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.26
    for i, m in enumerate(MAIN_METHODS):
        r = x[x.method == m].set_index("pattern").reindex(PATTERN_ORDER)
        if col == "coverage":
            yerr = np.vstack([(r.coverage - r.cov_lo).clip(lower=0), (r.cov_hi - r.coverage).clip(lower=0)])
        else:
            yerr = r["size_sd"]
        ax.bar(np.arange(7) + (i - 1) * w, r[col], w, yerr=yerr, capsize=2, color=COL[m], label=LAB[m])
    if target is not None:
        ax.axhline(target, color="black", ls="--", lw=1, label=f"Target {target:.2f}")
    ax.set_xticks(range(7)); ax.set_xticklabels(PATTERN_ORDER)
    ax.set_xlabel("Availability pattern (text, audio, video)"); ax.set_ylabel(ylabel)
    if col == "coverage":
        ax.set_ylim(max(0.6, float(x.coverage.min()) - 0.1), 1.0)
    ax.set_title(f"{task.capitalize()}, alpha = {alpha}, clean test data")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}_{task}.png"), dpi=200); plt.close(fig)
    print("wrote", name, task)


def fig_tradeoff(s, out, task, variant):
    base = s[(s.design == "A") & (s.condition == "clean") & (s.task == task) & (s.variant == variant)
             & (s.method.isin(MAIN_METHODS)) & (s.pattern.isin(PATTERN_ORDER))]
    if base.empty:
        print("skip Figure 8"); return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    style = {"conditional": ("o", 10, "-"), "global_111": ("s", 7, "-"), "global_pooled": ("^", 5, "-")}
    for m in MAIN_METHODS:
        g = base[base.method == m].groupby(base.alpha.round(4))[["coverage", "size"]].mean().sort_index()
        mk, ms, ls = style[m]
        ax.plot(g["size"], g["coverage"], ls, marker=mk, markersize=ms, markerfacecolor="none", markeredgewidth=1.5,
                color=COL[m], label=LAB[m])
        if m == "conditional":  # label each alpha once, below the curve, to avoid overlapping text
            for a, r in g.iterrows():
                ax.annotate(f"alpha {a:g}", (r["size"], r["coverage"]), textcoords="offset points", xytext=(6, -14), fontsize=7)
    ax.set_xlabel("Mean set size (unweighted mean over patterns)")
    ax.set_ylabel("Mean coverage (unweighted mean over patterns)")
    ax.set_title(f"{task.capitalize()}: coverage against set size")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(out, f"Figure8_tradeoff_{task}.png"), dpi=200); plt.close(fig)
    print("wrote Figure 8", task)


def fig_corruption(s, out, task, alpha, variant):
    x = s[(s.design == "A") & (s.task == task) & (s.variant == variant) & (s.alpha.round(4) == alpha)
          & (s.method.isin(["conditional", "global_pooled"])) & (s.pattern.isin(PATTERN_ORDER))].copy()
    if not (x.condition != "clean").any():
        print("skip Figure 9"); return
    x["cor"], x["cal"] = zip(*x.condition.map(split_condition))
    panels = [("audio", 1, [("audio_snr20", "20 dB"), ("audio_snr10", "10 dB"), ("audio_snr0", "0 dB")]),
              ("text", 0, [("text_drop10", "drop 10%"), ("text_drop30", "drop 30%")])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (mod, bit, levels) in zip(axes, panels):
        pats = [p for p in PATTERN_ORDER if p[bit] == "1"]
        sub = x[x.pattern.isin(pats)]
        labels = ["clean"] + [l for _, l in levels]
        for m, ls in (("conditional", "-"), ("global_pooled", "--")):
            for cal, mk in (("matched", "o"), ("mismatched", "s")):
                ys = [sub[(sub.condition == "clean") & (sub.method == m)].coverage.mean()]
                for name, _ in levels:
                    ys.append(sub[(sub.cor == name) & (sub.cal == cal) & (sub.method == m)].coverage.mean())
                ax.plot(range(len(labels)), ys, ls=ls, marker=mk, color=COL[m], alpha=0.9 if cal == "matched" else 0.55,
                        label=f"{LAB[m]}, {cal}")
        ax.axhline(1 - alpha, color="black", ls=":", lw=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_title(f"{mod.capitalize()} corruption (patterns with {mod}: {', '.join(pats)})", fontsize=9)
        ax.set_xlabel("Corruption level")
    axes[0].set_ylabel("Mean coverage over those patterns")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{task.capitalize()}, alpha = {alpha}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"Figure9_corruption_{task}.png"), dpi=200); plt.close(fig)
    print("wrote Figure 9", task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs_amendment")
    ap.add_argument("--out", default="report_out")
    ap.add_argument("--variant", default="T")
    a = ap.parse_args()
    s, c, w = load(a.dir)
    out = os.path.join(a.out, "figures"); os.makedirs(out, exist_ok=True)
    for task in ("emotion", "sentiment"):
        fig_bars(s, out, "coverage", "Figure6_coverage_by_pattern", "Coverage", task, 0.10, a.variant, target=0.90)
        fig_bars(s, out, "size", "Figure7_set_size_by_pattern", "Mean set size (labels)", task, 0.10, a.variant)
        fig_tradeoff(s, out, task, a.variant)
        fig_corruption(s, out, task, 0.10, a.variant)
    print("done ->", out)


if __name__ == "__main__":
    main()
