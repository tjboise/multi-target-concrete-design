"""
Diversity box plot: NSGA-II baseline vs Hybrid (full) vs w/o Elite vs w/o KT vs w/o Task.
Diversity = mean pairwise Euclidean distance in normalized objective space.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE = Path("results")
REPS = 30

# Normalization bounds (from README)
GWP_MIN, GWP_MAX   = 169, 534.5
STR_MIN, STR_MAX   = 17.4, 106.1


def normalize(gwp, strength):
    g = (gwp - GWP_MIN) / (GWP_MAX - GWP_MIN)
    s = 1 - (strength - STR_MIN) / (STR_MAX - STR_MIN)   # flip: lower strength = worse
    return np.column_stack([g, s])


def pairwise_diversity(pf_df):
    """Mean pairwise Euclidean distance in normalized objective space."""
    if len(pf_df) < 2:
        return np.nan
    pts = normalize(pf_df["GWP"].values, pf_df["28day"].values)
    n = len(pts)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(pts[i] - pts[j]))
    return np.mean(dists)


def load_diversity(tpl, reps=REPS):
    divs = []
    for rep in range(1, reps + 1):
        p = BASE / tpl.format(rep) / "pareto_front.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "GWP" in df.columns and "28day" in df.columns:
                divs.append(pairwise_diversity(df))
    return np.array([d for d in divs if not np.isnan(d)])


CONDITIONS = {
    "NSGA-II\nbaseline":   ("grid_everyg_base_rep{:02d}", REPS),
    "Hybrid\n(full prompt)": ("grid_everyg_hyb_rep{:02d}", REPS),
    "w/o Elite":           ("abl42_no_elite_rep{:02d}", REPS),
    "w/o KT":              ("abl42_no_kt_rep{:02d}", REPS),
    "w/o Task\n& gap":     ("abl42_no_task_rep{:02d}", REPS),
}

COLORS = {
    "NSGA-II\nbaseline":     "#9CA3AF",
    "Hybrid\n(full prompt)": "#1D4ED8",
    "w/o Elite":             "#60A5FA",
    "w/o KT":                "#F59E0B",
    "w/o Task\n& gap":       "#EF4444",
}

data = {}
for label, (tpl, reps) in CONDITIONS.items():
    d = load_diversity(tpl, reps)
    data[label] = d
    print(f"{label.replace(chr(10), ' '):25s}  n={len(d)}  mean={d.mean():.4f}  sd={d.std():.4f}")

# ── significance vs baseline ───────────────────────────────────
base = data["NSGA-II\nbaseline"]
ref  = data["Hybrid\n(full prompt)"]

print(f"\n{'Condition':25s}  {'vs baseline':>12}  {'vs hybrid full':>14}")
print("-" * 55)
for label, d in data.items():
    if label == "NSGA-II\nbaseline":
        continue
    _, p_vs_base = stats.wilcoxon(d[:len(base)], base[:len(d)], alternative="two-sided")
    if label != "Hybrid\n(full prompt)":
        _, p_vs_hyb  = stats.wilcoxon(d[:len(ref)], ref[:len(d)], alternative="two-sided")
    else:
        p_vs_hyb = None

    def sig(p): return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    hyb_str = sig(p_vs_hyb) if p_vs_hyb is not None else "  —"
    print(f"  {label.replace(chr(10), ' '):23s}  {sig(p_vs_base):>12}  {hyb_str:>14}")

# ── plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

labels = list(data.keys())
pos    = np.arange(len(labels))
dhv_list = [data[l] for l in labels]

bp = ax.boxplot(
    dhv_list, positions=pos, widths=0.5, patch_artist=True,
    medianprops=dict(color="white", linewidth=2.5),
    flierprops=dict(marker=".", markersize=4, alpha=0.4),
    whiskerprops=dict(linewidth=1.3),
    capprops=dict(linewidth=1.3),
)

for patch, label in zip(bp["boxes"], labels):
    c = COLORS[label]
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
    patch.set_edgecolor(c)

for whisker, color in zip(bp["whiskers"], [c for l in labels for c in [COLORS[l]] * 2]):
    whisker.set_color(color)
for cap, color in zip(bp["caps"], [c for l in labels for c in [COLORS[l]] * 2]):
    cap.set_color(color)

# jitter strip
rng = np.random.default_rng(42)
for i, (label, d) in enumerate(zip(labels, dhv_list)):
    jitter = rng.uniform(-0.15, 0.15, len(d))
    ax.scatter(i + jitter, d, s=18, color=COLORS[label], alpha=0.50, zorder=3)

# annotate mean + significance vs baseline
for i, (label, d) in enumerate(zip(labels, dhv_list)):
    m = d.mean()
    if label != "NSGA-II\nbaseline":
        paired_n = min(len(d), len(base))
        _, p = stats.wilcoxon(d[:paired_n], base[:paired_n], alternative="two-sided")
        def sig(p): return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(i, d.max() + 0.005, f"{m:.3f}\n({sig(p)})",
                ha="center", va="bottom", fontsize=8.5, color=COLORS[label], fontweight="bold")
    else:
        ax.text(i, d.max() + 0.005, f"{m:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=COLORS[label], fontweight="bold")

ax.set_xticks(pos)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Diversity (mean pairwise distance, normalized obj. space)", fontsize=10)
ax.set_title("Pareto Front Diversity: Ablation Comparison\n"
             "(significance vs NSGA-II baseline; n=30 paired reps)",
             fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axhline(base.mean(), color="#9CA3AF", ls="--", lw=0.9, alpha=0.6)

plt.tight_layout()
out = "results/figures/ablation_diversity.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
