"""
analyze_step41.py
=================
Step 4.1 figure: Pareto front + convergence curve for 4 methods at N=15.
  - NSGA-II baseline (grid_everyg_base_rep*)
  - Hybrid Augment N=15 (nsweep_n15_rep*)
  - Pure LLM N=15 (abl41_purellm_n15_rep*)
  - Hybrid Replace N=15 (abl41_replace_n15_rep*)

Outputs:
  results/figures/step41_pareto.png
  results/figures/step41_convergence.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE = Path("results")
REPS = 30
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

GWP_MIN, GWP_MAX = 169.0, 534.5
STR_MIN, STR_MAX = 17.4, 106.1
GWP_BINS = np.linspace(130, 325, 40)

METHODS = {
    "nsga2":   ("NSGA-II baseline",    "grid_everyg_base_rep{:02d}",      "#9CA3AF", "--"),
    "augment": ("Hybrid Augment N=15", "nsweep_n15_rep{:02d}",            "#1D4ED8", "-"),
    "replace": ("Hybrid Replace N=15", "abl41_replace_n15_rep{:02d}",     "#7C3AED", "-"),
    "purellm": ("Pure LLM N=15",       "abl41_purellm_n15_rep{:02d}",     "#DC2626", "-"),
}


def load_final_hv(tpl):
    return np.array([
        float(pd.read_csv(BASE / tpl.format(r) / "metrics.csv")["HV_hybrid"].iloc[0])
        for r in range(1, REPS+1)
        if (BASE / tpl.format(r) / "metrics.csv").exists()
    ])


def load_hv_curves(tpl):
    mat = np.full((REPS, 100), np.nan)
    for rep in range(1, REPS+1):
        p = BASE / tpl.format(rep) / "hv_history.csv"
        if p.exists():
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                g = int(row["gen"]) - 1
                if 0 <= g < 100:
                    mat[rep-1, g] = row["hv"]
    return mat


def rep_binned(pf_list, bins):
    n_bins = len(bins) - 1
    mat = np.full((len(pf_list), n_bins), np.nan)
    for ri, pf in enumerate(pf_list):
        for bi in range(n_bins):
            sub = pf[(pf["GWP"] >= bins[bi]) & (pf["GWP"] < bins[bi+1])]
            if len(sub):
                mat[ri, bi] = sub["28day"].max()
    return mat


def load_pareto_fronts(tpl):
    return [
        pd.read_csv(BASE / tpl.format(r) / "pareto_front.csv")
        for r in range(1, REPS+1)
        if (BASE / tpl.format(r) / "pareto_front.csv").exists()
    ]


# Check data availability
print("Data availability:")
for key, (label, tpl, color, ls) in METHODS.items():
    n = sum(1 for r in range(1, REPS+1) if (BASE / tpl.format(r) / "metrics.csv").exists())
    print(f"  {label}: {n}/30 reps")

# ── Figure 1: Pareto front ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
bin_centers = (GWP_BINS[:-1] + GWP_BINS[1:]) / 2

mats = {}
for key, (label, tpl, color, ls) in METHODS.items():
    pfs = load_pareto_fronts(tpl)
    if pfs:
        mats[key] = rep_binned(pfs, GWP_BINS)

# each method plotted over its own full range — no clipping
if mats:
    for key, (label, tpl, color, ls) in METHODS.items():
        if key not in mats:
            continue
        mat  = mats[key]
        mean = np.nanmean(mat, axis=0)
        sd   = np.nanstd(mat,  axis=0)
        mask = ~np.isnan(mean)
        ax.plot(bin_centers[mask], mean[mask],
                color=color, ls=ls, lw=2.2, label=label, zorder=3)
        ax.fill_between(bin_centers[mask],
                        mean[mask] - sd[mask],
                        mean[mask] + sd[mask],
                        color=color, alpha=0.14, zorder=2)

ax.set_xlabel("GWP (kg CO2/m3)", fontsize=11)
ax.set_ylabel("28-day compressive strength (MPa)", fontsize=11)
ax.set_title("Pareto Front: Method Comparison (N=15)\n"
             "Mean +/- 1 SD of max strength per GWP bin, n=30 replicates", fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUTDIR / "step41_pareto.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved step41_pareto.png")

# ── Figure 2: Convergence curves ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
gens = np.arange(1, 101)

for key, (label, tpl, color, ls) in METHODS.items():
    mat = load_hv_curves(tpl)
    if np.all(np.isnan(mat)):
        continue
    m = np.nanmean(mat, axis=0)
    s = np.nanstd(mat,  axis=0)
    ax.plot(gens, m, color=color, ls=ls, lw=2.0, label=label)
    ax.fill_between(gens, m-s, m+s, color=color, alpha=0.10)

ax.set_xlabel("Generation", fontsize=11)
ax.set_ylabel("Hypervolume", fontsize=11)
ax.set_title("HV Convergence: Method Comparison (N=15)\nmean +/- 1 SD, n=30 paired replicates", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(1, 100)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUTDIR / "step41_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved step41_convergence.png")

# ── Stats summary ─────────────────────────────────────────────────────────────
hv_base = load_final_hv(METHODS["nsga2"][1])
print(f"\n{'Method':25s}  {'n':>3}  {'Mean HV':>8}  {'Mean dHV':>9}  {'p (Wilcoxon)':>14}  {'%pos':>5}")
print("-" * 70)
print(f"{'NSGA-II baseline':25s}  {len(hv_base):>3}  {hv_base.mean():>8.4f}  {'—':>9}  {'—':>14}  {'—':>5}")
for key in ["purellm", "replace", "augment"]:
    label, tpl, _, _ = METHODS[key]
    hvs = load_final_hv(tpl)
    if len(hvs) == 0:
        print(f"  {label:23s}: no data")
        continue
    n   = min(len(hvs), len(hv_base))
    dhv = hvs[:n] - hv_base[:n]
    _, pw = stats.wilcoxon(dhv, alternative="two-sided")
    sig = "***" if pw < 0.001 else "**" if pw < 0.01 else "*" if pw < 0.05 else "ns"
    pct = int((dhv > 0).mean() * 100)
    print(f"  {label:23s}  {n:>3}  {hvs[:n].mean():>8.4f}  {dhv.mean():>+9.4f}  {pw:>8.4f} {sig:<5}  {pct:>4}%")
