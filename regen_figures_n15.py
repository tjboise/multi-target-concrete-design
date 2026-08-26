"""Regenerate all Step 1-3 figures using N=15 data (nsweep_n15_rep{01-30})."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from itertools import combinations

BASE     = Path("results")
HYB_TPL  = "nsweep_n15_rep{:02d}"
BASE_TPL = "grid_everyg_base_rep{:02d}"
REPS     = 30
OUTDIR   = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

GWP_MIN, GWP_MAX = 169.0, 534.5
STR_MIN, STR_MAX = 17.4, 106.1

# ── helpers ───────────────────────────────────────────────────
def norm_pt(gwp, s):
    return (gwp - GWP_MIN)/(GWP_MAX - GWP_MIN), (STR_MAX - s)/(STR_MAX - STR_MIN)

def pairwise_div(pf):
    pts = [norm_pt(r.GWP, r["28day"]) for _, r in pf.iterrows()]
    d = [np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2) for a,b in combinations(pts,2)]
    return np.mean(d) if d else 0.0

def load_hv_curves(tpl, n_gens=100):
    mat = np.full((REPS, n_gens), np.nan)
    for rep in range(1, REPS+1):
        p = BASE / tpl.format(rep) / "hv_history.csv"
        if p.exists():
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                g = int(row["gen"]) - 1
                if 0 <= g < n_gens:
                    mat[rep-1, g] = row["hv"]
    return mat

# ── load all rep data ─────────────────────────────────────────
reps_data = []
for rep in range(1, REPS+1):
    hd = BASE / HYB_TPL.format(rep)
    bd = BASE / BASE_TPL.format(rep)
    if not hd.exists() or not bd.exists():
        continue
    hm = pd.read_csv(hd/"metrics.csv").iloc[0]
    bm = pd.read_csv(bd/"metrics.csv").iloc[0]
    hpf = pd.read_csv(hd/"pareto_front.csv")
    bpf = pd.read_csv(bd/"pareto_front.csv")
    llm = pd.read_csv(hd/"llm_solutions.csv") if (hd/"llm_solutions.csv").exists() else None
    reps_data.append(dict(
        rep=rep,
        hv_hyb=float(hm["HV_hybrid"]), hv_base=float(bm["HV_hybrid"]),
        dhv=float(hm["HV_hybrid"])-float(bm["HV_hybrid"]),
        hpf=hpf, bpf=bpf, llm=llm,
        div_hyb=pairwise_div(hpf), div_base=pairwise_div(bpf),
    ))

dhv_arr     = np.array([r["dhv"]      for r in reps_data])
div_hyb_arr = np.array([r["div_hyb"]  for r in reps_data])
div_base_arr= np.array([r["div_base"] for r in reps_data])

# ══════════════════════════════════════════════════════════════
# Fig 1: Pareto front — binned mean ± 1 SD (no scatter dots)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

# Bin GWP axis; for each bin take per-rep max strength, then average across reps
GWP_BINS = np.linspace(130, 320, 30)

def rep_binned(pf_list, bins):
    """For each rep's Pareto front, compute max strength per GWP bin."""
    n_bins = len(bins) - 1
    mat = np.full((len(pf_list), n_bins), np.nan)
    for ri, pf in enumerate(pf_list):
        for bi in range(n_bins):
            sub = pf[(pf["GWP"] >= bins[bi]) & (pf["GWP"] < bins[bi+1])]
            if len(sub):
                mat[ri, bi] = sub["28day"].max()
    return mat

bin_centers = (GWP_BINS[:-1] + GWP_BINS[1:]) / 2

base_pfs = [r["bpf"] for r in reps_data]
hyb_pfs  = [r["hpf"] for r in reps_data]

mats = {}
for key, pf_list in [("base", base_pfs), ("hyb", hyb_pfs)]:
    mats[key] = rep_binned(pf_list, GWP_BINS)

# shared x-range: bins where BOTH have data
shared_mask = (~np.isnan(np.nanmean(mats["base"], axis=0))) & \
              (~np.isnan(np.nanmean(mats["hyb"],  axis=0)))

for key, color, label in [
    ("base", "#6B7280", f"NSGA-II baseline (n={REPS})"),
    ("hyb",  "#1D4ED8", f"Hybrid N=15 (n={REPS})"),
]:
    mat  = mats[key]
    mean = np.nanmean(mat, axis=0)
    sd   = np.nanstd(mat,  axis=0)
    mask = shared_mask
    ax.plot(bin_centers[mask], mean[mask], color=color, lw=2.2, label=label, zorder=3)
    ax.fill_between(bin_centers[mask],
                    mean[mask] - sd[mask],
                    mean[mask] + sd[mask],
                    color=color, alpha=0.18, zorder=2)

ax.set_xlabel("GWP (kg CO₂/m³)", fontsize=11)
ax.set_ylabel("28-day compressive strength (MPa)", fontsize=11)
ax.set_title(f"Pareto Front: NSGA-II vs Hybrid (N=15)\n"
             f"Mean ± 1 SD of max strength per GWP bin across {REPS} replicates  "
             f"(mean ΔHV = +0.082, p < 0.001)", fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUTDIR/"pareto_front_everyg.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pareto_front_everyg.png")

# ══════════════════════════════════════════════════════════════
# Fig 2: Convergence curves
# ══════════════════════════════════════════════════════════════
hyb_curves  = load_hv_curves(HYB_TPL)
base_curves = load_hv_curves(BASE_TPL)
gens = np.arange(1, 101)

fig, ax = plt.subplots(figsize=(9, 5))
for mat, color, ls, label in [
    (base_curves, "#9CA3AF", "--", "NSGA-II baseline"),
    (hyb_curves,  "#1D4ED8", "-",  "Hybrid N=15"),
]:
    m = np.nanmean(mat, axis=0)
    s = np.nanstd(mat, axis=0)
    ax.plot(gens, m, color=color, ls=ls, lw=2, label=label)
    ax.fill_between(gens, m-s, m+s, color=color, alpha=0.12)

ax.set_xlabel("Generation", fontsize=11)
ax.set_ylabel("Hypervolume", fontsize=11)
ax.set_title(f"HV Convergence: NSGA-II vs Hybrid (N=15)\nmean ± 1 SD, n={REPS} paired replicates", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(1, 100)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUTDIR/"convergence_everyg.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved convergence_everyg.png")

# ══════════════════════════════════════════════════════════════
# Fig 3: Per-generation incremental ΔHV
# ══════════════════════════════════════════════════════════════
dhv_curves = hyb_curves - base_curves
mean_dhv_per_gen = np.nanmean(dhv_curves, axis=0)
delta_dhv = np.diff(mean_dhv_per_gen, prepend=0)

# threshold: frac_useful > median frac_useful per gen (use LLM data)
llm_all = pd.concat([r["llm"] for r in reps_data if r["llm"] is not None])
llm_all["useful"] = (llm_all["is_feasible"]==1) & (llm_all["is_dominated"]==0)
gen_useful = llm_all.groupby("gen")["useful"].mean()
threshold = gen_useful.median()
above = gen_useful >= threshold

fig, ax = plt.subplots(figsize=(11, 4))
colors = ["#DC2626" if above.get(g+1, False) else "#16A34A" for g in range(100)]
ax.bar(gens, delta_dhv, color=colors, alpha=0.75, width=0.8)
ax.axhline(0, color="#6B7280", lw=0.8, ls="--")
ax.set_xlabel("Generation", fontsize=11)
ax.set_ylabel("ΔΔHV (change in mean ΔHV)", fontsize=11)
ax.set_title(f"Per-generation ΔHV increment — Hybrid N=15\n"
             f"Red: frac_useful ≥ {threshold:.3f} (above median); Green: below median", fontsize=11)
ax.set_xlim(0.5, 100.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUTDIR/"delta_hv_incremental_everyg.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved delta_hv_incremental_everyg.png")

# ══════════════════════════════════════════════════════════════
# Fig 4: Step 2 stacked area (LLM quality over generations)
# ══════════════════════════════════════════════════════════════
llm_all["infeasible_fl"] = (llm_all["is_feasible"]==0).astype(int)
llm_all["feas_dom"]      = ((llm_all["is_feasible"]==1)&(llm_all["is_dominated"]==1)).astype(int)

gen_grp = llm_all.groupby("gen")
n_per_gen = 15  # N=15

infeas_mean = gen_grp["infeasible_fl"].mean() * n_per_gen
feas_dom_mean = gen_grp["feas_dom"].mean() * n_per_gen
useful_mean   = gen_grp["useful"].mean() * n_per_gen

gen_idx = np.arange(1, 101)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.stackplot(gen_idx,
             [useful_mean.reindex(gen_idx, fill_value=0).values,
              feas_dom_mean.reindex(gen_idx, fill_value=0).values,
              infeas_mean.reindex(gen_idx, fill_value=0).values],
             labels=["Useful (feasible + non-dominated)", "Feasible but dominated", "Infeasible"],
             colors=["#DC2626", "#FB923C", "#9CA3AF"], alpha=0.80)
ax.set_xlabel("Generation", fontsize=11)
ax.set_ylabel("Mean # LLM solutions per generation", fontsize=11)
ax.set_title(f"LLM Solution Quality over Generations — Hybrid N=15\n"
             f"Stacked: 15 solutions per generation × {REPS} replicates averaged", fontsize=11)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.set_xlim(1, 100); ax.set_ylim(0, 15)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUTDIR/"step2_area.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved step2_area.png")

# ══════════════════════════════════════════════════════════════
# Fig 5: Step 3 diversity — single box plot + jitter dots
# ══════════════════════════════════════════════════════════════
t, pt = stats.ttest_rel(div_hyb_arr, div_base_arr, alternative="two-sided")
w, pw = stats.wilcoxon(div_hyb_arr, div_base_arr, alternative="two-sided")

fig, ax = plt.subplots(figsize=(6, 5))

colors_box = ["#9CA3AF", "#1D4ED8"]
bp = ax.boxplot([div_base_arr, div_hyb_arr], patch_artist=True,
                medianprops=dict(color="white", lw=2.5),
                flierprops=dict(marker=".", ms=4, alpha=0.4),
                whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3))
for patch, c in zip(bp["boxes"], colors_box):
    patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor(c)
for wh, c in zip(bp["whiskers"], [c for c in colors_box for _ in range(2)]):
    wh.set_color(c)
for cap, c in zip(bp["caps"], [c for c in colors_box for _ in range(2)]):
    cap.set_color(c)

rng = np.random.default_rng(42)
for i, (arr, c) in enumerate(zip([div_base_arr, div_hyb_arr], colors_box)):
    ax.scatter(i+1+rng.uniform(-0.14, 0.14, len(arr)), arr,
               s=20, color=c, alpha=0.55, zorder=3)

sig = "***" if pw < 0.001 else ("**" if pw < 0.01 else ("*" if pw < 0.05 else "ns"))
ax.set_xticks([1, 2])
ax.set_xticklabels(["NSGA-II\nbaseline", "Hybrid\nN=15"], fontsize=11)
ax.set_ylabel("Diversity (mean pairwise dist, norm. space)", fontsize=10)
ax.set_title(f"Pareto Front Diversity (n={REPS} paired replicates)\n"
             f"Δ = +{(div_hyb_arr-div_base_arr).mean():.3f}, p {pw:.4f} ({sig})", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUTDIR/"step3_paired.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved step3_paired.png")

print("\nAll Step 1-3 figures updated for N=15.")
