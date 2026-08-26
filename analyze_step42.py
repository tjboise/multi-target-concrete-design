"""
analyze_step42.py
=================
Step 4.2 analysis: prompt ablation study.

Outputs:
  results/figures/step42_dhv_table.csv   -- delta-HV summary table
  results/figures/step42_composition.png -- material composition: full vs w/o KT
  results/figures/step42_elite_effect.png -- frac_useful and diversity: full vs w/o elite
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from itertools import combinations

BASE = Path("results")
REPS = 30

GWP_MIN, GWP_MAX = 169.0, 534.5
STR_MIN, STR_MAX = 17.4, 106.1

CONDITIONS = {
    "base":     "grid_everyg_base_rep{:02d}",
    "full":     "grid_everyg_hyb_rep{:02d}",
    "no_obj":   "abl42_no_obj_rep{:02d}",
    "no_kt":    "abl42_no_kt_rep{:02d}",
    "no_con":   "abl42_no_con_rep{:02d}",
    "no_elite": "abl42_no_elite_rep{:02d}",
    "no_task":  "abl42_no_task_rep{:02d}",
}

LABELS = {
    "base":     "NSGA-II baseline",
    "full":     "Full prompt",
    "no_obj":   "w/o Objectives",
    "no_kt":    "w/o Knowledge Table",
    "no_con":   "w/o Constraints",
    "no_elite": "w/o Elite solutions",
    "no_task":  "w/o Task/Gap",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_hv(tpl, reps=REPS):
    out = []
    for r in range(1, reps + 1):
        p = BASE / tpl.format(r) / "metrics.csv"
        if p.exists():
            out.append(float(pd.read_csv(p)["HV_hybrid"].iloc[0]))
    return np.array(out)


def pairwise_div(pf_df):
    if len(pf_df) < 2:
        return np.nan
    g = (pf_df["GWP"].values   - GWP_MIN) / (GWP_MAX - GWP_MIN)
    s = 1 - (pf_df["28day"].values - STR_MIN) / (STR_MAX - STR_MIN)
    pts = np.column_stack([g, s])
    n   = len(pts)
    d   = [np.linalg.norm(pts[i] - pts[j]) for i in range(n) for j in range(i+1, n)]
    return np.mean(d)


def load_pareto_fronts(tpl, reps=REPS):
    out = []
    for r in range(1, reps + 1):
        p = BASE / tpl.format(r) / "pareto_front.csv"
        if p.exists():
            out.append(pd.read_csv(p))
    return out


def load_frac_useful(tpl, reps=REPS):
    out = []
    for r in range(1, reps + 1):
        p = BASE / tpl.format(r) / "llm_solutions.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["useful"] = (df["is_feasible"] == 1) & (df["is_dominated"] == 0)
            out.append(df["useful"].mean())
    return np.array(out)


OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. ΔHV summary table
# ══════════════════════════════════════════════════════════════════════════════
hv_base = load_hv(CONDITIONS["base"])

table_rows = []
for cond, tpl in CONDITIONS.items():
    if cond == "base":
        continue
    hvs = load_hv(tpl)
    n   = min(len(hvs), len(hv_base))
    dhv = hvs[:n] - hv_base[:n]
    w, pw = stats.wilcoxon(dhv, alternative="two-sided")
    sig   = "***" if pw < 0.001 else ("**" if pw < 0.01 else ("*" if pw < 0.05 else "ns"))
    pct   = int((dhv > 0).mean() * 100)
    table_rows.append({
        "Condition":       LABELS[cond],
        "Mean HV":         round(hvs[:n].mean(), 4),
        "Mean dHV":        round(dhv.mean(), 4),
        "SD dHV":          round(dhv.std(),  4),
        "p-value":         round(pw, 4),
        "Sig":             sig,
        "% reps improved": pct,
    })

df_table = pd.DataFrame(table_rows)
df_table.to_csv(OUTDIR / "step42_dhv_table.csv", index=False)
print("ΔHV table:")
print(df_table.to_string(index=False))
print(f"Saved step42_dhv_table.csv\n")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Material composition: full vs w/o KT
# ══════════════════════════════════════════════════════════════════════════════
MIX_VARS   = ["PC", "FA", "SC", "FAGG", "CAGG", "WATER"]
MIX_LABELS = {
    "PC":    "Portland Cement",
    "FA":    "Fly Ash",
    "SC":    "Slag Cement",
    "FAGG":  "Fine Aggregate",
    "CAGG":  "Coarse Aggregate",
    "WATER": "Water",
}

pfs_full  = load_pareto_fronts(CONDITIONS["full"])
pfs_nokt  = load_pareto_fronts(CONDITIONS["no_kt"])

def binder_stats(pf_list, var):
    """Mean proportion of <var> in binder (PC+FA+SC) per rep, then across reps."""
    vals = []
    for pf in pf_list:
        binder = pf["PC"] + pf["FA"] + pf["SC"]
        if var in ["PC", "FA", "SC"]:
            prop = (pf[var] / binder.replace(0, np.nan)).dropna()
        else:
            prop = pf[var]
        if len(prop):
            vals.append(prop.mean())
    return np.array(vals)

# Also compute w/b ratio
def wb_ratio(pf_list):
    vals = []
    for pf in pf_list:
        binder = pf["PC"] + pf["FA"] + pf["SC"]
        wb = (pf["WATER"] / binder.replace(0, np.nan)).dropna()
        if len(wb):
            vals.append(wb.mean())
    return np.array(vals)

print("Material composition comparison (full vs w/o KT):")
results_comp = {}
for var in ["SC", "FA", "PC"]:
    full_v = binder_stats(pfs_full, var)
    nokt_v = binder_stats(pfs_nokt, var)
    n = min(len(full_v), len(nokt_v))
    t, p = stats.ttest_rel(full_v[:n], nokt_v[:n])
    results_comp[var] = (full_v, nokt_v, p)
    print(f"  {var} binder frac: full={full_v.mean():.3f}  no_kt={nokt_v.mean():.3f}  p={p:.4f}")

wb_full = wb_ratio(pfs_full)
wb_nokt = wb_ratio(pfs_nokt)
n = min(len(wb_full), len(wb_nokt))
t_wb, p_wb = stats.ttest_rel(wb_full[:n], wb_nokt[:n])
print(f"  w/b ratio:      full={wb_full.mean():.3f}  no_kt={wb_nokt.mean():.3f}  p={p_wb:.4f}")
print()

fig, axes = plt.subplots(1, 4, figsize=(14, 5.5))
plot_vars = [("SC", "Slag Cement\nfraction"), ("FA", "Fly Ash\nfraction"),
             ("PC", "Portland Cement\nfraction"), ("wb", "w/b ratio")]

for ax, (var, ylabel) in zip(axes, plot_vars):
    if var == "wb":
        d_full, d_nokt = wb_full, wb_nokt
        p_val = p_wb
    else:
        d_full, d_nokt, p_val = results_comp[var]

    bp = ax.boxplot([d_full, d_nokt], patch_artist=True,
                    medianprops=dict(color="white", lw=2.2),
                    flierprops=dict(marker=".", ms=4, alpha=0.4),
                    whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3))
    colors = ["#1D4ED8", "#F59E0B"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.78); patch.set_edgecolor(c)
    for wh, c in zip(bp["whiskers"], [c for c in colors for _ in range(2)]):
        wh.set_color(c)
    for cap, c in zip(bp["caps"], [c for c in colors for _ in range(2)]):
        cap.set_color(c)

    rng = np.random.default_rng(42)
    for i, (arr, c) in enumerate(zip([d_full, d_nokt], colors)):
        ax.scatter(i+1+rng.uniform(-0.12, 0.12, len(arr)), arr,
                   s=16, color=c, alpha=0.50, zorder=3)

    sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Full\nprompt", "w/o KT"], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"{sig_str} (p={p_val:.3f})", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Material Composition: Full Prompt vs w/o Knowledge Table\n"
             "(Pareto front solutions, n=30 reps; paired t-test)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTDIR / "step42_composition.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved step42_composition.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Elite effect: frac_useful + diversity
# ══════════════════════════════════════════════════════════════════════════════
fu_full   = load_frac_useful(CONDITIONS["full"])
fu_noelite = load_frac_useful(CONDITIONS["no_elite"])

div_base_arr   = np.array([pairwise_div(pd.read_csv(BASE / CONDITIONS["base"].format(r) / "pareto_front.csv"))
                            for r in range(1, REPS+1)
                            if (BASE / CONDITIONS["base"].format(r) / "pareto_front.csv").exists()])
div_full_arr   = np.array([pairwise_div(pf) for pf in pfs_full])
div_noelite_arr = np.array([pairwise_div(pf)
                             for pf in load_pareto_fronts(CONDITIONS["no_elite"])])

print("Elite effect:")
n_fu = min(len(fu_full), len(fu_noelite))
_, p_fu = stats.wilcoxon(fu_full[:n_fu], fu_noelite[:n_fu])
print(f"  frac_useful: full={fu_full.mean():.4f}  no_elite={fu_noelite.mean():.4f}  p={p_fu:.4e}")

n_dv = min(len(div_full_arr), len(div_noelite_arr))
_, p_dv = stats.wilcoxon(div_full_arr[:n_dv], div_noelite_arr[:n_dv])
print(f"  diversity:   full={div_full_arr.mean():.4f}  no_elite={div_noelite_arr.mean():.4f}  p={p_dv:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5))

# Panel A: frac_useful
BLUE, AMBER, GRAY = "#1D4ED8", "#60A5FA", "#9CA3AF"
bp1 = ax1.boxplot([fu_full, fu_noelite], patch_artist=True,
                   medianprops=dict(color="white", lw=2.2),
                   flierprops=dict(marker=".", ms=4, alpha=0.4),
                   whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3))
colors1 = [BLUE, AMBER]
for patch, c in zip(bp1["boxes"], colors1):
    patch.set_facecolor(c); patch.set_alpha(0.78); patch.set_edgecolor(c)
for wh, c in zip(bp1["whiskers"], [c for c in colors1 for _ in range(2)]):
    wh.set_color(c)
for cap, c in zip(bp1["caps"], [c for c in colors1 for _ in range(2)]):
    cap.set_color(c)
rng = np.random.default_rng(42)
for i, (arr, c) in enumerate(zip([fu_full, fu_noelite], colors1)):
    ax1.scatter(i+1+rng.uniform(-0.12, 0.12, len(arr)), arr,
                s=16, color=c, alpha=0.50, zorder=3)
sig_fu = "***" if p_fu < 0.001 else "**" if p_fu < 0.01 else "*" if p_fu < 0.05 else "ns"
ax1.set_xticks([1, 2]); ax1.set_xticklabels(["Full prompt", "w/o Elite"], fontsize=10)
ax1.set_ylabel("Fraction of useful LLM solutions\n(feasible + non-dominated)", fontsize=9)
ax1.set_title(f"(A) LLM Proposal Quality\n{sig_fu} (p={p_fu:.1e})", fontsize=10)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
for i, (arr, c) in enumerate(zip([fu_full, fu_noelite], colors1)):
    ax1.text(i+1, arr.max() + 0.002, f"{arr.mean():.3f}", ha="center", va="bottom",
             fontsize=8.5, color=c, fontweight="bold")

# Panel B: Pareto diversity
bp2 = ax2.boxplot([div_base_arr, div_full_arr, div_noelite_arr], patch_artist=True,
                   medianprops=dict(color="white", lw=2.2),
                   flierprops=dict(marker=".", ms=4, alpha=0.4),
                   whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3))
colors2 = [GRAY, BLUE, AMBER]
for patch, c in zip(bp2["boxes"], colors2):
    patch.set_facecolor(c); patch.set_alpha(0.78); patch.set_edgecolor(c)
for wh, c in zip(bp2["whiskers"], [c for c in colors2 for _ in range(2)]):
    wh.set_color(c)
for cap, c in zip(bp2["caps"], [c for c in colors2 for _ in range(2)]):
    cap.set_color(c)
rng2 = np.random.default_rng(0)
for i, (arr, c) in enumerate(zip([div_base_arr, div_full_arr, div_noelite_arr], colors2)):
    ax2.scatter(i+1+rng2.uniform(-0.12, 0.12, len(arr)), arr,
                s=16, color=c, alpha=0.50, zorder=3)

_, p_dv_base_full = stats.wilcoxon(div_full_arr[:len(div_base_arr)],
                                    div_base_arr[:len(div_full_arr)])
_, p_dv_base_noelite = stats.wilcoxon(div_noelite_arr[:len(div_base_arr)],
                                       div_base_arr[:len(div_noelite_arr)])

ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(["NSGA-II\nbaseline", "Full\nprompt", "w/o Elite"], fontsize=9)
ax2.set_ylabel("Diversity (mean pairwise dist,\nnormalized obj. space)", fontsize=9)
ax2.set_title("(B) Pareto Front Diversity", fontsize=10)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
for i, (arr, c, p) in enumerate(zip(
        [div_base_arr, div_full_arr, div_noelite_arr],
        colors2,
        [None, p_dv_base_full, p_dv_base_noelite])):
    ypos = arr.max() + 0.005
    if p is not None:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax2.text(i+1, ypos, f"{arr.mean():.3f}\n({sig})",
                 ha="center", va="bottom", fontsize=8.5, color=c, fontweight="bold")
    else:
        ax2.text(i+1, ypos, f"{arr.mean():.3f}",
                 ha="center", va="bottom", fontsize=8.5, color=c, fontweight="bold")

fig.suptitle("Effect of Elite Solutions in Prompt (n=30 paired reps)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTDIR / "step42_elite_effect.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved step42_elite_effect.png")
print("\nStep 4.2 analysis complete.")
