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
# 2. Material composition: all conditions
# ══════════════════════════════════════════════════════════════════════════════

COMP_CONDITIONS = {
    "NSGA-II":         CONDITIONS["base"],
    "Full prompt":     CONDITIONS["full"],
    "w/o Objectives":  CONDITIONS["no_obj"],
    "w/o KT":          CONDITIONS["no_kt"],
    "w/o Constraints": CONDITIONS["no_con"],
    "w/o Elite":       CONDITIONS["no_elite"],
    "w/o Task/Gap":    CONDITIONS["no_task"],
}
COMP_COLORS = {
    "NSGA-II":         "#9CA3AF",
    "Full prompt":     "#1D4ED8",
    "w/o Objectives":  "#60A5FA",
    "w/o KT":          "#F59E0B",
    "w/o Constraints": "#34D399",
    "w/o Elite":       "#A78BFA",
    "w/o Task/Gap":    "#EF4444",
}

def binder_frac(pf_list, var):
    vals = []
    for pf in pf_list:
        binder = pf["PC"] + pf["FA"] + pf["SC"]
        prop = (pf[var] / binder.replace(0, np.nan)).dropna()
        if len(prop):
            vals.append(prop.mean())
    return np.array(vals)

def wb_ratio(pf_list):
    vals = []
    for pf in pf_list:
        binder = pf["PC"] + pf["FA"] + pf["SC"]
        wb = (pf["WATER"] / binder.replace(0, np.nan)).dropna()
        if len(wb):
            vals.append(wb.mean())
    return np.array(vals)

comp_data = {}
for name, tpl in COMP_CONDITIONS.items():
    pfs = load_pareto_fronts(tpl)
    comp_data[name] = {
        "SC": binder_frac(pfs, "SC"),
        "FA": binder_frac(pfs, "FA"),
        "PC": binder_frac(pfs, "PC"),
        "wb": wb_ratio(pfs),
    }

ref = comp_data["Full prompt"]

print("Material composition (all conditions):")
def sig(p): return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
for name, d in comp_data.items():
    sc = d["SC"].mean()*100; fa = d["FA"].mean()*100
    pc = d["PC"].mean()*100; wb = d["wb"].mean()
    if name in ("Full prompt", "NSGA-II"):
        print(f"  {name:<22}: SC={sc:.1f}%  FA={fa:.1f}%  PC={pc:.1f}%  w/b={wb:.3f}")
    else:
        n = min(len(d["SC"]), len(ref["SC"]))
        _, p_sc = stats.ttest_rel(d["SC"][:n], ref["SC"][:n])
        _, p_fa = stats.ttest_rel(d["FA"][:n], ref["FA"][:n])
        _, p_pc = stats.ttest_rel(d["PC"][:n], ref["PC"][:n])
        _, p_wb = stats.ttest_rel(d["wb"][:n], ref["wb"][:n])
        print(f"  {name:<22}: SC={sc:.1f}%({sig(p_sc)})  FA={fa:.1f}%({sig(p_fa)})  PC={pc:.1f}%({sig(p_pc)})  w/b={wb:.3f}({sig(p_wb)})")
print()

# grouped bar chart: 4 properties × 7 conditions
prop_keys   = ["SC", "FA", "PC", "wb"]
prop_labels = ["Slag cement\nbinder fraction", "Fly ash\nbinder fraction",
               "Portland cement\nbinder fraction", "w/b ratio"]
prop_scale  = [100, 100, 100, 1]   # SC/FA/PC as %, wb as raw

cond_names = list(COMP_CONDITIONS.keys())
x = np.arange(len(cond_names))
width = 0.6

fig, axes = plt.subplots(1, 4, figsize=(16, 5.5), sharey=False)
rng = np.random.default_rng(42)

for ax, prop, ylabel, scale in zip(axes, prop_keys, prop_labels, prop_scale):
    ref_val = comp_data["Full prompt"][prop].mean() * scale
    ax.axhline(ref_val, color="#1D4ED8", lw=1.2, ls="--", alpha=0.5, zorder=1)

    for i, name in enumerate(cond_names):
        d   = comp_data[name][prop]
        col = COMP_COLORS[name]
        mean_v = d.mean() * scale

        bp = ax.boxplot([d * scale], positions=[i], widths=0.55, patch_artist=True,
                        medianprops=dict(color="white", lw=2.0),
                        flierprops=dict(marker=".", ms=3, alpha=0.3),
                        whiskerprops=dict(lw=1.1), capprops=dict(lw=1.1))
        bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][0].set_edgecolor(col)
        bp["whiskers"][0].set_color(col); bp["whiskers"][1].set_color(col)
        bp["caps"][0].set_color(col);     bp["caps"][1].set_color(col)

        jitter = rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(i + jitter, d * scale, s=12, color=col, alpha=0.45, zorder=3)

        # significance marker vs Full prompt
        if name not in ("Full prompt", "NSGA-II"):
            n  = min(len(d), len(ref[prop]))
            _, p = stats.ttest_rel(d[:n], ref[prop][:n])
            s  = sig(p)
            if s != "ns":
                ypos = (d * scale).max() + (scale * 0.005 if scale == 1 else 0.3)
                ax.text(i, ypos, s, ha="center", va="bottom", fontsize=7.5,
                        color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in cond_names], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Pareto Front Material Composition: All Ablation Conditions\n"
             "(mean per-rep; dashed line = Full prompt mean; markers = significance vs Full prompt)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTDIR / "step42_composition.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved step42_composition.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Elite effect: frac_useful + diversity
# ══════════════════════════════════════════════════════════════════════════════
fu_full   = load_frac_useful(CONDITIONS["full"])
fu_noelite = load_frac_useful(CONDITIONS["no_elite"])

pfs_base_   = load_pareto_fronts(CONDITIONS["base"])
pfs_full_   = load_pareto_fronts(CONDITIONS["full"])
pfs_noelite_= load_pareto_fronts(CONDITIONS["no_elite"])

div_base_arr    = np.array([pairwise_div(pf) for pf in pfs_base_])
div_full_arr    = np.array([pairwise_div(pf) for pf in pfs_full_])
div_noelite_arr = np.array([pairwise_div(pf) for pf in pfs_noelite_])

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
