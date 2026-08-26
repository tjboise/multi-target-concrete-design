"""N-sweep final analysis: N=5,10,15,20,25 each 30 reps."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE = Path("results")
BASE_TPL = "grid_everyg_base_rep{:02d}"

N_TPLS = {
    5:  "grid_everyg_hyb_rep{:02d}",
    10: "nsweep_n10_rep{:02d}",
    15: "nsweep_n15_rep{:02d}",
    20: "nsweep_n20_rep{:02d}",
    25: "nsweep_n25_rep{:02d}",
}
N_VALS = [5, 10, 15, 20, 25]

hv_base = np.array([
    float(pd.read_csv(BASE / BASE_TPL.format(r) / "metrics.csv")["HV_hybrid"].iloc[0])
    for r in range(1, 31)
])

def load_final_hv(tpl, reps=30):
    out = []
    for rep in range(1, reps + 1):
        p = BASE / tpl.format(rep) / "metrics.csv"
        if p.exists():
            out.append(float(pd.read_csv(p)["HV_hybrid"].iloc[0]))
    return np.array(out)

def load_hv_curves(tpl, reps=30, n_gens=100):
    mat = np.full((reps, n_gens), np.nan)
    for rep in range(1, reps + 1):
        p = BASE / tpl.format(rep) / "hv_history.csv"
        if p.exists():
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                g = int(row["gen"]) - 1
                if 0 <= g < n_gens:
                    mat[rep - 1, g] = row["hv"]
    return mat

data = {}
for n, tpl in N_TPLS.items():
    hvs   = load_final_hv(tpl)
    dhv   = hvs - hv_base[:len(hvs)]
    curves = load_hv_curves(tpl)
    data[n] = dict(hvs=hvs, dhv=dhv, curves=curves)

# palette: light -> dark blue, N=20 slightly muted to highlight the dip
COLORS = {5: "#93C5FD", 10: "#60A5FA", 15: "#1D4ED8", 20: "#A78BFA", 25: "#1E3A8A"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
gens = np.arange(1, 101)

# ── Panel A: HV convergence curves ────────────────────────────
ax = axes[0]
base_curves = np.full((30, 100), np.nan)
for rep in range(1, 31):
    p = BASE / BASE_TPL.format(rep) / "hv_history.csv"
    if p.exists():
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            g = int(row["gen"]) - 1
            if 0 <= g < 100:
                base_curves[rep - 1, g] = row["hv"]

bm = np.nanmean(base_curves, axis=0)
bs = np.nanstd(base_curves, axis=0)
ax.plot(gens, bm, color="#9CA3AF", ls="--", lw=1.5, label="NSGA-II baseline", zorder=1)
ax.fill_between(gens, bm - bs, bm + bs, color="#9CA3AF", alpha=0.08)

for n in N_VALS:
    mat  = data[n]["curves"]
    mean = np.nanmean(mat, axis=0)
    sd   = np.nanstd(mat, axis=0)
    ax.plot(gens, mean, color=COLORS[n], lw=2.0, label=f"N={n}", zorder=2)
    ax.fill_between(gens, mean - sd, mean + sd, color=COLORS[n], alpha=0.10)

ax.set_xlabel("Generation", fontsize=11)
ax.set_ylabel("Hypervolume", fontsize=11)
ax.set_title("(A) HV Convergence (mean +/- 1 SD, n=30)", fontsize=11)
ax.legend(fontsize=9, framealpha=0.8)
ax.set_xlim(1, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Panel B: dHV box plot ─────────────────────────────────────
ax = axes[1]
dhv_list = [data[n]["dhv"] for n in N_VALS]
pos = np.arange(len(N_VALS))

bp = ax.boxplot(
    dhv_list, positions=pos, widths=0.5, patch_artist=True,
    medianprops=dict(color="white", linewidth=2.5),
    flierprops=dict(marker=".", markersize=4, alpha=0.4),
    whiskerprops=dict(linewidth=1.3),
    capprops=dict(linewidth=1.3),
)
for patch, n in zip(bp["boxes"], N_VALS):
    patch.set_facecolor(COLORS[n]); patch.set_alpha(0.78); patch.set_edgecolor(COLORS[n])
for w, c in zip(bp["whiskers"], [c for n in N_VALS for c in [COLORS[n]] * 2]):
    w.set_color(c)
for cap, c in zip(bp["caps"], [c for n in N_VALS for c in [COLORS[n]] * 2]):
    cap.set_color(c)

rng = np.random.default_rng(42)
for i, (n, dhv) in enumerate(zip(N_VALS, dhv_list)):
    jitter = rng.uniform(-0.15, 0.15, len(dhv))
    ax.scatter(i + jitter, dhv, s=18, color=COLORS[n], alpha=0.50, zorder=3)

ax.axhline(0, color="#9CA3AF", ls="--", lw=0.9)

for i, (n, dhv) in enumerate(zip(N_VALS, dhv_list)):
    m  = dhv.mean()
    se = dhv.std() / np.sqrt(len(dhv))
    w, pw = stats.wilcoxon(dhv, alternative="greater")
    sig = "***" if pw < 0.001 else ("**" if pw < 0.01 else ("*" if pw < 0.05 else "ns"))
    pct = int((dhv > 0).mean() * 100)
    ypos = dhv.max() + 0.005
    ax.text(i, ypos, f"{m:+.3f}\n{sig} ({pct}%+)",
            ha="center", va="bottom", fontsize=8.5,
            color=COLORS[n], fontweight="bold")

ax.set_xticks(pos)
ax.set_xticklabels([f"N={n}\n(n=30)" for n in N_VALS], fontsize=10)
ax.set_ylabel("Delta HV (hybrid - baseline, paired)", fontsize=11)
ax.set_title("(B) HV Gain Distribution per N", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("N-Sweep: LLM Solutions Injected per Generation\n"
             "Every-gen augment, pop=50, 100 gens, n=30 paired replicates each",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = "results/figures/nsweep_final.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")

# stats table
print(f"\n{'N':>4}  {'mean HV':>8}  {'mean dHV':>9}  {'SD':>6}  {'p(Wilcoxon)':>12}  {'%pos':>5}")
print("-" * 55)
print(f"{'base':>4}  {hv_base.mean():>8.4f}  {'—':>9}  {'—':>6}  {'—':>12}  {'—':>5}")
for n in N_VALS:
    d = data[n]
    w, pw = stats.wilcoxon(d["dhv"], alternative="greater")
    sig = "***" if pw < 0.001 else ("**" if pw < 0.01 else ("*" if pw < 0.05 else "ns"))
    pct = int((d["dhv"] > 0).mean() * 100)
    print(f"{n:>4}  {d['hvs'].mean():>8.4f}  {d['dhv'].mean():>+9.4f}  "
          f"{d['dhv'].std():>6.4f}  {pw:>8.4f} {sig:<3}  {pct:>4}%")
