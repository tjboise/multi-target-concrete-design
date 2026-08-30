"""
Verify lean-binder hypothesis across the full FLAME global Pareto front.
GWP is driven more by total binder content than by SC substitution rate.

Approach:
1. Correlation: GWP vs binder content, vs SC%, vs effective GWP factor
2. Partial regression: decompose GWP variance into binder vs substitution
3. Visualize: scatter GWP vs binder (color = SC%) + GWP vs SC% (color = binder)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE = Path(r"C:\Users\Tianjie Zhang\OneDrive - Rutgers University\Documents\GitHub\multi-target concrete design\results")
FIGDIR = Path(r"C:\Users\Tianjie Zhang\OneDrive - Rutgers University\Documents\GitHub\multi-target concrete design\results\figures")
REPS = 30

GWP_PC, GWP_FA, GWP_SC = 0.82, 0.027, 0.052

def load_all_pf(tpl, reps=REPS):
    frames = []
    for r in range(1, reps+1):
        p = BASE / tpl.format(r) / "pareto_front.csv"
        if p.exists():
            df = pd.read_csv(p); df["rep"] = r; frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def non_dominated(df):
    pts = df[["GWP","28day"]].values
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if pts[j,0] <= pts[i,0] and pts[j,1] >= pts[i,1] and \
               (pts[j,0] < pts[i,0] or pts[j,1] > pts[i,1]):
                dominated[i] = True; break
    return df[~dominated].copy()

flame_all = load_all_pf("nsweep_n15_rep{:02d}")
flame_nd = non_dominated(flame_all).sort_values("GWP").reset_index(drop=True)
print(f"Global FLAME Pareto front: {len(flame_nd)} solutions")

# ── Compute mix properties ────────────────────────────────────────────────────
df = flame_nd.copy()
df["binder"]   = df["PC"] + df["FA"] + df["SC"]
df["SC_pct"]   = df["SC"] / df["binder"] * 100
df["FA_pct"]   = df["FA"] / df["binder"] * 100
df["PC_pct"]   = df["PC"] / df["binder"] * 100
df["wb"]       = df["WATER"] / df["binder"]
# Effective GWP factor (kg CO2 per kg binder)
df["eff_gwp"]  = (df["PC"]*GWP_PC + df["FA"]*GWP_FA + df["SC"]*GWP_SC) / df["binder"]
# GWP from binder only (to isolate binder contribution from aggregates etc.)
df["gwp_binder"] = df["PC"]*GWP_PC + df["FA"]*GWP_FA + df["SC"]*GWP_SC

# ── 1. Correlations ───────────────────────────────────────────────────────────
print("\n=== Spearman correlations with GWP ===")
for col, label in [
    ("binder",   "Total binder content (kg/m3)"),
    ("SC_pct",   "SC substitution rate (%)"),
    ("FA_pct",   "FA fraction (%)"),
    ("PC_pct",   "PC fraction (%)"),
    ("eff_gwp",  "Effective GWP factor (kg CO2/kg binder)"),
    ("wb",       "w/b ratio"),
]:
    r, p = stats.spearmanr(df["GWP"], df[col])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {label:<42}: r={r:+.3f}  p={p:.4f} {sig}")

# ── 2. Partial regression (OLS): GWP ~ binder + eff_gwp ──────────────────────
# Standardize predictors for comparable coefficients
from numpy.linalg import lstsq

def standardize(x): return (x - x.mean()) / x.std()

X = np.column_stack([
    np.ones(len(df)),
    standardize(df["binder"].values),
    standardize(df["eff_gwp"].values),
])
y = standardize(df["GWP"].values)
coeffs, _, _, _ = lstsq(X, y, rcond=None)
# R^2
y_pred = X @ coeffs
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_full = 1 - ss_res/ss_tot

# R^2 with only binder
Xb = np.column_stack([np.ones(len(df)), standardize(df["binder"].values)])
cb, _, _, _ = lstsq(Xb, y, rcond=None)
r2_binder_only = 1 - np.sum((y - Xb@cb)**2)/ss_tot

# R^2 with only eff_gwp
Xe = np.column_stack([np.ones(len(df)), standardize(df["eff_gwp"].values)])
ce, _, _, _ = lstsq(Xe, y, rcond=None)
r2_effgwp_only = 1 - np.sum((y - Xe@ce)**2)/ss_tot

print(f"\n=== OLS regression: standardized GWP ~ binder + eff_gwp ===")
print(f"  Full model R2:          {r2_full:.3f}")
print(f"  Binder-only R2:         {r2_binder_only:.3f}")
print(f"  Eff_gwp-only R2:        {r2_effgwp_only:.3f}")
print(f"  Beta(binder):           {coeffs[1]:+.3f}")
print(f"  Beta(eff_gwp):          {coeffs[2]:+.3f}")

# ── 3. Ranges to quantify relative variation ──────────────────────────────────
print(f"\n=== Range of key predictors across the front ===")
for col, label in [("binder","Binder (kg/m3)"), ("SC_pct","SC% (pct)"),
                   ("eff_gwp","Eff GWP factor")]:
    lo, hi = df[col].min(), df[col].max()
    rng = hi - lo
    cv  = df[col].std() / df[col].mean() * 100
    print(f"  {label:<22}: {lo:.2f} – {hi:.2f}  (range={rng:.2f}, CV={cv:.1f}%)")

# ── 4. Figure: two panels ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: GWP vs binder, color = SC%
sc1 = ax1.scatter(df["binder"], df["GWP"], c=df["SC_pct"],
                  cmap="RdYlGn_r", vmin=15, vmax=75, s=40, alpha=0.85, zorder=3)
cb1 = fig.colorbar(sc1, ax=ax1)
cb1.set_label("SC substitution rate (%)", fontsize=8)
# Add regression line
bx = np.linspace(df["binder"].min(), df["binder"].max(), 200)
m, b_int, r, p, _ = stats.linregress(df["binder"], df["GWP"])
ax1.plot(bx, m*bx + b_int, color="#9CA3AF", lw=1.5, ls="--", zorder=2)
r_s, p_s = stats.spearmanr(df["binder"], df["GWP"])
ax1.set_xlabel("Total binder content (kg/m\u00b3)", fontsize=10)
ax1.set_ylabel("GWP (kg CO\u2082/m\u00b3)", fontsize=10)
ax1.text(0.05, 0.93, f"$r_s$ = {r_s:+.2f}, p < 0.001",
         transform=ax1.transAxes, fontsize=9, color="#1D4ED8")
for spine in ax1.spines.values():
    spine.set_visible(True); spine.set_linewidth(0.8); spine.set_color("black")

# Panel B: GWP vs SC%, color = binder
sc2 = ax2.scatter(df["SC_pct"], df["GWP"], c=df["binder"],
                  cmap="Blues", vmin=150, vmax=500, s=40, alpha=0.85, zorder=3)
cb2 = fig.colorbar(sc2, ax=ax2)
cb2.set_label("Total binder (kg/m\u00b3)", fontsize=8)
r_s2, p_s2 = stats.spearmanr(df["SC_pct"], df["GWP"])
sig2 = "p < 0.001" if p_s2 < 0.001 else f"p = {p_s2:.3f}"
ax2.set_xlabel("SC substitution rate (%)", fontsize=10)
ax2.set_ylabel("GWP (kg CO\u2082/m\u00b3)", fontsize=10)
ax2.text(0.05, 0.93, f"$r_s$ = {r_s2:+.2f}, {sig2}",
         transform=ax2.transAxes, fontsize=9, color="#D97706")
for spine in ax2.spines.values():
    spine.set_visible(True); spine.set_linewidth(0.8); spine.set_color("black")

plt.tight_layout()
FIG_PATH = FIGDIR / "pareto_gwp_decomposition.png"
plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {FIG_PATH}")
