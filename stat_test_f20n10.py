"""
Statistical significance test for F=20, N=10 hybrid vs NSGA-II baseline.
  - Wilcoxon signed-rank test (one-tailed: H1: hybrid > baseline)
  - Cliff's delta effect size
  - Bootstrap 95% CI on mean HV gain
"""

import glob, pandas as pd, numpy as np
from scipy.stats import wilcoxon

# ── load all grid CSVs and filter F=20, N=10 ────────────────────
dfs = [pd.read_csv(f) for f in sorted(glob.glob("results/grid_fn_nokt_*.csv"))]
df  = pd.concat(dfs, ignore_index=True)
d   = df[(df["F"] == 20) & (df["N"] == 10)].copy().reset_index(drop=True)
print(f"Loaded {len(d)} paired observations (F=20, N=10)\n")

hv_base = d["HV_base"].values
hv_hyb  = d["HV_hybrid"].values
diff    = hv_hyb - hv_base

# ── per-rep summary ──────────────────────────────────────────────
print("Per-rep results:")
print(f"  {'Rep':>4}  {'Baseline HV':>13}  {'Hybrid HV':>13}  {'Δ HV':>10}  {'Winner'}")
print(f"  {'─'*4}  {'─'*13}  {'─'*13}  {'─'*10}  {'─'*6}")
for i, (b, h, dv) in enumerate(zip(hv_base, hv_hyb, diff), 1):
    winner = "Hybrid ✓" if dv > 0 else "Baseline"
    print(f"  {i:>4}  {b:>13.1f}  {h:>13.1f}  {dv:>+10.1f}  {winner}")

n_wins = (diff > 0).sum()
print(f"\n  Hybrid wins: {n_wins}/{len(diff)} reps")

# ── Wilcoxon signed-rank test (one-tailed) ───────────────────────
stat, p_two = wilcoxon(diff, alternative="two-sided")
stat, p_one = wilcoxon(diff, alternative="greater")
print(f"\n── Wilcoxon signed-rank test ──────────────────────────────")
print(f"  n                : {len(diff)}")
print(f"  W statistic      : {stat:.1f}")
print(f"  p-value (1-tail) : {p_one:.4f}  {'✓ p<0.05' if p_one < 0.05 else '✗ p≥0.05'}")
print(f"  p-value (2-tail) : {p_two:.4f}  {'✓ p<0.05' if p_two < 0.05 else '✗ p≥0.05'}")

# ── Cliff's delta ────────────────────────────────────────────────
def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    more = sum(xi > yj for xi in x for yj in y)
    less = sum(xi < yj for xi in x for yj in y)
    return (more - less) / (n1 * n2)

delta = cliffs_delta(hv_hyb, hv_base)
if   abs(delta) >= 0.474: magnitude = "large"
elif abs(delta) >= 0.330: magnitude = "medium"
elif abs(delta) >= 0.147: magnitude = "small"
else:                       magnitude = "negligible"
print(f"\n── Cliff's delta ──────────────────────────────────────────")
print(f"  δ = {delta:+.3f}  ({magnitude})")
print(f"  Interpretation: hybrid solution dominates baseline in "
      f"{100*(delta+1)/2:.1f}% of paired comparisons")

# ── Bootstrap 95% CI on mean Δ HV ───────────────────────────────
rng = np.random.default_rng(42)
boot_means = [rng.choice(diff, size=len(diff), replace=True).mean()
              for _ in range(10_000)]
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
print(f"\n── Bootstrap 95% CI on mean HV gain (10,000 iterations) ──")
print(f"  Mean Δ HV        : {diff.mean():+.1f}")
print(f"  95% CI           : [{ci_lo:+.1f}, {ci_hi:+.1f}]")
contains_zero = ci_lo <= 0 <= ci_hi
print(f"  CI contains 0    : {'Yes → inconclusive' if contains_zero else 'No → significant'}")

# ── Mean pct gain ────────────────────────────────────────────────
pct = d["HV_pct_gain"].values
print(f"\n── Summary ────────────────────────────────────────────────")
print(f"  Mean HV gain     : {pct.mean():+.2f}%  (std {pct.std():.2f}pp)")
print(f"  Median HV gain   : {np.median(diff):+.1f}")
