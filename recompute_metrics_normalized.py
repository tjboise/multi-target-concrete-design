"""
Recompute normalized HV for all grid configs from saved pareto_front.csv files.
Uses dataset-derived normalization bounds (Concrete_Data_SI.csv).
Updates metrics.csv and regenerates grid_fn_nokt_*.csv summary.
"""
import os, glob
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

RESULTS = "results"
HV_GWP_MIN, HV_GWP_MAX = 169.0, 534.5
HV_STR_MIN, HV_STR_MAX = 17.4,  106.1
HV_REF = np.array([1.0, 1.0])

def normalized_hv(df_pareto):
    if df_pareto is None or len(df_pareto) == 0:
        return 0.0
    gwp = df_pareto["GWP"].values
    d28 = df_pareto["28day"].values
    gwp_n = (gwp - HV_GWP_MIN) / (HV_GWP_MAX - HV_GWP_MIN)
    str_n = (HV_STR_MAX - d28) / (HV_STR_MAX - HV_STR_MIN)
    F = np.column_stack([gwp_n, str_n])
    # clip to avoid points outside [0,1] range
    F = np.clip(F, 0.0, None)
    try:
        return round(float(HV(ref_point=HV_REF)(F)), 6)
    except:
        return 0.0

FS = [5, 10, 20]
NS = [5, 10, 20]
REPS = [1, 2, 3, 4, 5]

rows = []
for F in FS:
    for N in NS:
        for rep in REPS:
            tag = f"f{F:02d}_n{N:02d}_rep{rep:02d}"
            base_dir = os.path.join(RESULTS, f"grid_base_{tag}")
            hyb_dir  = os.path.join(RESULTS, f"grid_hyb_{tag}")

            def load_pareto(d):
                p = os.path.join(d, "pareto_front.csv")
                if os.path.exists(p):
                    return pd.read_csv(p)
                return None

            def load_metrics(d):
                p = os.path.join(d, "metrics.csv")
                if os.path.exists(p):
                    return pd.read_csv(p).iloc[0].to_dict()
                return {}

            base_pareto = load_pareto(base_dir)
            hyb_pareto  = load_pareto(hyb_dir)
            base_m = load_metrics(base_dir)
            hyb_m  = load_metrics(hyb_dir)

            hv_base = normalized_hv(base_pareto)
            hv_hyb  = normalized_hv(hyb_pareto)
            hv_adv  = round(hv_hyb - hv_base, 6)
            hv_pct  = round((hv_adv / hv_base * 100) if hv_base > 0 else 0.0, 4)

            # Update metrics.csv for both dirs
            for d, hv_val in [(base_dir, hv_base), (hyb_dir, hv_hyb)]:
                mp = os.path.join(d, "metrics.csv")
                if os.path.exists(mp):
                    mdf = pd.read_csv(mp)
                    mdf["HV"] = hv_val
                    mdf.to_csv(mp, index=False)

            rows.append({
                "F": F, "N": N, "rep": rep,
                "HV_base":    hv_base,
                "HV_hybrid":  hv_hyb,
                "HV_advantage": hv_adv,
                "HV_pct_gain":  hv_pct,
                "parse_fails":  hyb_m.get("parse_fails", 0),
                "llm_calls":    hyb_m.get("llm_calls", 0),
                "n_pareto_base": len(base_pareto) if base_pareto is not None else 0,
                "n_pareto_hyb":  len(hyb_pareto)  if hyb_pareto  is not None else 0,
            })
            print(f"F={F} N={N} rep={rep}: base={hv_base:.4f} hyb={hv_hyb:.4f} pct={hv_pct:+.2f}%")

df = pd.DataFrame(rows)
out = "results/grid_fn_normalized_20260813.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")

# Summary table
print("\nMean HV % gain by F x N:")
pivot = df.groupby(["F","N"])["HV_pct_gain"].mean().unstack("N")
print(pivot.round(2).to_string())
print(f"\nBest config: F={df.groupby(['F','N'])['HV_pct_gain'].mean().idxmax()}")
