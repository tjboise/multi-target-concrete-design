"""
run_grid_fn_mt.py
=================
Full 3x3 grid search over F (llm_frequency) and N (llm_n_solutions),
with pop=50, gen=100 fixed, no-knowledge-table prompt.

Grid:
  F  ∈ {5, 10, 20}   (injection frequency: every F generations)
  N  ∈ {5, 10, 20}   (solutions per injection)

9 combos × repeat=5 × 2 modes (baseline + hybrid) = 90 total runs.
Baseline is shared across all F×N combos (same pop/gen seed pool).

Usage:
  python run_grid_fn_mt.py              # all 9 combos, repeat=5
  python run_grid_fn_mt.py --repeat 1  # quick test
"""

import argparse
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from optimizer_core_mt import load_df, get_bounds, get_physics_bounds, load_surrogate
from optimizer_hybrid_mt import (
    HybridConfig, run_hybrid, compute_hybrid_metrics, save_hybrid_results,
)

# ── Shared settings ────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash-lite"
DATA_PATH      = r"Concrete_Data_SI.csv"
MODEL_PKL      = r"..\low_carbon_concrete\concrete_catboost_optimized.pkl"
NSGA_REF_CSV   = r"results\nsga2_reference.csv"

POP   = 50
GEN   = 100
F_LEVELS = [5, 10, 20]
N_LEVELS = [5, 10, 20]
DEFAULT_REPEAT = 5


def _make_cfg(F: int, N: int, rep: int, mode: str) -> HybridConfig:
    tag  = f"f{F:02d}_n{N:02d}_rep{rep:02d}"
    name = f"grid_{'base' if mode == 'baseline' else 'hyb'}_{tag}"
    return HybridConfig(
        name=name,
        description=f"Grid ({mode}): pop={POP}, gen={GEN}, F={F}, N={N}, rep={rep}",
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        pop_size=POP,
        max_generations=GEN,
        llm_frequency=0 if mode == "baseline" else F,
        llm_n_solutions=N,
        llm_n_elite=10,
        data_path=DATA_PATH,
        model_pkl=MODEL_PKL,
        output_prefix=os.path.join("results", name),
        use_knowledge_table=False,
        use_json_mode=True,
    )


def run_cell(F: int, N: int, rep: int,
             raw_b, der_b, phys_b, meta, nsga_ref) -> dict:
    # ── Baseline ───────────────────────────────────────────────
    base_cfg = _make_cfg(F, N, rep, "baseline")
    base_res = run_hybrid(raw_b, der_b, phys_b, meta, base_cfg)
    base_m   = compute_hybrid_metrics(base_res, nsga_ref)
    save_hybrid_results(base_res, base_m, base_cfg, nsga_ref)

    # ── Hybrid ─────────────────────────────────────────────────
    hyb_cfg = _make_cfg(F, N, rep, "hybrid")
    hyb_res = run_hybrid(raw_b, der_b, phys_b, meta, hyb_cfg)
    hyb_m   = compute_hybrid_metrics(hyb_res, nsga_ref)
    save_hybrid_results(hyb_res, hyb_m, hyb_cfg, nsga_ref)

    hv_adv = hyb_m["HV_hybrid"] - base_m["HV_hybrid"]
    print(
        f"    [F={F:2d} N={N:2d} rep={rep}] "
        f"Base={base_m['HV_hybrid']:.4f}  Hyb={hyb_m['HV_hybrid']:.4f}  d={hv_adv:+.4f}"
    )
    return {
        "F": F, "N": N, "rep": rep,
        "N_over_pop": round(N / POP, 3),
        "HV_base":     base_m["HV_hybrid"],
        "HV_hybrid":   hyb_m["HV_hybrid"],
        "HV_ratio":    hyb_m["HV_ratio"],
        "HV_advantage": hv_adv,
        "HV_pct_gain":  100.0 * hv_adv / base_m["HV_hybrid"] if base_m["HV_hybrid"] > 0 else 0.0,
        "llm_calls":   hyb_m["llm_calls"],
        "parse_fails": hyb_m["parse_fails"],
    }


def print_summary(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print(f"  FxN GRID SUMMARY  pop={POP} gen={GEN}  no-knowledge-table")
    print(f"  (mean ± std over {df['rep'].max()} repeats)")
    print(f"{'='*70}")
    grp = df.groupby(["F", "N", "N_over_pop"]).agg(
        HV_base_mean  =("HV_base",     "mean"),
        HV_hyb_mean   =("HV_hybrid",   "mean"),
        HV_hyb_std    =("HV_hybrid",   "std"),
        HV_adv_mean   =("HV_advantage","mean"),
        HV_pct_mean   =("HV_pct_gain", "mean"),
    ).reset_index()
    grp["HV_pct_mean"] = grp["HV_pct_mean"].map("{:+.2f}%".format)
    print(grp.to_string(index=False))

    print(f"\n{'='*70}")
    print("  FACTOR MARGINAL MEANS (HV_pct_gain)")
    print(f"{'='*70}")
    for factor in ["F", "N"]:
        fa = df.groupby(factor)["HV_pct_gain"].mean().reset_index()
        fa.columns = [factor, "mean_pct_gain"]
        fa["mean_pct_gain"] = fa["mean_pct_gain"].map("{:+.2f}%".format)
        print(f"\n  {factor}:")
        print(fa.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="F×N grid search (no-KT prompt)")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--rep-start", type=int, default=1,
                        help="First rep index (default: 1); use >1 to continue a previous run")
    parser.add_argument("--f-levels", type=int, nargs="+", default=F_LEVELS,
                        metavar="F", help="Injection frequencies to test (default: 5 10 20)")
    parser.add_argument("--n-levels", type=int, nargs="+", default=N_LEVELS,
                        metavar="N", help="Injection sizes to test (default: 5 10 20)")
    args = parser.parse_args()
    f_levels = sorted(args.f_levels)
    n_levels = sorted(args.n_levels)

    print(f"\n[Setup] Loading data and surrogate ...")
    df           = load_df(DATA_PATH)
    raw_b, der_b = get_bounds(df)
    phys_b       = get_physics_bounds(df)
    meta         = load_surrogate(MODEL_PKL)
    print(f"  Dataset rows : {len(df)}")
    print(f"  pop={POP}  gen={GEN}  prompt=no-knowledge-table")
    print(f"  F levels: {f_levels}  N levels: {n_levels}")

    os.makedirs("results", exist_ok=True)
    nsga_ref = []
    if os.path.exists(NSGA_REF_CSV):
        nsga_ref = pd.read_csv(NSGA_REF_CSV).to_dict("records")
        print(f"\n[Reference] Loaded {len(nsga_ref)} NSGA-II Pareto solutions")
    else:
        print("\n[Warning] nsga2_reference.csv not found")

    rep_range = range(args.rep_start, args.rep_start + args.repeat)
    total = len(f_levels) * len(n_levels) * args.repeat
    print(f"\n[Plan] {len(f_levels)}x{len(n_levels)} grid x reps {args.rep_start}-{args.rep_start+args.repeat-1} "
          f"x 2 modes = {total * 2} total runs\n")

    all_rows = []
    for F in f_levels:
        for N in n_levels:
            print(f"\n{'='*70}")
            print(f"  F={F}  N={N}  (N/pop={N/POP:.0%})")
            print(f"{'='*70}")
            for rep in rep_range:
                result = run_cell(F, N, rep, raw_b, der_b, phys_b, meta, nsga_ref)
                all_rows.append(result)

    df_out   = pd.DataFrame(all_rows)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join("results", f"grid_fn_nokt_{stamp}.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n  Combined results -> {csv_path}")

    print_summary(df_out)


if __name__ == "__main__":
    main()
