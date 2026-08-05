"""
run_orthogonal_mt.py
====================
L9(3^4) orthogonal experiment for LLM-NSGA-II hybrid optimizer.

Factors and levels:
  A  gen (max_generations): 50, 100, 150
  B  pop (pop_size):        30,  50, 100
  C  F   (llm_frequency):    5,  10,  20
  D  N   (llm_n_solutions):  5,  10,  20

L9 array (9 runs × repeat=5 = 45 hybrid + 45 baseline = 90 total runs).
Each run saves individual result directories + a combined CSV.

Usage:
  python run_orthogonal_mt.py                # all 9 runs, repeat=5
  python run_orthogonal_mt.py --repeat 1    # quick test (1 rep per run)
  python run_orthogonal_mt.py --run 4       # single L9 row (1-indexed)
"""

import argparse
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from optimizer_core_mt import load_df, get_bounds, load_surrogate
from optimizer_hybrid_mt import (
    HybridConfig, run_hybrid, compute_hybrid_metrics, save_hybrid_results,
)

# ── Shared settings ────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash-lite"
DATA_PATH      = r"Super_Cleaned_Concrete_Data - backup.csv"
MODEL_PKL      = r"..\low_carbon_concrete\concrete_catboost_optimized.pkl"
NSGA_REF_CSV   = r"results\nsga2_reference.csv"

DEFAULT_REPEAT = 5

# ── L9(3^4) orthogonal array ───────────────────────────────────
# Columns: gen, pop, F (llm_frequency), N (llm_n_solutions)
L9_TABLE = [
    {"gen":  50, "pop":  30, "F":  5, "N":  5},   # Run 1
    {"gen":  50, "pop":  50, "F": 10, "N": 10},   # Run 2
    {"gen":  50, "pop": 100, "F": 20, "N": 20},   # Run 3
    {"gen": 100, "pop":  30, "F": 10, "N": 20},   # Run 4
    {"gen": 100, "pop":  50, "F": 20, "N":  5},   # Run 5
    {"gen": 100, "pop": 100, "F":  5, "N": 10},   # Run 6
    {"gen": 150, "pop":  30, "F": 20, "N": 10},   # Run 7
    {"gen": 150, "pop":  50, "F":  5, "N": 20},   # Run 8
    {"gen": 150, "pop": 100, "F": 10, "N":  5},   # Run 9
]


def _make_cfg(run_id: int, row: dict, rep: int, mode: str,
              use_kt: bool = True) -> HybridConfig:
    """Build a HybridConfig for one (run, repeat, mode) combination."""
    kt_tag = "" if use_kt else "_nokt"
    tag = f"r{run_id:02d}_g{row['gen']:03d}_p{row['pop']:03d}_f{row['F']:02d}_n{row['N']:02d}"
    if mode == "baseline":
        name = f"ortho_base{kt_tag}_{tag}_rep{rep:02d}"
    else:
        name = f"ortho_hyb{kt_tag}_{tag}_rep{rep:02d}"

    return HybridConfig(
        name=name,
        description=(
            f"L9 Run {run_id} ({mode}, kt={use_kt}): gen={row['gen']}, pop={row['pop']}, "
            f"F={row['F']}, N={row['N']}, rep={rep}"
        ),
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        pop_size=row["pop"],
        max_generations=row["gen"],
        llm_frequency=0 if mode == "baseline" else row["F"],
        llm_n_solutions=row["N"],
        llm_n_elite=10,
        data_path=DATA_PATH,
        model_pkl=MODEL_PKL,
        output_prefix=os.path.join("results", name),
        use_knowledge_table=use_kt,
        use_json_mode=True,
    )


def run_row(run_id: int, row: dict, rep: int,
            raw_b, der_b, meta, nsga_ref, use_kt: bool = True) -> dict:
    """Run one (L9 row, repeat) pair — baseline + hybrid. Return metrics dict."""
    tag = f"r{run_id:02d}_g{row['gen']:03d}_p{row['pop']:03d}_f{row['F']:02d}_n{row['N']:02d}_rep{rep:02d}"

    # ── Baseline (pure NSGA-II) ────────────────────────────────
    base_cfg = _make_cfg(run_id, row, rep, "baseline", use_kt=use_kt)
    base_res = run_hybrid(raw_b, der_b, meta, base_cfg)
    base_m   = compute_hybrid_metrics(base_res, nsga_ref)
    save_hybrid_results(base_res, base_m, base_cfg, nsga_ref)

    # ── Hybrid (LLM-NSGA-II) ──────────────────────────────────
    hyb_cfg = _make_cfg(run_id, row, rep, "hybrid", use_kt=use_kt)
    hyb_res = run_hybrid(raw_b, der_b, meta, hyb_cfg)
    hyb_m   = compute_hybrid_metrics(hyb_res, nsga_ref)
    save_hybrid_results(hyb_res, hyb_m, hyb_cfg, nsga_ref)

    hv_adv = hyb_m["HV_hybrid"] - base_m["HV_hybrid"]
    print(
        f"    [{tag}] Base HV={base_m['HV_hybrid']:.1f} | "
        f"Hyb HV={hyb_m['HV_hybrid']:.1f} | Δ={hv_adv:+.1f}"
    )

    return {
        "run_id": run_id, "rep": rep,
        "gen": row["gen"], "pop": row["pop"],
        "F": row["F"], "N": row["N"],
        # Baseline metrics
        "HV_base":    base_m["HV_hybrid"],
        "GD_base":    base_m["GD"],
        "IGD_base":   base_m["IGD"],
        # Hybrid metrics
        "HV_hybrid":  hyb_m["HV_hybrid"],
        "HV_ratio":   hyb_m["HV_ratio"],
        "GD_hybrid":  hyb_m["GD"],
        "IGD_hybrid": hyb_m["IGD"],
        "n_pareto":   hyb_m["n_pareto"],
        "llm_calls":  hyb_m["llm_calls"],
        "parse_fails":hyb_m["parse_fails"],
        # Derived
        "HV_advantage": hv_adv,
        "HV_pct_gain":  100.0 * hv_adv / base_m["HV_hybrid"] if base_m["HV_hybrid"] > 0 else 0.0,
    }


def print_summary(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("  L9 ORTHOGONAL EXPERIMENT SUMMARY  (mean ± std over repeats)")
    print(f"{'='*70}")
    grp = df.groupby(["run_id", "gen", "pop", "F", "N"]).agg(
        HV_base_mean  =("HV_base",     "mean"),
        HV_hyb_mean   =("HV_hybrid",   "mean"),
        HV_hyb_std    =("HV_hybrid",   "std"),
        HV_adv_mean   =("HV_advantage","mean"),
        HV_pct_mean   =("HV_pct_gain", "mean"),
    ).reset_index()
    grp["HV_pct_mean"] = grp["HV_pct_mean"].map("{:+.2f}%".format)
    print(grp.to_string(index=False))

    print(f"\n{'='*70}")
    print("  FACTOR-LEVEL ANALYSIS (mean HV_hybrid across all runs)")
    print(f"{'='*70}")
    for factor in ["gen", "pop", "F", "N"]:
        fa = df.groupby(factor)["HV_hybrid"].mean().reset_index()
        fa.columns = [factor, "mean_HV"]
        print(f"\n  {factor}:")
        print(fa.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="L9 orthogonal experiment for LLM-NSGA-II")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help=f"Repeats per L9 row (default {DEFAULT_REPEAT})")
    parser.add_argument("--run", type=int, default=None,
                        help="Run only this L9 row (1–9)")
    parser.add_argument("--nokt", action="store_true",
                        help="Use no-knowledge-table prompt (ablation)")
    args = parser.parse_args()
    use_kt = not args.nokt

    # ── Load shared resources ──────────────────────────────────
    print("\n[Setup] Loading data and surrogate ...")
    df           = load_df(DATA_PATH)
    raw_b, der_b = get_bounds(df)
    meta         = load_surrogate(MODEL_PKL)
    print(f"  Dataset rows : {len(df)}")
    print(f"  PC  range    : [{raw_b['PC']['min']:.1f}, {raw_b['PC']['max']:.1f}] kg/m3")
    print(f"  28d range    : [{df['28day'].min():.1f}, {df['28day'].max():.1f}] MPa")

    os.makedirs("results", exist_ok=True)
    if os.path.exists(NSGA_REF_CSV):
        nsga_ref = pd.read_csv(NSGA_REF_CSV).to_dict("records")
        print(f"\n[Reference] Loaded {len(nsga_ref)} NSGA-II Pareto solutions from {NSGA_REF_CSV}")
    else:
        print("\n[Warning] nsga2_reference.csv not found — HV_ratio will be 0")
        nsga_ref = []

    # ── Select rows ────────────────────────────────────────────
    rows_to_run = [(i + 1, row) for i, row in enumerate(L9_TABLE)]
    if args.run is not None:
        if not (1 <= args.run <= 9):
            print("--run must be between 1 and 9"); return
        rows_to_run = [(args.run, L9_TABLE[args.run - 1])]

    kt_label = "WITH knowledge table" if use_kt else "WITHOUT knowledge table (--nokt)"
    print(f"\n[Plan] {len(rows_to_run)} L9 row(s) × {args.repeat} repeat(s) × 2 modes = "
          f"{len(rows_to_run) * args.repeat * 2} total runs  [{kt_label}]\n")

    all_rows = []
    for run_id, row in rows_to_run:
        print(f"\n{'─'*70}")
        print(f"  L9 Run {run_id}/9 │ gen={row['gen']}  pop={row['pop']}  F={row['F']}  N={row['N']}")
        print(f"{'─'*70}")
        for rep in range(1, args.repeat + 1):
            print(f"  Repeat {rep}/{args.repeat}")
            result = run_row(run_id, row, rep, raw_b, der_b, meta, nsga_ref, use_kt=use_kt)
            all_rows.append(result)

    # ── Save combined results ──────────────────────────────────
    df_out = pd.DataFrame(all_rows)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M")
    kt_suffix = "" if use_kt else "_nokt"
    csv_path = os.path.join("results", f"orthogonal_L9{kt_suffix}_{stamp}.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n  Combined results → {csv_path}")

    print_summary(df_out)


if __name__ == "__main__":
    main()
