"""
run_hybrid_mt.py
================
Experiment runner for LLM-assisted NSGA-II (hybrid) vs. pure NSGA-II.

Experiment grid (mirrors Lu et al. 2026 Table 3):
  - Baseline: pure NSGA-II (llm_frequency=0)
  - Vary F (intervention frequency): 5, 10, 20
  - Vary N (solutions per intervention): 5, 10, 20

Usage:
  python run_hybrid_mt.py                          # run all
  python run_hybrid_mt.py --exp nsga2_only         # single experiment
  python run_hybrid_mt.py --exp hybrid_f10_n10
  python run_hybrid_mt.py --gens 20                # quick test
  python run_hybrid_mt.py --repeat 3               # repeated runs
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

# ──────────────────────────────────────────────────────────────
# SHARED SETTINGS
# ──────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash-lite"
DATA_PATH      = r"Super_Cleaned_Concrete_Data - backup.csv"
MODEL_PKL      = r"..\low_carbon_concrete\concrete_catboost_optimized.pkl"
NSGA_REF_CSV   = r"results\nsga2_reference.csv"

POP_SIZE        = 50
MAX_GENERATIONS = 100


def make_cfg(name: str, description: str, **kwargs) -> HybridConfig:
    defaults = dict(
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        data_path=DATA_PATH,
        model_pkl=MODEL_PKL,
        output_prefix=os.path.join("results", name),
        use_knowledge_table=True,
        use_json_mode=True,
    )
    defaults.update(kwargs)
    return HybridConfig(name=name, description=description, **defaults)


# ──────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# ──────────────────────────────────────────────────────────────

EXPERIMENTS = {

    # Baseline: pure NSGA-II (no LLM)
    "nsga2_only": make_cfg(
        "nsga2_only",
        "Pure NSGA-II — no LLM intervention (baseline)",
        llm_frequency=0,
    ),

    # ── Vary intervention frequency (N fixed at 10) ───────────

    "hybrid_f5_n10": make_cfg(
        "hybrid_f5_n10",
        "LLM-NSGA-II: intervene every 5 generations, inject 10 solutions",
        llm_frequency=5,
        llm_n_solutions=10,
        llm_n_elite=10,
    ),

    "hybrid_f10_n10": make_cfg(
        "hybrid_f10_n10",
        "LLM-NSGA-II: intervene every 10 generations, inject 10 solutions",
        llm_frequency=10,
        llm_n_solutions=10,
        llm_n_elite=10,
    ),

    "hybrid_f20_n10": make_cfg(
        "hybrid_f20_n10",
        "LLM-NSGA-II: intervene every 20 generations, inject 10 solutions",
        llm_frequency=20,
        llm_n_solutions=10,
        llm_n_elite=10,
    ),

    # ── Vary number of LLM solutions (F fixed at 10) ──────────

    "hybrid_f10_n5": make_cfg(
        "hybrid_f10_n5",
        "LLM-NSGA-II: intervene every 10 generations, inject 5 solutions",
        llm_frequency=10,
        llm_n_solutions=5,
        llm_n_elite=10,
    ),

    "hybrid_f10_n20": make_cfg(
        "hybrid_f10_n20",
        "LLM-NSGA-II: intervene every 10 generations, inject 20 solutions",
        llm_frequency=10,
        llm_n_solutions=20,
        llm_n_elite=10,
    ),

    # ── Ablation: no knowledge table (prompt only has bounds + elite) ─
    "hybrid_f10_n10_nokt": make_cfg(
        "hybrid_f10_n10_nokt",
        "LLM-NSGA-II: F=10, N=10, no knowledge table (ablation)",
        llm_frequency=10,
        llm_n_solutions=10,
        llm_n_elite=10,
        use_knowledge_table=False,
    ),

    "hybrid_f20_n10_nokt": make_cfg(
        "hybrid_f20_n10_nokt",
        "LLM-NSGA-II: F=20, N=10, no knowledge table",
        llm_frequency=20,
        llm_n_solutions=10,
        llm_n_elite=10,
        use_knowledge_table=False,
    ),
}


# ──────────────────────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────────────────────

def run_one(cfg: HybridConfig, raw_b, der_b, meta, nsga_ref: list) -> dict:
    print(f"\n{'#'*62}")
    print(f"# {cfg.name}")
    print(f"# {cfg.description}")
    print(f"{'#'*62}")

    result  = run_hybrid(raw_b, der_b, meta, cfg)
    metrics = compute_hybrid_metrics(result, nsga_ref)
    save_hybrid_results(result, metrics, cfg, nsga_ref)

    return {"experiment": cfg.name, **metrics}


def main():
    parser = argparse.ArgumentParser(description="LLM-assisted NSGA-II experiments")
    parser.add_argument("--exp",    default=None,
                        help=f"Single experiment to run. Choices: {list(EXPERIMENTS)}")
    parser.add_argument("--gens",   type=int, default=None,
                        help="Override max_generations (e.g. 20 for quick test)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of repeated runs per experiment")
    parser.add_argument("--pop",    type=int, default=None,
                        help="Override pop_size (e.g. 100)")
    args = parser.parse_args()

    pop_suffix = f"_p{args.pop}" if args.pop else ""
    if args.pop:
        for cfg in EXPERIMENTS.values():
            cfg.pop_size = args.pop
    if args.gens:
        for cfg in EXPERIMENTS.values():
            cfg.max_generations = args.gens
            cfg.output_prefix = os.path.join("results", f"{cfg.name}{pop_suffix}_g{args.gens:03d}")

    # ── Load shared resources ──────────────────────────────────
    print("\n[Setup] Loading data and surrogate ...")
    df           = load_df(DATA_PATH)
    raw_b, der_b = get_bounds(df)
    meta         = load_surrogate(MODEL_PKL)
    print(f"  Dataset rows : {len(df)}")
    print(f"  PC  range    : [{raw_b['PC']['min']:.1f}, {raw_b['PC']['max']:.1f}] kg/m3")
    print(f"  28d range    : [{df['28day'].min():.1f}, {df['28day'].max():.1f}] MPa")

    # ── NSGA-II reference (reuse saved if available) ───────────
    os.makedirs("results", exist_ok=True)
    if os.path.exists(NSGA_REF_CSV):
        print(f"\n[Reference] Loading saved NSGA-II front from {NSGA_REF_CSV} ...")
        nsga_ref = pd.read_csv(NSGA_REF_CSV).to_dict("records")
        print(f"  Loaded {len(nsga_ref)} Pareto solutions")
    else:
        print("\n[Reference] No saved NSGA-II front found.")
        print("  Run optimizer_core_mt experiments first, or run:")
        print("  python run_experiment_mt.py --exp e0_baseline")
        nsga_ref = []

    # ── Select experiments ─────────────────────────────────────
    if args.exp:
        if args.exp not in EXPERIMENTS:
            print(f"Unknown experiment '{args.exp}'. Available: {list(EXPERIMENTS)}")
            return
        to_run = {args.exp: EXPERIMENTS[args.exp]}
    else:
        to_run = EXPERIMENTS

    # ── Run ────────────────────────────────────────────────────
    all_metrics = []
    gen_suffix = f"_g{args.gens:03d}" if args.gens else ""
    for exp_name, cfg in to_run.items():
        for rep in range(args.repeat):
            if args.repeat > 1:
                cfg.output_prefix = os.path.join("results", f"{cfg.name}{pop_suffix}{gen_suffix}_r{rep+1}")
                print(f"\n  >>> Repeat {rep+1}/{args.repeat}")

            row = run_one(cfg, raw_b, der_b, meta, nsga_ref)
            row["repeat"] = rep + 1
            all_metrics.append(row)

    # ── Summary ────────────────────────────────────────────────
    if len(all_metrics) > 1:
        df_m = pd.DataFrame(all_metrics)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        summary_path = os.path.join("results", f"hybrid_summary_{stamp}.csv")
        df_m.to_csv(summary_path, index=False)

        print(f"\n{'='*62}")
        print("  HYBRID EXPERIMENT SUMMARY")
        print(f"{'='*62}")
        cols = ["HV_hybrid", "HV_ratio", "GD", "IGD", "n_pareto", "llm_calls", "parse_fails"]
        for _, row in df_m.iterrows():
            print(f"\n  {row['experiment']}:")
            for col in cols:
                if col in row:
                    val = row[col]
                    fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                    print(f"    {col:<20}: {fmt}")

        print(f"\n  Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
