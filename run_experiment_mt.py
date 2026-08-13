"""
run_experiment_mt.py
====================
Entry point for multi-objective ablation experiments.
Edit only this file between experiments; core logic stays in optimizer_core_mt.py.

Ablation ladder (build from E0 upward):
  E0: Baseline      — LLM with no domain knowledge (just objective + bounds)
  E1: + Knowledge   — add material-effects table
  E2: + Rules       — add Pareto navigation situation rules
  E3: + Few-shot    — add examples from dataset
  E4: + Gap-target  — instruct LLM to fill largest Pareto gap each iteration

Usage:
  python run_experiment_mt.py                  # run all experiments
  python run_experiment_mt.py --exp e0_baseline
  python run_experiment_mt.py --skip-nsga      # reuse saved NSGA-II reference
  python run_experiment_mt.py --iters 10       # quick test
  python run_experiment_mt.py --exp e0_baseline --repeat 3
"""

import argparse
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from optimizer_core_mt import (
    ExperimentConfig,
    load_df, get_bounds, get_physics_bounds, load_surrogate,
    run_nsga2, run_llm_pareto,
    compute_metrics, save_results,
)

# ─────────────────────────────────────────────────────────────
# SHARED SETTINGS
# ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
MAX_ITERS      = 30
NSGA_GENS      = 200
NSGA_POP       = 100

DATA_PATH  = r"Concrete_Data_SI.csv"
MODEL_PKL  = r"..\low_carbon_concrete\concrete_catboost_optimized.pkl"
NSGA_CSV   = r"results\nsga2_reference.csv"


def make_cfg(name: str, description: str, **kwargs) -> ExperimentConfig:
    defaults = dict(
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        max_iters=MAX_ITERS,
        nsga_gens=NSGA_GENS,
        nsga_pop=NSGA_POP,
        data_path=DATA_PATH,
        model_pkl=MODEL_PKL,
        output_prefix=os.path.join("results", name),
    )
    defaults.update(kwargs)   # kwargs override defaults (e.g. max_iters=60)
    return ExperimentConfig(name=name, description=description, **defaults)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# ─────────────────────────────────────────────────────────────

EXPERIMENTS = {

    # E0: Bare-minimum baseline — zero domain injection
    "e0_baseline": make_cfg(
        "e0_baseline",
        "LLM with no domain knowledge — only objective description and bounds",
        use_knowledge_table=False,
        use_situation_rules=False,
        use_few_shot=False,
        targeting_mode="none",
        rag_mode="none",
    ),

    # E1: Add material-effects knowledge table
    "e1_knowledge": make_cfg(
        "e1_knowledge",
        "E0 + material effects knowledge table",
        use_knowledge_table=True,
        use_situation_rules=False,
        use_few_shot=False,
        targeting_mode="none",
        rag_mode="none",
    ),

    # E2: Add Pareto navigation rules on top of E1
    "e2_rules": make_cfg(
        "e2_rules",
        "E1 + Pareto navigation situation rules",
        use_knowledge_table=True,
        use_situation_rules=True,
        use_few_shot=False,
        targeting_mode="none",
        rag_mode="none",
    ),

    # E3: Add few-shot examples on top of E2
    "e3_fewshot": make_cfg(
        "e3_fewshot",
        "E2 + few-shot examples from dataset",
        use_knowledge_table=True,
        use_situation_rules=True,
        use_few_shot=True,
        targeting_mode="none",
        rag_mode="none",
    ),

    # E4: Add gap-filling targeting on top of E3
    "e4_gap_target": make_cfg(
        "e4_gap_target",
        "E3 + gap-filling targeting (direct LLM to largest Pareto gap)",
        use_knowledge_table=True,
        use_situation_rules=True,
        use_few_shot=True,
        targeting_mode="gap",
        rag_mode="none",
    ),

    # ── JSON-mode variants (fix parse failures) ───────────────

    # E3-json: E3 + JSON mode with response_schema (enforces {"reasoning","mix"} structure)
    "e3_json30": make_cfg(
        "e3_json30",
        "E3 + JSON mode with response_schema, 30 iters",
        use_knowledge_table=True,
        use_situation_rules=True,
        use_few_shot=True,
        targeting_mode="none",
        rag_mode="none",
        use_json_mode=True,
    ),

    # E3-json-60: JSON mode + schema, 60 iterations to observe HV convergence curve
    "e3_json60": make_cfg(
        "e3_json60",
        "E3 + JSON mode with response_schema, 60 iters — HV convergence curve",
        use_knowledge_table=True,
        use_situation_rules=True,
        use_few_shot=True,
        targeting_mode="none",
        rag_mode="none",
        use_json_mode=True,
        max_iters=60,
    ),

    # E3-60: no JSON mode, 60 iterations (baseline convergence reference)
    "e3_60": make_cfg(
        "e3_60",
        "E3, no JSON mode, 60 iters — baseline HV convergence reference",
        use_knowledge_table=True,
        use_situation_rules=True,
        use_few_shot=True,
        targeting_mode="none",
        rag_mode="none",
        use_json_mode=False,
        max_iters=60,
    ),
}


# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────

def run_one(cfg: ExperimentConfig, df, raw_b, der_b, meta,
            nsga_front: list) -> dict:
    print(f"\n{'#'*62}")
    print(f"# {cfg.name}")
    print(f"# {cfg.description}")
    print(f"{'#'*62}")

    traj, llm_front, n_calls = run_llm_pareto(raw_b, der_b, meta, cfg, df=df)
    metrics = compute_metrics(llm_front, nsga_front, traj, n_calls)
    save_results(traj, llm_front, nsga_front, metrics, cfg)

    print(f"\n  METRICS — {cfg.name}")
    for k, v in metrics.items():
        print(f"    {k:<20}: {v}")

    return {"experiment": cfg.name, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Multi-objective concrete ablation")
    parser.add_argument("--exp",       default=None,
                        help=f"Experiment to run. Choices: {list(EXPERIMENTS)}")
    parser.add_argument("--skip-nsga", action="store_true",
                        help="Load saved NSGA-II reference instead of re-running")
    parser.add_argument("--iters",     type=int, default=None,
                        help="Override MAX_ITERS (e.g. 10 for quick test)")
    parser.add_argument("--repeat",    type=int, default=1,
                        help="Number of repeated runs per experiment")
    args = parser.parse_args()

    if args.iters:
        for cfg in EXPERIMENTS.values():
            cfg.max_iters = args.iters

    # ── Load shared resources ─────────────────────────────────
    print("\n[Setup] Loading data and surrogate ...")
    df       = load_df(DATA_PATH)
    raw_b, der_b = get_bounds(df)
    phys_b   = get_physics_bounds(df)
    meta     = load_surrogate(MODEL_PKL)
    print(f"  Dataset rows : {len(df)}")
    print(f"  PC  range    : [{raw_b['PC']['min']:.1f}, {raw_b['PC']['max']:.1f}] kg/m3")
    print(f"  28d range    : [{df['28day'].min():.1f}, {df['28day'].max():.1f}] MPa")

    # ── NSGA-II reference ─────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    if args.skip_nsga and os.path.exists(NSGA_CSV):
        print(f"\n[NSGA-II] Loading saved reference from {NSGA_CSV} ...")
        nsga_front = pd.read_csv(NSGA_CSV).to_dict("records")
        print(f"  Loaded {len(nsga_front)} Pareto solutions")
    else:
        first_cfg  = next(iter(EXPERIMENTS.values()))
        nsga_front = run_nsga2(raw_b, der_b, phys_b, meta, first_cfg)
        if nsga_front:
            pd.DataFrame(nsga_front).to_csv(NSGA_CSV, index=False)
            print(f"  Saved NSGA reference -> {NSGA_CSV}")

    if not nsga_front:
        print("[Warning] NSGA-II found no feasible Pareto solutions.")

    # ── Select experiments ────────────────────────────────────
    if args.exp:
        if args.exp not in EXPERIMENTS:
            print(f"Unknown: '{args.exp}'. Available: {list(EXPERIMENTS)}")
            return
        to_run = {args.exp: EXPERIMENTS[args.exp]}
    else:
        to_run = EXPERIMENTS

    # ── Run and collect ───────────────────────────────────────
    all_metrics = []
    for exp_name, cfg in to_run.items():
        for rep in range(args.repeat):
            if args.repeat > 1:
                cfg.output_prefix = os.path.join("results", f"{cfg.name}_r{rep+1}")
                print(f"\n  >>> Repeat {rep+1}/{args.repeat}")

            result = run_one(cfg, df, raw_b, der_b, meta, nsga_front)
            result["repeat"] = rep + 1
            all_metrics.append(result)

    # ── Summary across experiments ────────────────────────────
    if len(all_metrics) > 1:
        df_m = pd.DataFrame(all_metrics)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        summary_path = os.path.join("results", f"ablation_summary_{stamp}.csv")
        df_m.to_csv(summary_path, index=False)

        print(f"\n{'='*62}")
        print("  ABLATION SUMMARY")
        print(f"{'='*62}")
        metric_cols = ["HV_llm", "HV_ratio", "GD", "IGD", "spread", "non_dom_rate"]
        for _, row in df_m.iterrows():
            print(f"\n  {row['experiment']}  (rep={row['repeat']}):")
            for col in metric_cols:
                if col in row:
                    print(f"    {col:<18}: {row[col]:.4f}")

        print(f"\n  Summary -> {summary_path}")


if __name__ == "__main__":
    main()
