"""
run_ablation_mt.py
==================
Ablation study for the LLM-hybrid concrete mix design optimizer.
All conditions use every-generation injection (stagnation_window=0), augment mode
-- matching the final grid_everyg strategy.

Step 4.1:  Four-way method comparison (n=30 paired replicates, seed=rep)
             - nsga2   : NSGA-II baseline            [reused from grid_everyg_base_rep*]
             - purellm : Pure-LLM (no NSGA-II evo)   [new run]
             - augment : LLM-hybrid, augment mode     [reused from grid_everyg_hyb_rep*]
             - replace : LLM-hybrid, replace mode     [new run]

Step 4.2:  Leave-one-out prompt ablation (n=30 paired replicates, seed=rep)
           Full prompt = Objectives + KT + Constraints + Elite + Task/Gap
             - full       : complete prompt            [reused from grid_everyg_hyb_rep*]
             - no_obj     : remove Objectives section  [new run]
             - no_kt      : remove Knowledge Table     [new run]
             - no_con     : remove Constraints section [new run]
             - no_elite   : remove Pareto Elite list   [new run]
             - no_task    : remove Task/Gap targeting  [new run]

Usage:
    python run_ablation_mt.py --study 41 --repeat 30
    python run_ablation_mt.py --study 42 --repeat 30
    python run_ablation_mt.py --study all --repeat 30
"""

import argparse, os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from optimizer_core_mt import load_df, get_bounds, get_physics_bounds, load_surrogate
from optimizer_hybrid_mt import (
    HybridConfig, run_hybrid, run_pure_llm,
    compute_hybrid_metrics, save_hybrid_results,
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash-lite"
DATA_PATH      = r"Concrete_Data_SI_clean.csv"
MODEL_PKL      = r"..\low_carbon_concrete\concrete_catboost_optimized_clean.pkl"
NSGA_REF_CSV   = r"results\nsga2_reference.csv"
POP, GEN = 50, 100

EVERYG_BASE_DIR = "results/grid_everyg_base_rep{:02d}"
EVERYG_HYB_DIR  = "results/grid_everyg_hyb_rep{:02d}"


def _load_hv(dir_template: str, rep: int) -> float:
    """Load final HV from an existing grid_everyg result directory."""
    path = os.path.join(dir_template.format(rep), "metrics.csv")
    return float(pd.read_csv(path)["HV_hybrid"].iloc[0])


def _base_cfg(name: str, rep: int, inject_mode: str = "augment",
              use_objectives: bool = True, use_kt: bool = True,
              use_constraints: bool = True, use_elite: bool = True,
              use_gap: bool = True) -> HybridConfig:
    """Every-generation injection config (stagnation_window=0)."""
    return HybridConfig(
        name=name,
        description=f"Ablation: rep={rep}",
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        pop_size=POP,
        max_generations=GEN,
        llm_frequency=1,
        stagnation_window=0,
        stagnation_threshold=0.0,
        llm_inject_mode=inject_mode,
        llm_n_solutions=5,
        llm_n_elite=10,
        data_path=DATA_PATH,
        model_pkl=MODEL_PKL,
        output_prefix=os.path.join("results", name),
        use_objectives=use_objectives,
        use_knowledge_table=use_kt,
        use_constraints=use_constraints,
        use_elite=use_elite,
        use_gap_targeting=use_gap,
        use_json_mode=True,
        seed=rep,
        constraint_mode="feasibility_first",
    )


def run_study_41(reps, rep_start, raw_b, der_b, phys_b, meta, nsga_ref):
    """Step 4.1: four-way comparison — NSGA-II / Pure-LLM / Augment / Replace."""
    print("\n" + "="*70)
    print("  ABLATION 4.1 — NSGA-II  vs  Pure-LLM  vs  Augment  vs  Replace")
    print("  (every-gen injection, n=%d paired reps)" % reps)
    print("="*70)
    rows = []
    for rep in range(rep_start, rep_start + reps):
        print(f"\n  === Rep {rep} ===")

        # NSGA-II baseline — reuse grid_everyg_base
        hv_nsga2 = _load_hv(EVERYG_BASE_DIR, rep)
        print(f"    nsga2   = {hv_nsga2:.4f}  [loaded]")

        # Hybrid augment — reuse grid_everyg_hyb
        hv_aug = _load_hv(EVERYG_HYB_DIR, rep)
        print(f"    augment = {hv_aug:.4f}  [loaded]")

        # Pure LLM — new run
        cfg_p = _base_cfg(f"abl41_purellm_rep{rep:02d}", rep)
        res_p = run_pure_llm(raw_b, der_b, phys_b, meta, cfg_p)
        met_p = compute_hybrid_metrics(res_p, nsga_ref)
        save_hybrid_results(res_p, met_p, cfg_p, nsga_ref)
        hv_pllm = met_p["HV_hybrid"]
        print(f"    purellm = {hv_pllm:.4f}")

        # Hybrid replace — new run
        cfg_r = _base_cfg(f"abl41_replace_rep{rep:02d}", rep, inject_mode="replace")
        res_r = run_hybrid(raw_b, der_b, phys_b, meta, cfg_r)
        met_r = compute_hybrid_metrics(res_r, nsga_ref)
        save_hybrid_results(res_r, met_r, cfg_r, nsga_ref)
        hv_rep = met_r["HV_hybrid"]
        print(f"    replace = {hv_rep:.4f}")

        rows.append({
            "rep":        rep,
            "HV_nsga2":   hv_nsga2,
            "HV_purellm": hv_pllm,
            "HV_augment": hv_aug,
            "HV_replace": hv_rep,
            "dHV_purellm": hv_pllm - hv_nsga2,
            "dHV_augment": hv_aug  - hv_nsga2,
            "dHV_replace": hv_rep  - hv_nsga2,
        })
    return pd.DataFrame(rows)


def run_study_42(reps, rep_start, raw_b, der_b, phys_b, meta, nsga_ref):
    """Step 4.2: leave-one-out prompt ablation (5 components, every-gen augment)."""
    print("\n" + "="*70)
    print("  ABLATION 4.2 — Leave-one-out prompt ablation")
    print("  Full = Objectives + KT + Constraints + Elite + Task/Gap")
    print("  (every-gen injection, n=%d paired reps)" % reps)
    print("="*70)

    VARIANTS = [
        # (label, dir_name, kwargs_override)
        ("no_obj",   "abl42_no_obj_rep{:02d}",   dict(use_objectives=False)),
        ("no_kt",    "abl42_no_kt_rep{:02d}",    dict(use_kt=False)),
        ("no_con",   "abl42_no_con_rep{:02d}",   dict(use_constraints=False)),
        ("no_elite", "abl42_no_elite_rep{:02d}", dict(use_elite=False)),
        ("no_task",  "abl42_no_task_rep{:02d}",  dict(use_gap=False)),
    ]

    rows = []
    for rep in range(rep_start, rep_start + reps):
        print(f"\n  === Rep {rep} ===")

        hv_base = _load_hv(EVERYG_BASE_DIR, rep)
        hv_full = _load_hv(EVERYG_HYB_DIR,  rep)
        print(f"    base = {hv_base:.4f}  [loaded]   full = {hv_full:.4f}  [loaded]")

        row = {"rep": rep, "HV_base": hv_base, "HV_full": hv_full,
               "dHV_full": hv_full - hv_base}

        for label, tpl, kwargs in VARIANTS:
            cfg = _base_cfg(tpl.format(rep), rep, **kwargs)
            res = run_hybrid(raw_b, der_b, phys_b, meta, cfg)
            met = compute_hybrid_metrics(res, nsga_ref)
            save_hybrid_results(res, met, cfg, nsga_ref)
            hv  = met["HV_hybrid"]
            row[f"HV_{label}"]  = hv
            row[f"dHV_{label}"] = hv - hv_base
            print(f"    {label:<10} = {hv:.4f}  (dHV={hv-hv_base:+.4f})")

        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, study: str):
    from scipy import stats
    print(f"\n{'='*70}")
    print(f"  ABLATION {study} SUMMARY")
    print(f"{'='*70}")
    for col in df.columns:
        if col == "rep" or not col.startswith("dHV"):
            continue
        d = df[col]
        t, p = stats.ttest_1samp(d, 0)
        print(f"  {col:<20}: mean={d.mean():+.4f}  std={d.std():.4f}  p={p:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study",     default="all", choices=["41", "42", "all"])
    parser.add_argument("--repeat",    type=int, default=30)
    parser.add_argument("--rep-start", type=int, default=1)
    args = parser.parse_args()

    print("[Setup] Loading data and surrogate...")
    df           = load_df(DATA_PATH)
    raw_b, der_b = get_bounds(df)
    phys_b       = get_physics_bounds(df)
    meta         = load_surrogate(MODEL_PKL)
    nsga_ref     = pd.read_csv(NSGA_REF_CSV).to_dict("records") if os.path.exists(NSGA_REF_CSV) else []
    os.makedirs("results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    if args.study in ("41", "all"):
        df41 = run_study_41(args.repeat, args.rep_start, raw_b, der_b, phys_b, meta, nsga_ref)
        print_summary(df41, "4.1")
        df41.to_csv(f"results/ablation_41_{stamp}.csv", index=False)
        print(f"  -> results/ablation_41_{stamp}.csv")

    if args.study in ("42", "all"):
        df42 = run_study_42(args.repeat, args.rep_start, raw_b, der_b, phys_b, meta, nsga_ref)
        print_summary(df42, "4.2")
        df42.to_csv(f"results/ablation_42_{stamp}.csv", index=False)
        print(f"  -> results/ablation_42_{stamp}.csv")


if __name__ == "__main__":
    main()
