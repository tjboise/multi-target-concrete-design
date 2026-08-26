"""
run_abl41_n15.py
================
Re-run Step 4.1 Pure-LLM and Hybrid-Replace at N=15 (30 paired reps).
Results saved to:
  results/abl41_purellm_n15_rep{01-30}/
  results/abl41_replace_n15_rep{01-30}/

Usage:
    python run_abl41_n15.py [--repeat 30] [--rep-start 1]
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
POP, GEN, N    = 50, 100, 15


def _cfg(name: str, rep: int, inject_mode: str = "augment") -> HybridConfig:
    return HybridConfig(
        name=name,
        description=f"abl41_n15: rep={rep}",
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        pop_size=POP,
        max_generations=GEN,
        llm_frequency=1,
        stagnation_window=0,
        stagnation_threshold=0.0,
        llm_inject_mode=inject_mode,
        llm_n_solutions=N,
        llm_n_elite=10,
        data_path=DATA_PATH,
        model_pkl=MODEL_PKL,
        output_prefix=os.path.join("results", name),
        use_objectives=True,
        use_knowledge_table=True,
        use_constraints=True,
        use_elite=True,
        use_gap_targeting=True,
        use_json_mode=True,
        seed=rep,
        constraint_mode="feasibility_first",
    )


def _skip(name: str) -> bool:
    return os.path.exists(os.path.join("results", name, "metrics.csv"))


def main():
    parser = argparse.ArgumentParser()
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

    rows = []
    for rep in range(args.rep_start, args.rep_start + args.repeat):
        print(f"\n=== Rep {rep:02d} ===")

        # Pure LLM N=15
        pname = f"abl41_purellm_n15_rep{rep:02d}"
        if _skip(pname):
            hv_p = float(pd.read_csv(os.path.join("results", pname, "metrics.csv"))["HV_hybrid"].iloc[0])
            print(f"  purellm_n15 = {hv_p:.4f}  [loaded]")
        else:
            cfg_p = _cfg(pname, rep)
            res_p = run_pure_llm(raw_b, der_b, phys_b, meta, cfg_p)
            met_p = compute_hybrid_metrics(res_p, nsga_ref)
            save_hybrid_results(res_p, met_p, cfg_p, nsga_ref)
            hv_p  = met_p["HV_hybrid"]
            print(f"  purellm_n15 = {hv_p:.4f}")

        # Hybrid Replace N=15
        rname = f"abl41_replace_n15_rep{rep:02d}"
        if _skip(rname):
            hv_r = float(pd.read_csv(os.path.join("results", rname, "metrics.csv"))["HV_hybrid"].iloc[0])
            print(f"  replace_n15 = {hv_r:.4f}  [loaded]")
        else:
            cfg_r = _cfg(rname, rep, inject_mode="replace")
            res_r = run_hybrid(raw_b, der_b, phys_b, meta, cfg_r)
            met_r = compute_hybrid_metrics(res_r, nsga_ref)
            save_hybrid_results(res_r, met_r, cfg_r, nsga_ref)
            hv_r  = met_r["HV_hybrid"]
            print(f"  replace_n15 = {hv_r:.4f}")

        rows.append({"rep": rep, "HV_purellm_n15": hv_p, "HV_replace_n15": hv_r})

    df_out = pd.DataFrame(rows)
    stamp  = datetime.now().strftime("%Y%m%d_%H%M")
    out    = f"results/abl41_n15_{stamp}.csv"
    df_out.to_csv(out, index=False)

    from scipy import stats
    print(f"\n{'='*60}")
    print(f"  Summary (n={len(df_out)})")
    print(f"{'='*60}")
    for col in ["HV_purellm_n15", "HV_replace_n15"]:
        arr = df_out[col].values
        print(f"  {col}: mean={arr.mean():.4f}  std={arr.std():.4f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
