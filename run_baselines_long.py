"""
run_baselines_long.py
=====================
Run MOEA/D and MOPSO for 500 generations, 30 replicates each.
Results saved to:
  results/moead_long_g500_rep{01-30}/
  results/mopso_long_g500_rep{01-30}/

Usage:
    python run_baselines_long.py --algo moead --start 1 --end 30
    python run_baselines_long.py --algo mopso  --start 1 --end 30
"""
import os, argparse
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from optimizer_core_mt import load_df, get_bounds, get_physics_bounds, load_surrogate

DATA_PATH = r"Concrete_Data_SI_clean.csv"
MODEL_PKL = r"..\low_carbon_concrete\concrete_catboost_optimized_clean.pkl"

parser = argparse.ArgumentParser()
parser.add_argument("--algo",  choices=["moead", "mopso", "nsga2"], required=True)
parser.add_argument("--start", type=int, default=1)
parser.add_argument("--end",   type=int, default=30)
args = parser.parse_args()

print("[Setup] Loading data and surrogate...")
df     = load_df(DATA_PATH)
raw_b, der_b = get_bounds(df)
phys_b = get_physics_bounds(df)
meta   = load_surrogate(MODEL_PKL)

if args.algo == "moead":
    from optimizer_moead import run_moead as run_algo
    label = "moead_long_g500"
    use_hybrid = False
elif args.algo == "mopso":
    from optimizer_mopso import run_mopso as run_algo
    label = "mopso_long_g500"
    use_hybrid = False
else:  # nsga2
    from optimizer_hybrid_mt import HybridConfig, run_hybrid
    label = "baseline_long_g500"
    use_hybrid = True

for rep in range(args.start, args.end + 1):
    name    = f"{label}_rep{rep:02d}"
    out_dir = os.path.join("results", name)
    pf_path = os.path.join(out_dir, "pareto_front.csv")

    if os.path.exists(pf_path):
        print(f"[skip] {name} already done.")
        continue

    print(f"\n=== {name} ===")
    if use_hybrid:
        cfg = HybridConfig(name=name, llm_frequency=0, max_generations=500,
                           seed=rep, constraint_mode="feasibility_first")
        result = run_hybrid(raw_b, der_b, phys_b, meta, cfg)
    else:
        result = run_algo(raw_b, der_b, phys_b, meta,
                          pop_size=50, max_generations=500, seed=rep,
                          constraint_mode="feasibility_first")

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(result["hv_history"]).to_csv(
        os.path.join(out_dir, "hv_history.csv"), index=False)
    pd.DataFrame(result["final_pareto"]).to_csv(
        os.path.join(out_dir, "pareto_front.csv"), index=False)

    final_hv = result["hv_history"][-1]["hv"]
    print(f"  Done — final HV: {final_hv:.4f}  Pareto: {len(result['final_pareto'])} pts")

print("\nAll done.")
