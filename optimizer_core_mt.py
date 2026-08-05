"""
optimizer_core_mt.py
====================
Multi-objective concrete mix design via iterative LLM Pareto search.

Objectives (2-objective):
    1. Minimize GWP  (kg CO2-eq / m3)
    2. Maximize 28-day compressive strength (MPa)

These conflict: more binder -> higher strength but higher GWP.
Goal: discover the Pareto front where neither objective can improve
      without the other worsening.

Surrogate : CatBoost-Chain (concrete_catboost_optimized.pkl, unit=kg/m3)
Reference  : NSGA-II via pymoo
Metric     : Hypervolume (HV), GD, IGD, spread

Ablation flags in ExperimentConfig (all off = E0 baseline):
    use_knowledge_table  : inject material-effects domain knowledge
    use_situation_rules  : inject Pareto-navigation rules
    use_few_shot         : inject examples from dataset
    targeting_mode       : "none" | "region" | "gap"
    rag_mode             : "none" | "static" | "dynamic"
"""

import json, re, time, warnings, joblib, os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

LB_YD3_TO_KG_M3 = 0.5933

GWP_FACTORS = {
    "PC": 1.048, "FA": 0.328, "SC": 0.264,
    "FAGG": 0.0026, "CAGG": 0.0037,
    "WATER": 0.0, "AEA": 0.0, "WR_HR": 0.0, "WR": 0.0, "ACC": 0.0,
}

RAW_VARS     = ["PC", "FA", "SC", "FAGG", "CAGG", "WATER", "AEA", "WR_HR", "WR", "ACC"]
DERIVED_VARS = ["w/b", "b/a", "SCM%", "CAGG%", "FAGG%", "PC%", "FA%", "SC%"]

# HV reference point in [GWP, -28d_strength] minimization space.
# Must be WORSE (larger) than any feasible solution in both dimensions.
#   GWP ref = 500  → all solutions have GWP < 500
#   -28d ref = -10 → all solutions have -28d < -10 (i.e. strength > 10 MPa)
HV_REF_POINT = np.array([500.0, -10.0])


# ─────────────────────────────────────────────────────────────
# EXPERIMENT CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    # Identity
    name: str = "e0_baseline"
    description: str = "Simplest LLM: no domain knowledge, no rules, no examples"

    # Problem
    min_strength_filter: float = 20.0   # discard solutions weaker than this
    max_iters: int = 30                 # LLM proposals to collect

    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    temperature: float = 0.9

    # NSGA-II reference
    nsga_gens: int = 200
    nsga_pop: int = 100

    # Ablation flags (all False = E0 baseline)
    use_knowledge_table: bool = False
    use_situation_rules: bool = False
    use_few_shot: bool = False
    targeting_mode: str = "none"    # "none" | "region" | "gap"
    rag_mode: str = "none"          # "none" | "static" | "dynamic"
    rag_k: int = 3
    use_json_mode: bool = False     # force JSON response via response_mime_type

    # Paths
    data_path: str = r"Super_Cleaned_Concrete_Data - backup.csv"
    model_pkl: str = r"..\low_carbon_concrete\concrete_catboost_optimized.pkl"
    output_prefix: str = r"results\e0_baseline"


# ─────────────────────────────────────────────────────────────
# 1. DATA & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in RAW_VARS:
        if col in df.columns:
            df[col] = df[col] * LB_YD3_TO_KG_M3
    return _add_derived(df)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tb  = df["PC"] + df["FA"] + df["SC"]
    agg = df["FAGG"] + df["CAGG"]
    df["TOTAL_BINDER"] = tb
    df["w/b"]   = df["WATER"] / tb
    df["b/a"]   = tb / agg
    df["SCM%"]  = (df["FA"] + df["SC"]) / tb
    df["CAGG%"] = df["CAGG"] / agg
    df["FAGG%"] = df["FAGG"] / agg
    df["PC%"]   = df["PC"]   / tb
    df["FA%"]   = df["FA"]   / tb
    df["SC%"]   = df["SC"]   / tb
    return df


def get_bounds(df: pd.DataFrame):
    raw = {v: {"min": float(df[v].min()), "max": float(df[v].max())} for v in RAW_VARS}
    der = {v: {"min": float(df[v].min()), "max": float(df[v].max())} for v in DERIVED_VARS}
    return raw, der


# ─────────────────────────────────────────────────────────────
# 2. CATBOOST SURROGATE  (same as low_carbon_concrete)
# ─────────────────────────────────────────────────────────────

def load_surrogate(pkl: str) -> dict:
    meta = joblib.load(pkl)
    assert "models" in meta and "feature_names" in meta, \
        f"Invalid surrogate format in {pkl}"
    print(f"  Surrogate unit : {meta.get('unit', 'lb/yd3')}")
    print(f"  Feature count  : {len(meta['feature_names'])}")
    return meta


def _engineer_one(mix: dict) -> dict:
    m  = dict(mix)
    tb = m["PC"] + m["FA"] + m["SC"]
    ag = m["FAGG"] + m["CAGG"]
    e  = 1e-9
    m["TOTAL_BINDER"] = tb
    m["w/b"]   = m["WATER"] / (tb + e)
    m["b/a"]   = tb / (ag + e)
    m["SCM%"]  = (m["FA"] + m["SC"]) / (tb + e)
    m["CAGG%"] = m["CAGG"] / (ag + e)
    m["FAGG%"] = m["FAGG"] / (ag + e)
    m["PC%"]   = m["PC"]   / (tb + e)
    m["FA%"]   = m["FA"]   / (tb + e)
    m["SC%"]   = m["SC"]   / (tb + e)
    return m


def predict(meta: dict, mix: dict) -> dict:
    fn  = meta["feature_names"]
    mdl = meta["models"]
    # concrete_catboost_optimized.pkl has unit='kg/m3' — no conversion needed
    if meta.get("unit", "lb/yd3") == "kg/m3":
        mix_in = mix
    else:
        mix_in = {k: (v / LB_YD3_TO_KG_M3 if k in RAW_VARS else v)
                  for k, v in mix.items()}
    m = _engineer_one(mix_in)

    r7  = pd.DataFrame([{k: m.get(k, 0.) for k in fn}])
    p7  = float(mdl["7day"].predict(r7)[0])

    m["7day"] = p7
    r28 = pd.DataFrame([{k: m.get(k, 0.) for k in fn + ["7day"]}])
    p28 = float(mdl["28day"].predict(r28)[0])

    m["28day"] = p28
    r56 = pd.DataFrame([{k: m.get(k, 0.) for k in fn + ["28day"]}])
    p56 = float(mdl["56day"].predict(r56)[0])

    return {"7day": round(p7, 2), "28day": round(p28, 2), "56day": round(p56, 2)}


def compute_gwp(mix: dict) -> float:
    return round(sum(mix.get(k, 0.) * v for k, v in GWP_FACTORS.items()), 2)


def get_derived(mix: dict) -> dict:
    m = _engineer_one(mix)
    return {k: round(m[k], 5) for k in DERIVED_VARS}


# ─────────────────────────────────────────────────────────────
# 3. BOUNDS CHECK  (no strength constraint — it is now an objective)
# ─────────────────────────────────────────────────────────────

def check_bounds(mix: dict, raw_b: dict, der_b: dict) -> dict:
    rv = {v: {"val": mix[v], "min": b["min"], "max": b["max"]}
          for v, b in raw_b.items()
          if mix.get(v, 0) < b["min"] - 0.5 or mix.get(v, 0) > b["max"] + 0.5}

    dv_vals = get_derived(mix)
    dv = {}
    for v, b in der_b.items():
        val = dv_vals[v]
        tol = (b["max"] - b["min"]) * 0.01 + 1e-6
        if val < b["min"] - tol or val > b["max"] + tol:
            dv[v] = {"val": round(val, 4), "min": round(b["min"], 4),
                     "max": round(b["max"], 4)}

    return {"raw_v": rv, "der_v": dv, "in_bounds": not rv and not dv}


# ─────────────────────────────────────────────────────────────
# 4. PARETO FRONT MANAGEMENT
# ─────────────────────────────────────────────────────────────

def dominates(a: dict, b: dict) -> bool:
    """Does a dominate b? (minimize GWP, maximize 28d strength)"""
    return (
        a["gwp"] <= b["gwp"]
        and a["pred_28day"] >= b["pred_28day"]
        and (a["gwp"] < b["gwp"] or a["pred_28day"] > b["pred_28day"])
    )


def update_pareto_front(front: list, sol: dict):
    """
    Add sol to front if non-dominated; remove any solutions sol dominates.
    Returns (new_front, is_non_dominated).
    """
    for existing in front:
        if dominates(existing, sol):
            return front, False         # sol is dominated — skip
    new_front = [s for s in front if not dominates(sol, s)]
    new_front.append(sol)
    return new_front, True


def compute_hypervolume(pareto_front: list) -> float:
    if not pareto_front:
        return 0.0
    try:
        from pymoo.indicators.hv import HV
        F = np.array([[s["gwp"], -s["pred_28day"]] for s in pareto_front])
        return round(float(HV(ref_point=HV_REF_POINT)(F)), 4)
    except Exception as exc:
        print(f"  [Warning] HV failed: {exc}")
        return 0.0


def pareto_table_str(front: list) -> str:
    if not front:
        return "  (empty — no non-dominated solutions yet)"
    rows = sorted(front, key=lambda s: s["gwp"])
    lines = ["  #    GWP (kg CO2/m3)   28d (MPa)"]
    lines += [f"  {i:<4} {s['gwp']:>14.2f}   {s['pred_28day']:>9.2f}"
              for i, s in enumerate(rows, 1)]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 5. NSGA-II REFERENCE BASELINE
# ─────────────────────────────────────────────────────────────

def run_nsga2(raw_b: dict, der_b: dict, meta: dict, cfg: ExperimentConfig) -> list:
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize as pymoo_min
        from pymoo.termination import get_termination
    except ImportError:
        raise ImportError("pymoo not installed — run: pip install pymoo")

    xl  = np.array([raw_b[v]["min"] for v in RAW_VARS])
    xu  = np.array([raw_b[v]["max"] for v in RAW_VARS])
    n_c = len(DERIVED_VARS) * 2     # lower + upper for each derived ratio

    class MixMOProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(RAW_VARS), n_obj=2,
                             n_ieq_constr=n_c, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            F, G = [], []
            for row in X:
                mix = dict(zip(RAW_VARS, row))
                pr  = predict(meta, mix)
                gwp = compute_gwp(mix)
                F.append([gwp, -pr["28day"]])    # minimize GWP, minimize -28d

                dv = get_derived(mix)
                gc = []
                for v in DERIVED_VARS:
                    b = der_b[v]
                    gc += [b["min"] - dv[v], dv[v] - b["max"]]
                G.append(gc)
            out["F"] = np.array(F)
            out["G"] = np.array(G)

    print(f"\n[NSGA-II] {cfg.nsga_gens} generations × pop={cfg.nsga_pop} ...")
    res = pymoo_min(
        MixMOProblem(), NSGA2(pop_size=cfg.nsga_pop),
        termination=get_termination("n_gen", cfg.nsga_gens),
        seed=42, verbose=False,
    )

    raw_candidates = []
    if res.X is not None:
        if res.G is not None:
            feasible_mask = res.G.max(axis=1) <= 0
            X_f = res.X[feasible_mask]
        else:
            X_f = res.X
        for x in X_f:
            mix = {k: round(float(v), 2) for k, v in zip(RAW_VARS, x)}
            pr  = predict(meta, mix)
            gwp = compute_gwp(mix)
            raw_candidates.append({
                **mix,
                "pred_7day":  pr["7day"],
                "pred_28day": pr["28day"],
                "pred_56day": pr["56day"],
                "gwp":        gwp,
            })

    # Distil to true Pareto front
    true_front: list = []
    for sol in raw_candidates:
        true_front, _ = update_pareto_front(true_front, sol)

    hv = compute_hypervolume(true_front)
    print(f"  Pareto solutions : {len(true_front)}")
    if true_front:
        print(f"  GWP  range : {min(s['gwp'] for s in true_front):.1f} – "
              f"{max(s['gwp'] for s in true_front):.1f}  kg/m3")
        print(f"  28d  range : {min(s['pred_28day'] for s in true_front):.1f} – "
              f"{max(s['pred_28day'] for s in true_front):.1f}  MPa")
    print(f"  HV             : {hv:.4f}")
    return true_front


# ─────────────────────────────────────────────────────────────
# 6. PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────

# ── Optional domain-knowledge blocks (injected in ablation E1+) ──

KNOWLEDGE_TABLE = """\
MATERIAL EFFECTS TABLE
=======================
PC  (Portland cement)  GWP 1.048 kg CO2/kg  — primary strength driver; highest CO2
SC  (Slag cement/GGBS) GWP 0.264 kg CO2/kg  — moderate strength; best CO2 substitute
FA  (Fly ash)          GWP 0.328 kg CO2/kg  — slow 28d gain; second CO2 substitute
WATER                  GWP 0.000            — more water = higher w/b = lower strength
FAGG/CAGG (aggregates) GWP ~0.003           — nearly zero CO2; dilutes binder paste
WR / WR_HR             GWP 0.000            — enables water reduction at zero CO2 cost
ACC (accelerator)      GWP 0.000            — direct strength boost at zero CO2 cost

GWP (kg CO2/m3) = PC*1.048 + FA*0.328 + SC*0.264 + CAGG*0.0037 + FAGG*0.0026

TO PUSH THE LOW-GWP END OF THE PARETO FRONT:
  - Substitute PC with SC  (saves 1.048-0.264=0.784 kg CO2 per kg, small strength loss)
  - Increase FAGG+CAGG (dilutes total binder, cuts GWP, reduces strength)
  - Use WR/WR_HR to cut WATER (no GWP cost; raises strength, creates headroom)

TO PUSH THE HIGH-STRENGTH END OF THE PARETO FRONT:
  - Increase PC (strong strength boost, ~1.048 kg CO2 per kg added)
  - Reduce WATER (raises strength at zero GWP cost)
  - Add ACC 100–400 kg/m3 (direct strength boost, zero GWP)
"""

SITUATION_RULES = """\
PARETO NAVIGATION RULES
========================
Before proposing, examine the current Pareto front table:

SITUATION 1 — Low-GWP end can go lower:
  The minimum GWP point still has relatively high PC content.
  -> Swap more PC->SC, or increase FAGG+CAGG to push GWP below current minimum.

SITUATION 2 — High-strength end can go higher:
  The maximum strength point doesn't use high PC or has high water.
  -> Increase PC, reduce WATER, or add ACC to push strength above current maximum.

SITUATION 3 — Large gap between adjacent Pareto points (GWP gap > 25 kg/m3):
  -> Target the midpoint (GWP, 28d) of the gap; blend ingredients between the two points.

SITUATION 4 — Stagnation (last 4 proposals all dominated):
  -> Make a bold move: change at least 2 ingredients by >20 kg/m3.
  -> Or try a very different binder type (e.g. pure PC+SC without FA, or high FA).
"""


def build_system_prompt(raw_b: dict, der_b: dict, few_shot: list,
                        cfg: ExperimentConfig) -> str:
    raw_lines = "\n".join(
        f"  {v:<8} [{b['min']:8.2f}, {b['max']:8.2f}]  kg/m3"
        for v, b in raw_b.items()
    )
    der_lines = "\n".join(
        f"  {v:<8} [{b['min']:8.5f}, {b['max']:8.5f}]"
        for v, b in der_b.items()
    )

    if cfg.use_knowledge_table:
        knowledge_block = KNOWLEDGE_TABLE
    else:
        knowledge_block = (
            "GWP (kg CO2/m3) = PC*1.048 + FA*0.328 + SC*0.264 "
            "+ CAGG*0.0037 + FAGG*0.0026\n"
        )

    situation_block = SITUATION_RULES if cfg.use_situation_rules else ""

    if cfg.use_few_shot and few_shot:
        ex_lines = ["REFERENCE MIXES FROM DATASET (real historical data)\n" + "="*60]
        for ex in few_shot:
            mix_str = "  ".join(f"{k}={ex[k]}" for k in RAW_VARS)
            ex_lines.append(
                f"  {mix_str}\n"
                f"  -> GWP={ex['gwp']:.1f} kg CO2/m3   28d={ex['pred_28day']:.1f} MPa"
            )
        few_shot_block = "\n".join(ex_lines)
    else:
        few_shot_block = ""

    return (
        "You are an expert concrete mix design engineer.\n\n"
        "MULTI-OBJECTIVE OPTIMISATION\n"
        "=============================\n"
        "OBJECTIVE 1 : MINIMIZE GWP (kg CO2-eq/m3)\n"
        "OBJECTIVE 2 : MAXIMIZE 28-day compressive strength (MPa)\n\n"
        "These objectives CONFLICT. More cementitious binder raises both strength and GWP.\n"
        "Your task is to discover the PARETO FRONT — mixes where improving one\n"
        "objective necessarily worsens the other.\n\n"
        "DOMINANCE RULE: Mix A dominates Mix B when:\n"
        "  A.GWP <= B.GWP  AND  A.28d >= B.28d  (at least one strict inequality)\n"
        "Propose mixes that are NON-DOMINATED by all existing Pareto points.\n\n"
        + knowledge_block + "\n"
        "VARIABLE BOUNDS (kg/m3)\n"
        "========================\n"
        + raw_lines + "\n\n"
        "DERIVED RATIO BOUNDS (must be satisfied)\n"
        "=========================================\n"
        + der_lines + "\n"
        "  w/b   = WATER / (PC+FA+SC)\n"
        "  b/a   = (PC+FA+SC) / (FAGG+CAGG)\n"
        "  SCM%  = (FA+SC) / (PC+FA+SC)\n"
        "  CAGG% = CAGG / (FAGG+CAGG)\n"
        "  FAGG% = FAGG / (FAGG+CAGG)\n"
        "  PC%   = PC / (PC+FA+SC)\n"
        "  FA%   = FA / (PC+FA+SC)\n"
        "  SC%   = SC / (PC+FA+SC)\n\n"
        + situation_block + "\n"
        + few_shot_block + "\n"
        "OUTPUT FORMAT — STRICTLY REQUIRED\n"
        "===================================\n"
        "Return ONLY a valid JSON object. No markdown, no extra text.\n\n"
        '{\n'
        '  "reasoning": "<what you changed and why, max 100 words>",\n'
        '  "mix": {\n'
        '    "PC": <number>, "FA": <number>, "SC": <number>,\n'
        '    "FAGG": <number>, "CAGG": <number>, "WATER": <number>,\n'
        '    "AEA": <number>, "WR_HR": <number>, "WR": <number>, "ACC": <number>\n'
        '  }\n'
        '}\n'
    )


def _build_target_block(front: list, targeting_mode: str) -> str:
    if targeting_mode == "none" or len(front) < 2:
        return ""
    sorted_f = sorted(front, key=lambda s: s["gwp"])

    if targeting_mode == "region":
        n   = len(sorted_f)
        mid = n // 2
        return (
            "SUGGESTED TARGET REGION (pick one to explore):\n"
            f"  LOW-GWP  : GWP < {sorted_f[0]['gwp']:.1f} kg/m3  (current minimum)\n"
            f"  MIDDLE   : GWP ~ {sorted_f[mid]['gwp']:.1f}  28d ~ {sorted_f[mid]['pred_28day']:.1f} MPa\n"
            f"  HIGH-STR : 28d > {sorted_f[-1]['pred_28day']:.1f} MPa  (current maximum)\n"
        )

    elif targeting_mode == "gap":
        # Find the largest GWP gap between adjacent Pareto points
        gaps = [(sorted_f[i+1]["gwp"] - sorted_f[i]["gwp"], i)
                for i in range(len(sorted_f) - 1)]
        _, idx = max(gaps)
        lo, hi = sorted_f[idx], sorted_f[idx + 1]
        return (
            f"FILL THE LARGEST GAP between Pareto points #{idx+1} and #{idx+2}:\n"
            f"  Point #{idx+1}: GWP={lo['gwp']:.1f}  28d={lo['pred_28day']:.1f} MPa\n"
            f"  Point #{idx+2}: GWP={hi['gwp']:.1f}  28d={hi['pred_28day']:.1f} MPa\n"
            f"  -> Target: GWP ~ {(lo['gwp']+hi['gwp'])/2:.1f}  "
            f"28d ~ {(lo['pred_28day']+hi['pred_28day'])/2:.1f} MPa\n"
        )
    return ""


def build_feedback(it: int, max_it: int, mix: dict, preds: dict,
                   gwp: float, bc: dict, front: list,
                   is_nd: bool, hv: float,
                   cfg: ExperimentConfig) -> str:

    dom_status = (
        "NON-DOMINATED -- added to Pareto front"
        if is_nd else
        "DOMINATED -- a Pareto point has both lower GWP and higher strength"
    )
    bounds_warn = ""
    if not bc["in_bounds"]:
        violated = list(bc["raw_v"]) + list(bc["der_v"])
        bounds_warn = (
            "*** BOUNDS VIOLATION ***\n"
            f"  Variables out of range: {', '.join(violated)}\n"
            "  Fix these in your next proposal.\n\n"
        )

    mix_line = "  " + "  ".join(f"{k}={mix.get(k,0):.1f}" for k in RAW_VARS)
    target_block = _build_target_block(front, cfg.targeting_mode)

    return (
        f"=== ITERATION {it} / {max_it} ===\n\n"
        "Your last proposal:\n"
        f"{mix_line}\n"
        f"  GWP    = {gwp:.2f} kg CO2/m3\n"
        f"  28-day = {preds['28day']:.2f} MPa\n"
        f"  7-day  = {preds['7day']:.2f} MPa\n"
        f"  Status : {dom_status}\n\n"
        + bounds_warn
        + f"Current Pareto Front ({len(front)} solutions, sorted by GWP):\n"
        + pareto_table_str(front) + "\n\n"
        f"Hypervolume : {hv:.4f}  (larger = more objective space covered)\n\n"
        + target_block
        + "=== YOUR TASK ===\n"
        "Propose a mix that adds a NON-DOMINATED point to the Pareto front.\n"
        "Non-dominated: no Pareto point has BOTH lower GWP AND higher 28d strength.\n\n"
        "Options:\n"
        "  -> Lower GWP  : substitute PC with SC/FA, or increase FAGG+CAGG\n"
        "  -> Higher 28d : increase PC, reduce WATER, or add ACC (zero GWP)\n"
        "  -> Fill gap   : target a (GWP, 28d) midpoint between two adjacent Pareto points\n\n"
        "Output ONLY the JSON object."
    )


FIRST_TURN = """\
Start Pareto search. Propose your first concrete mix.

Your goal: explore the trade-off between GWP (minimize) and 28-day strength (maximize).
Over {max_iters} iterations you will build a Pareto front.

For your first proposal, choose ONE strategy:
  A) LOW-GWP    : PC ~ 100, SC ~ 200 kg/m3; aim for GWP < 220, moderate strength
  B) HIGH-STR   : PC ~ 250, SC ~ 150 kg/m3; aim for 28d > 60 MPa, higher GWP acceptable
  C) BALANCED   : somewhere in between

Ensure all variable bounds and derived ratio bounds are satisfied.
Output ONLY the JSON object.\
"""


# ─────────────────────────────────────────────────────────────
# 7. HELPERS
# ─────────────────────────────────────────────────────────────

def clip_mix(mix: dict, raw_b: dict):
    clean, notes = {}, []
    for v in RAW_VARS:
        b   = raw_b[v]
        val = float(mix.get(v, b["min"]))
        clp = float(np.clip(val, b["min"], b["max"]))
        if abs(clp - val) > 0.5:
            notes.append(f"  {v}: {val:.1f} clipped to {clp:.1f}")
        clean[v] = round(clp, 2)
    return clean, notes


def parse_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def select_few_shot(df: pd.DataFrame, n: int = 5) -> list:
    """Pick n examples spread across the GWP-strength trade-off space."""
    sub = df.dropna(subset=["28day"]).copy()
    sub["gwp"] = sub.apply(compute_gwp, axis=1)
    sub = sub.sort_values("gwp").reset_index(drop=True)
    indices = np.linspace(0, len(sub) - 1, min(n, len(sub)), dtype=int)
    result = []
    for idx in indices:
        row = sub.iloc[idx]
        ex = {k: round(float(row[k]), 2) for k in RAW_VARS}
        ex["pred_28day"] = round(float(row["28day"]), 1)
        ex["gwp"]        = round(float(row["gwp"]), 1)
        result.append(ex)
    return result


# ─────────────────────────────────────────────────────────────
# 8. MAIN LLM PARETO LOOP
# ─────────────────────────────────────────────────────────────

def run_llm_pareto(raw_b: dict, der_b: dict, meta: dict,
                   cfg: ExperimentConfig,
                   df: pd.DataFrame = None) -> tuple:
    """
    Run the LLM iterative Pareto search.
    Returns: (trajectory, pareto_front, total_calls)
    """
    from google import genai
    from google.genai import types as genai_types

    few_shot   = select_few_shot(df) if (cfg.use_few_shot and df is not None) else []
    sys_prompt = build_system_prompt(raw_b, der_b, few_shot, cfg)

    _client = genai.Client(api_key=cfg.gemini_api_key)

    # Build response schema once so _make_chat can reference it
    _json_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(type=genai_types.Type.STRING),
            "mix": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    v: genai_types.Schema(type=genai_types.Type.NUMBER)
                    for v in RAW_VARS
                },
                required=RAW_VARS,
            ),
        },
        required=["reasoning", "mix"],
    ) if cfg.use_json_mode else None

    def _make_chat(temp):
        extra = (
            {
                "response_mime_type": "application/json",
                "response_schema": _json_schema,
            }
            if cfg.use_json_mode else {}
        )
        return _client.chats.create(
            model=cfg.gemini_model,
            config=genai_types.GenerateContentConfig(
                system_instruction=sys_prompt,
                temperature=temp,
                max_output_tokens=1024,
                **extra,
            ),
        )

    print(f"\n{'='*62}")
    print(f"  Experiment : {cfg.name}")
    print(f"  Model      : {cfg.gemini_model}  |  Iters: {cfg.max_iters}")
    print(f"  Knowledge  : {cfg.use_knowledge_table}  |  "
          f"Rules: {cfg.use_situation_rules}  |  "
          f"Few-shot: {cfg.use_few_shot}  |  "
          f"Targeting: {cfg.targeting_mode}")
    print(f"{'='*62}\n")

    chat         = _make_chat(cfg.temperature)
    trajectory   = []
    pareto_front = []
    total_calls  = 0
    parse_fails  = 0

    print(f"  {'It':>3}  {'GWP':>8}  {'28d MPa':>8}  {'Status':<22}  {'HV':>8}  Pareto#")
    print("  " + "-"*70)

    it = 0
    while it < cfg.max_iters:

        # ── Build message ─────────────────────────────────────
        if it == 0:
            user_msg = FIRST_TURN.format(max_iters=cfg.max_iters)
        else:
            last = trajectory[-1]
            user_msg = build_feedback(
                it, cfg.max_iters,
                {k: last[k] for k in RAW_VARS},
                {"7day": last["pred_7day"], "28day": last["pred_28day"],
                 "56day": last["pred_56day"]},
                last["gwp"],
                {"in_bounds": last["in_bounds"],
                 "raw_v": last.get("_raw_v", {}),
                 "der_v": last.get("_der_v", {})},
                pareto_front,
                last["is_non_dominated"],
                compute_hypervolume(pareto_front),
                cfg,
            )

        # ── Call LLM ─────────────────────────────────────────
        try:
            resp     = chat.send_message(user_msg)
            raw_text = resp.text
        except Exception as exc:
            err  = str(exc)
            wait = 60 if "429" in err else 15
            print(f"  API error ({err[:50]}) — waiting {wait}s ...")
            time.sleep(wait)
            continue

        # ── Parse ─────────────────────────────────────────────
        parsed = parse_json(raw_text)
        if parsed is None or "mix" not in parsed:
            parse_fails += 1
            try:
                resp2  = chat.send_message(
                    "Output ONLY the JSON object with keys 'reasoning' and 'mix'.")
                parsed = parse_json(resp2.text)
            except Exception:
                pass
            if parsed is None or "mix" not in parsed:
                continue

        # ── Evaluate ──────────────────────────────────────────
        mix, clip_notes = clip_mix(parsed["mix"], raw_b)
        preds = predict(meta, mix)
        total_calls += 1
        gwp   = compute_gwp(mix)
        bc    = check_bounds(mix, raw_b, der_b)
        it   += 1

        # Try to add to Pareto front (only if in-bounds and strength > filter)
        is_nd = False
        if bc["in_bounds"] and preds["28day"] >= cfg.min_strength_filter:
            sol_for_pareto = {
                **{k: mix[k] for k in RAW_VARS},
                "pred_7day":  preds["7day"],
                "pred_28day": preds["28day"],
                "pred_56day": preds["56day"],
                "gwp":        gwp,
                "iteration":  it,
            }
            pareto_front, is_nd = update_pareto_front(pareto_front, sol_for_pareto)

        hv = compute_hypervolume(pareto_front)

        record = {
            "iteration":       it,
            "reasoning":       parsed.get("reasoning", ""),
            **{k: mix[k] for k in RAW_VARS},
            **{f"d_{k}": v for k, v in get_derived(mix).items()},
            "pred_7day":       preds["7day"],
            "pred_28day":      preds["28day"],
            "pred_56day":      preds["56day"],
            "gwp":             gwp,
            "in_bounds":       bc["in_bounds"],
            "violations":      ",".join(list(bc["raw_v"]) + list(bc["der_v"])),
            "is_non_dominated": is_nd,
            "hv_after":        hv,
            "pareto_size":     len(pareto_front),
            # stash for feedback rebuild (not saved to CSV)
            "_raw_v":          bc["raw_v"],
            "_der_v":          bc["der_v"],
        }
        trajectory.append(record)

        status = ("NON-DOM" if is_nd
                  else ("OOB" if not bc["in_bounds"] else "dominated"))
        print(f"  {it:3d}  {gwp:8.2f}  {preds['28day']:8.2f}  "
              f"{status:<22}  {hv:8.4f}  {len(pareto_front)}")

        if clip_notes:
            print("  [clipped]", " | ".join(clip_notes))

        time.sleep(2)

    print(f"\n  Calls: {total_calls}  Parse fails: {parse_fails}")
    print(f"  Pareto front: {len(pareto_front)} solutions  HV: {compute_hypervolume(pareto_front):.4f}")

    return trajectory, pareto_front, total_calls


# ─────────────────────────────────────────────────────────────
# 9. METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(llm_front: list, nsga_front: list,
                    trajectory: list, total_calls: int) -> dict:
    """
    HV_ratio   : HV(LLM) / HV(NSGA)  — fraction of NSGA hypervolume achieved
    GD         : mean distance LLM->NSGA (proximity to reference front)
    IGD        : mean distance NSGA->LLM (coverage of reference front)
    spread     : LLM GWP range / NSGA GWP range
    non_dom_rate : fraction of proposals that were non-dominated
    """
    hv_llm  = compute_hypervolume(llm_front)
    hv_nsga = compute_hypervolume(nsga_front) if nsga_front else float("nan")
    hv_ratio = (hv_llm / hv_nsga) if hv_nsga > 0 else float("nan")

    # Normalized objective space for distance metrics
    all_sols = llm_front + nsga_front
    if len(all_sols) > 1:
        gwp_range = max(s["gwp"] for s in all_sols) - min(s["gwp"] for s in all_sols) + 1e-9
        str_range = (max(s["pred_28day"] for s in all_sols) -
                     min(s["pred_28day"] for s in all_sols) + 1e-9)
    else:
        gwp_range = str_range = 1.0

    def dist(a, b):
        return np.sqrt(((a["gwp"] - b["gwp"]) / gwp_range) ** 2 +
                       ((a["pred_28day"] - b["pred_28day"]) / str_range) ** 2)

    GD = IGD = float("nan")
    if nsga_front and llm_front:
        GD  = round(float(np.mean([min(dist(l, n) for n in nsga_front) for l in llm_front])), 6)
        IGD = round(float(np.mean([min(dist(n, l) for l in llm_front) for n in nsga_front])), 6)

    llm_gwp_range  = ((max(s["gwp"] for s in llm_front) - min(s["gwp"] for s in llm_front))
                      if len(llm_front) > 1 else 0.0)
    nsga_gwp_range = ((max(s["gwp"] for s in nsga_front) - min(s["gwp"] for s in nsga_front))
                      if len(nsga_front) > 1 else 1.0)
    spread = round(llm_gwp_range / (nsga_gwp_range + 1e-9), 4)

    nd_rate = round(sum(1 for r in trajectory if r["is_non_dominated"]) / max(len(trajectory), 1), 4)

    return {
        "HV_llm":        round(hv_llm, 4),
        "HV_nsga":       round(hv_nsga, 4) if not np.isnan(hv_nsga) else float("nan"),
        "HV_ratio":      round(hv_ratio, 4) if not np.isnan(hv_ratio) else float("nan"),
        "GD":            GD,
        "IGD":           IGD,
        "spread":        spread,
        "non_dom_rate":  nd_rate,
        "n_pareto_llm":  len(llm_front),
        "n_pareto_nsga": len(nsga_front),
        "total_calls":   total_calls,
    }


# ─────────────────────────────────────────────────────────────
# 10. SAVE RESULTS
# ─────────────────────────────────────────────────────────────

def save_results(trajectory: list, pareto_front: list, nsga_front: list,
                 metrics: dict, cfg: ExperimentConfig) -> None:
    os.makedirs(cfg.output_prefix, exist_ok=True)
    prefix = cfg.output_prefix

    # Trajectory CSV — drop internal dicts used for feedback rebuild
    traj_rows = [
        {k: v for k, v in r.items()
         if k not in ("_raw_v", "_der_v", "reasoning")}
        for r in trajectory
    ]
    pd.DataFrame(traj_rows).to_csv(f"{prefix}/trajectory.csv", index=False)

    if pareto_front:
        pd.DataFrame(pareto_front).to_csv(
            f"{prefix}/llm_pareto_front.csv", index=False)

    if nsga_front:
        pd.DataFrame(nsga_front).to_csv(
            f"{prefix}/nsga_pareto_front.csv", index=False)

    metrics_row = {"experiment": cfg.name, **metrics}
    pd.DataFrame([metrics_row]).to_csv(f"{prefix}/metrics.csv", index=False)

    print(f"\n  Saved -> {prefix}/")
    print(f"    trajectory.csv        ({len(trajectory)} rows)")
    print(f"    llm_pareto_front.csv  ({len(pareto_front)} solutions)")
    print(f"    nsga_pareto_front.csv ({len(nsga_front)} solutions)")
    print(f"    metrics.csv")
