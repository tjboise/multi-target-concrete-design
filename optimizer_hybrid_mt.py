"""
optimizer_hybrid_mt.py
======================
LLM-assisted NSGA-II hybrid optimizer for multi-objective concrete mix design.

Based on Lu et al. (2026), Expert Systems With Applications:
  "A knowledge-intensive LLM-assisted evolutionary framework for
   multi-objective geometric design of high-performance concrete structures"

Architecture:
  - NSGA-II runs the main evolutionary loop (SBX + polynomial mutation)
  - Every F generations, LLM intervention:
      1. Top Pareto-elite solutions from current population -> structured prompt
      2. LLM generates N candidate mixes (informed by material knowledge + elite context)
      3. Replace worst N offspring with LLM solutions before environmental selection
  - Continues for max_generations

Key difference from optimizer_core_mt.py:
  LLM supplements NSGA-II instead of replacing it.  The Pareto elite solutions
  from each generation serve as dynamic few-shot examples for the LLM, combining
  the global search power of NSGA-II with LLM's domain reasoning.
"""

import json, re, time, warnings, os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

from optimizer_core_mt import (
    RAW_VARS, HV_REF_POINT, PHYSICS_VARS,
    HV_GWP_MIN, HV_GWP_MAX, HV_STR_MIN, HV_STR_MAX,
    load_df, get_bounds, load_surrogate,
    compute_gwp, get_derived, get_physics,
)


def _hv(pareto: list) -> float:
    """Compute normalized hypervolume from list of dicts with 'GWP' and '28day' keys."""
    if not pareto:
        return 0.0
    try:
        from pymoo.indicators.hv import HV
        gwp = np.array([s["GWP"]   for s in pareto])
        d28 = np.array([s["28day"] for s in pareto])
        gwp_n = (gwp - HV_GWP_MIN) / (HV_GWP_MAX - HV_GWP_MIN)
        str_n = (HV_STR_MAX - d28) / (HV_STR_MAX - HV_STR_MIN)
        F = np.column_stack([gwp_n, str_n])
        return round(float(HV(ref_point=HV_REF_POINT)(F)), 6)
    except Exception as exc:
        print(f"  [Warning] HV failed: {exc}")
        return 0.0


def normalize_ref(nsga_ref: list) -> list:
    """Convert nsga_ref rows (gwp/pred_28day keys) to GWP/28day keys."""
    out = []
    for s in nsga_ref:
        gwp = s.get("GWP", s.get("gwp", 0.0))
        d28 = s.get("28day", s.get("pred_28day", 0.0))
        out.append({"GWP": gwp, "28day": d28})
    return out


# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

@dataclass
class HybridConfig:
    # Identity
    name: str = "hybrid_f10_n10"
    description: str = "LLM-assisted NSGA-II, F=10, N=10"

    # NSGA-II
    pop_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.9
    eta_c: float = 15.0          # SBX distribution index
    eta_m: float = 20.0          # polynomial mutation distribution index

    # LLM intervention (set llm_frequency=0 for pure NSGA-II baseline)
    llm_frequency: int = 10      # F: call LLM every F generations
    llm_n_solutions: int = 10    # N: solutions generated per LLM call
    llm_n_elite: int = 10        # elite solutions shown to LLM as few-shot context

    # Constraint handling strategy
    # "feasibility_first" : infeasible solutions stay in population but are dominated
    #                        by any feasible solution (Deb 2002 NSGA-II standard)
    # "death_penalty"     : infeasible solutions receive obj = [inf, inf] and are
    #                        driven out within 1-2 generations
    constraint_mode: str = "feasibility_first"

    # Prompt components
    use_knowledge_table: bool = True  # material GWP factors + strength effects

    # API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.7     # lower than standalone LLM for mix precision
    use_json_mode: bool = True   # enforce output schema

    # Reproducibility
    # Same seed for baseline and hybrid of the same rep → identical initial population,
    # so any HV difference is due to LLM injection only, not initialization luck.
    # None means do not seed (legacy behaviour).
    seed: int = None

    # Paths
    data_path: str = r"Concrete_Data_SI.csv"
    model_pkl: str = r"..\low_carbon_concrete\concrete_catboost_optimized.pkl"
    output_prefix: str = r"results\hybrid_f10_n10"


# ──────────────────────────────────────────────────────────────
# SURROGATE HELPERS
# ──────────────────────────────────────────────────────────────

def _predict(mix: dict, meta: dict) -> dict:
    from optimizer_core_mt import predict
    return predict(meta, mix)

def _to_dict(x: np.ndarray) -> dict:
    return {v: float(x[i]) for i, v in enumerate(RAW_VARS)}

def _to_array(mix: dict) -> np.ndarray:
    return np.array([mix[v] for v in RAW_VARS])


# ──────────────────────────────────────────────────────────────
# NSGA-II COMPONENTS
# ──────────────────────────────────────────────────────────────

def evaluate_population(pop: np.ndarray, raw_b: dict, der_b: dict,
                        phys_b: dict, meta: dict,
                        constraint_mode: str = "feasibility_first") -> tuple:
    """
    Evaluate all individuals.

    Returns:
        obj      : (n, 2) array of [GWP, -28d_strength]  (both minimized)
                   In death_penalty mode, infeasible rows are set to [inf, inf].
        feasible : (n,)   bool array — True if all Layer-2 and Layer-3 constraints pass.
                   Layer-1 (raw bounds) is enforced by construction (clipping) and never
                   checked here.

    constraint_mode:
        "feasibility_first" — infeasible solutions remain with true objective values but
                              are dominated by any feasible solution in _dominance().
        "death_penalty"     — infeasible solutions get obj = [inf, inf]; they are
                              dominated by every feasible solution automatically, no
                              special logic needed in _dominance().
    """
    n = len(pop)
    obj = np.empty((n, 2))
    feasible = np.ones(n, dtype=bool)

    for i in range(n):
        mix = _to_dict(pop[i])
        obj[i, 0] = compute_gwp(mix)
        obj[i, 1] = -_predict(mix, meta)["28day"]

        der = get_derived(mix)
        for k, b in der_b.items():
            v = der.get(k, 0.0)
            if not (b["min"] - 1e-6 <= v <= b["max"] + 1e-6):
                feasible[i] = False
                break
        if feasible[i]:
            pv = get_physics(mix)
            for k, b in phys_b.items():
                v = pv.get(k, 0.0)
                if not (b["min"] - 1e-6 <= v <= b["max"] + 1e-6):
                    feasible[i] = False
                    break

    if constraint_mode == "death_penalty":
        obj[~feasible] = np.inf

    return obj, feasible


def _dominance(p: int, q: int, obj: np.ndarray, feas: np.ndarray) -> int:
    """Return 1 if p dominates q, -1 if q dominates p, 0 otherwise."""
    fp, fq = feas[p], feas[q]
    if fp and not fq:
        return 1
    if fq and not fp:
        return -1
    if not fp and not fq:
        return 0
    # Both feasible: standard Pareto dominance
    if np.all(obj[p] <= obj[q]) and np.any(obj[p] < obj[q]):
        return 1
    if np.all(obj[q] <= obj[p]) and np.any(obj[q] < obj[p]):
        return -1
    return 0


def fast_nondominated_sort(obj: np.ndarray, feas: np.ndarray) -> list:
    """
    Deb's fast non-dominated sort with feasibility-first dominance.
    Returns list of fronts, each front is a list of indices.
    """
    n = len(obj)
    dominated_by_count = np.zeros(n, dtype=int)
    dominates_set = [[] for _ in range(n)]

    for p in range(n):
        for q in range(p + 1, n):
            d = _dominance(p, q, obj, feas)
            if d == 1:
                dominates_set[p].append(q)
                dominated_by_count[q] += 1
            elif d == -1:
                dominates_set[q].append(p)
                dominated_by_count[p] += 1

    fronts = [[p for p in range(n) if dominated_by_count[p] == 0]]
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                dominated_by_count[q] -= 1
                if dominated_by_count[q] == 0:
                    next_front.append(q)
        fronts.append(next_front)
        i += 1

    return [f for f in fronts if f]


def crowding_distance(obj: np.ndarray, front: list) -> np.ndarray:
    """Crowding distance for a single front; returns array indexed by position in front."""
    n = len(front)
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf
        return dist
    f_obj = obj[front]
    for m in range(f_obj.shape[1]):
        order = np.argsort(f_obj[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = f_obj[order[-1], m] - f_obj[order[0], m]
        if span < 1e-10:
            continue
        for j in range(1, n - 1):
            dist[order[j]] += (f_obj[order[j + 1], m] - f_obj[order[j - 1], m]) / span
    return dist


def assign_ranks_and_crowding(obj: np.ndarray, feas: np.ndarray,
                               fronts: list) -> tuple:
    """Returns (ranks, crowding) arrays of length len(obj)."""
    n = len(obj)
    ranks = np.zeros(n, dtype=int)
    crowding = np.zeros(n)
    for rank, front in enumerate(fronts):
        cd = crowding_distance(obj, front)
        for j, idx in enumerate(front):
            ranks[idx] = rank
            crowding[idx] = cd[j]
    return ranks, crowding


def tournament_select(n: int, ranks: np.ndarray, crowding: np.ndarray) -> int:
    """Binary tournament selection."""
    a, b = np.random.choice(n, 2, replace=False)
    if ranks[a] < ranks[b]:
        return a
    if ranks[b] < ranks[a]:
        return b
    return a if crowding[a] >= crowding[b] else b


def sbx_crossover(p1: np.ndarray, p2: np.ndarray,
                  lb: np.ndarray, ub: np.ndarray,
                  pc: float = 0.9, eta: float = 15.0) -> tuple:
    """Simulated Binary Crossover."""
    c1, c2 = p1.copy(), p2.copy()
    if np.random.random() > pc:
        return c1, c2
    for i in range(len(p1)):
        if np.random.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-10:
            continue
        u = np.random.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c1[i] = np.clip(0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i]), lb[i], ub[i])
        c2[i] = np.clip(0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i]), lb[i], ub[i])
    return c1, c2


def polynomial_mutation(x: np.ndarray, lb: np.ndarray,
                        ub: np.ndarray, eta: float = 20.0) -> np.ndarray:
    """Polynomial Mutation with pm = 1/n_vars."""
    y, pm = x.copy(), 1.0 / len(x)
    for i in range(len(x)):
        if np.random.random() >= pm:
            continue
        u = np.random.random()
        span = ub[i] - lb[i]
        if span < 1e-10:
            continue
        delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        y[i] = np.clip(x[i] + delta * span, lb[i], ub[i])
    return y


def generate_offspring(pop: np.ndarray, ranks: np.ndarray, crowding: np.ndarray,
                       lb: np.ndarray, ub: np.ndarray, cfg: HybridConfig) -> np.ndarray:
    """Generate pop_size offspring via tournament selection + SBX + PM."""
    n = cfg.pop_size
    offspring = []
    while len(offspring) < n:
        i1 = tournament_select(n, ranks, crowding)
        i2 = tournament_select(n, ranks, crowding)
        c1, c2 = sbx_crossover(pop[i1], pop[i2], lb, ub, cfg.crossover_rate, cfg.eta_c)
        offspring.append(polynomial_mutation(c1, lb, ub, cfg.eta_m))
        if len(offspring) < n:
            offspring.append(polynomial_mutation(c2, lb, ub, cfg.eta_m))
    return np.array(offspring[:n])


def environmental_selection(combined: np.ndarray, comb_obj: np.ndarray,
                             comb_feas: np.ndarray, n: int) -> tuple:
    """
    Select n survivors from combined parent+offspring population
    via non-dominated sort + crowding distance (NSGA-II rule).
    Returns (pop, obj, feas, ranks, crowding).
    """
    fronts = fast_nondominated_sort(comb_obj, comb_feas)
    ranks, crowding = assign_ranks_and_crowding(comb_obj, comb_feas, fronts)

    selected = []
    for front in fronts:
        if len(selected) + len(front) <= n:
            selected.extend(front)
        else:
            remaining = n - len(selected)
            sorted_front = sorted(front, key=lambda x: crowding[x], reverse=True)
            selected.extend(sorted_front[:remaining])
            break

    sel = np.array(selected)
    return combined[sel], comb_obj[sel], comb_feas[sel], ranks[sel], crowding[sel]


# ──────────────────────────────────────────────────────────────
# LLM PROMPT & CALL
# ──────────────────────────────────────────────────────────────

_KNOWLEDGE_TABLE = """\
## Material Reference: GWP Emission Factors and Strength Effects

### Cementitious Materials (binders) — dominate both GWP and strength
| Material | GWP (kg CO2/kg) | Strength Effect | Optimization Strategy |
|----------|----------------|-----------------|----------------------|
| PC  (Portland Cement)  | 1.048 | Primary strength driver at all ages | Minimize; each 100 kg/m3 reduction saves ~105 kg CO2/m3 GWP |
| FA  (Fly Ash)          | 0.328 | Slow pozzolanic reaction; moderate 28d gain | Substitute for PC on low-GWP path; SC preferred for 28d strength |
| SC  (Slag Cement/GGBS) | 0.264 | Latent hydraulic; strong 28d contribution | Best PC substitute: lower GWP than FA, better 28d strength |

### Aggregates — negligible GWP, structural role
| Material | GWP (kg CO2/kg) | Role |
|----------|----------------|------|
| FAGG (Fine Aggregate / Sand) | 0.0026 | Fills voids; higher content dilutes binder → lower GWP |
| CAGG (Coarse Aggregate)      | 0.0037 | Load-bearing skeleton; higher content with WR_HR maintains strength at lower binder |

### Water and Admixtures
| Material | GWP | Role and Mechanism |
|----------|-----|--------------------|
| WATER    | 0.000 | w/b ratio governs strength; reducing WATER with WR_HR lowers w/b without losing workability |
| WR_HR (Superplasticizer) | 0.000 | Enables w/b < 0.38 at constant workability; most powerful strength lever per unit |
| WR    (Water Reducer)    | 0.000 | Moderate water reduction; use when WR_HR budget is exhausted |
| ACC   (Accelerator)      | 0.000 | Boosts early-age hydration; modest 28d effect |
| AEA   (Air-Entraining Agent) | 0.000 | Freeze-thaw resistance; slightly reduces strength |

## Key Formulas and Derived Ratios

  GWP ≈ 1.048·PC + 0.328·FA + 0.264·SC + 0.0026·FAGG + 0.0037·CAGG  (kg CO2-eq/m3)

  Binder composition:
    Binder  = PC + FA + SC                           (total cementitious content, kg/m3)
    w/b     = WATER / Binder                         (water-to-binder ratio; governs strength via Abrams' law)
    PC%     = PC / Binder                            (Portland cement fraction; high PC% → high GWP)
    FA%     = FA / Binder                            (fly ash fraction)
    SC%     = SC / Binder                            (slag cement fraction; preferred substitute for PC)

  Mix density ratios:
    b/agg   = Binder / (FAGG + CAGG)                (binder-to-aggregate ratio; lower → lower GWP risk: lower strength)

"""


_DERIVED_FORMULAS = [
    ("w/b",       "WATER / Binder"),
    ("b/a",       "Binder / Agg"),
    ("SCM%",      "(FA+SC) / Binder"),
    ("CAGG%",     "CAGG / Agg"),
    ("FAGG%",     "FAGG / Agg"),
    ("PC%",       "PC / Binder"),
    ("FA%",       "FA / Binder"),
    ("SC%",       "SC / Binder"),
    ("AEA_pct",   "AEA / Binder"),
    ("WR_HR_pct", "WR_HR / Binder"),
    ("WR_pct",    "WR / Binder"),
    ("ACC_pct",   "ACC / Binder"),
]


def build_hybrid_prompt(elite: list, raw_b: dict, der_b: dict,
                        phys_b: dict, cfg: HybridConfig) -> str:
    """Build the LLM prompt for one intervention step."""
    lines = [
        "You are an expert concrete mix designer assisting a multi-objective genetic algorithm.",
        "",
        "## Optimization Objectives",
        "Simultaneously optimize two competing objectives:",
        "  1. MINIMIZE GWP (kg CO2-eq/m3) — greenhouse gas emission footprint",
        "  2. MAXIMIZE 28-day compressive strength (MPa) — structural performance",
        "",
        "These objectives conflict: cementitious binder drives both strength and GWP.",
        "A Pareto-optimal mix cannot improve one objective without worsening the other.",
        "",
    ]

    if cfg.use_knowledge_table:
        lines.append(_KNOWLEDGE_TABLE)

    # ── Layer 1 ──────────────────────────────────────────────
    lines += [
        "## Constraints",
        "",
        "### Layer 1 — Raw Ingredient Bounds (kg/m3)",
        "All 10 ingredient values must stay within these dataset-derived bounds:",
        "",
    ]
    col = max(len(v) for v in RAW_VARS) + 2
    for v in RAW_VARS:
        b = raw_b[v]
        lines.append(f"  {v:<{col}}: [{b['min']:.1f}, {b['max']:.1f}]")

    # ── Layer 2 ──────────────────────────────────────────────
    lines += [
        "",
        "### Layer 2 — Derived Ratio Constraints",
        "  Binder = PC + FA + SC   (total cementitious content, kg/m3)",
        "  Agg    = FAGG + CAGG    (total aggregate, kg/m3)",
        "",
        f"  {'Ratio':<12} {'Formula':<25} {'Min':>8}  {'Max':>8}",
        f"  {'-'*12} {'-'*25} {'-'*8}  {'-'*8}",
    ]
    for var, formula in _DERIVED_FORMULAS:
        if var in der_b:
            b = der_b[var]
            lines.append(f"  {var:<12} {formula:<25} {b['min']:>8.3f}  {b['max']:>8.3f}")

    # ── Layer 3 ──────────────────────────────────────────────
    lines += [
        "",
        "### Layer 3 — Physics Constraints (Pfeiffer et al. 2024, Eq. 19–20)",
        "  Material densities (kg/m3):",
        "    PC=3150  FA=2200  SC=2900  FAGG=2650  CAGG=2650",
        "    WATER=1000  AEA=1010  WR_HR=1080  WR=1140  ACC=1340",
        "",
        "  Vm = PC/3150 + FA/2200 + SC/2900 + FAGG/2650 + CAGG/2650",
        "     + WATER/1000 + AEA/1010 + WR_HR/1080 + WR/1140 + ACC/1340",
        "",
        "  Vfinal = Vm + 0.07   if AEA/PC >= 0.000244  (7% entrained air — AEA present)",
        "         = Vm + 0.03   if AEA/PC <  0.000244  (3% entrapped air — no AEA)",
        "  Constraint: 0.950 <= Vfinal <= 1.050",
    ]
    if "Vagg" in phys_b:
        b = phys_b["Vagg"]
        lines += [
            "",
            "  Vagg = FAGG/2650 + CAGG/2650",
            f"  Constraint: {b['min']:.3f} <= Vagg <= {b['max']:.3f}",
        ]
    if "TOTAL_BINDER" in phys_b:
        b = phys_b["TOTAL_BINDER"]
        lines += [
            "",
            "  TOTAL_BINDER = PC + FA + SC",
            f"  Constraint: {b['min']:.1f} <= TOTAL_BINDER <= {b['max']:.1f} kg/m3",
        ]

    # ── Elite ────────────────────────────────────────────────
    lines += [
        "",
        "## Current Pareto-Elite Solutions (sorted by GWP, low → high)",
        "These are the best non-dominated feasible mixes found by NSGA-II so far.",
        "",
    ]
    for i, e in enumerate(elite):
        mix = e["mix"]
        pc  = mix.get("PC", 0); fa = mix.get("FA", 0); sc = mix.get("SC", 0)
        tb  = pc + fa + sc
        wb   = mix.get("WATER", 0) / tb if tb > 0 else 0
        agg  = mix.get("FAGG", 0) + mix.get("CAGG", 0)
        bagg = tb / agg if agg > 0 else 0
        lines.append(
            f"  [{i+1}] GWP={e['GWP']:.1f} | 28d={e['28d_MPa']:.1f} MPa"
            f" | w/b={wb:.3f} | binder={tb:.0f} kg/m3 | b/agg={bagg:.3f}"
        )
        lines.append(
            f"       PC%={100*pc/tb if tb>0 else 0:.0f}%"
            f"  FA%={100*fa/tb if tb>0 else 0:.0f}%"
            f"  SC%={100*sc/tb if tb>0 else 0:.0f}%"
        )
        lines.append("       " + "  ".join(f"{k}={v:.1f}" for k, v in mix.items()))
    lines.append("")

    # ── Task ─────────────────────────────────────────────────
    lines += [
        f"## Task: Generate {cfg.llm_n_solutions} New Candidate Mixes",
        "",
        "You are augmenting NSGA-II's offspring pool. The mixes you generate will be",
        "injected into the next generation's environmental selection alongside the genetic",
        "algorithm's own offspring. Mixes that are Pareto-dominated will be selected against;",
        "those that push or extend the current Pareto front will survive and propagate into",
        "future generations.",
        "",
        f"Generate {cfg.llm_n_solutions} mixes that you believe have the best chance of being",
        "non-dominated relative to the current Pareto front shown above.",
        "",
        "Study the elite solutions carefully: identify their compositional patterns, note where",
        "the GWP–strength trade-off is densely covered and where it is sparse, and reason about",
        "what changes to ingredient proportions could push the front outward in any direction.",
        "Draw on your material science knowledge and the formulas above to verify that your",
        "proposals are physically reasonable and satisfy all three constraint layers.",
        "",
        "Every proposed mix must satisfy ALL constraints listed above. Verify each mix",
        "before including it in your response.",
        "",
        f"Return exactly {cfg.llm_n_solutions} mixes as a JSON array. Follow this format exactly:",
        "[",
        '  {"mix": {"PC": 200.0, "FA": 60.0, "SC": 110.0, "FAGG": 800.0, "CAGG": 900.0,',
        '           "WATER": 148.0, "AEA": 0.0, "WR_HR": 2.5, "WR": 0.0, "ACC": 0.0}},',
        '  {"mix": {"PC": ..., "FA": ..., "SC": ..., "FAGG": ..., "CAGG": ...,',
        '           "WATER": ..., "AEA": ..., "WR_HR": ..., "WR": ..., "ACC": ...}},',
        f"  ... ({cfg.llm_n_solutions} elements total)",
        "]",
        "All values in kg/m3. Output only the JSON array — no surrounding text, no markdown fences.",
    ]

    return "\n".join(lines)


def _parse_llm_array(text: str):
    """Parse LLM text response into a list of mix dicts."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("mixes", "solutions", "candidates"):
                if key in data and isinstance(data[key], list):
                    return data[key]
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def call_llm_for_solutions(elite: list, raw_b: dict, der_b: dict, phys_b: dict,
                            cfg: HybridConfig, client, genai_types) -> tuple:
    """
    Single LLM call to generate N candidate solutions.
    Returns (list of np.ndarray, n_parse_fails).
    """
    prompt = build_hybrid_prompt(elite, raw_b, der_b, phys_b, cfg)
    lb = np.array([raw_b[v]["min"] for v in RAW_VARS])
    ub = np.array([raw_b[v]["max"] for v in RAW_VARS])

    # JSON schema: array of {mix: {PC, FA, ...}}
    mix_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={v: genai_types.Schema(type=genai_types.Type.NUMBER)
                    for v in RAW_VARS},
        required=RAW_VARS,
    )
    array_schema = genai_types.Schema(
        type=genai_types.Type.ARRAY,
        items=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={"mix": mix_schema},
            required=["mix"],
        ),
    )

    extra = ({"response_mime_type": "application/json", "response_schema": array_schema}
             if cfg.use_json_mode else {})

    parse_fails = 0
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=cfg.gemini_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=cfg.temperature,
                    max_output_tokens=2048,
                    **extra,
                ),
            )
            data = _parse_llm_array(resp.text)
            if data:
                solutions = []
                for item in data:
                    mix = item.get("mix", item)
                    x = np.array([
                        float(mix.get(v, (lb[i] + ub[i]) / 2))
                        for i, v in enumerate(RAW_VARS)
                    ])
                    solutions.append(np.clip(x, lb, ub))
                return solutions, parse_fails
            else:
                parse_fails += 1
        except Exception as e:
            parse_fails += 1
            wait = 60 if "429" in str(e) else 5
            time.sleep(wait)

    return [], parse_fails


# ──────────────────────────────────────────────────────────────
# MAIN HYBRID LOOP
# ──────────────────────────────────────────────────────────────

def run_hybrid(raw_b: dict, der_b: dict, phys_b: dict,
               meta: dict, cfg: HybridConfig) -> dict:
    """
    Run LLM-assisted NSGA-II (or pure NSGA-II if llm_frequency == 0).

    Returns dict:
        final_pareto : list of dicts with mix + GWP + 28day
        hv_history   : list of {gen, hv, n_pareto, llm_injected}
        llm_calls    : int
        parse_fails  : int
    """
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    use_llm = cfg.llm_frequency > 0 and cfg.gemini_api_key

    if use_llm:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=cfg.gemini_api_key)
    else:
        client = genai_types = None

    lb = np.array([raw_b[v]["min"] for v in RAW_VARS])
    ub = np.array([raw_b[v]["max"] for v in RAW_VARS])

    # Initialize population
    pop = np.column_stack([
        np.random.uniform(raw_b[v]["min"], raw_b[v]["max"], cfg.pop_size)
        for v in RAW_VARS
    ])
    obj, feas = evaluate_population(pop, raw_b, der_b, phys_b, meta, cfg.constraint_mode)

    hv_history = []
    llm_calls = 0
    parse_fails = 0

    label = cfg.name if not use_llm else f"{cfg.name} (F={cfg.llm_frequency}, N={cfg.llm_n_solutions})"
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"  Pop={cfg.pop_size} | Gen={cfg.max_generations} | LLM={'yes' if use_llm else 'no'}")
    print(f"{'='*62}")
    print(f"  {'Gen':>4}  {'HV':>10}  {'Pareto#':>7}  Note")
    print(f"  {'-'*45}")

    for gen in range(cfg.max_generations):
        # ── Rank + crowding for parent population ─────────────
        fronts = fast_nondominated_sort(obj, feas)
        ranks, crowding = assign_ranks_and_crowding(obj, feas, fronts)

        # ── Generate offspring ─────────────────────────────────
        offspring = generate_offspring(pop, ranks, crowding, lb, ub, cfg)

        # ── LLM intervention ───────────────────────────────────
        note = ""
        if use_llm and (gen + 1) % cfg.llm_frequency == 0:
            # Collect elite: non-dominated feasible individuals, sorted by GWP
            front0_feasible = [i for i in fronts[0] if feas[i]]
            front0_feasible.sort(key=lambda i: obj[i, 0])
            elite_idx = front0_feasible[:cfg.llm_n_elite]

            if elite_idx:
                elite = []
                for idx in elite_idx:
                    mix = _to_dict(pop[idx])
                    elite.append({
                        "mix": {v: round(float(mix[v]), 1) for v in RAW_VARS},
                        "GWP": round(float(obj[idx, 0]), 2),
                        "28d_MPa": round(float(-obj[idx, 1]), 2),
                    })

                new_sols, n_fails = call_llm_for_solutions(
                    elite, raw_b, der_b, phys_b, cfg, client, genai_types
                )
                llm_calls += 1
                parse_fails += n_fails

                # Replace worst N offspring (last N rows) with LLM solutions
                n_inject = min(len(new_sols), cfg.llm_n_solutions)
                for j in range(n_inject):
                    offspring[-(j + 1)] = new_sols[j]
                note = f"+LLM×{n_inject}"

        # ── Evaluate offspring ─────────────────────────────────
        off_obj, off_feas = evaluate_population(offspring, raw_b, der_b, phys_b, meta, cfg.constraint_mode)

        # ── Environmental selection (parent + offspring) ───────
        combined = np.vstack([pop, offspring])
        comb_obj = np.vstack([obj, off_obj])
        comb_feas = np.concatenate([feas, off_feas])
        pop, obj, feas, ranks, crowding = environmental_selection(
            combined, comb_obj, comb_feas, cfg.pop_size
        )

        # ── Track HV ──────────────────────────────────────────
        fronts_new = fast_nondominated_sort(obj, feas)
        pareto_for_hv = [
            {"GWP": float(obj[i, 0]), "28day": float(-obj[i, 1])}
            for i in fronts_new[0] if feas[i]
        ]
        hv = _hv(pareto_for_hv)
        n_pareto = len(pareto_for_hv)
        hv_history.append({
            "gen": gen + 1,
            "hv": hv,
            "n_pareto": n_pareto,
            "llm_injected": int(bool(note)),
        })

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  {gen+1:4d}  {hv:10.1f}  {n_pareto:7d}  {note}")

    # ── Collect final Pareto front ─────────────────────────────
    fronts_final = fast_nondominated_sort(obj, feas)
    final_pareto = []
    for i in fronts_final[0]:
        if feas[i]:
            mix = _to_dict(pop[i])
            final_pareto.append({
                **{v: round(float(mix[v]), 3) for v in RAW_VARS},
                "GWP": round(float(obj[i, 0]), 3),
                "28day": round(float(-obj[i, 1]), 3),
            })

    hv_final = _hv(final_pareto)
    print(f"\n  LLM calls: {llm_calls}  Parse fails: {parse_fails}")
    print(f"  Final Pareto: {len(final_pareto)} solutions  HV: {hv_final:.1f}")

    return {
        "final_pareto": final_pareto,
        "hv_history": hv_history,
        "llm_calls": llm_calls,
        "parse_fails": parse_fails,
    }


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────

def compute_hybrid_metrics(result: dict, nsga_ref: list) -> dict:
    """
    Compute evaluation metrics for a hybrid run vs. pure NSGA-II reference.
    nsga_ref: list of dicts (accepts both gwp/pred_28day and GWP/28day keys).
    """
    pareto   = result["final_pareto"]
    nsga_ref = normalize_ref(nsga_ref)
    hv_llm   = _hv(pareto) if pareto else 0.0

    if not pareto or not nsga_ref:
        return {
            "HV_hybrid":  round(hv_llm, 4),
            "HV_nsga":    0.0,
            "HV_ratio":   0.0,
            "GD":         0.0,
            "IGD":        0.0,
            "spread":     0.0,
            "n_pareto":   len(pareto),
            "n_nsga":     len(nsga_ref),
            "llm_calls":  result["llm_calls"],
            "parse_fails": result["parse_fails"],
        }

    hv_nsga = _hv(nsga_ref)

    def pts(front):
        return np.array([[s["GWP"], s["28day"]] for s in front])

    p_llm  = pts(pareto)
    p_nsga = pts(nsga_ref)

    # GD: mean distance from each LLM point to nearest NSGA point
    from scipy.spatial.distance import cdist
    d = cdist(p_llm, p_nsga)
    gd  = float(np.mean(np.min(d, axis=1)))
    igd = float(np.mean(np.min(d, axis=0)))

    gwp_llm  = p_llm[:, 0]
    gwp_nsga = p_nsga[:, 0]
    span_nsga = gwp_nsga.max() - gwp_nsga.min()
    spread = float((gwp_llm.max() - gwp_llm.min()) / span_nsga) if span_nsga > 0 else 0.0

    return {
        "HV_hybrid":    round(hv_llm, 4),
        "HV_nsga":      round(hv_nsga, 4),
        "HV_ratio":     round(hv_llm / hv_nsga, 4) if hv_nsga > 0 else 0.0,
        "GD":           round(gd, 6),
        "IGD":          round(igd, 6),
        "spread":       round(spread, 4),
        "n_pareto":     len(pareto),
        "n_nsga":       len(nsga_ref),
        "llm_calls":    result["llm_calls"],
        "parse_fails":  result["parse_fails"],
    }


# ──────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────

def save_hybrid_results(result: dict, metrics: dict, cfg: HybridConfig,
                        nsga_ref: list) -> None:
    os.makedirs(cfg.output_prefix, exist_ok=True)

    pd.DataFrame(result["hv_history"]).to_csv(
        os.path.join(cfg.output_prefix, "hv_history.csv"), index=False)

    pd.DataFrame(result["final_pareto"]).to_csv(
        os.path.join(cfg.output_prefix, "pareto_front.csv"), index=False)

    pd.DataFrame(nsga_ref).to_csv(
        os.path.join(cfg.output_prefix, "nsga_reference.csv"), index=False)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(cfg.output_prefix, "metrics.csv"), index=False)

    print(f"\n  Saved -> {cfg.output_prefix}/")
    for k, v in metrics.items():
        print(f"    {k:<20}: {v}")
