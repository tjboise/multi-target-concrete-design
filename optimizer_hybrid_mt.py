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
    llm_start_gen: int = 0       # earliest generation to inject (0 = from the start)
    llm_n_solutions: int = 10    # N: solutions generated per LLM call
    llm_n_elite: int = 10        # elite solutions shown to LLM as few-shot context
    # Stagnation-based injection: only inject when HV has not improved by
    # more than `stagnation_threshold` over the last `stagnation_window` gens.
    # Set stagnation_window=0 to disable stagnation detection (always inject).
    stagnation_window: int = 0      # look-back window (gens); 0 = disabled
    stagnation_threshold: float = 0.005  # max tolerated HV gain over the window

    # Constraint handling strategy
    # "death_penalty"     : infeasible solutions receive obj = [inf, inf] and are
    #                        eliminated in the same generation's environmental selection.
    #                        Consistent with LLM solution pre-filtering (infeasible LLM
    #                        solutions are discarded before injection).
    # "feasibility_first" : infeasible solutions stay with true objective values but are
    #                        dominated by any feasible solution (Deb 2002 NSGA-II standard);
    #                        they are gradually pushed out over several generations.
    constraint_mode: str = "feasibility_first"

    # Prompt components (leave-one-out ablation flags)
    use_objectives: bool = True        # Section 1: optimization objectives + conflict statement
    use_knowledge_table: bool = True   # Section 2: material GWP factors + strength effects
    use_constraints: bool = True       # Section 3: all three constraint layers
    use_elite: bool = True             # Section 4: current Pareto elite solutions
    use_gap_targeting: bool = True     # Section 5: gap/extreme analysis + structured task allocation

    # Injection strategy:
    #   "replace" (default): LLM solutions overwrite the N worst offspring before
    #                        environmental selection.
    #   "augment": LLM solutions are appended to the offspring pool; NSGA-II offspring
    #              are never discarded, so the selection pool is pop_size + offspring_size + N.
    llm_inject_mode: str = "replace"

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
    data_path: str = r"Concrete_Data_SI_clean.csv"
    model_pkl: str = r"..\low_carbon_concrete\concrete_catboost_optimized_clean.pkl"
    output_prefix: str = r"results\hybrid_f10_n10"


# ──────────────────────────────────────────────────────────────
# SURROGATE HELPERS
# ──────────────────────────────────────────────────────────────

def _predict(mix: dict, meta: dict) -> dict:
    from optimizer_core_mt import predict
    return predict(meta, mix)

def _to_dict(x: np.ndarray) -> dict:
    return {v: float(x[i]) for i, v in enumerate(RAW_VARS)}

def _is_feasible(x: np.ndarray, der_b: dict, phys_b: dict) -> bool:
    """Check Layer 2 + 3 feasibility for a single solution vector.
    Layer 1 (raw bounds) is guaranteed by np.clip in the caller."""
    mix = _to_dict(x)
    der = get_derived(mix)
    for k, b in der_b.items():
        if not (b["min"] - 1e-6 <= der.get(k, 0.0) <= b["max"] + 1e-6):
            return False
    pv = get_physics(mix)
    for k, b in phys_b.items():
        if not (b["min"] - 1e-6 <= pv.get(k, 0.0) <= b["max"] + 1e-6):
            return False
    return True

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


def _compute_pareto_gaps(coords: list, top_n: int = 3) -> list:
    """Return the top_n largest gaps between adjacent Pareto front points.

    coords : list of (gwp, strength) tuples
    Returns list of (mid_gwp, mid_str, p1, p2) sorted largest-gap first.
    """
    if len(coords) < 2:
        return []
    pts = sorted(coords, key=lambda p: p[0])
    gwp_rng = pts[-1][0] - pts[0][0] + 1e-6
    str_rng = max(p[1] for p in pts) - min(p[1] for p in pts) + 1e-6
    gaps = []
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        dist = ((p2[0]-p1[0])/gwp_rng)**2 + ((p2[1]-p1[1])/str_rng)**2
        mid  = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
        gaps.append((dist**0.5, mid, p1, p2))
    gaps.sort(reverse=True)
    return [(mid, p1, p2) for _, mid, p1, p2 in gaps[:top_n]]


def build_hybrid_prompt(elite: list, raw_b: dict, der_b: dict,
                        phys_b: dict, cfg: HybridConfig,
                        n_solutions: int = None,
                        feedback: dict = None) -> str:
    """Build the LLM prompt for one intervention step.

    feedback (optional): dict with keys prev_hv, curr_hv,
        prev_gwp_min, prev_gwp_max, prev_str_min, prev_str_max.
        When provided, a 'Previous Injection Outcome' section is prepended.
    """
    lines = [
        "You are an expert concrete mix designer assisting a multi-objective genetic algorithm.",
        "",
    ]

    # ── Section 1: Objectives ────────────────────────────────
    if cfg.use_objectives:
        lines += [
            "## Optimization Objectives",
            "Simultaneously optimize two competing objectives:",
            "  1. MINIMIZE GWP (kg CO2-eq/m3) — greenhouse gas emission footprint",
            "  2. MAXIMIZE 28-day compressive strength (MPa) — structural performance",
            "",
            "These objectives conflict: cementitious binder drives both strength and GWP.",
            "A Pareto-optimal mix cannot improve one objective without worsening the other.",
            "",
        ]

    # ── Section 2: Knowledge Table ───────────────────────────
    if cfg.use_knowledge_table:
        lines.append(_KNOWLEDGE_TABLE)

    # ── Section 3: Constraints ───────────────────────────────
    if cfg.use_constraints:
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

        if "TOTAL_BINDER" in phys_b:
            b = phys_b["TOTAL_BINDER"]
            lines += [
                "",
                "### Layer 3 — Binder Content",
                f"  TOTAL_BINDER = PC + FA + SC  must be in [{b['min']:.0f}, {b['max']:.0f}] kg/m3",
                "  (Volume physics — Vfinal, Vagg — are checked automatically; focus on Layers 1–2.)",
            ]

    # ── Section 4: Pareto Elite ──────────────────────────────
    if cfg.use_elite:
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

    # ── Feedback from previous injection (if available) ──────
    if feedback:
        dv = feedback["curr_hv"] - feedback["prev_hv"]
        sign = "+" if dv >= 0 else ""
        lines += [
            "## Pareto Front Evolution Since Last Injection",
            f"  Hypervolume change: {sign}{dv:.4f}"
            + (" [IMPROVED]" if dv > 0 else " [NOT IMPROVED -- try a different strategy]"),
            f"  (Hypervolume measures objective-space coverage — larger is better.)",
            "",
        ]

        def _format_coords(coords):
            parts = [f"({g:.1f}, {s:.1f})" for g, s in coords]
            rows = []
            for i in range(0, len(parts), 5):
                rows.append("  " + "  ".join(parts[i:i + 5]))
            return rows

        prev_coords = feedback.get("prev_pareto_coords", [])
        curr_coords = feedback.get("curr_pareto_coords", [])
        if prev_coords:
            lines.append(f"  Front BEFORE last injection ({len(prev_coords)} points, GWP / Strength MPa):")
            lines += _format_coords(prev_coords)
            lines.append("")
        if curr_coords:
            lines.append(f"  Front NOW ({len(curr_coords)} points, GWP / Strength MPa):")
            lines += _format_coords(curr_coords)
            lines.append("")

            if cfg.use_gap_targeting:
                # Gap analysis
                gaps = _compute_pareto_gaps(curr_coords, top_n=3)
                if gaps:
                    lines.append("  Largest GAPS in current front (sparse/uncovered regions — high-priority targets):")
                    for mid, p1, p2 in gaps:
                        lines.append(
                            f"    Between ({p1[0]:.1f}, {p1[1]:.1f}) and ({p2[0]:.1f}, {p2[1]:.1f})"
                            f"  →  target ≈ GWP {mid[0]:.1f}, Strength {mid[1]:.1f} MPa"
                        )
                    lines.append("")

                # Extreme analysis
                min_gwp = min(g for g, _ in curr_coords)
                max_str = max(s for _, s in curr_coords)
                lines += [
                    "  Current EXTREMES of the front:",
                    f"    Lowest GWP:      {min_gwp:.1f} kg CO2e/m3"
                    + f"  →  try to reach below {max(raw_b['PC']['min']*1.5, min_gwp - 12):.0f}",
                    f"    Highest Strength: {max_str:.1f} MPa"
                    + f"  →  try to push above {min(raw_b['SC']['max']*0.5 + 80, max_str + 6):.0f}",
                    "",
                ]

    # ── Task ─────────────────────────────────────────────────
    elite_gwps = [e["GWP"] for e in elite]
    elite_strs = [e["28d_MPa"] for e in elite]
    n = n_solutions if n_solutions is not None else cfg.llm_n_solutions
    n_gap   = max(1, n // 3)    # at least 1/3 targeting gaps
    n_ext   = max(1, n // 4)    # at least 1/4 pushing extremes
    n_free  = max(0, n - n_gap - n_ext)

    if cfg.use_gap_targeting:
        lines += [
            f"## Task: Generate {n} New Candidate Mixes",
            "",
            "Goal: INCREASE HYPERVOLUME by placing solutions in uncovered or sparse regions of the objective space.",
            "",
            f"Allocate your {n} solutions as follows:",
            f"  • {n_gap} solution(s) — fill the LARGEST GAPS identified above (interpolate between the bounding elite mixes)",
            f"  • {n_ext} solution(s) — push EXTREME ends: very low GWP (< current minimum) OR very high strength",
            f"  • {n_free} solution(s) — free diversity (any non-dominated region not yet covered)",
            "",
            "For gap-filling: look at the two elite mixes bounding the gap and generate intermediate/beyond mixes.",
            "For extreme GWP: use maximum fly ash + slag replacement, minimum PC content, low water content.",
            "For extreme strength: use high binder content, low w/b ratio, high GGBS or silica fume.",
            "",
        ]
    else:
        lines += [
            f"## Task: Generate {n} New Candidate Mixes",
            "",
            "Goal: INCREASE HYPERVOLUME by generating diverse, non-dominated mixes that extend or fill the Pareto front.",
            "",
            f"Generate {n} candidate mixes covering a range of trade-offs between low GWP and high strength.",
            "Aim for variety: some should focus on very low GWP, some on very high strength, and others on intermediate trade-offs.",
            "",
        ]

    lines += [
        "Verify every mix satisfies all constraint layers before returning.",
        "",
        f"Return exactly {n} mixes as a JSON array. Follow this format exactly:",
        "[",
        '  {"mix": {"PC": 200.0, "FA": 60.0, "SC": 110.0, "FAGG": 800.0, "CAGG": 900.0,',
        '           "WATER": 148.0, "AEA": 0.0, "WR_HR": 2.5, "WR": 0.0, "ACC": 0.0}},',
        '  {"mix": {"PC": ..., "FA": ..., "SC": ..., "FAGG": ..., "CAGG": ...,',
        '           "WATER": ..., "AEA": ..., "WR_HR": ..., "WR": ..., "ACC": ...}},',
        f"  ... ({n} elements total)",
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
                            cfg: HybridConfig, client, genai_types,
                            n_solutions: int = None,
                            feedback: dict = None) -> tuple:
    """
    Single LLM call requesting n_solutions candidates (defaults to cfg.llm_n_solutions).
    Returns (list of np.ndarray, n_parse_fails).
    """
    prompt = build_hybrid_prompt(elite, raw_b, der_b, phys_b, cfg,
                                 n_solutions=n_solutions, feedback=feedback)
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


def collect_feasible_llm_solutions(
    elite: list, raw_b: dict, der_b: dict, phys_b: dict,
    cfg: HybridConfig, client, genai_types,
    target_n: int = None, max_attempts: int = 4,
) -> tuple:
    """
    Repeatedly call the LLM until exactly target_n Layer-2+3-feasible solutions
    are collected (or max_attempts is reached).

    Each call requests extra solutions to compensate for expected rejections:
      - First call: ask for target_n * 2 (generous buffer)
      - Subsequent calls: ask for only what is still needed

    Returns (solutions[:target_n], total_llm_calls, total_parse_fails, total_infeasible).
    """
    target_n = target_n or cfg.llm_n_solutions
    collected = []
    total_parse_fails = 0
    total_infeasible = 0
    total_calls = 0

    for attempt in range(max_attempts):
        if len(collected) >= target_n:
            break
        needed = target_n - len(collected)
        # First attempt: ask for 2× to get a good batch; subsequent: ask exactly needed
        ask = min(needed * 2, 40) if attempt == 0 else needed
        sols, n_fails = call_llm_for_solutions(
            elite, raw_b, der_b, phys_b, cfg, client, genai_types, n_solutions=ask
        )
        total_calls += 1
        total_parse_fails += n_fails
        for x in sols:
            if _is_feasible(x, der_b, phys_b):
                collected.append(x)
                if len(collected) >= target_n:
                    break
            else:
                total_infeasible += 1

    return collected[:target_n], total_calls, total_parse_fails, total_infeasible


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
    llm_infeasible = 0
    _prev_feedback = None   # feedback dict passed to the next injection
    llm_solution_log = []   # per-injection record of LLM solution quality
    _last_n_inject = 0      # how many LLM solutions were injected last call

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
        _freq_ok = (cfg.llm_frequency > 0 and gen >= cfg.llm_start_gen
                    and (gen + 1 - cfg.llm_start_gen) % cfg.llm_frequency == 0)
        _stag_ok = True
        if _freq_ok and cfg.stagnation_window > 0 and len(hv_history) >= cfg.stagnation_window:
            recent_gain = hv_history[-1]["hv"] - hv_history[-cfg.stagnation_window]["hv"]
            _stag_ok = recent_gain <= cfg.stagnation_threshold
        if use_llm and _freq_ok and _stag_ok:
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

                # Build feedback from previous injection
                curr_hv_val = _hv([{
                    "GWP": float(obj[i, 0]), "28day": float(-obj[i, 1])
                } for i in front0_feasible])
                curr_gwps = [obj[i, 0] for i in front0_feasible]
                curr_strs = [-obj[i, 1] for i in front0_feasible]
                curr_feedback = {
                    "curr_hv":      curr_hv_val,
                    "curr_gwp_min": float(min(curr_gwps)) if curr_gwps else 0,
                    "curr_gwp_max": float(max(curr_gwps)) if curr_gwps else 0,
                    "curr_str_min": float(min(curr_strs)) if curr_strs else 0,
                    "curr_str_max": float(max(curr_strs)) if curr_strs else 0,
                    "curr_pareto_coords": [
                        (round(float(obj[i, 0]), 1), round(float(-obj[i, 1]), 1))
                        for i in front0_feasible
                    ],
                }
                feedback_to_pass = None
                if _prev_feedback is not None:
                    feedback_to_pass = {
                        "prev_hv":            _prev_feedback["curr_hv"],
                        "prev_gwp_min":       _prev_feedback["curr_gwp_min"],
                        "prev_gwp_max":       _prev_feedback["curr_gwp_max"],
                        "prev_str_min":       _prev_feedback["curr_str_min"],
                        "prev_str_max":       _prev_feedback["curr_str_max"],
                        "prev_pareto_coords": _prev_feedback.get("curr_pareto_coords", []),
                        **curr_feedback,
                    }
                _prev_feedback = curr_feedback

                new_sols, n_fails = call_llm_for_solutions(
                    elite, raw_b, der_b, phys_b, cfg, client, genai_types,
                    feedback=feedback_to_pass
                )
                llm_calls += 1
                parse_fails += n_fails

                # Inject LLM solutions into the offspring pool.
                n_infeas_llm = sum(1 for x in new_sols if not _is_feasible(x, der_b, phys_b))
                llm_infeasible += n_infeas_llm
                n_inject = min(len(new_sols), cfg.llm_n_solutions)
                if cfg.llm_inject_mode == "augment":
                    # Append LLM solutions — NSGA-II offspring are untouched.
                    # Environmental selection then picks best from pop + offspring + LLM.
                    if n_inject > 0:
                        llm_arr = np.array([new_sols[j] for j in range(n_inject)])
                        offspring = np.vstack([offspring, llm_arr])
                else:
                    # "replace": overwrite the N worst offspring (original behavior).
                    for j in range(n_inject):
                        offspring[-(j + 1)] = new_sols[j]
                _last_n_inject = n_inject
                note = f"+LLM×{n_inject}"
                if n_infeas_llm:
                    note += f"({n_infeas_llm} infeas,kept)"
            else:
                _last_n_inject = 0

        # ── Evaluate offspring ─────────────────────────────────
        off_obj, off_feas = evaluate_population(offspring, raw_b, der_b, phys_b, meta, cfg.constraint_mode)

        # ── Log LLM solution quality (before environmental selection) ──
        if _last_n_inject > 0:
            # LLM solutions occupy the last _last_n_inject slots of offspring
            front0_f = [i for i in fast_nondominated_sort(obj, feas)[0] if feas[i]]
            curr_front_pts = [(obj[i, 0], -obj[i, 1]) for i in front0_f]
            for j in range(_last_n_inject):
                llm_gwp = float(off_obj[-(j + 1), 0])
                llm_str = float(-off_obj[-(j + 1), 1])
                is_feas = bool(off_feas[-(j + 1)])
                # Is this solution non-dominated by the current Pareto front?
                dominated = any(
                    fg <= llm_gwp and fs >= llm_str
                    for fg, fs in curr_front_pts
                ) if curr_front_pts else False
                # Min Euclidean distance to current front (globally normalized)
                _GWP_RNG = 534.5 - 169.0   # global normalization bounds
                _STR_RNG = 106.1 - 17.4
                if curr_front_pts:
                    min_dist = min(
                        (((llm_gwp - fg) / _GWP_RNG) ** 2 + ((llm_str - fs) / _STR_RNG) ** 2) ** 0.5
                        for fg, fs in curr_front_pts
                    )
                else:
                    min_dist = float("nan")
                llm_solution_log.append({
                    "gen": gen + 1,
                    "llm_gwp": round(llm_gwp, 2),
                    "llm_str": round(llm_str, 2),
                    "is_feasible": int(is_feas),
                    "is_dominated": int(dominated),
                    "min_dist_to_front": round(min_dist, 4),
                })
            _last_n_inject = 0

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
    print(f"\n  LLM calls: {llm_calls}  Parse fails: {parse_fails}  LLM infeasible dropped: {llm_infeasible}")
    print(f"  Final Pareto: {len(final_pareto)} solutions  HV: {hv_final:.1f}")

    return {
        "final_pareto": final_pareto,
        "hv_history": hv_history,
        "llm_calls": llm_calls,
        "parse_fails": parse_fails,
        "llm_infeasible": llm_infeasible,
        "llm_solution_log": llm_solution_log,
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
            "llm_calls":      result["llm_calls"],
            "parse_fails":    result["parse_fails"],
            "llm_infeasible": result.get("llm_infeasible", 0),
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

    if result.get("llm_solution_log"):
        pd.DataFrame(result["llm_solution_log"]).to_csv(
            os.path.join(cfg.output_prefix, "llm_solutions.csv"), index=False)

    print(f"\n  Saved -> {cfg.output_prefix}/")
    for k, v in metrics.items():
        print(f"    {k:<20}: {v}")


# ──────────────────────────────────────────────────────────────
# PURE-LLM OPTIMIZER  (Step 4.1 ablation)
# ──────────────────────────────────────────────────────────────

def run_pure_llm(raw_b: dict, der_b: dict, phys_b: dict,
                 meta: dict, cfg: HybridConfig) -> dict:
    """
    Pure-LLM optimizer: no NSGA-II evolution.  The LLM is called every generation,
    and its outputs are evaluated and added to a maintained Pareto archive.

    Uses the same number of LLM calls as the hybrid (llm_frequency=1 by design
    when called via the ablation runner).  Seed controls initial archive population.

    Returns the same dict format as run_hybrid() for drop-in comparability.
    """
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    from google import genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=cfg.gemini_api_key)

    lb = np.array([raw_b[v]["min"] for v in RAW_VARS])
    ub = np.array([raw_b[v]["max"] for v in RAW_VARS])

    # Seed the archive with a random initial population (same as NSGA-II)
    archive = np.column_stack([
        np.random.uniform(raw_b[v]["min"], raw_b[v]["max"], cfg.pop_size)
        for v in RAW_VARS
    ])
    arc_obj, arc_feas = evaluate_population(archive, raw_b, der_b, phys_b, meta, cfg.constraint_mode)

    hv_history = []
    llm_calls = 0
    parse_fails = 0
    llm_solution_log = []

    print(f"\n{'='*62}")
    print(f"  {cfg.name}  [PURE-LLM MODE]")
    print(f"  Pop={cfg.pop_size} | Gen={cfg.max_generations} | LLM calls=every gen")
    print(f"{'='*62}")

    for gen in range(cfg.max_generations):
        # Get current Pareto front from archive
        fronts = fast_nondominated_sort(arc_obj, arc_feas)
        front0 = [i for i in fronts[0] if arc_feas[i]]
        front0.sort(key=lambda i: arc_obj[i, 0])
        elite_idx = front0[:cfg.llm_n_elite]

        elite = []
        for idx in elite_idx:
            mix = _to_dict(archive[idx])
            elite.append({
                "mix": {v: round(float(mix[v]), 1) for v in RAW_VARS},
                "GWP": round(float(arc_obj[idx, 0]), 2),
                "28d_MPa": round(float(-arc_obj[idx, 1]), 2),
            })

        # Build feedback
        curr_hv_val = _hv([{"GWP": float(arc_obj[i,0]), "28day": float(-arc_obj[i,1])} for i in front0])
        curr_coords  = [(round(float(arc_obj[i,0]),1), round(float(-arc_obj[i,1]),1)) for i in front0]

        feedback = {
            "prev_hv":            curr_hv_val,
            "curr_hv":            curr_hv_val,
            "prev_pareto_coords": curr_coords,
            "curr_pareto_coords": curr_coords,
        }

        new_sols, n_fails = call_llm_for_solutions(
            elite, raw_b, der_b, phys_b, cfg, client, genai_types,
            feedback=feedback
        )
        llm_calls += 1
        parse_fails += n_fails

        if new_sols:
            # Evaluate new solutions and add to archive
            new_arr = np.array(new_sols)
            new_obj, new_feas = evaluate_population(new_arr, raw_b, der_b, phys_b, meta, cfg.constraint_mode)

            # Log LLM solution quality
            front_pts = [(arc_obj[i,0], -arc_obj[i,1]) for i in front0]
            gwp_rng = max((p[0] for p in front_pts), default=1) - min((p[0] for p in front_pts), default=0) + 1e-6
            str_rng = max((p[1] for p in front_pts), default=1) - min((p[1] for p in front_pts), default=0) + 1e-6
            for j in range(len(new_sols)):
                lg, ls = float(new_obj[j,0]), float(-new_obj[j,1])
                dominated = any(fp<=lg and fs>=ls for fp,fs in front_pts)
                min_dist = min((((lg-fp)/gwp_rng)**2+((ls-fs)/str_rng)**2)**0.5 for fp,fs in front_pts) if front_pts else float("nan")
                llm_solution_log.append({
                    "gen": gen+1, "llm_gwp": round(lg,2), "llm_str": round(ls,2),
                    "is_feasible": int(new_feas[j]), "is_dominated": int(dominated),
                    "min_dist_to_front": round(min_dist,4),
                })

            # Merge into archive and trim to pop_size by Pareto ranking + crowding
            combined  = np.vstack([archive, new_arr])
            comb_obj  = np.vstack([arc_obj, new_obj])
            comb_feas = np.concatenate([arc_feas, new_feas])
            archive, arc_obj, arc_feas, _, _ = environmental_selection(
                combined, comb_obj, comb_feas, cfg.pop_size
            )

        # HV tracking
        fronts2   = fast_nondominated_sort(arc_obj, arc_feas)
        pareto_hv = [{"GWP": float(arc_obj[i,0]), "28day": float(-arc_obj[i,1])} for i in fronts2[0] if arc_feas[i]]
        hv  = _hv(pareto_hv)
        hv_history.append({"gen": gen+1, "hv": hv, "n_pareto": len(pareto_hv), "llm_injected": 1})

        if (gen+1) % 10 == 0 or gen == 0:
            print(f"  {gen+1:4d}  {hv:10.4f}  {len(pareto_hv):7d}")

    # Final Pareto
    fronts_f = fast_nondominated_sort(arc_obj, arc_feas)
    final_pareto = []
    for i in fronts_f[0]:
        if arc_feas[i]:
            mix = _to_dict(archive[i])
            final_pareto.append({
                **{v: round(float(mix[v]),3) for v in RAW_VARS},
                "GWP": round(float(arc_obj[i,0]),3),
                "28day": round(float(-arc_obj[i,1]),3),
            })
    hv_final = _hv(final_pareto)
    print(f"\n  LLM calls: {llm_calls}  Parse fails: {parse_fails}")
    print(f"  Final Pareto: {len(final_pareto)} solutions  HV: {hv_final:.4f}")

    return {
        "final_pareto": final_pareto,
        "hv_history": hv_history,
        "llm_calls": llm_calls,
        "parse_fails": parse_fails,
        "llm_infeasible": 0,
        "llm_solution_log": llm_solution_log,
    }
