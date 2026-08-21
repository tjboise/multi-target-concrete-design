# Multi-Target Concrete Mix Design via LLM-Augmented NSGA-II

Simultaneously minimizes GWP (kg CO₂e/m³) and maximizes 28-day compressive strength (MPa) using an **LLM-hybrid NSGA-II** optimizer. Every generation, Gemini proposes 5 new candidate mixes that are appended to NSGA-II's offspring pool before environmental selection. A paired baseline (same seeds, no LLM) runs in parallel for significance testing.

---

## Table of Contents

1. [Research Design](#research-design)
   - [Problem Formulation](#problem-formulation)
   - [Constraints](#constraints)
   - [Surrogate Model](#surrogate-model)
2. [LLM-NSGA-II Hybrid Optimizer](#llm-nsga-ii-hybrid-optimizer)
   - [Architecture](#architecture)
   - [Prompt Design](#prompt-design)
3. [Results](#results)
   - [Step 1 — Statistical Significance](#step-1--statistical-significance)
   - [Step 2 — LLM Solution Quality](#step-2--llm-solution-quality)
   - [Step 3 — Pareto Front Diversity](#step-3--pareto-front-diversity)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Repository Structure](#repository-structure)
6. [Setup and Usage](#setup-and-usage)
7. [Related Work](#related-work)

---

## Research Design

### Problem Formulation

**Objectives (two conflicting):**
- **Minimize GWP** (kg CO₂-eq/m³): `GWP = PC×1.048 + FA×0.328 + SC×0.264 + CAGG×0.0037 + FAGG×0.0026`
- **Maximize 28-day compressive strength** (MPa): predicted by CatBoost-Chain surrogate

**Variables (10 raw ingredients, kg/m³):**
`PC`, `FA`, `SC`, `FAGG`, `CAGG`, `WATER`, `AEA`, `WR_HR`, `WR`, `ACC`

**Constraints:** three layers enforced during optimization (all bounds derived from the training dataset, except where noted). See [Constraints](#constraints) below.

A solution **A dominates** solution **B** when:
`A.GWP ≤ B.GWP` AND `A.28d ≥ B.28d` (with at least one strict inequality).

The **Pareto front** is the set of all non-dominated solutions.

---

### Constraints

Optimization constraints are applied in three layers. All bounds are derived from the 756-mix training dataset (`Concrete_Data_SI_clean.csv`, SI units: kg/m³, MPa) unless stated otherwise.

#### Layer 1 — Raw ingredient bounds

| Variable | Description | Min (kg/m³) | Max (kg/m³) |
|----------|-------------|------------:|------------:|
| PC | Portland cement | 97.3 | 504.3 |
| FA | Fly ash | 0.0 | 162.0 |
| SC | Slag cement (GGBS) | 0.0 | 332.2 |
| FAGG | Fine aggregate (sand) | 473.5 | 1067.9 |
| CAGG | Coarse aggregate | 400.5 | 1364.6 |
| WATER | Mix water | 90.8 | 214.8 |
| AEA | Air-entraining agent | 0.0 | 1.5 |
| WR\_HR | High-range water reducer | 0.0 | 4.7 |
| WR | Water reducer | 0.0 | 7.8 |
| ACC | Accelerator | 0.0 | 28.5 |

#### Layer 2 — Derived ratio constraints

| Ratio | Formula | Min | Max |
|-------|---------|----:|----:|
| w/b | WATER / (PC+FA+SC) | 0.235 | 0.714 |
| b/a | (PC+FA+SC) / (FAGG+CAGG) | 0.105 | 0.488 |
| SCM% | (FA+SC) / (PC+FA+SC) | 0.000 | 0.765 |
| CAGG% | CAGG / (FAGG+CAGG) | 0.315 | 0.721 |
| FAGG% | FAGG / (FAGG+CAGG) | 0.279 | 0.685 |
| PC% | PC / (PC+FA+SC) | 0.235 | 1.000 |
| FA% | FA / (PC+FA+SC) | 0.000 | 0.375 |
| SC% | SC / (PC+FA+SC) | 0.000 | 0.717 |
| AEA\_pct | AEA / (PC+FA+SC) | 0.000 | 0.003 |
| WR\_HR\_pct | WR\_HR / (PC+FA+SC) | 0.000 | 0.012 |
| WR\_pct | WR / (PC+FA+SC) | 0.000 | 0.020 |
| ACC\_pct | ACC / (PC+FA+SC) | 0.000 | 0.061 |

#### Layer 3 — Physics constraints

| Constraint | Formula | Min | Max | Physical meaning |
|------------|---------|----:|----:|-----------------|
| `Vfinal` | Vm + 0.07 (AEA present) or Vm + 0.03 | 0.950 | 1.050 | Final concrete volume per m³ (Pfeiffer et al. 2024, Eq. 19–20) |
| `Vagg` | FAGG/2630 + CAGG/2710 | 0.553 m³/m³ | 0.768 m³/m³ | Volume fraction of aggregates |
| `TOTAL_BINDER` | PC + FA + SC | 207.7 kg/m³ | 590.3 kg/m³ | Total cementitious content |

---

### Surrogate Model

CatBoost-Chain surrogate trained on `Concrete_Data_SI_clean.csv` (756 mixes, kg/m³), reused from [`low_carbon_concrete`](../low_carbon_concrete):

```
Stage 1: f(mix features)          → pred_7day
Stage 2: f(mix features, pred_7d) → pred_28day
Stage 3: f(mix features, pred_28d)→ pred_56day
```

**Performance:** 28-day Test R² = 0.923.

---

## LLM-NSGA-II Hybrid Optimizer

### Architecture

Every generation, 5 LLM-proposed solutions are **appended** to NSGA-II's offspring pool (augment mode — the genetic offspring are never replaced). NSGA-II's environmental selection then decides which solutions survive based on non-domination rank and crowding distance.

```
NSGA-II generation loop
  ├── SBX crossover + polynomial mutation  →  offspring (pop_size solutions)
  ├── every generation:
  │     elite solutions → Gemini prompt → 5 new candidates
  │     append 5 candidates to offspring pool  (pool size = pop_size + 5)
  └── fast non-dominated sort + crowding distance → next population (pop_size)
```

**Key design choices:**
- **Augment mode** (not replace): LLM solutions compete alongside genetic offspring; NSGA-II selection pressure decides their fate.
- **Every generation** injection: simpler than stagnation-based triggers and allows LLM to seed domain knowledge during NSGA-II's formative early phase.
- **Paired design**: each replicate runs a hybrid and a baseline with the same random seed, so any HV difference is attributable solely to LLM injection.

---

### Prompt Design

Each LLM call sends a structured prompt with five sections:

| Section | Content |
|---------|---------|
| **Objectives** | Minimize GWP, maximize 28-day strength; why they conflict |
| **Material Reference** | GWP factors per ingredient, strength mechanisms, two-path optimization strategy |
| **Constraints** | All three constraint layers with exact bounds and formulas |
| **Pareto Elite** | Up to 10 current best non-dominated solutions from NSGA-II |
| **Task** | Generate 5 mixes targeting under-explored regions of the Pareto front |

<details>
<summary>Full example prompt</summary>

```
You are an expert concrete mix designer assisting a multi-objective genetic algorithm.

## Optimization Objectives
Simultaneously optimize two competing objectives:
  1. MINIMIZE GWP (kg CO2-eq/m3) — greenhouse gas emission footprint
  2. MAXIMIZE 28-day compressive strength (MPa) — structural performance

These objectives conflict: cementitious binder drives both strength and GWP.
A Pareto-optimal mix cannot improve one objective without worsening the other.

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

[... constraints section omitted for brevity ...]

## Task: Generate 5 New Candidate Mixes
Generate 5 mixes that push or extend the current Pareto front.
Return exactly 5 mixes as a JSON array.
```

</details>

---

## Results

**Experiment:** `grid_everyg` — 30 paired replicates, pop=50, gen=100, 5 LLM solutions injected every generation (augment mode), `seed=rep`, `constraint_mode=feasibility_first`, `gemini-2.5-flash-lite`.

HV normalized in [0,1]² using dataset-derived bounds (GWP: [169, 534.5] kg CO₂/m³; strength: [17.4, 106.1] MPa; reference point: [1,1]).

---

### Step 1 — Statistical Significance

| Metric | Value |
|--------|-------|
| Mean ΔHV (hybrid − baseline) | **+0.046** |
| % replicates improved | **77%** |
| Paired t-test (one-sided) | t = 3.16, **p = 0.0018** |
| Wilcoxon signed-rank (one-sided) | W = 378, **p = 0.0010** |

#### Pareto front (all 30 replicates)

![Pareto front: NSGA-II vs LLM-Hybrid, 30 replicates](results/figures/pareto_front_everyg.png)

Scatter of all 1,500 final Pareto solutions per method. Lines show the binned mean of maximum achievable strength per GWP interval (±1 SD shaded). The hybrid extends the front toward lower GWP and higher strength at both extremes.

#### Convergence curves

![HV convergence: NSGA-II vs LLM-Hybrid, mean ± 1 SD, 30 replicates](results/figures/convergence_everyg.png)

Mean HV ± 1 SD across 30 paired replicates. The hybrid maintains a positive lead from generation 1 onward.

#### Per-generation ΔHV increment

![Incremental ΔHV per generation](results/figures/delta_hv_incremental_everyg.png)

Change in mean ΔHV from one generation to the next. **Red bars**: LLM contributed above-average useful solutions at that generation (frac_useful > 13.8%); **green bars**: below-average. Red bars cluster in early generations when the Pareto front is sparse and easy to extend.

---

### Step 2 — LLM Solution Quality

15,000 LLM solutions logged across 30 replicates (500 solutions per rep: 5 solutions × 100 generations).

| Quality Metric | Mean | SD |
|---------------|-----:|---:|
| Fraction feasible (all constraints satisfied) | 59.4% | 7.6% |
| Fraction non-dominated (vs. current front) | 26.6% | 4.6% |
| Fraction useful (feasible **and** non-dominated) | 13.8% | 3.3% |
| Median distance to Pareto front (normalized) | 0.020 | 0.009 |

#### LLM solution quality over generations

![LLM solution quality stacked area — 100 generations, 30 replicates](results/figures/step2_area.png)

Each generation, 5 LLM solutions are injected. The stacked area shows their breakdown: useful (feasible and non-dominated, red), feasible but dominated (orange), and infeasible (grey). Quality peaks in early generations when the Pareto front is sparse, then declines as NSGA-II refines it.

**Correlations with per-replicate ΔHV** (n = 30):

| Predictor | r | p |
|-----------|--:|--:|
| Fraction non-dominated | +0.45 | 0.013 ★ |
| Fraction useful (feasible + non-dominated) | +0.38 | 0.041 ★ |
| Fraction feasible | +0.05 | 0.807 |

Replicates in which the LLM generated more non-dominated solutions showed larger HV improvement, confirming that the effect is mechanistic rather than coincidental.

---

### Step 3 — Pareto Front Diversity

**Diversity metric:** mean pairwise Euclidean distance between all solutions on the final Pareto front, computed in normalized objective space:

$$D = \frac{2}{n(n-1)} \sum_{i < j} \left\| \tilde{x}_i - \tilde{x}_j \right\|_2$$

where $\tilde{x}_i = \bigl(\frac{\text{GWP}_i - 169}{534.5 - 169},\; 1 - \frac{\text{str}_i - 17.4}{106.1 - 17.4}\bigr)$ normalizes each solution to [0, 1]². Larger *D* means solutions are spread more broadly across the trade-off curve.

![Pareto front diversity — box plot with individual replicates](results/figures/step3_paired.png)

| | Baseline | Hybrid | Δ |
|--|:--------:|:------:|:-:|
| Mean diversity | 0.174 | 0.198 | **+0.024 (+13.6%)** |
| SD | 0.034 | 0.035 | — |
| % replicates with higher diversity | — | — | **70%** |

- Paired t-test (two-sided): t = 3.14, **p = 0.0039**
- Wilcoxon (two-sided): W = 112, **p = 0.0120**
- Correlation ΔDiversity vs ΔHV: **r = +0.593, p = 0.001**

LLM injection not only improves the hypervolume but also spreads the Pareto front more broadly across the GWP–strength trade-off space. Replicates with larger diversity gains also achieved larger HV gains.

---

## Evaluation Metrics

### HV — Hypervolume

Measures the volume of objective space dominated by the Pareto front, relative to a fixed reference point [1,1] in normalized space.

- GWP: normalized to [0,1] over [169, 534.5] kg CO₂/m³
- Strength: inverted and normalized to [0,1] over [17.4, 106.1] MPa

Larger HV = front pushes further toward low GWP **and** high strength simultaneously.

### ΔHV = HV(hybrid) − HV(baseline)

Primary outcome variable. Positive = hybrid outperforms the paired baseline sharing the same random seed.

### Diversity

Average pairwise Euclidean distance between all Pareto front solutions in normalized objective space. Higher diversity = broader coverage of the trade-off curve.

---

## Repository Structure

```
├── optimizer_core_mt.py          # Core NSGA-II optimizer (constraints, bounds, physics)
├── optimizer_hybrid_mt.py        # LLM-augmented NSGA-II hybrid
├── run_grid_late_mt.py           # Runner for grid_everyg experiments
├── run_ablation_mt.py            # Ablation study runner (4.1: NSGA vs LLM vs hybrid; 4.2: prompt variants)
├── analyze_grid_gap.py           # Steps 1–3 analysis (significance, LLM quality, diversity)
├── Concrete_Data_SI_clean.csv    # Dataset (756 mixes filtered by Vfinal, kg/m³)
├── results/
│   ├── figures/                  # Figures for README
│   │   ├── pareto_front_everyg.png
│   │   ├── convergence_everyg.png
│   │   ├── delta_hv_incremental_everyg.png
│   │   ├── step2_area.png
│   │   └── step3_paired.png
│   ├── grid_everyg_hyb_rep{01-30}/   # Hybrid runs (pareto_front.csv, hv_history.csv, metrics.csv, llm_solutions.csv)
│   ├── grid_everyg_base_rep{01-30}/  # Baseline NSGA-II runs
│   └── grid_everyg_analysis.csv      # Per-replicate summary
└── README.md
```

---

## Setup and Usage

### Installation

```bash
pip install pymoo catboost joblib google-genai python-dotenv pandas numpy scipy matplotlib
```

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

### Run grid_everyg (30 paired replicates)

```bash
python run_grid_late_mt.py --mode late --start-gen 0 --inject-mode augment \
    --prefix grid_everyg --repeat 30 --rep-start 1
```

### Analyze results (Steps 1–3)

```bash
python analyze_grid_gap.py --prefix grid_everyg --reps 30
```

### Run ablation study (Step 4)

```bash
python run_ablation_mt.py --study all --repeat 10
```

---

## Related Work

This project extends LLM-augmented optimization to the multi-objective setting, benchmarking a Gemini-powered NSGA-II hybrid against a pure evolutionary baseline.

Companion project: [`low_carbon_concrete`](../low_carbon_concrete) — single-objective LLM optimizer (minimize GWP with a 28d strength constraint).
