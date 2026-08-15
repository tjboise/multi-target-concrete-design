# Multi-Target Concrete Mix Design via LLM Pareto Search

Extends LLM-based concrete mix design from single-objective optimization to **true multi-objective Pareto optimization**: simultaneously minimizing GWP and maximizing 28-day compressive strength. An LLM-augmented NSGA-II hybrid is benchmarked against a pure NSGA-II baseline across a 3×3 hyperparameter grid.

---

## Table of Contents

1. [Research Design](#research-design)
   - [Problem Formulation](#problem-formulation)
   - [Constraints](#constraints)
   - [Surrogate Model](#surrogate-model)
2. [LLM-NSGA-II Hybrid Optimizer](#llm-nsga-ii-hybrid-optimizer)
   - [Architecture](#architecture)
   - [Prompt Design](#prompt-design)
   - [Hyperparameter Grid Search](#hyperparameter-grid-search)
   - [Results — Physics Constrained](#results--physics-constrained)
   - [Statistical Significance](#statistical-significance)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Repository Structure](#repository-structure)
5. [Setup and Usage](#setup-and-usage)
6. [Related Work](#related-work)

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

Optimization constraints are applied in three layers. All bounds are derived from the 756-mix training dataset (`Concrete_Data_SI.csv`, SI units: kg/m³, MPa) unless stated otherwise. Material densities follow Pfeiffer et al. (2024), Table 4.

#### Layer 1 — Raw ingredient bounds

Each of the 10 decision variables is bounded by the observed min/max in the dataset:

| Variable | Description | Dataset min (kg/m³) | Dataset max (kg/m³) |
|----------|-------------|--------------------:|--------------------:|
| PC | Portland cement | 97.3 | 504.3 |
| FA | Fly ash | 0.0 | 162.0 |
| SC | Slag cement (GGBS) | 0.0 | 332.2 |
| FAGG | Fine aggregate (sand) | 473.5 | 1067.9 |
| CAGG | Coarse aggregate | 400.5 | 1364.6 |
| WATER | Mix water | 90.8 | 214.8 |
| AEA | Air-entraining agent | 0.0 | 1.5 |
| WR\_HR | High-range water reducer (superplasticizer) | 0.0 | 4.7 |
| WR | Water reducer | 0.0 | 7.8 |
| ACC | Accelerator | 0.0 | 28.5 |

#### Layer 2 — Derived ratio constraints

Twelve dimensionless ratios must stay within their dataset ranges. The first eight prevent physically degenerate mixes (e.g., zero binder, pure-aggregate mixes); the last four cap admixture dosages as a fraction of total binder, preventing proportions that are technically within raw bounds but are not observed in practice.

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

Three constraints grounded in concrete volume physics. Material densities follow Pfeiffer et al. (2024), Table 4. The `Vfinal` bound is fixed (not dataset-derived); `Vagg` and `TOTAL_BINDER` bounds come from the dataset distribution.

| Constraint | Formula | Min | Max | Physical meaning |
|------------|---------|----:|----:|-----------------|
| `Vfinal` | Vm + 0.07 if AEA/PC ≥ 0.000244; else Vm + 0.03 | 0.950 | 1.050 | Final concrete volume per m³, accounting for entrained air (Pfeiffer et al. 2024, Eq. 19–20). Vm = Σ(massᵢ / ρᵢ). Air fraction: 7 % when AEA is present, 3 % otherwise. Covers 88 % of the dataset (5th–95th percentile); the paper's design target is the tighter [0.99, 1.01]. |
| `Vagg` | (FAGG + CAGG) / 2650 | 0.413 m³/m³ | 0.778 m³/m³ | Volume of aggregates in the mix (ρFAGG = ρCAGG = 2650 kg/m³) |
| `TOTAL_BINDER` | PC + FA + SC | 207.7 kg/m³ | 590.3 kg/m³ | Total cementitious content; prevents extreme binder reduction that fools the surrogate |

**Rationale for Layer 3:** Without these constraints, the optimizer exploits the CatBoost surrogate by simultaneously minimizing binder (low GWP) and maximizing accelerator (ACC) to predict high strength — producing mixes with `TOTAL_BINDER` below 170 kg/m³ and `ACC` above 22 kg/m³, both far outside any real-world mix. The physics constraints close this exploitation gap. The `Vfinal` constraint replaces the earlier `solid_vol` constraint and is more physically accurate: it separates the material volume fraction (Vm) from entrained air, matching the volume model used in the source dataset paper.

---

### Surrogate Model

The CatBoost-Chain surrogate is trained on `Concrete_Data_SI.csv` (756 mixes, native kg/m³) and reused from the companion project [`low_carbon_concrete`](../low_carbon_concrete). It predicts 7-day, 28-day, and 56-day compressive strength via a chained architecture:

```
Stage 1: f(mix features)          → pred_7day
Stage 2: f(mix features, pred_7d) → pred_28day
Stage 3: f(mix features, pred_28d)→ pred_56day
```

**Performance:** 28-day Test R² = 0.923, chained R² = 0.913.  
Model path (relative): `../low_carbon_concrete/concrete_catboost_optimized.pkl`  
Input unit: kg/m³

---

## LLM-NSGA-II Hybrid Optimizer

Rather than replacing NSGA-II with the LLM, this approach **injects LLM-proposed solutions into a running NSGA-II loop**. Every F generations, the top Pareto-elite solutions are sent to Gemini; the LLM returns N new candidate mixes that are inserted into the offspring pool before environmental selection.

### Architecture

```
NSGA-II generation loop
  ├── SBX crossover + polynomial mutation  →  offspring
  ├── every F generations:
  │     elite solutions → LLM prompt → N new candidates
  │     replace worst N offspring with LLM candidates
  └── fast non-dominated sort + crowding distance → next population
```

Injection timing: the first LLM call occurs at generation F (not at gen 0). With F=10, injections occur at gen 10, 20, 30, …, 100 (10 calls total per run). The HV recorded at each of those generations already includes the injected solutions.

---

### Prompt Design

Each LLM call sends a structured prompt with five sections:

| Section | Content |
|---------|---------|
| **Objectives** | Minimize GWP, maximize 28-day strength; why they conflict |
| **Material Reference** | GWP factors per ingredient, strength mechanisms, two-path optimization strategy — optional, controlled by `use_knowledge_table` |
| **Constraints** | All three constraint layers: raw bounds (Layer 1), 12 derived ratio constraints with formulas and bounds (Layer 2), volume physics Vfinal/Vagg/TOTAL_BINDER (Layer 3) |
| **Pareto Elite** | Up to 10 current best non-dominated solutions from NSGA-II, sorted by GWP, with w/b, binder, b/agg, and binder composition (PC%/FA%/SC%) computed inline |
| **Task** | Ask the LLM to generate N mixes it believes can push the Pareto front, with free reasoning about composition — no prescribed mix types or distribution |

<details>
<summary>Full example prompt (use_knowledge_table=True, N=10, 4 elite solutions)</summary>

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

  GWP sensitivity (approximate, per 10% binder fraction shift):
    Shifting 10% of Binder from PC → SC saves ≈ (1.048-0.264)·0.10·Binder kg CO2/m3
    Shifting 10% of Binder from PC → FA saves ≈ (1.048-0.328)·0.10·Binder kg CO2/m3

## Two-Path Optimization Strategy

  PATH 1 — Binder Substitution (maintain binder volume, change composition):
    Replace PC with SC (preferred) or FA to lower GWP while keeping total binder roughly constant.
    If strength drops, reduce WATER or increase WR_HR to recover w/b.
    Best for: improving GWP without sacrificing strength; easier to control.

  PATH 2 — Binder Volume Reduction (reduce b/agg ratio):
    Reduce total binder content; compensate with higher FAGG+CAGG, higher WR_HR dosage.
    This pushes the low-GWP frontier further than Path 1 alone.
    Risk: strength may drop — counter with aggressive WR_HR and lower WATER simultaneously.
    Best for: mixes where binder fraction is still high (b/agg > 0.3).

  COMBINED: SC-dominant binder (SC% 40-70%) + low PC% + WR_HR to reduce w/b below 0.42.
            Prevents premature convergence to pure Path 1 (high-SC but still high-binder) solutions.

## Constraints

### Layer 1 — Raw Ingredient Bounds (kg/m3)
All 10 ingredient values must stay within these dataset-derived bounds:

  PC     : [97.3, 504.3]
  FA     : [0.0, 162.0]
  SC     : [0.0, 332.2]
  FAGG   : [473.5, 1067.9]
  CAGG   : [400.5, 1364.6]
  WATER  : [90.8, 214.8]
  AEA    : [0.0, 1.5]
  WR_HR  : [0.0, 4.7]
  WR     : [0.0, 7.8]
  ACC    : [0.0, 28.5]

### Layer 2 — Derived Ratio Constraints
  Binder = PC + FA + SC   (total cementitious content, kg/m3)
  Agg    = FAGG + CAGG    (total aggregate, kg/m3)

  Ratio        Formula                        Min       Max
  ------------ ------------------------- --------  --------
  w/b          WATER / Binder               0.235     0.714
  b/a          Binder / Agg                 0.105     0.488
  SCM%         (FA+SC) / Binder             0.000     0.765
  CAGG%        CAGG / Agg                   0.315     0.721
  FAGG%        FAGG / Agg                   0.279     0.685
  PC%          PC / Binder                  0.235     1.000
  FA%          FA / Binder                  0.000     0.375
  SC%          SC / Binder                  0.000     0.717
  AEA_pct      AEA / Binder                 0.000     0.003
  WR_HR_pct    WR_HR / Binder               0.000     0.012
  WR_pct       WR / Binder                  0.000     0.020
  ACC_pct      ACC / Binder                 0.000     0.061

### Layer 3 — Physics Constraints (Pfeiffer et al. 2024, Eq. 19-20)
  Material densities (kg/m3):
    PC=3150  FA=2200  SC=2900  FAGG=2650  CAGG=2650
    WATER=1000  AEA=1010  WR_HR=1080  WR=1140  ACC=1340

  Vm = PC/3150 + FA/2200 + SC/2900 + FAGG/2650 + CAGG/2650
     + WATER/1000 + AEA/1010 + WR_HR/1080 + WR/1140 + ACC/1340

  Vfinal = Vm + 0.07   if AEA/PC >= 0.000244  (7% entrained air — AEA present)
         = Vm + 0.03   if AEA/PC <  0.000244  (3% entrapped air — no AEA)
  Constraint: 0.950 <= Vfinal <= 1.050

  Vagg = FAGG/2650 + CAGG/2650
  Constraint: 0.413 <= Vagg <= 0.778

  TOTAL_BINDER = PC + FA + SC
  Constraint: 207.7 <= TOTAL_BINDER <= 590.3 kg/m3

## Current Pareto-Elite Solutions (sorted by GWP, low → high)
These are the best non-dominated feasible mixes found by NSGA-II so far.

  [1] GWP=195.2 | 28d=55.8 MPa | w/b=0.392 | binder=370 kg/m3 | b/agg=0.220
       PC%=38%  FA%=22%  SC%=41%
       PC=140.0  FA=80.0  SC=150.0  FAGG=810.0  CAGG=870.0  WATER=145.0  AEA=0.0  WR_HR=1.8  WR=0.0  ACC=0.0
  [2] GWP=248.6 | 28d=67.1 MPa | w/b=0.403 | binder=370 kg/m3 | b/agg=0.218
       PC%=54%  FA%=15%  SC%=31%
       PC=200.0  FA=55.0  SC=115.0  FAGG=800.0  CAGG=895.0  WATER=149.0  AEA=0.0  WR_HR=2.3  WR=0.0  ACC=0.0
  [3] GWP=318.4 | 28d=81.3 MPa | w/b=0.346 | binder=390 kg/m3 | b/agg=0.227
       PC%=69%  FA%=12%  SC%=19%
       PC=270.0  FA=45.0  SC=75.0  FAGG=775.0  CAGG=940.0  WATER=135.0  AEA=0.5  WR_HR=3.5  WR=0.0  ACC=0.0
  [4] GWP=403.7 | 28d=93.5 MPa | w/b=0.293 | binder=430 kg/m3 | b/agg=0.260
       PC%=81%  FA%=6%  SC%=13%
       PC=350.0  FA=25.0  SC=55.0  FAGG=740.0  CAGG=915.0  WATER=126.0  AEA=0.9  WR_HR=4.3  WR=0.0  ACC=0.0

## Task: Generate 10 New Candidate Mixes

You are augmenting NSGA-II's offspring pool. The mixes you generate will be
injected into the next generation's environmental selection alongside the genetic
algorithm's own offspring. Mixes that are Pareto-dominated will be selected against;
those that push or extend the current Pareto front will survive and propagate into
future generations.

Generate 10 mixes that you believe have the best chance of being
non-dominated relative to the current Pareto front shown above.

Study the elite solutions carefully: identify their compositional patterns, note where
the GWP-strength trade-off is densely covered and where it is sparse, and reason about
what changes to ingredient proportions could push the front outward in any direction.
Draw on your material science knowledge and the formulas above to verify that your
proposals are physically reasonable and satisfy all three constraint layers.

Every proposed mix must satisfy ALL constraints listed above. Verify each mix
before including it in your response.

Return exactly 10 mixes as a JSON array:
  [{"mix": {"PC": ..., "FA": ..., "SC": ..., "FAGG": ..., "CAGG": ...,
            "WATER": ..., "AEA": ..., "WR_HR": ..., "WR": ..., "ACC": ...}}, ...]
All values in kg/m3.
```

</details>

---

### Hyperparameter Grid Search

A 3×3 grid over injection frequency F ∈ {5, 10, 20} and injection size N ∈ {5, 10, 20} was run with pop=50, gen=100, 5 repeats per cell (90 total runs).

### Results — Physics Constrained

After adding all three constraint layers, the grid was re-run with the surrogate model retrained on `Concrete_Data_SI.csv` (756 rows, native kg/m³) to eliminate the unit-mismatch from the earlier lb/yd³ → kg/m³ conversion. HV is computed in the normalized [0, 1]² objective space using dataset-derived bounds (GWP: 169–534.5 kg CO₂/m³; 28d strength: 17.4–106.1 MPa), making values directly comparable across experiments.

**Grid summary (mean normalized HV % gain, hybrid vs within-cell baseline):**

| | N=5 | N=10 | N=20 |
|---|:---:|:---:|:---:|
| **F=5**  | +1.46% | +0.86% | −4.05%\* |
| **F=10** | +3.53% | +4.30% | **+5.21%**\* |
| **F=20** | −1.11% | −1.23% | −0.77%\* |

\* N=20 averaged 57 / 27 / 13 JSON parse failures per run (Gemini fails to return 20 valid solutions in one call), reducing effective injection count.

| Configuration | Role |
|---|---|
| **F=10, N=20** | Selected best (mean **+5.21% HV**, highest gain; 27 parse failures/run) |
| F=10, N=5 | Best parse-failure-free (mean +3.53% HV, std ±7.72%, zero parse failures) |
| F=20, N=5 | Most stable (mean −1.11% HV, std ±5.44%, only 5 LLM calls/run) |

#### Pareto front: Hybrid vs Baseline (F=10, N=20)

![Pareto front: hybrid F=10,N=20 vs baseline, physics-constrained](results/figures/pareto_constrained_f10n20.png)

All 5 runs pooled; curves are PCHIP-smoothed non-dominated fronts. The hybrid pushes the high-strength end of the Pareto front: max 28-day strength reaches **101.2 MPa** vs **98.8 MPa** for the baseline (+2.4%). The hybrid also narrows the GWP upper end (hybrid [136.5, 282.6], baseline [136.2, 307.5] kg CO₂/m³), indicating the LLM's injections both raise achievable strength and reduce coverage of high-GWP, low-value mixes.

#### Convergence curves (F=10, N=20)

![HV convergence: hybrid F=10,N=20 vs baseline, mean ± 1 SD](results/figures/convergence_constrained_f10n20.png)

The hybrid leads from the very first injection (generation 10) and maintains a consistent advantage throughout. Final normalized HV: **0.8696 (hybrid)** vs **0.8273 (baseline)**, a **+5.1% gain**.

**HV gap at each injection checkpoint** (mean across 5 reps, F=10, N=20):

| Generation | Event | Baseline HV | Hybrid HV | Gap (%) |
|:----------:|:------|------------:|----------:|--------:|
| 10  | After injection 1  | 0.5626 | 0.6042 | **+7.39%** |
| 20  | After injection 2  | 0.6169 | 0.6770 | **+9.74%** |
| 30  | After injection 3  | 0.6485 | 0.7036 | **+8.51%** |
| 40  | After injection 4  | 0.6952 | 0.7397 | +6.39% |
| 50  | After injection 5  | 0.7191 | 0.7701 | +7.09% |
| 60  | After injection 6  | 0.7521 | 0.8068 | +7.28% |
| 70  | After injection 7  | 0.7673 | 0.8318 | **+8.40%** |
| 80  | After injection 8  | 0.7974 | 0.8476 | +6.29% |
| 90  | After injection 9  | 0.8118 | 0.8546 | +5.27% |
| 100 | Final              | 0.8273 | 0.8696 | +5.11% |

Unlike smaller N configurations (which show 2–3 generations of disruption before the hybrid advantage emerges), F=10, N=20 shows **a positive gap from the very first injection**. With N=20 solutions injected per call, the LLM's influence is strong enough to immediately shift the population toward higher-quality regions. The gap peaks at **+9.74% at generation 20** and stabilises around +5–7% thereafter.

---

### Statistical Significance

#### Constrained grid (n=5 reps per cell)

Friedman test across 9 hybrid configurations: χ²=9.867, p=0.274 (ns).  
Wilcoxon signed-rank test (paired within each cell, two-sided):

| Configuration | HV baseline | HV hybrid | Δ% | p-value |
|---|:---:|:---:|:---:|:---:|
| F=5,  N=5  | 0.7963 | 0.7982 | +0.24% | 1.0000 |
| F=5,  N=10 | 0.7873 | 0.7859 | −0.18% | 0.6250 |
| F=5,  N=20\* | 0.8344 | 0.7971 | −4.47% | 0.4375 |
| F=10, N=5  | 0.7963 | 0.8219 | +3.21% | 0.4375 |
| F=10, N=10 | 0.7983 | 0.8295 | +3.90% | 0.6250 |
| **F=10, N=20**\* | 0.7972 | **0.8339** | **+4.61%** | 0.8125 |
| F=20, N=5  | 0.8322 | 0.8221 | −1.22% | 1.0000 |
| F=20, N=10 | 0.8556 | 0.8421 | −1.58% | 1.0000 |
| F=20, N=20\* | 0.8491 | 0.8407 | −0.98% | 0.8125 |

\* High parse-failure rate; effective N is much lower than nominal.

> **Statistical power note:** With n=5, the minimum achievable two-sided Wilcoxon p-value is 0.0625. No configuration reaches p<0.05 at this sample size. The reference paper (Lu et al. 2026) used n=10 and achieved p=0.002 for their best config. **Planned: rerun with n=10 reps for publication-grade significance.**

---

## Evaluation Metrics

### HV — Hypervolume

Measures the **volume of objective space dominated by the Pareto front**, relative to a fixed reference point. Computed in the normalized [0, 1]² space using dataset-derived bounds:
- GWP: normalized to [0, 1] over [169, 534.5] kg CO₂/m³
- 28d strength: inverted and normalized to [0, 1] over [17.4, 106.1] MPa
- Reference point: [1, 1] (worst-case corner in minimization space)

A larger HV means the front pushes further toward low GWP **and** high strength simultaneously.

### HV ratio = HV(LLM) / HV(NSGA-II)

The fraction of the NSGA-II hypervolume achieved by the LLM.
- `1.0` = LLM matches NSGA-II perfectly
- `0.64` (E3 best) = LLM covers 64% of the NSGA-II objective space

### GD — Generational Distance

Measures how **close the LLM solutions are to the NSGA-II reference front** (LLM → NSGA direction).

```
GD = mean over all LLM Pareto points of: distance to the nearest NSGA-II point
```

- Lower GD = LLM solutions are nearer to the true Pareto front
- Answers: *"Are the LLM solutions high quality?"*

### IGD — Inverted Generational Distance

Measures how **well the LLM front covers the NSGA-II reference front** (NSGA → LLM direction).

```
IGD = mean over all NSGA-II Pareto points of: distance to the nearest LLM point
```

- Lower IGD = the LLM front covers more of the reference front
- Answers: *"Does the LLM explore the full range of the Pareto front, or only part of it?"*

**GD vs IGD:** A front can have low GD (high-quality solutions) but high IGD (only covers one end of the trade-off). Both should be small for a good result.

### Spread

The GWP range covered by the LLM front divided by the GWP range of the NSGA-II front:

```
spread = (max_GWP_llm - min_GWP_llm) / (max_GWP_nsga - min_GWP_nsga)
```

A spread > 1 means the LLM explored a wider GWP range than NSGA-II.

### Non-dom rate

The fraction of LLM proposals that were **non-dominated** (successfully added to the Pareto front at the time of proposal). Higher is better, though a low rate is not always bad — if the LLM finds a few excellent solutions early, later proposals will be dominated by them.

### Parse fails

The number of times the LLM's response **could not be parsed as valid JSON**, requiring a retry. Ideal = 0. Parse failures increase with prompt complexity and reduce effective injection count.

---

## Repository Structure

```
├── optimizer_core_mt.py          # Core multi-objective optimizer (constraints, bounds, physics)
├── optimizer_hybrid_mt.py        # LLM-NSGA-II hybrid optimizer
├── run_grid_fn_mt.py             # F×N hyperparameter grid search runner
├── recompute_metrics_normalized.py  # Post-processing: recompute normalized HV
├── Concrete_Data_SI.csv          # Dataset (756 mixes, kg/m³)
├── results/
│   ├── nsga2_reference.csv       # NSGA-II Pareto front (shared reference)
│   ├── figures/                  # All generated figures
│   ├── grid_base_f??_n??_rep??/  # Baseline NSGA-II runs (90 total)
│   ├── grid_hyb_f??_n??_rep??/   # Hybrid runs (90 total)
│   └── grid_fn_normalized_*.csv  # Grid summary with normalized HV
└── README.md
```

Each `results/grid_*/` folder contains:
- `hv_history.csv` — HV recorded at every generation (gen 1–100)
- `pareto_front.csv` — final non-dominated solutions
- `metrics.csv` — summary metrics

---

## Setup and Usage

### Installation

```bash
pip install pymoo catboost joblib google-genai python-dotenv pandas numpy scipy
```

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

### Hybrid optimizer grid search

```bash
# Full 3×3 grid, 5 repeats (90 runs — takes several hours)
python run_grid_fn_mt.py

# Single cell, e.g. F=10 N=20 only
python run_grid_fn_mt.py --f-levels 10 --n-levels 20 --repeat 5

# Partial rerun starting from rep 3
python run_grid_fn_mt.py --f-levels 10 --n-levels 20 --rep-start 3 --repeat 3
```

---

## Related Work

This project extends LLM-augmented optimization to the multi-objective setting, benchmarking a Gemini-powered NSGA-II hybrid against a pure evolutionary baseline across a 3×3 hyperparameter grid.

Companion project: [`low_carbon_concrete`](../low_carbon_concrete) — single-objective LLM optimizer (minimize GWP with a 28d strength constraint), Gemini + CatBoost-Chain surrogate.
