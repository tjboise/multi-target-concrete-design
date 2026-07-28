# Multi-Target Concrete Mix Design via LLM Pareto Search

This project extends LLM-based concrete mix design from single-objective optimization (minimize GWP subject to a strength constraint) to **true multi-objective Pareto optimization**: simultaneously minimizing GWP and maximizing 28-day compressive strength.

The LLM iteratively proposes concrete mixes and receives feedback about where each proposal sits relative to the current Pareto front. Different prompt strategies are compared via ablation study, with NSGA-II as the reference baseline.

---

## Problem Formulation

**Objectives (two conflicting):**
- **Minimize GWP** (kg CO₂-eq/m³): `GWP = PC×1.048 + FA×0.328 + SC×0.264 + CAGG×0.0037 + FAGG×0.0026`
- **Maximize 28-day compressive strength** (MPa): predicted by CatBoost-Chain surrogate

**Variables (10 raw ingredients, kg/m³):**
`PC`, `FA`, `SC`, `FAGG`, `CAGG`, `WATER`, `AEA`, `WR_HR`, `WR`, `ACC`

**Constraints:** derived ratio bounds (w/b, b/a, SCM%, CAGG%, FAGG%, PC%, FA%, SC%) must stay within the ranges observed in the training dataset.

A solution **A dominates** solution **B** when:
`A.GWP ≤ B.GWP` AND `A.28d ≥ B.28d` (with at least one strict inequality).

The **Pareto front** is the set of all non-dominated solutions.

---

## Repository Structure

```
├── optimizer_core_mt.py          # Core multi-objective optimizer (stable, do not edit between experiments)
├── run_experiment_mt.py          # Ablation experiment runner (edit this to configure experiments)
├── Super_Cleaned_Concrete_Data - backup.csv   # Dataset (756 mixes, building & pavement, PA)
├── results/
│   ├── nsga2_reference.csv       # NSGA-II Pareto front (shared reference for all experiments)
│   ├── e0_baseline/              # E0 results
│   ├── e1_knowledge/             # E1 results
│   ├── e2_rules/                 # E2 results
│   ├── e3_fewshot/               # E3 results
│   └── e4_gap_target/            # E4 results
└── README.md
```

Each `results/<experiment>/` folder contains:
- `trajectory.csv` — all 30 LLM proposals (dominated + non-dominated)
- `llm_pareto_front.csv` — the final non-dominated set found by LLM
- `nsga_pareto_front.csv` — the NSGA-II reference Pareto front
- `metrics.csv` — evaluation metrics

---

## Surrogate Model

The CatBoost-Chain surrogate is reused from the companion project [`low_carbon_concrete`](../low_carbon_concrete).  
It predicts 7-day, 28-day, and 56-day compressive strength via a chained architecture:

```
Stage 1: f(mix features)          → pred_7day
Stage 2: f(mix features, pred_7d) → pred_28day
Stage 3: f(mix features, pred_28d)→ pred_56day
```

Model path (relative): `../low_carbon_concrete/concrete_catboost_optimized.pkl`  
Input unit: kg/m³

---

## Ablation Experiments

Each experiment adds one more prompt component on top of the previous:

| ID | Name | Knowledge Table | Nav. Rules | Few-shot | Targeting |
|----|------|:-:|:-:|:-:|:-:|
| E0 | `e0_baseline`   | ✗ | ✗ | ✗ | none |
| E1 | `e1_knowledge`  | ✓ | ✗ | ✗ | none |
| E2 | `e2_rules`      | ✓ | ✓ | ✗ | none |
| E3 | `e3_fewshot`    | ✓ | ✓ | ✓ | none |
| E4 | `e4_gap_target` | ✓ | ✓ | ✓ | gap  |

**Prompt components:**
- **Knowledge Table**: material-level GWP factors and their effect on strength — tells the LLM what each ingredient does.
- **Navigation Rules**: situation-based Pareto strategies (push the low-GWP end, push the high-strength end, fill a gap).
- **Few-shot examples**: 5 real mixes from the dataset spread across the GWP spectrum, showing the LLM realistic trade-off examples.
- **Gap targeting**: each iteration, the LLM is directed toward the largest GWP gap in the current Pareto front.

---

## Evaluation Metrics

### HV — Hypervolume

Measures the **volume of objective space dominated by the Pareto front**, relative to a fixed reference point.

```
Strength ↑
  │    ████████ Pareto front
  │  ████
  │ ██
  │██          ← reference point (GWP=500, strength=10 MPa)
  └────────────── GWP →
       ↑
    HV = shaded area between front and reference point
```

The reference point is set to `[GWP=500, strength=10 MPa]`, representing the worst-case corner.  
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
- GD = 0 means every LLM solution lies exactly on the NSGA-II front
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

A spread > 1 means the LLM explored a wider GWP range than NSGA-II — often caused by the LLM proposing extreme high-GWP solutions that happen to be non-dominated within its own small front.

### Non-dom rate

The fraction of the 30 LLM proposals that were **non-dominated** (i.e., successfully added to the Pareto front at the time of proposal).

- Higher = the LLM more efficiently proposes solutions that advance the front
- A low rate is not always bad: if the LLM finds a few excellent solutions early, later proposals will be dominated by them

### Parse fails

The number of times the LLM's response **could not be parsed as valid JSON**, requiring a retry.  
- Ideal = 0 (every response is correctly formatted)
- High parse-fail counts reduce the effective number of productive iterations
- Parse failures increase with prompt complexity — a sign of prompt engineering overhead

---

## Results Summary

NSGA-II reference: **98 Pareto solutions**, GWP 113.6–240.3 kg/m³, 28d 44.5–75.3 MPa, **HV = 23,859**

| Experiment | HV | HV ratio | GD ↓ | IGD ↓ | Non-dom rate | Parse fails |
|------------|------|----------|-------|-------|:---:|:---:|
| NSGA-II (ref) | 23,859 | 1.000 | — | — | — | — |
| E0 baseline | 13,389 | 0.561 | 0.381 | 0.395 | 30% | 32 |
| E1 +knowledge | 12,683 | 0.532 | 0.479 | 0.381 | 27% | 49 |
| E2 +rules | 13,631 | 0.571 | 0.518 | 0.365 | 40% | 30 |
| **E3 +fewshot** | **15,259** | **0.640** | **0.320** | **0.283** | 13% | 60 |
| E4 +gap-target | 12,915 | 0.541 | 0.616 | 0.468 | 30% | 66 |

**Key findings:**
- **E3 (few-shot) achieves the best HV, GD, and IGD** — real historical examples are more effective than textual rules alone for guiding Pareto exploration.
- **E1 (knowledge table alone) slightly degrades performance** — the longer prompt increases parse failures without providing actionable Pareto navigation guidance.
- **E4 (gap-targeting) performs poorly** — with only 30 iterations, the Pareto front is too sparse for gap-filling to be effective; the LLM over-focuses on narrow regions.
- **Parse failures are consistently high (30–66)** across all experiments — an inherent cost of complex prompts in structured-output tasks.

---

## Usage

### Setup

```bash
pip install pymoo catboost joblib google-genai python-dotenv pandas numpy
```

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

### Run a single experiment

```bash
# Run E0 baseline (also runs NSGA-II reference)
python run_experiment_mt.py --exp e0_baseline

# Reuse saved NSGA-II reference for subsequent runs
python run_experiment_mt.py --exp e3_fewshot --skip-nsga

# Quick test with fewer iterations
python run_experiment_mt.py --exp e0_baseline --iters 5

# Run multiple repeats for statistical analysis
python run_experiment_mt.py --exp e0_baseline --repeat 3 --skip-nsga
```

### Run all experiments

```bash
python run_experiment_mt.py --skip-nsga
```

### Add a new ablation experiment

Edit `run_experiment_mt.py`: add a new entry to the `EXPERIMENTS` dict using `make_cfg()`.

---

## Related Work

This project is Phase 1 of a two-phase study:
- **Phase 1 (this repo):** LLM as Pareto front explorer — compare LLM strategies vs. NSGA-II
- **Phase 2 (planned):** LLM as decision-maker — select the best mix from the Pareto front given an application context (building vs. pavement), replacing the TOPSIS weight-selection step

Companion project: [`low_carbon_concrete`](../low_carbon_concrete) — single-objective LLM optimizer (minimize GWP, 28d strength constraint), Gemini + CatBoost-Chain surrogate.
