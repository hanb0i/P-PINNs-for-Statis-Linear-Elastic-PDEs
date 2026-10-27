# Paper Reproducibility Guide

This repository contains the code and selected artifacts needed to reproduce
the result tables and figures for the manuscript.

## Environment

```bash
python3 -m pip install -r requirements.txt
```

For CPU reproduction:

```bash
export PINN_FORCE_CPU=1
export PYTHONPYCACHEPREFIX=.pycache
export MPLBACKEND=Agg
export MPLCONFIGDIR=.mplconfig
export XDG_CACHE_HOME=.cache
```

Timing values are hardware-dependent. Accuracy artifacts are tied to the saved
checkpoints and the scripts in this repository.

## Fast Paper Check

```bash
PYTHONPYCACHEPREFIX=.pycache python3 scripts/check_paper_reproducibility.py
```

The checker validates the paper-facing CSV/JSON artifacts, required checkpoints,
and selected figures.

## Paper Tables

### PINN-vs-FEM Verification

Artifact:

```text
graphs/data/paper_verification_results.csv
```

Metric: volume MAE (VolMAE) over 100 random interior parameter cases
(seed 20260430, 16x16x8 FEM mesh).

| Configuration | Mean VolMAE (%) | Worst VolMAE (%) |
| --- | ---: | ---: |
| One-layer | 1.56 | 2.87 |
| Three-layer | 2.53 | 4.66 |

The three-layer row is evaluated from the shipped checkpoint
`three-layer-workflow/pinn_model_final.pth`:

```bash
PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 \
python3 scripts/run_random_interior_generalization.py \
  --model-path three-layer-workflow/pinn_model_final.pth \
  --n-cases 100 \
  --seed 20260430 \
  --out-csv graphs/data/three_layer_final_supervised_random100.csv \
  --out-summary graphs/data/three_layer_final_supervised_random100_summary.json
```

The same run reports 5.74% mean / 10.81% worst *top-surface* MAE; the worst
VolMAE case is `random_interior_046`.

**Training provenance (important).** This checkpoint was not trained with the
recipe used in the ablation/seed study. Per `three-layer-workflow/retrain_log.txt`
and `supervised_retrain_log.txt` (commit `9e0a736`) it was trained in two stages
on a different machine: (1) SOAP with the 64-corner supervision cache, then
(2) a warm-started SOAP + 100 L-BFGS stage on
`random_fem_supervision_7dbdac38a653c901.pt`, which holds 36,000 points from
**64 random interior parameter cases** (supervision seed 20260429, evaluation
seed 20260430). Stage 2 is therefore interior-supervised, which contradicts the
manuscript's statement that supervision is at parameter extremes only. Training
itself is not reproducible from this repository (stage-1 checkpoint and exact
settings are not recorded); evaluation of the shipped checkpoint is.

### One-Layer Ablation

Artifact:

```text
graphs/data/one_layer_paper_ablation.csv
```

Regenerate:

```bash
PYTHONPYCACHEPREFIX=.pycache python3 graphs/scripts/eval_one_layer_ablation.py
```

Current reproducible values with the checkpoints in this workspace:

| Variant | Mean MAE (%) | Worst MAE (%) |
| --- | ---: | ---: |
| Full framework | 1.62 | 3.15 |
| - Compliance-aware scaling | 10.76 | 17.73 |
| Vanilla PINN | 22.17 | 24.01 |

Note: an earlier manuscript draft used `19.45 / 23.34` for the vanilla PINN
row. That exact value requires restoring the external vanilla checkpoint used
for that run. The release package uses the available vanilla checkpoint under
`graphs/data/one_layer_ablation_400_from_scratch/vanilla_parameterized_pinn/`.

### Three-Layer Ablation (multi-seed)

Artifacts:

```text
graphs/data/three_layer_seed_study_gpu/<variant>/seed_<k>/result.json  (35 runs, with checkpoints)
graphs/data/three_layer_seed_study_summary.csv                          (full statistics)
graphs/data/three_layer_paper_ablation.csv                              (paper table, generated)
```

Recipe (`scripts/run_three_layer_volume_ablation.py`): trained from scratch,
400 SOAP epochs, no L-BFGS, no warm start, corner-only FEM supervision
(36,000 points from the 64 extreme-parameter cases). Training seeds vary
(0-9 for Full framework, 0-4 for the other variants); every model is scored on
the same 100 random interior cases (evaluation seed 20260430).

| Variant | Seeds | Mean VolMAE (%) | Worst VolMAE (%) |
| --- | ---: | ---: | ---: |
| Full framework | 10 | 6.35 +/- 1.22 | 7.74 +/- 1.12 |
| - Compliance-aware scaling | 5 | 33.17 +/- 14.95 | 102.74 +/- 37.49 |
| - Layerwise PDE decomposition | 5 | 5.60 +/- 0.33 | 7.19 +/- 0.30 |
| - Interface continuity | 5 | 6.02 +/- 0.56 | 7.48 +/- 0.46 |
| - Sparse FEM supervision | 5 | 8.04 +/- 0.73 | 9.12 +/- 0.79 |
| Vanilla PINN baseline | 5 | 27.67 +/- 10.26 | 82.26 +/- 32.03 |

No Full-framework seed reached a worst-case VolMAE below 5% (range 7.07-10.79%).
Removing layerwise decomposition or interface continuity is within seed noise of
the Full framework.

Regenerate (GPU, about 2.5 min per run; CPU is about 6-10 min per run and agrees
to within about 0.05 points on seed 0):

```bash
cd scripts
python3 run_three_layer_seed_study.py --variants "Full framework" --train-seeds 0 1 2 3 4 5 6 7 8 9 --device cuda --out-dir ../graphs/data/three_layer_seed_study_gpu
python3 run_three_layer_seed_study.py --variants all --train-seeds 0 1 2 3 4 --device cuda --out-dir ../graphs/data/three_layer_seed_study_gpu
python3 aggregate_seed_study.py --study-dir ../graphs/data/three_layer_seed_study_gpu
```

`graphs/data/three_layer_seed_study/` holds an independent CPU run of Full
framework seed 0 (5.65 / 7.25) used as a cross-machine check against the GPU
run of the same seed (5.61 / 7.19).

### Efficiency Tables

Artifacts:

```text
graphs/data/one_layer_efficiency_timing_summary.json
graphs/data/efficiency_timing_summary.json
```

Regenerate:

```bash
PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 python3 scripts/report_one_layer_efficiency_timing.py --repeats 200
PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 python3 scripts/report_efficiency_timing.py --repeats 200
```

The current three-layer timing artifact records one-time training as 94.3 s
(1.57 min), while the manuscript text says 4.5 min. Either cite a log for
4.5 min or update the manuscript.

## Figures

Regenerate scripted figures:

```bash
PYTHONPYCACHEPREFIX=.pycache python3 graphs/make_all_graphs.py --mode plot-only
```

Primary figure artifacts are in `graphs/figures/`. The release package also
creates a top-level `Graphs/` directory matching the manuscript's LaTeX paths.

## Manuscript Cleanup Before Submission

- Remove the LaTeX `To-Do List` section.
- Make figure paths consistent (`Graphs/` in LaTeX, generated from `graphs/`).
- Align the one-layer vanilla ablation row with the checkpoint included in the
  release, or restore the exact earlier vanilla checkpoint.
- Replace the three-layer ablation table with the multi-seed results above.
- Describe the Table 5 checkpoint's two-stage, interior-supervised training
  honestly, or restate Table 5 from the multi-seed Full-framework runs.
- The "best case 2.09%" three-layer figure does not match the CSV minimum
  (1.97%, `random_interior_039`).
- The one-layer 6.2 min training time has no timing artifact.
- Align the three-layer training time with the timing artifact or supply the
  supporting log for 4.5 min.
