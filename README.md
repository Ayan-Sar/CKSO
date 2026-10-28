# CKSO: Constrained Koopman Spectral Operator

Reference implementation of **"CKSO: A Constrained Koopman Spectral Operator for Spectral
Estimation of Echocardiographic Cine Dynamics"** (IEEE Signal Processing Letters manuscript).

CKSO treats ejection-fraction (EF) estimation as modal estimation of a periodically driven
system. The latent transition is a block-diagonal operator of scaled planar rotations:

```
K(ω₁, Θ) = [1] ⊕ ⊕_k ρ_k R(k ω₁) ⊕ ⊕_p ρ⊥_p R(ω⊥_p) ⊕ ⊕_q [ϱ_q]
```

* One subject-conditioned fundamental frequency ω₁, with harmonics locked at kω₁ (Theorem 1).
* A DC eigenvalue that is exactly 1, and every modulus ≤ 1 **by parameterisation**.
* K is normal, so ‖Kˢ‖₂ = 1 for every horizon s at every step, and Kˢ is closed-form (Proposition 2).
* The Nyquist-safe fundamental ω₁ = π/(H+½) · σ(u) gives Hω₁ < π for any u.
* The volume trace is a degree-H trigonometric polynomial (Theorem 3). EF, ED/ES and heart rate
  are analytic functionals of the spectrum (exact unit-circle root finding at inference,
  soft extrema with an explicit error bound during training). There is no EF regression head.

> **Results.** This repository contains **no experimental results** from the manuscript.
> Every number in a results table must be produced by running the scripts below on the
> real datasets. The synthetic data in this repository is for verifying the pipeline only.

---

## Contents

```
ckso/
  layout.py          latent coordinate layout r = 1 + 2H + 2P + Q
  operator.py        CKSO operator (closed-form powers), ablation operators, ||K^s|| diagnostics
  frequency.py       autocorrelation detector, Table IV baselines, refinement head
  networks.py        encoder / decoder, delay embedding
  readout.py         gauge-fixed linear volume observable, Theorem-3 coefficients
  functionals.py     soft / exact extrema, EF, ED/ES phases, error bound
  model.py           CKSOModel (all Table V variants by configuration)
  losses.py          L_rec, L_lin, L_pred, L_vol, L_ef, L_freq, L_perp, L_pos, L_var
  curriculum.py      four-stage curriculum, S / beta annealing
  training.py        trainer (AdamW, two param groups, logging, checkpoints)
  evaluation.py      per-clip inference, beat protocols, aFD, diagnostics
  metrics.py         MAE / RMSE / R2, HR, octave errors, aFD, AUC, stratified MAE
  stats.py           paired bootstrap, Holm, ICC(2,1), seed aggregation
  perturbations.py   robustness perturbations
  complexity.py      parameters, analytic GFLOPs, latency
  data/              EchoNet-Dynamic, CAMUS, EchoNet-Pediatric, HMC-QU, synthetic
configs/             base + per-dataset configs, ablations/
scripts/             train, evaluate, ablations, sweeps, stability, statistics, analyses
tests/               68 automated tests (operator theory, functionals, data, model, training)
examples/            synthetic_demo.py
docs/IMPLEMENTATION_NOTES.md   code map, open choices, manuscript inconsistencies, limitations
```

## Installation

```bash
pip install -r requirements.txt
```

Or install it as a package with `pip install -e .[test]` (add `[camus]` for NIfTI/MHD readers).
It runs on CPU or CUDA; the device defaults to `auto`.

## Quick start (synthetic, CPU, minutes)

```bash
python -m pytest tests -q
```

```bash
python examples/synthetic_demo.py
```

```bash
python scripts/train.py --config configs/synthetic_smoke.yaml --out runs/smoke --device cpu
```

```bash
python scripts/evaluate.py --ckpt runs/smoke/best.pt --out runs/smoke/eval --device cpu
```

`configs/synthetic.yaml` is a longer synthetic run (256 clips, 20 epochs) that is practical on a GPU.

## Datasets

Download each dataset from its official source under its own licence; none are redistributed here.
Set `data.root` in the config or pass `-o data.root=/path`.

| Dataset | Config | Expected layout | Notes |
|---|---|---|---|
| EchoNet-Dynamic | `echonet_dynamic.yaml` | `FileList.csv`, `VolumeTracings.csv`, `Videos/*.avi` | official split. ED/ES are ordered by method-of-disks volume. **No HR field**: pass `data.kwargs.hr_csv` (FileName, HR) for heart-rate metrics |
| CAMUS | `camus.yaml` | `patientXXXX/…_{2CH,4CH}_half_sequence.nii.gz` (or `.mhd`), `Info_*.cfg` | patient-wise 7:1:2 split. Half cycles (ED→ES), see the notes |
| EchoNet-Pediatric | `echonet_pediatric[_psax].yaml` | `{A4C,PSAX}/FileList.csv`, `VolumeTracings.csv`, `Videos/` | fold-based patient-wise split. H = 4. Fine-tunes from an adult checkpoint |
| HMC-QU | `hmcqu.yaml` | `A4C/` videos, `A4C.xlsx`, optional `.mat` wall masks | 25 fps, H = 3, P = 3, Q = 2. Stratified patient-level 5-fold CV (`data.kwargs.fold`). Column names configurable |

To test the EchoNet code path without the real data:

```bash
python scripts/make_synthetic.py --out data/syn_echonet --n 64 16 16
```

```bash
python scripts/train.py --config configs/echonet_dynamic.yaml --out runs/syn_echonet -o data.root=data/syn_echonet data.kwargs.hr_csv=data/syn_echonet/HeartRate.csv train.epochs=10
```

## Reproducing the manuscript's experiments

The commands below regenerate each table. The outputs (CSV/JSON) are the only valid source of numbers.

| Manuscript item | Command |
|---|---|
| Table I (CKSO row), 5 seeds | `python scripts/train.py --config configs/echonet_dynamic.yaml --out runs/ed --all-seeds -o data.root=...`, then `scripts/evaluate.py` per seed and `scripts/aggregate_seeds.py "runs/ed/seed_*/eval_test/summary.json" --out runs/ed/seeds.json` |
| Table II (paired bootstrap, Holm) | `python scripts/compare_methods.py --ckso runs/ed/seed_0/eval_test/predictions.csv --baseline NAME=per_video_preds.csv ... --out table2.csv` (needs **per-video** baseline predictions) |
| Table III (beat protocol) | `scripts/evaluate.py --protocol central`, `--protocol three`, `--protocol all` |
| Table IV (HR, octave errors, ED/ES aFD) | `summary.json` → `hr` (learned ω₁ and three detector baselines), `ed_afd`, `es_afd`. Needs a reference HR |
| Table V (ablations) | `python scripts/run_ablations.py --out runs/ablations -o data.root=...` |
| Section IV-E (‖Kˢ‖ during training, gradient ratio, cost vs S) | train with `-o train.log_operator_every=10`, then `scripts/stability_experiments.py --logs "runs/ablations/*/seed_0"` and `--benchmark --horizons 1 4 16 32 48 64` |
| Efficiency (params, GFLOPs, latency) | `python scripts/complexity.py --config configs/echonet_dynamic.yaml --device cuda` (`--device cpu --threads 1` for CPU) |
| Section IV-F HMC-QU | train each fold (`-o data.kwargs.fold=k`), then evaluate, `scripts/hmcqu_analysis.py trace ...` and `scripts/hmcqu_analysis.py mi --pred "runs/hmcqu/fold*/eval_test/predictions.csv"` |
| Section IV-F CAMUS | train/evaluate with `configs/camus.yaml`, then `scripts/camus_cross_view.py` |
| Section IV-F pediatric | `configs/echonet_pediatric.yaml` (set `train.init_checkpoint`). Zero-shot: `scripts/evaluate.py --ckpt adult.pt -o data.dataset=echonet_pediatric data.root=...` |
| Section V (failure prediction, abstention) | `python scripts/failure_analysis.py --pred .../predictions.csv --out failure.json [--subgroup-csv rhythm.csv]` |
| Appendix E sweeps | `scripts/sweep.py --param model.H --values 1 3 5 7 9`, `--param model.m --values 1 2 4 8`, `--param horizon --values 1 4 16 32 48` |
| Appendix E readout comparison | `scripts/evaluate.py --readout roots`, `--readout soft --beta 40`, `--readout soft --beta 10`, `--readout grid` |
| Appendix E robustness | `scripts/robustness.py --ckpt ... --out ...` (25 fps / HR > 120: `--perturb downsample2 --min-hr 120` on an H = 3 model) |
| Appendix F stratified EF | `summary.json` → `ef_stratified` |

All scripts take `-o key.sub=value` overrides and `--device {auto,cpu,cuda}`.

## Method details as implemented

* **Training curriculum** (`configs/base.yaml`):
  * A: ℒ_rec, ℒ_var for 15 epochs.
  * B: adds ℒ_lin, ℒ_freq, ℒ_⊥ and ℒ_pred with S = 2, for 25 epochs.
  * C: S annealed geometrically 2 → 48, for 30 epochs.
  * D: adds ℒ_vol, ℒ_ef, ℒ_pos, with β annealed 2 → 40, for 30 epochs.
  * Final weights follow Appendix D. AdamW at lr 3e-4 for the networks and 1e-3 for the operator scalars (no weight decay), batch 16.
* **Inference**: the frequency head refines the detector's ω̂ to ω₁. The clip is cut into beats of period 2π/ω₁. For each beat, the state at the beat start gives an on-cycle trigonometric polynomial, and EDV/ESV/ED/ES are its exact extrema. EF is the mean over beats (protocol `all`). Each clip also gets diagnostics: the linearity residual, the frequency disagreement |ω₁ − ω̂|/ω̂, and per-block empirical moduli.
* **Ablations** are configuration switches on the same model; how each Table V row is realised is described in `docs/IMPLEMENTATION_NOTES.md`.

## Important caveats

Read `docs/IMPLEMENTATION_NOTES.md` before running experiments. In short:

* Several loss terms and hyper-parameters are not fully specified in the manuscript; our choices are listed there and marked `[impl]` in the configs.
* Some manuscript statements cannot be reproduced as written and need attention:
  * the parameter count: the described architecture has 0.23 M inference parameters, not 0.72 M;
  * ρ₁₋₃ in the MI descriptor are global parameters;
  * EchoNet-Dynamic has no HR field;
  * CAMUS contains half cycles;
  * HMC-QU has no volume labels;
  * the dense-K cost claims.
* The dataset adapters follow the public release formats but were not run on the real data during development.
* In short synthetic runs the learned ω₁ drifted down once stage D began: to ≈ ω̂/6 with the manuscript's ℒ_lin, and to ω̂/2 with the optional `loss.lin_normalize`. Monitor `freq_disagreement` and read section 6 of the notes before running full experiments.


## License

Add a license file before publishing. The datasets keep their own licences and must not be redistributed.
