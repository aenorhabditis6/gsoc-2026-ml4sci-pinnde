# Tina — flow-matching track (GSoC 2026, ML4SCI / GENIE PINNDE)

Two self-contained Python packages for the PINNDE project:

| Package | What it is |
|---|---|
| **`pinnde_eval/`** | A shared, quantitative **evaluation module** — one function compares any two point clouds (real vs. generated) with three tiers of metrics. Meant to be the common yardstick for *both* tracks (score-based and flow-matching). |
| **`flow_matching/`** | A **conditional flow-matching generator** — learns a velocity field and generates by integrating an ODE from noise to data. A simulation-free alternative to the score-based track's O(N·M) Monte-Carlo score estimate. |

Everything is plain functions + a couple of small `nn.Module`s, fully seeded, and
tested. Inputs are torch tensors **or** numpy arrays of shape `(N, d)`.

---

## Install

```bash
pip install torch numpy scipy scikit-learn matplotlib pytest
pip install jetnet          # OPTIONAL — only needed for Tier-2 FPD/KPD (see note below)
```

or:

```bash
pip install -r requirements.txt
```

Runs on CPU (a GPU helps only the flow-matching training, which is tiny here).
Nothing is pip-installed as a package — put this `Tina/` folder on the path:

```python
import sys; sys.path.append(".../GSOC_2026_PINNDE/Tina")
import pinnde_eval, flow_matching
```

(The scripts and tests already insert the path automatically when run from this folder.)

---

## Part 1 — `pinnde_eval`: how good is a generator?

One entry point returns a flat dict of metrics, grouped in three tiers:

| Tier | Question it answers | Metrics |
|------|---------------------|---------|
| **Live monitors** | *Is it converging?* (cheap, run during training) | `mmd`, `swd` |
| **Distribution distances** | *How far apart, with error bars?* | `fpd`, `kpd`, `wasserstein_per_feature` |
| **Standard tests** | *Can you even tell real from generated?* | `classifier_two_sample_test` (AUC), `histogram_chi2` |

```python
import numpy as np
from pinnde_eval import evaluate, report, mmd, swd

real = np.random.default_rng(1).normal(size=(5000, 3))
gen  = np.random.default_rng(2).normal(size=(5000, 3))

# cheap monitors only (use inside a training loop)
evaluate(real, gen, tier="monitor")     # {"mmd": ..., "swd": ...}
swd(real, gen)                           # or call a metric directly

# full suite (everything) -> dict, then a printed table
res = evaluate(real, gen, tier="full")
report(res)
```

`report` prints:

```
            mmd : 0.0001
            swd : 0.049
            auc : 0.502 +/- 0.004     # ~0.5  -> indistinguishable (good)
chi2_per_feature: [1.5, 0.6, 1.1]
      chi2_mean : 1.09                # ~1    -> histograms agree (good)
        w1_mean : 0.046               # ->0   -> distributions close (good)
            fpd : 0.0003 +/- 4e-05    # (None if jetnet not installed)
            kpd : ...
```

**Interpretation cheat-sheet:** AUC → 0.5, χ² → 1, MMD/SWD/W1/FPD → 0 all mean *good*.

### Per-condition evaluation
A global score can hide a bad slice (e.g. one energy bin). Pass one label per sample:

```python
from pinnde_eval import evaluate_by_condition
by_bin = evaluate_by_condition(real, gen, energy_bin_labels, tier="monitor")
print(by_bin["20-50 GeV"]["swd"])
```

### Calorimeter hook (no API change needed)
`evaluate(..., features_fn=...)` maps raw samples → high-level features *before* any
metric runs. For toys the features are the raw coordinates; for the calorimeter,
pass a function that returns shower observables (layer energies, widths, …):

```python
res = evaluate(real, gen, tier="full", features_fn=shower_observables)
```

### Validate the metrics on toys
```bash
python -m pinnde_eval.validate_toys
```
Three checks (all asserted): **null** (same distribution → AUC≈0.5, χ²≈1, distances≈0),
**sensitivity** (perturb → every metric grows monotonically), **speed** (Tier-3 < 1 s on 5k).
Calibration details and the chosen thresholds are in `pinnde_eval/DEVLOG.md`.

---

## Part 2 — `flow_matching`: a toy generator

Learn a velocity field `vθ(x, t)` by regressing it onto the straight-line target,
then generate by integrating the ODE from noise (`t=0`) to data (`t=1`):

```
x_t   = (1 - t)*x0 + t*x1          # x0 ~ N(0,I) noise, x1 data
target = x1 - x0                    # constant velocity along the path
loss   = || vθ(x_t, t) - target ||^2
```

```python
from flow_matching import train_flow_matching, sample

# data: a (N, d) torch tensor
model, history = train_flow_matching(data, dim=2, n_steps=4000, seed=0)
gen = sample(model, n=5000, dim=2, steps=50)     # integrate noise -> data
```

Run the end-to-end toy demo (trains on a 2-D Gaussian mixture, scores with `pinnde_eval`):

```bash
python -m flow_matching.demo
```

**Time convention:** `t=0` = noise, `t=1` = data, sample `0 → 1` (the flow-matching /
Lipman convention). The diffusion / score track uses the opposite labelling
(`t=0` data → `t=1` noise); the two are related by `t → 1−t`.

---

## Tests

```bash
python -m pytest pinnde_eval/tests flow_matching/tests -q     # 27 tests
```

## Reproduce the figures (in `figures/`)
```bash
python make_figures.py        # sensitivity + flow-matching result figures
python make_flow_figure.py    # velocity field + noise->data trajectories
```

| Figure | Shows |
|---|---|
| `sensitivity.png` | every metric grows as the distribution is perturbed (module is calibrated) |
| `fm_convergence.png` | flow-matching MMD/SWD falling to the null floor during training |
| `fm_scatter.png`, `fm_histograms.png` | generated vs. true (2-D GMM) |
| `fm_flow.png` | the learned velocity field and noise→data trajectories |

### Toy result (2-D GMM, 6 modes)
Generated-vs-truth lands near the statistical floor:
**SWD 0.066** (floor ≈0.049) · **MMD ~1e-4** · **AUC 0.54 ± 0.003** (0.5 = perfect).

---

## Note: FPD / KPD need `jetnet`
FPD and KPD wrap `jetnet.evaluation`. If `jetnet` is not installed they return
`None` and everything else still runs (the module degrades gracefully — this path
is tested). `jetnet` installs cleanly on Python 3.11/3.12 (e.g. Google Colab); it
fails to build on Python 3.14. To get FPD/KPD numbers:

```bash
pip install jetnet
python -m pinnde_eval.validate_toys     # FPD/KPD now reported instead of "n/a"
```

---

## Layout
```
Tina/
├── pinnde_eval/        evaluation module
│   ├── evaluate.py     evaluate(), evaluate_by_condition(), report()
│   ├── tier1.py        classifier AUC, histogram chi^2
│   ├── tier2.py        FPD, KPD, per-feature Wasserstein
│   ├── tier3.py        MMD, sliced Wasserstein (pure torch)
│   ├── data.py         seeded GMM toys
│   ├── validate_toys.py null / sensitivity / speed checks
│   ├── DEVLOG.md       calibration record + chosen thresholds
│   └── tests/
├── flow_matching/      conditional flow-matching generator
│   ├── model.py        VelocityField (Fourier time embedding + GELU)
│   ├── core.py         fm_loss, sample (Euler/Heun ODE)
│   ├── train.py        train_flow_matching (Adam + cosine LR, optional monitoring)
│   ├── demo.py         toy GMM demo, scored with pinnde_eval
│   └── tests/
├── figures/            generated result figures
├── make_figures.py     reproduce the result figures
└── make_flow_figure.py reproduce the velocity-field / trajectory figure
```
