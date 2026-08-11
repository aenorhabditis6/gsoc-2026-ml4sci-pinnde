# pinnde_eval

A small, shared, **quantitative** evaluation module for the PINNDE project, so the
score-based and flow-matching tracks can be compared with the same numbers. It
works on toy distributions today and is designed to take calorimeter high-level
features later without an API change.

Everything is plain functions plus the seeded toy generators. Inputs are torch
tensors **or** numpy arrays of shape `(N, d)`; conversion is internal.

## Install

```bash
pip install torch numpy scipy scikit-learn matplotlib pytest
pip install jetnet          # OPTIONAL -- only Tier-2 FPD/KPD need it
```

Runs on Google Colab (CPU or single GPU). `jetnet` provides the peer-reviewed
FPD/KPD reference implementations; if it is missing, every other metric still
runs and FPD/KPD come back as `None`. A plain `pip install jetnet` works on
Colab but fails on macOS (its `wasserstein` dependency wants an OpenMP
compiler); the local no-compiler recipe is in the top-level `Tina/README.md`.

> **Threading gotcha.** If numpy links a threaded OpenBLAS (numpy 1.26 wheels on
> macOS), run everything with `OPENBLAS_NUM_THREADS=1`. The Tier-1 classifier
> issues thousands of tiny matrix multiplies, and OpenBLAS's thread busy-wait
> turns a 9-second `evaluate()` into >30 minutes of spinning that looks like a
> hang. Colab's default numpy is unaffected.

The package is not pip-installed; put the folder that contains `pinnde_eval/` on
the path, e.g. in Colab after cloning the repo:

```python
import sys
sys.path.append(".../GSOC_2026_PINNDE/Tina")   # the folder that contains pinnde_eval/
import pinnde_eval
```

## The three tiers

| Tier | Metric | What it tells you | Cost |
|------|--------|-------------------|------|
| 1 | `classifier_two_sample_test` | Test AUC (mean ± std over k retrainings) of an MLP separating real vs generated. **0.5 = indistinguishable.** | medium |
| 1 | `histogram_chi2` | Reduced χ² per feature between real/generated 1D histograms. **≈1 = agree within stats.** | cheap |
| 1 | `separation_power` | CaloChallenge separation power per feature, in [0, 1]. **0 = identical shapes.** Has a strong finite-N floor — see below. | cheap |
| 2 | `fpd` | Fréchet physics distance (2-Wasserstein between Gaussian fits) with infinite-sample extrapolation → `(value, error)`. | medium |
| 2 | `kpd` | Kernel physics distance (polynomial-kernel MMD) with batched uncertainty → `(median, error)`. | medium |
| 2 | `wasserstein_per_feature` | 1D Wasserstein-1 distance per coordinate (vector + mean). | cheap |
| 3 | `mmd` | RBF-kernel squared MMD, median-heuristic bandwidth. Pure torch. | cheap |
| 3 | `swd` | Sliced Wasserstein distance (random 1D projections). Pure torch. | cheap |

**Which tier when?**

- **Tier 3** — inside the training loop, every few hundred steps on ~1–5k
  samples. Fast, pure-torch, no extra dependencies. Use it to watch convergence.
- **Tier 2** — the headline numbers you report and compare between tracks. FPD
  and KPD come with error bars, so two models are comparable.
- **Tier 1** — the CaloChallenge-standard sanity checks. AUC near 0.5 and
  reduced χ² near 1 are the bar a good generator must clear.

## Usage

```python
from pinnde_eval import evaluate, report, mmd, swd

# Full suite (Tiers 1+2+3) -> flat dict
results = evaluate(real, gen, tier="full")
report(results)
#            mmd : 0.0021
#            swd : 0.013
#            auc : 0.52 +/- 0.01
#      chi2_mean : 1.08
#        w1_mean : 0.046
#            fpd : 0.00031 +/- 4e-05
#            kpd : ...

# Tier 3 only -- cheap monitors for the training loop
results = evaluate(real, gen, tier="monitor")   # {"mmd": ..., "swd": ...}
loss_monitor = mmd(real_batch, gen_batch)        # or call directly
```

**Reading the numbers:** AUC → 0.5, χ² → 1, and MMD/SWD/W1/FPD/KPD → their null
floor all mean *good*. See the stability section below for why the comparison is
against the floor and not against zero.

### Conditional evaluation

Use `evaluate_by_condition` when a global score may hide failures in one class,
energy range, or detector region. Pass one label/bin per sample:

```python
from pinnde_eval import evaluate_by_condition

# real and gen share the same per-sample labels (the usual case)
by_energy = evaluate_by_condition(real, gen, energy_bin, tier="monitor")
print(by_energy["20-50 GeV"]["swd"])

# or give gen its own labels when the two sets are binned separately
by_energy = evaluate_by_condition(real, gen, real_energy_bin, gen_energy_bin)
```

### Calorimeter hook (deferred, no API change needed)

`evaluate(..., features_fn=...)` maps raw samples → high-level features before any
metric runs. For toys the features are the raw coordinates; later, layer
energies / shower widths (or a wrapper around the official CaloChallenge
`evaluate.py`) plug in here:

```python
results = evaluate(real, gen, tier="full", features_fn=shower_observables)
```

## How many samples do you need? (`stability.py`)

Every metric has a finite-N **null floor** — the non-zero value a *perfect*
generator still shows at that sample size — and a finite-N **spread**. The
stability study measures both, then reports the smallest N at which a fixed
perturbation separates from the floor by `z_min` combined error bars.

```python
from pinnde_eval import stability_study, separation_z, min_resolvable_n

study = stability_study(n_grid=(250, 1000, 4000), kind="mean", eps=0.2)
separation_z(study)       # {metric: z per N}  -- z >= 2 means "resolvable"
min_resolvable_n(study)   # {metric: smallest N that resolves it, or None}
```

```bash
python -m pinnde_eval.stability     # full table + ../figures/stability.png
```

**The rule this buys you: never compare a metric value against zero — compare it
against the null floor at your N.** SWD/W1 floors are pure finite-N artifacts
falling as roughly N^(−1/2), so "SWD = 0.05" is *excellent* at N=8000 and
*terrible* at N=500. The measured floors are tabulated in `DEVLOG.md` §8.

## Where does the generator fail? (`local.py`)

A single global score hides a localized failure — a dropped or shifted mode
barely moves SWD. Three complementary maps localize the disagreement, all
sharing one sign convention: **positive = real over-dense (the generator misses
that region), negative = generated over-dense (it hallucinates there).**

| Function | Question it answers | How to read it |
|---|---|---|
| `mmd_witness(real, gen, points)` | where is the probability mass wrong? | the RBF-MMD witness at each query point; same kernel/bandwidth as `tier3.mmd`, so it decomposes the global monitor in space |
| `classifier_discrepancy(real, gen)` | which samples give the generator away? | out-of-fold P(real\|x) per sample plus the oof AUC; sort `gen` ascending to rank the worst fakes |
| `binned_residual_map(real, gen, features=(0, 1))` | which histogram bins disagree? | per-bin residual, ~N(0,1) under the null, so \|r\| > 3 is a genuine local failure; this is the Tier-1 χ² decomposed bin by bin |

```python
from pinnde_eval import mmd_witness, classifier_discrepancy, binned_residual_map

w = mmd_witness(real, gen)                    # defaults to the pooled sample
p_real, p_gen, oof_auc = classifier_discrepancy(real, gen)
residuals, edges = binned_residual_map(real, gen, features=(0, 1))
```

```bash
python ../make_local_figure.py      # demo: all three light up on a broken generator
```

All three run in feature space, so for showers they apply to any observable pair
(layer energy vs. width, …) through the same `features_fn` hook.

## Real calorimeter showers (`observables.py`)

`observables.py` is the `features_fn` for CaloChallenge data: it turns an
`(N, n_voxels)` array of energy deposits into the high-level observables
physicists judge showers by — `E_tot`, sampling fraction, longitudinal centre
of gravity and width, transverse centre of gravity and width, and sparsity.

```python
from pinnde_eval import load_calochallenge, shower_observables, shower_features_fn, evaluate

showers, e_inc = load_calochallenge("dataset_2_1.hdf5", n=8000)   # reads a slice
feats, names = shower_observables(showers, e_inc, geometry="ds2")

# or plug straight into evaluate
fn = shower_features_fn(e_inc, geometry="ds2")
res = evaluate(real_showers, gen_showers, features_fn=fn, standardize=True)
```

Get the data from [Zenodo](https://zenodo.org/records/6366271) (ds2, electrons,
two files of 100k showers) and put it in `Tina/`. It is gitignored — the files
are ~1.4 GB each and GitHub rejects blobs over 100 MB.

**Two things that will bite you:**

*Voxel ordering is `(layer, alpha, r)`.* ds2's 6480 voxels are 45 layers × 16
angular × 9 radial. The dataset description reads "9 radial and 16 angular",
which invites `reshape(45, 9, 16)` — that is wrong, and it leaves `<z>` looking
plausible while silently corrupting `sigma_r`. The convention here was
determined empirically (the radial axis must fall off monotonically, the
angular axis must be flat) and is asserted in the tests. See `DEVLOG.md` §11.

*Pass `standardize=True` for real observables.* `E_tot` is tens of thousands of
MeV, `sparsity` is in [0, 1]. SWD and W1 are scale-dependent, so without
standardization they measure `E_tot` and nothing else — on the Geant4 null,
`swd` reads 1128 unstandardized and 0.022 standardized. `evaluate` warns when
feature scales span more than 100×.

### Validate on real data

```bash
python -m pinnde_eval.validate_calo     # needs both ds2 files in Tina/
```

`dataset_2_1` and `dataset_2_2` are independent Geant4 draws, so comparing them
is a true null — every metric must sit at its floor. That floor is what a
*perfect* generator looks like on real showers (N=8000, standardized):

```
mmd  -2.3e-05    swd 0.0222    auc 0.4971 +/- 0.0101
chi2  1.096      w1  0.0221    sep 0.0030    fpd 0.0002 +/- 1.1e-04
```

## Separation power and its floor

`separation_power` is the CaloChallenge's own per-observable statistic,
`S = ½ Σ (p−q)²/(p+q)` on normalized histograms — bounded in [0, 1], **0 =
identical, 1 = no overlap**. It is what CaloChallenge submissions report, so
including it makes results here directly comparable to the published table.

It also has a finite-N floor, and a sharp one:

```
S_floor  ≈  n_occupied_bins / (2N)
```

Measured on independent Geant4 draws, the floor falls as **1/N** (not the
1/√N that SWD and W1 follow) and is linear in the binning. At N=500 with 50
bins a *perfect* generator scores 0.044 — larger than many published
per-observable separation powers. So a bare S is not interpretable: quote it
against `n_bins/(2N)`, and never compare two values computed at different N or
binning. Derivation and the measured table are in `DEVLOG.md` §12.

## Reproducibility

Everything stochastic (classifier splits/inits, projections, bandwidth
subsampling, FPD/KPD batching) takes a `seed`. The default `seed=0` makes runs
deterministic.

## Validation

`validate_toys.py` demonstrates the module is calibrated on Gaussian-mixture
toys (run `python -m pinnde_eval.validate_toys`):

1. **Null test** — two independent draws from the same 3D GMM give AUC ≈ 0.5,
   reduced χ² ≈ 1, and Tier-3 distances ≈ 0.
2. **Sensitivity test** — perturbing the GMM (mean shift, variance scale, drop a
   component) makes every metric grow with the perturbation size.
3. **Speed check** — Tier-3 monitors run in well under a second on 5k samples.

Every threshold was calibrated against a *measured* noise floor rather than
chosen by preference. `DEVLOG.md` records each one, the number behind it, and
the baseline table to bisect against if a change moves a metric.

## Tests

```bash
pytest pinnde_eval/tests -q
```

## Files

- `tier1.py`, `tier2.py`, `tier3.py` — the metrics, grouped by tier.
- `evaluate.py` — the `evaluate()` entry point, `evaluate_by_condition()`,
  `report()`, `plot_histograms()`, and the shared standardization.
- `observables.py` — CaloChallenge shower observables, dataset geometries, and
  the HDF5 loader. This is the `features_fn` for real data.
- `validate_calo.py` — null test and separation-power floor law on real Geant4
  showers.
- `stability.py` — metric behaviour vs. sample size: null floor, spread,
  separation z, minimum resolvable N. Runnable as a module.
- `local.py` — local discrepancy maps: MMD witness, out-of-fold P(real|x),
  binned residuals.
- `data.py` — seeded toy generators: `gmm_params` / `sample_gmm` /
  `perturb_params` (Gaussian mixtures for validation) and `sample_shower_toy`
  (a calo-flavoured conditional toy: sampling fraction, depth, width vs.
  normalized log-energy).
- `validate_toys.py` — null / sensitivity / speed checks.
- `_utils.py` — array conversion, pair checking, seeding helpers.
- `DEVLOG.md` — calibration record: every threshold, the measurement behind it,
  and the baseline numbers used to catch regressions.
- `tests/` — pytest edge-case tests.
