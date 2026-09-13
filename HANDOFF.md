# Handoff — flow-matching track

For the next working session (likely on the cluster). Read this first.
Written 2026-09-13, after the meeting that covered the classical tests and Sinkhorn.

---

## 1. What is in the repo

Pushed on 2026-09-13 (the push promised at the meeting):

- `pinnde_eval/classical.py` — KS, Cramér–von Mises, Anderson–Darling, `combine_pvalues`
- `pinnde_eval/validate_classical.py` — calibration, energy difference between
  the two files, power, and p-value independence checks
- `pinnde_eval/tests/test_classical.py` — tests for the above
- `pinnde_eval/tier3.py` — `sinkhorn()`; `evaluate.py`, `__init__.py` — wired in
- `flow_matching/demo_calo.py` — E_tot histogram binning fix (the per-layer option was pushed earlier)
- `make_classical_figures.py`, `make_postmidterm_figures.py` and their figures
- `meeting_2026-08-17.md`, `meeting_2026-08-31.md` — as presented
- READMEs updated for the new module; this file

Kept local on purpose: `slides_postmidterm.md` (an early draft, superseded by
the meeting documents) and `sijil_data/` (gitignored).

Tests: 91 passing with the two ds2 files present; without them expect 90 passed,
1 skipped (the real-data test skips itself). `python -m pinnde_eval.validate_classical`
passes all four checks.

---

## 2. Setting up on the cluster

**Host:** `macha.hep.fsu.edu`. Connection from mainland China is
unstable (needs VPN). Harrison is sending documentation on using the cluster.
From the meeting: you log into the cluster first, then to the GPU node. There
is **one GPU on that node, shared with Sijil**, so coordinate before long runs.
Use `tmux` or `screen` so jobs survive a dropped connection.

```bash
git clone https://github.com/sijil-jose/GSOC_2026_PINNDE.git
cd GSOC_2026_PINNDE/Tina
python3.12 -m venv .venv          # laptop uses 3.12.13
.venv/bin/pip install torch "numpy<2" "scipy<1.14" scikit-learn matplotlib pytest h5py
# jetnet (only for FPD/KPD) -- no-compiler recipe from README.md:
.venv/bin/pip install --no-deps jetnet
.venv/bin/pip install numba energyflow tables pandas awkward coffea pyyaml requests tqdm
```

Laptop versions for reference: numpy 1.26.4, scipy 1.13.1, torch 2.13.0,
scikit-learn 1.9.0.

**Data is not in git** (1.36 GB each, over GitHub's limit). Download into `Tina/`:

```bash
wget -O dataset_2_1.hdf5 "https://zenodo.org/records/6366271/files/dataset_2_1.hdf5?download=1"
wget -O dataset_2_2.hdf5 "https://zenodo.org/records/6366271/files/dataset_2_2.hdf5?download=1"
```

(Check the file links on the Zenodo record page if those 404.)

**Always run with** `OPENBLAS_NUM_THREADS=1`. Without it the sklearn classifier
busy-waits and a 9-second evaluation looks like a 30-minute hang.

**Verify the setup:**

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest pinnde_eval/tests flow_matching/tests -q
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pinnde_eval.validate_classical
```

**GPU note:** `train_flow_matching` and `sample` take `device=`, but
`flow_matching/demo_calo.py`'s `main()` does not pass one, so it runs on CPU.
Add a `device` argument before using the GPU. Evaluation metrics (sklearn,
scipy) stay on CPU regardless.

---

## 3. Decisions from the meeting

1. **Keep both kinds of metric.** For direct comparison with published
   submissions we must use the CaloChallenge's own metrics *exactly* as they did,
   including their binning. The classical tests (KS, CvM, AD) are kept alongside
   for readers who are statisticians. Harrison's point: different tests test
   different things, so conclusions can differ between them.
2. **Move to the real dimensionality now.** Agreed that further analysis of the
   7-observable space has little value. The goal is generated showers that agree,
   not just 7 summaries.
3. **Energy shift between ds2_1 and ds2_2:** investigate. If it is real,
   **correct dataset 1** (not dataset 2) and send Sijil the corrected version.
   Harrison and Sijil both agreed.
4. Next meeting: next Monday, 10am Central, every week for the next month.

---

## 4. Action items, in order

### A. Get the cluster working (section 2)

### B. The energy shift — probably sampling noise, decide before correcting

The numbers shown at the meeting came from 30,000 showers per file. On **all
100,000 per file** the effect is smaller:

```
                          30k per file     100k per file
mean log E_inc diff          0.034             0.018
significance                 2.10σ             2.00σ
KS on log E_inc              p = 0.107         p = 0.101
```

On the full data only `z_mean` disagrees across files at p < 0.05 (p = 0.034).
The "matching the energy band fixes it" pattern is not clean either: `E_tot`
and `sparsity` improve, but two energy-independent controls get worse
(`r_mean` 0.35 → 0.08), probably because the matched band holds only ~10% of the
showers and is noisier.

**Conclusion so far:** a 2σ difference between two independent random draws
happens about 1 time in 20, so this is consistent with ordinary sampling noise.
It is not clearly a problem that needs correcting. Say this at the next meeting,
since the meeting agreed to correct dataset 1.

**If it is corrected anyway — do not "rescale" the energies.** In the meeting this was
described as rescaling, and Harrison agreed to rescaling. But the two files do
not differ by a calibration offset; they each drew a different random sample of
incident energies. Multiplying shower energies would change the physics in each
shower. The correct fix is to **reweight or resample dataset 1 so its log E_inc
distribution matches dataset 2's** (histogram-ratio weights, or rejection
resampling). Then:

- confirm with a KS test on log E_inc after reweighting
- rerun the null floors (`validate_calo`, `validate_classical`)
- send Sijil the **weights or kept indices**, not a new 1.4 GB file

If this changes the plan from what was agreed, say so at the next meeting.

### C. Move to the real feature space

- **~360 features** is what the CaloChallenge classifier uses for ds2. We have
  180 per-layer columns (energy, sparsity, radial centre, radial width per
  layer). The missing 180 are per-layer centres and widths in eta and phi, which
  need the detector maps in `binning.xml`.
- Get the official evaluation code from github.com/CaloChallenge/homepage:
  `evaluate.py`, `HighLevelFeatures.py`, `XMLHandler.py`, `binning_dataset_2.xml`.
  The copy in `Tina/sijil_data/` has `HighLevelFeatures.py` but not
  `XMLHandler.py` or the binning file, so it cannot run.
- The known blocker: at 187 columns the flow failed (out-of-fold AUC 0.9963)
  because empty layers put a point mass at exactly 0. Up to 31.6% of generated
  radial widths came out negative; correlation between a layer's atom mass and
  its impossible-value rate was +0.989. Continuous flows cannot represent this at
  any capacity. Options: a two-part (hurdle) model — occupancy per layer, then
  shape only where lit — or Sijil's approach of a hit/no-hit BCE loss plus
  `clamp(expm1(x), min=0)` on output. See DEVLOG §15–16.

### D. Outstanding from earlier meetings

- **Floors: rerun with ≥10 repeats.** In the meeting I said the floors were
  repeated 10–15 times. That is not right for any of them:
  the headline null floor (AUC 0.4971, SWD 0.0222, sep 0.0030) is a **single**
  N=8000 comparison, and AUC's ± comes from 5 classifier retrainings on that one
  pair; the separation-power floor used up to 4 repeats per N; the midterm
  stability study used 6; the Sinkhorn floor (0.4151 ± 0.077) used 3. Rerun the
  headline floor and the Sinkhorn floor over ≥10 disjoint repeats and correct
  the numbers at the next meeting.
- **Sinkhorn vs the paper's metrics** (Harrison's original question, still not
  answered): run Sinkhorn and separation power on the earlier failing model
  configuration (`hidden=128, depth=3, n_steps=12000`, AUC 0.787 in the lowest
  energy quartile) and see whether Sinkhorn catches what separation power missed.
- **Compare against CaloChallenge submissions.** ds2 submission samples are on
  Zenodo (record 15962050). Harrison noted two flow-matching submissions exist.
- **Sijil's loss function.** The code is not in the shared repo, only the
  description (`Sijil/Calorie_shower_autoencoder/Readme.md` §3). Ask him for the
  training script. His loss operates on voxels, so it does not apply to the
  7-observable model; it becomes relevant for item D.
- **Fixed-condition test** is in `flow_matching/demo_conditional.py` (the toy)
  but was never ported to `demo_calo.py`: generate at unseen fixed energies and
  compare against Geant4 at exactly those energies.
- **Write DEVLOG §17** for the classical tests and Sinkhorn. The figure script
  refers to §17 but DEVLOG stops at §16.

---

## 5. Corrections to what was said at the meeting

So they are not repeated:

| Said | Actually |
|---|---|
| floors repeated 10–15 times | none were: headline floor is 1 comparison (AUC ± from 5 retrainings), separation-power floor ≤4 repeats, stability study 6, Sinkhorn 3 |
| real dimension "around 136-ish" | **~360** features for the CaloChallenge classifier on ds2 |
| AD "can tell 60% at N=250", most powerful | that table combined p-values with Fisher, which over-rejects. With Bonferroni and 50 repeats: all three tests near chance at N=250; AD modestly ahead at every N (0.62 vs KS 0.50 at N=2000). A correction note is now in `meeting_2026-08-31.md` |
| shift in "total energy", "0.03 or 0.3" | shift in **incident** energy; mean log E_inc differs by 0.034 on 30k showers, **0.018 on all 100k** (2.00σ) |
| dataset 2 "a little bit bigger" | file 2's mean log E_inc is higher, but KS on E_inc gives p = 0.101 on the full data — consistent with noise |

---

## 6. Things that will bite you

- **Voxel order is `(layer, alpha, r)`** = `reshape(N, 45, 16, 9)`. The
  dataset description invites `(45, 9, 16)`, which silently corrupts every radial
  width.
- **Units:** Zenodo data and our code are in **MeV**. Sijil loads in **GeV**
  (divides by 1e3). He just found a MeV/GeV mismatch bug in his own loss. Only
  `E_tot` and thresholds are affected; the other observables are scale-free.
- **Pass `standardize=True`** to `evaluate` for observables in mixed units, or
  SWD, W1 and Sinkhorn measure only `E_tot`.
- **Null checks need disjoint splits.** Drawing splits independently reuses
  showers, correlates the p-values, and produces wrong conclusions. This
  happened once already.
- **Combine p-values with Bonferroni, not Fisher.** The 7 observables' p-values
  are correlated (E_tot/sparsity +0.94; 14 of 21 pairs outside the independence
  band), and Fisher rejected true nulls 14–29% of the time.
- **Anderson–Darling's p-value is clipped by scipy to [0.001, 0.25].** Fine for
  deciding, not for quoting a precise significance.
- **Sinkhorn caps both samples at 2000 points** (`max_points`), so its floor
  stops falling above N=2000.
- **The 7-observable result is not comparable to published numbers.** AUC
  0.4998 is on a 7-number summary with ~4 effective dimensions, three of which
  largely track the given incident energy. Published AUCs use ~360 features plus
  a CNN on voxels.
- **The two ds2 files drew their incident energies independently** (item B).
  The difference is ~2σ, consistent with noise, so file-vs-file floors are at
  most slightly generous.
- **ds2 energies look pre-calibrated:** 3.4% of showers have E_tot > E_inc.
  Not verified how.

---

## 7. Where things are

| What | Where |
|---|---|
| Full calibration record, findings §1–16 | `pinnde_eval/DEVLOG.md` |
| Last meeting document (as presented) | `meeting_2026-08-31.md` |
| Classical tests | `pinnde_eval/classical.py` |
| Their validation | `python -m pinnde_eval.validate_classical` (loads all 200k showers, a few minutes) |
| Real-data null floor | `python -m pinnde_eval.validate_calo` |
| Model on real data | `python -m flow_matching.demo_calo` (~20 min CPU), `--per-layer` for 187 columns |
| Observable extraction | `pinnde_eval/observables.py` |
| Sijil's files (gitignored, laptop only) | `sijil_data/` — `Dataset2.pt` is byte-identical to `dataset_2_1.hdf5`, not needed |
| Meeting minutes (shared) | `../meeting_minutes.md` — still ends at June 26 |
