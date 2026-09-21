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

Added 2026-09-14 to 09-16:

- `cluster/` — `check_node.sh`, `get_data.sh`, `setup.sh`, `requirements.txt`
  (section 2)
- `pinnde_eval/calochallenge.py` — the challenge's own 362 features for ds2,
  computed with their code, plus `tests/test_calochallenge.py`
- `pinnde_eval/floors.py` — the pairing rules behind every floor (disjoint
  pairs inside a file, across files, or matched in incident energy), plus
  `tests/test_floors.py`
- `pinnde_eval/validate_floors.py`, `validate_official.py`,
  `validate_matched.py` — the floors themselves over 10 disjoint repeats, and
  how large a difference the null test can actually see
- `flow_matching/demo_calo.py` — `--features {core,per-layer,official}`,
  `--device`, and `NULL_FLOORS` replaced by the energy-matched floors
- `pinnde_eval/DEVLOG.md` — §17 classical tests and Sinkhorn, §18 the 362
  features, §19 checking the claims before making them

Kept local on purpose: `slides_postmidterm.md` (an early draft, superseded by
the meeting documents) and `sijil_data/` (gitignored). Not committed either:
`calochallenge_code/` (their code, no licence — downloaded on demand),
`calochallenge_cache/` and `floor_results/` (large, regenerable).

Tests: 101 passing with the two ds2 files present; without them expect a few
skips (the real-data tests skip themselves).
`python -m pinnde_eval.validate_classical` passes all four checks.

---

## 2. Setting up on the cluster

Explored on 2026-09-13. Machines are `<name>.hep.fsu.edu`. There is **no job
scheduler**: jobs run directly on a machine. The connection from mainland China
is unstable (needs VPN). Harrison is sending documentation on using the cluster.

| Machine | Role | What matters |
|---|---|---|
| `dagda` | login machine, file server for `/home`, Kerberos server | Ubuntu 24.04, 32 CPUs, 31 GB RAM, no GPU. Has tmux. SSH keys work. Python 3.12 cannot create venvs (python3.12-venv not installed). |
| `macha` | CPU compute machine | AlmaLinux 10.1, 64 CPUs, 125 GB RAM. Python 3.12 venvs work. **No tmux.** Its only NVIDIA card (GeForce GT 730) cannot run CUDA. Password login only. |
| `credne` | **GPU machine**, shared with Sijil | Ubuntu 24.04, 32 CPUs, 251 GB RAM, **NVIDIA GeForce RTX 5090 (32 GB)**, driver 580.173.02 (CUDA 13.0). Has tmux. Python 3.12 (Miniforge) venvs work. Local `/scratch`, 7.3 TB. Password login only. |
| `vilya`, `gandalf`, `frodo` | not needed so far | password login only |

- **GPU work runs on credne** (`credne.hep.fsu.edu`; it is not in dagda's host
  list). The GPU is shared with Sijil, so check `nvidia-smi` before long runs.
  The standard torch 2.13.0 on PyPI is built for CUDA 13.0, which credne's
  driver supports. macha's GT 730 is too old for current CUDA and PyTorch.
- **Home folder** is dagda's disk (802 GB free), shared with macha over
  Kerberos-protected NFS. On macha it is readable only with a Kerberos ticket,
  which lasts 10 hours from login and renews without a password (`kinit -R`)
  for up to 2 days. A longer job must renew it or it loses the home folder.
- **Run long jobs inside tmux.** credne has it. macha does not: start tmux on
  dagda and `ssh macha` inside it. The dagda-to-macha link stays inside FSU, so
  it survives your own connection dropping.
- **The GitHub repo is private**, so `git clone` on the cluster asks for a
  login. `~/GSOC_2026_PINNDE` on the cluster was made from a git bundle of the
  laptop repo (`git bundle create repo.bundle main`, copy it over, `git clone
  repo.bundle`, `git checkout main`), with `origin` set back to GitHub.

Scripts in `cluster/`:

- `check_node.sh` reports OS, Python (and whether venvs work), home folder,
  disks, Kerberos ticket, GPU and internet access. Changes nothing.
- `get_data.sh` downloads both ds2 files from Zenodo, resuming partial files,
  and checks size and MD5 against the Zenodo record. Run it on dagda, whose own
  disk holds `/home`. Only one run at a time; a second run waits.
- `setup.sh`: a venv with the laptop's exact package versions
  (`cluster/requirements.txt`, with the CPU build of torch where there is no
  NVIDIA driver), tests, GPU check, data check, `validate_classical`. Safe to
  rerun. Each machine needs its own venv because the home folder is shared:
  macha uses the default `.venv`; on credne pass `VENV=` a local path. Its
  header has the tmux and ssh steps.

**Data is not in git** (1.36 GB each, over GitHub's limit). `cluster/get_data.sh`
downloads it from https://zenodo.org/records/6366271. The laptop copies match
the Zenodo MD5s. Zenodo was down for hours on 2026-09-13 (HTTP 504), so expect
to rerun it.

**Always run with** `OPENBLAS_NUM_THREADS=1`. Without it the sklearn classifier
busy-waits and a 9-second evaluation looks like a 30-minute hang.

**Verify the setup** (`setup.sh` runs both):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest pinnde_eval/tests flow_matching/tests -q
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pinnde_eval.validate_classical
```

**Status: working on both machines** (cluster time: macha 2026-09-14, credne
2026-09-15). `setup.sh` ran end to end on each, in 64 and 66 minutes.

- **macha:** 91 tests passed (55 s), both data files match the Zenodo MD5s, and
  `validate_classical` printed output identical to the laptop run. Venv in
  `Tina/.venv`, 46,182 files, 1.9 GB.
- **credne:** torch 2.13.0+cu130 on the RTX 5090, a 200-step training and 1,000
  samples on the GPU, 91 tests (33 s), data MD5s and `validate_classical` fine.
  Venv in `~/venvs/credne`, 47,162 files, 5.7 GB.

Most of each hour was pip writing those small files onto the network home
folder, which is the price of a shared home. `/scratch` on credne is
admin-only, so a local venv needs the admin to create `/scratch/<username>`.

**GPU note:** `python -m flow_matching.demo_calo --device cuda` trains and
samples on a GPU; the metrics stay on CPU. Tested end to end on the laptop's
Apple GPU (`device="mps"`), never on CUDA. A GPU run draws different random
numbers from a CPU run with the same seed, so it will not reproduce the CPU
numbers in DEVLOG exactly.

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

The calibration figure in `meeting_2026-08-31.md` is not evidence of a shift
either. Its red histogram (file 1 vs file 2, KS on the 7 observables, 30
disjoint 1,000-vs-1,000 splits) has 35 of 210 p-values below 0.1 against 21
expected, and the document put that down to the energy difference. The figure
uses the first 30,000 showers of each file; the next two blocks of 30,000 give
12 and 20. Inside a single 1,000-vs-1,000 split the shift is only 0.2 to 0.4
standard errors, too small to cause it.

**Settled on 2026-09-16, in the space that matters.**
`python -m pinnde_eval.validate_official` scores 10 disjoint pairs of 8,000
showers in the CaloChallenge's own 362 features, drawn both inside one file and
across the two:

| | same file | different files |
|---|---|---|
| classifier AUC | 0.4993 ± 0.0042 | 0.5006 ± 0.0059 |
| separation power | 0.00279 ± 0.00009 | 0.00280 ± 0.00010 |
| SWD | 0.0211 ± 0.0021 | 0.0217 ± 0.0035 |

The gap in AUC is +0.0012, half a standard error: a classifier reading all 362
features cannot tell which file a shower came from. The shift in the incident
energy marginal is real but changes nothing we measure.

**Conclusion: do not correct dataset 1.** A 2σ difference between two
independent random draws happens about 1 time in 20, and in the full feature
space it leaves no trace at all. Say this at the next meeting, since the meeting
agreed to correct dataset 1.

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

**Done (2026-09-16).** `pinnde_eval/calochallenge.py` computes the challenge's
**362 features** for ds2 with their own code, downloaded on demand (their
repository has no licence, so nothing from it is committed), pinned to commit
`3073d13` and checked by MD5. `tests/test_calochallenge.py` compares our column
assembly against their own `prepare_high_data_for_classifier` on real showers,
and our layer energies and sparsities against `observables.py`.
`flow_matching/demo_calo.py --features official` trains in that space, and
`python -m pinnde_eval.validate_official` measures the floor there over 10
disjoint repeats — both from pairs inside one file (sampling noise alone) and
across the two files (what a model is scored against), which also answers
item B in the space that matters.

**First model run there (2026-09-16, 96 seconds on credne's GPU).** The
configuration that works at d=7 fails comprehensively at d=362: AUC 0.9861
against the matched floor of 0.5003, separation power 75× the floor, combined
KS exactly 0, and up to 37% of generated deep-layer energies outside the Geant4
range. The surprise is **which** bin is worst: the highest energy (AUC 0.996),
where there are no empty layers at all. So zero-inflation is not the whole
story. DEVLOG §20.

**Located (2026-09-17), after proving the harness first** (DEVLOG §21):

- *The harness is sound.* `demo_calo --null` puts real Geant4 showers at matched
  energies in the model's place and lands on the floor at d=7 and d=362. It also
  exposed three bugs — KS rejecting values moved by 1e-17, the range check
  counting rounding as impossible, and the quantizer returning sparsity values
  one bit off the data's (which made KS reject even a perfect generator). All
  three are fixed and pinned by tests; only the model runs' KS and out-of-range
  numbers were affected.
- *Capacity helps but does not reach the floor.* 1024×8 for 100k steps takes
  AUC from 0.986 to 0.900; the top energy bin barely moves (0.996 → 0.992).
- *The classifier is mostly detecting correlations.* Real showers with every
  marginal kept exact but columns shuffled among showers of the same energy
  score AUC 0.978; with only whole layers shuffled, 0.872, and worst at high
  energy (0.905). Rebuilding E_tot from the layers changes nothing, so it is the
  soft structure of shower development, not the energy-sum identity.
- *So the main failure is how layers relate to each other,* not the empty-layer
  point masses. **The next architecture needs to represent cross-layer
  structure** (for example a transformer over the 45 layers, or generating
  voxels and computing features afterwards); the hurdle model is secondary.
- For the next runs: `--save-samples PATH` then
  `python -m pinnde_eval.diagnose_samples PATH` breaks a run down by column
  family and depth without retraining.
- *Diagnosis of the big run* (DEVLOG §21): at low energy the failure is sparsity
  and the back layers (the point masses); at high energy every family fails,
  worst in the middle layers. The **energy response is wrong**: at high energy
  the model's sampling-fraction spread is 0.044 against Geant4's 0.018 (2.4×
  too wide), and at low energy it misses the rise to 0.83. Likely cause: energies
  are modelled as absolute log10 values spanning three decades.
  `--relative-energy` models log10(E / E_inc) and takes E_inc from the
  condition. Its first run destabilized (loss spike, 2 of 8000 showers NaN).
  With `--clip-grad 1.0` on both, same seed: absolute energies AUC **0.892**
  (best so far), relative 0.943. Relative energies give the best energy
  response but make the layer-energy family much worse (0.771 → 0.961).
- **Use `--relative-energy total`** (only E_tot relative). AUC 0.8915, the same
  as the absolute baseline within noise, and it moves the totals family to the
  floor (0.691 → 0.503) with the high-energy energy resolution from 2.5× to 1.3×
  too wide. Nothing else changes.
- **Seed noise is ±0.016 in AUC** (the same absolute setup scored 0.8922 and
  0.9078 with two seeds). Quote it beside any comparison.
- *Worst family then:* η/φ widths (0.962), where **30% of generated deep-layer
  widths are negative**, which is impossible. `--positive-sqrt` models the
  widths and radial centres as sqrt(x): AUC 0.914, chi² 45.5 → 23.9, and no
  negative widths at all.
- **These fixes do not stack.** sqrt + E_tot-relative scores 0.929 with chi²
  26.6, worse than sqrt alone; adding both on top of `--atom-snap` gives 0.907
  and chi² 12.5 against 0.890 and 9.4 for `--atom-snap` alone. Measure each
  combination rather than assuming improvements add.

**The empty-layer wall: mostly solved for the marginals (2026-09-17).**

- **Run it as `--atom-snap --relative-energy total`.** `--atom-snap` alone gives
  the best overall agreement (chi² 9.4 vs 11.7); adding `--relative-energy
  total` costs that and buys the energy response, putting the totals family on
  the floor (0.502 vs 0.697/0.612, floor 0.5003). For a calorimeter the sampling
  fraction has to be right, so the trade is worth it — but say which one a
  number came from. The other earlier fixes no longer help on top of it: sqrt
  widths cost chi² 9.4 → 12.6, because the atom was what they were really
  fixing (23% of width values are the empty-layer 0).
- Use **`--atom-snap`**. It is the largest single improvement measured so far:
  chi² **45.5 → 9.4** and separation power **0.130 → 0.0247**, both a factor of
  5, at an AUC that does not move (0.890 vs the baseline's 0.892). Reproduced at
  a second seed (0.898, chi² 10.5). Every family improves or holds, sparsity by
  10 noise widths and the deep layers by 7. DEVLOG §24.
- What it does: an empty layer is an exact value, not a small one (log10 E =
  −8, centres and widths 0, sparsity 1), and that is 17.8% of all (layer,
  shower) pairs and 51% of the last layer. A continuous flow puts zero
  probability on any exact value, so it produced an exactly-empty layer **0.0%**
  of the time. `--atom-snap` spreads the point mass over the empty gap above it
  during training and snaps everything in that gap back when sampling — the
  treatment the sparsity comb already had.
- **Emptiness belongs to the layer, not the column.** Doing this column by
  column made things worse (AUC 0.968): it generated layers with no energy but a
  non-zero centre, which no real shower contains. In real data, when a layer has
  no energy, every one of its columns is at its empty value 100% of the time. The
  rule is now applied per layer.
- Not fixed: the **joint**. The pooled AUC is unchanged, because the classifier
  keys on correlations between layers that none of this touches. Marginals 5×
  closer, correlations untouched — that is the summary, and the case for
  a layer-aware architecture next.
- A dead end worth not repeating: modelling layer energies as
  `sqrt(E_layer / E_inc)` (`--energy-sqrt`) scores **0.992**. It removes every
  impossible energy and is still much worse, because a near-miss next to the
  atom in sqrt space is 7 log units away in the space the metrics read. Match
  the scale the evaluation uses.

**Still open: the empty-layer wall in the joint distribution.**

- At 187 columns the flow failed (out-of-fold AUC 0.9963)
  because empty layers put a point mass at exactly 0. Up to 31.6% of generated
  radial widths came out negative; correlation between a layer's atom mass and
  its impossible-value rate was +0.989. Continuous flows cannot represent this at
  any capacity. Options: a two-part (hurdle) model — occupancy per layer, then
  shape only where lit — or Sijil's approach of a hit/no-hit BCE loss plus
  `clamp(expm1(x), min=0)` on output. See DEVLOG §15–16.
- In the official space it is worse: an empty layer gives log10 E_layer exactly
  −8, sparsity exactly 1, and its four centre and width columns exactly 0.
  Measured on 100,000 ds2 showers (2026-09-16), by incident-energy quartile:

  | sample | columns with >50% of their mass on one value | empty layers |
  |---|---|---|
  | all 100k | 20 of 362 (5.5%) | 18.1% |
  | lowest E quartile | **198 of 362 (54.7%)** | 48.4% |
  | second quartile | 88 of 362 (24.3%) | 21.1% |
  | third / top quartile | 0 of 362 | 2.8% / 0.0% |

  So the target is continuous at high energy and more than half point masses at
  low energy. A hurdle model needs per-layer occupancy conditioned on incident
  energy, running from about half the layers empty at the bottom of the range to
  none at the top.
- **What the occupancy part has to look like** (measured 2026-09-16 on 100,000
  showers). Incident energy explains most of whether a layer is lit but not all
  of it: across 20 energy bins x 45 layers, two thirds of the cells are settled
  (under 2% or over 98% lit), and knowing the energy drops the uncertainty from
  0.52 to 0.22 bits per layer. It is **not** a simple depth cut, though: only
  48.6% of showers have their lit layers forming a solid block from layer 0, and
  in the lowest energy quartile only 7.5% do, with on average 6.25 empty layers
  scattered inside the lit range. So the discrete part needs per-layer occupancy
  with correlations, not a single "how deep did it reach" variable.

### D. Outstanding from earlier meetings

- **Floors: rerun with ≥10 repeats.** In the meeting I said the floors were
  repeated 10–15 times. That is not right for any of them:
  the headline null floor (AUC 0.4971, SWD 0.0222, sep 0.0030) is a **single**
  N=8000 comparison, and AUC's ± comes from 5 classifier retrainings on that one
  pair; the separation-power floor used up to 4 repeats per N; the midterm
  stability study used 6; the Sinkhorn floor (0.4151 ± 0.077) was recorded as 3
  repeats, but no script in the repo produces it, so that cannot be checked.
  Rerun the headline floor and the Sinkhorn floor over ≥10 disjoint repeats and
  correct the numbers at the next meeting.
  **Done 2026-09-16, in all three spaces.** `validate_floors.py` (7 and 187),
  `validate_official.py` (362) and `validate_matched.py` (energy-matched) each
  measure every floor over 10 disjoint repeats, Sinkhorn included, and print a
  gap column for every metric so none can be cherry-picked. The old
  single-comparison values sit inside the new spreads. The Sinkhorn floor has a
  script for the first time: 0.394 ± 0.040 at d=7, against the unscriptable
  0.4151 ± 0.077. Per-repeat numbers are saved under `floor_results/`.
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
- ~~Write DEVLOG §17 for the classical tests and Sinkhorn.~~ Done 2026-09-16,
  together with §18 on the CaloChallenge's 362 features and the two floors
  measured there.

---

## 5. Corrections to what was said at the meeting

So they are not repeated:

| Said | Actually |
|---|---|
| floors repeated 10–15 times | none were: headline floor is 1 comparison (AUC ± from 5 retrainings), separation-power floor ≤4 repeats, stability study 6, Sinkhorn recorded as 3 (no script to check). All three spaces were remeasured over 10 disjoint repeats on 2026-09-16; the old values sit inside the new spreads |
| real dimension "around 136-ish" | **~360** features for the CaloChallenge classifier on ds2 |
| AD "can tell 60% at N=250", most powerful | that table combined p-values with Fisher, which over-rejects. With Bonferroni and 50 repeats: all three tests near chance at N=250; AD modestly ahead at every N (0.62 vs KS 0.50 at N=2000). A correction note is now in `meeting_2026-08-31.md` |
| shift in "total energy", "0.03 or 0.3" | shift in **incident** energy; mean log E_inc differs by 0.034 on 30k showers, **0.018 on all 100k** (2.00σ) |
| dataset 2 "a little bit bigger" | file 2's mean log E_inc is higher, but KS on E_inc gives p = 0.101 on the full data — consistent with noise |
| the 7 observables are "the compression the CaloChallenge people do" (agreed) | they are our own whole-shower summaries. The CaloChallenge's high-level features are per layer: 362 for ds2, counted by running their `evaluate.py` feature code on 200 showers |
| KS, CvM and AD "produce different and independent insights" | on the same data they almost always agree: rank correlation of their statistics 0.86–0.97 per observable over 100 disjoint 1,000-vs-1,000 null splits (CvM–AD 0.95–0.97) |
| Sinkhorn floor from "a thousand points", repeated "15 times" | no script in the repo produces 0.4151 ± 0.077, so neither the sample size nor the repeat count can be checked. The ≥10-repeat rerun should be a committed script |
| cluster: "I tried to set up everything. It's currently okay." | nothing was installed then, and there is no usable GPU on the cluster (section 2) |

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
- **Score a conditional model against the energy-matched floor.** It generates
  at the evaluation set's own energies, so it never pays the energy-draw
  variance that two independent Geant4 samples pay. The unmatched floor sits 3
  to 5 standard errors higher on chi², SWD, W1 and separation power (DEVLOG
  §19), which is exactly what made the model look *better than Geant4*.
  `NULL_FLOORS` in `demo_calo.py` now holds the matched floors for d=7 and
  d=362; AUC and Sinkhorn are unaffected by the matching. Against the matched
  floor the model is simply at it (−0.4 to −1.5σ on every metric).
- **An AUC of 0.5 is weak evidence on its own.** Measured on known shifts
  (DEVLOG §19): separation power notices a 0.09 shift in mean log E_inc, while
  the classifier needs about 0.5 — thirty times the difference between the two
  ds2 files. Quote AUC beside a metric that has been shown to move.
- **ds2 energies are pre-calibrated** (checked 2026-09-16 on all 200,000
  showers): 3.42% have E_tot > E_inc, and the excess is entirely a low-energy
  effect — 12.9% in the lowest E_inc quartile, 0.79% in the second, none in the
  top half. The median E_tot/E_inc is 0.78 in every quartile, so the deposits
  carry a sampling-fraction calibration of about 0.78 and the over-unity showers
  are ordinary fluctuations where a shower is smallest.

---

## 7. Where things are

| What | Where |
|---|---|
| Full calibration record, findings §1–16 | `pinnde_eval/DEVLOG.md` |
| Last meeting document (as presented) | `meeting_2026-08-31.md` |
| Classical tests | `pinnde_eval/classical.py` |
| Their validation | `python -m pinnde_eval.validate_classical` (loads all 200k showers, a few minutes) |
| Real-data null floor | `python -m pinnde_eval.validate_calo` |
| Model on real data | `python -m flow_matching.demo_calo` (~20 min CPU, ~2 min on credne's GPU); `--features per-layer` (187) or `--features official` (362); `--device cuda` |
| Observable extraction | `pinnde_eval/observables.py` (ours), `pinnde_eval/calochallenge.py` (the challenge's 362) |
| Floor in the official space | `python -m pinnde_eval.validate_official` |
| Sijil's files (gitignored, laptop only) | `sijil_data/` — `Dataset2.pt` is byte-identical to `dataset_2_1.hdf5`, not needed |
| Meeting minutes (shared) | `../meeting_minutes.md` — still ends at June 26 |
