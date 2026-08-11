# flow_matching

Conditional flow-matching generator for the PINNDE flow track — a
simulation-free alternative to the score-based track's O(N·M) Monte-Carlo score
estimate.

**Idea.** Learn a velocity field `v_theta(x, t)` by regressing it onto a
straight-line interpolant. For noise `x0 ~ N(0,I)` and data `x1`:

```
x_t   = (1 - t) * x0 + t * x1          # straight-line (OT) path
target = x1 - x0                        # constant velocity along the path
loss   = || v_theta(x_t, t) - target ||^2
```

The target is exact (no kernel sum, no ODE solve in training), so it is cheap
and low-variance. Generate by integrating `dx/dt = v_theta(x, t)` from `t=0`
(noise) to `t=1` (data).

**Time convention:** `t=0` = noise, `t=1` = data, sample `0 → 1` (the
flow-matching / Lipman convention). The diffusion / score track labels it the
other way (`t=0` data → `t=1` noise); the two are related by `t → 1−t`.

## Files
- `model.py` — `VelocityField`: MLP with Fourier-feature time **and condition**
  embeddings + GELU.
- `core.py` — `fm_loss` (the objective) and `sample` (Euler/Heun ODE solver),
  both condition-aware.
- `train.py` — `train_flow_matching`: Adam + cosine LR decay; optional Tier-3
  (`pinnde_eval`) monitoring during training.
- `demo.py` — trains on a 2-D GMM toy and reports `pinnde_eval` metrics.
- `demo_conditional.py` — trains `p(observables | energy)` on the calo-flavoured
  shower toy and scores it pooled, per energy bin, and at fixed conditions.
- `demo_calo.py` — the same on **real CaloChallenge ds2 showers**: trains on
  `dataset_2_1`, scores against the held-out `dataset_2_2` and the measured
  Geant4 null floor.
- `_utils.py` — seeding and array helpers.
- `tests/` — edge-case + "it actually learns" tests.

## Run

```bash
cd Tina
python -m flow_matching.demo              # train on a 2-D GMM, score vs truth
python -m flow_matching.demo_conditional  # train p(observables | energy), toy
python -m flow_matching.demo_calo         # the same on real ds2 showers
pytest flow_matching/tests -q
```

```python
from flow_matching import train_flow_matching, sample
model, history = train_flow_matching(data, dim=2)   # data: (N, d) tensor
gen = sample(model, n=5000, dim=2)                  # integrate noise -> data
```

## Conditional generation

The actual calorimeter target is the *conditional* density `p(shower | E_inc)`,
not a marginal. Pass a per-sample condition and the field becomes
`v_theta(x, t, c)`; the condition enters as its raw value plus a low-frequency
Fourier embedding.

```python
model, history = train_flow_matching(x, dim=3, cond=c)   # c: (N, 1)

gen = sample(model, n=5000, dim=3, cond=0.9)      # everyone at c = 0.9
gen = sample(model, n=5000, dim=3, cond=c_eval)   # matched per-row conditions
```

`demo_conditional.py` scores the result three ways, strictest last:

1. **pooled** over all energies — the number a marginal model could also fake;
2. **per energy bin** via `evaluate_by_condition` — a bad bin cannot hide inside
   a good average;
3. **at fixed unseen conditions** `c* ∈ {0.1, 0.5, 0.9}` against fresh truth
   draws at exactly those `c*` — the interpolation test a lookup table fails.

## Results

**2-D GMM, k=6, 4000 steps (unconditional).** Generated-vs-truth lands near the
statistical null floor: **SWD 0.066** (floor ≈0.049 from two true draws),
**MMD ~1e-4**, **AUC 0.54 ± 0.003** (0.5 = perfect). Cosine LR decay was needed —
without it the samples drifted off the data manifold late in training
(final SWD 0.166 → 0.066 with decay).

**Shower toy, 40k events, 6000 steps (conditional).** Pooled: **SWD 0.0056**,
**MMD −8.4e-05**, **W1 0.0058**, FPD/KPD consistent with zero. **AUC 0.565 ±
0.007** — the transport distances sit at the floor but the classifier still
finds a residual imperfection, and χ² localizes it to the skewed
sampling-fraction feature (2.4 vs ~1.1–1.4 on the others): the flow slightly
under-models the gamma tail. This is kept as an honest example of *why* AUC
stays in the suite even when the distances look perfect.

Per energy bin the pooled number hides a trend — AUC rises 0.512 / 0.551 /
0.561 / 0.617 across the four condition quartiles, so the high-energy slice is
the weakest (narrow distributions make the same absolute error more visible).
At fixed `c*`, per-feature means track truth to ≲0.5%.

**Real CaloChallenge ds2, 100k events, 12000 steps (conditional, ~5 min CPU).**
Trained on `dataset_2_1`, scored against the held-out `dataset_2_2`. Pooled, it
is **indistinguishable from Geant4**: AUC 0.5048 against a measured null floor
of 0.4971, χ² 1.118 vs 1.096, SWD 0.0168 vs a floor of 0.0222.

Per energy bin tells a completely different story:

| energy bin | swd | auc | sep |
|---|---|---|---|
| E 0–25% | 0.0474 | **0.787 ± 0.010** | 0.01266 |
| E 25–50% | 0.0465 | 0.573 ± 0.016 | 0.01186 |
| E 50–75% | 0.0329 | 0.498 ± 0.015 | 0.01159 |
| E 75–100% | 0.0338 | 0.527 ± 0.018 | 0.00910 |

A quarter of the data is trivially separable while the pooled AUC sits at the
floor. Low-energy showers are sparse and nearly discrete, which a continuous
flow in observable space models badly — and note this is the *opposite* trend
to the toy, where AUC rose with energy. The toy did not predict where the real
model fails.

The local maps only see it once you condition: pooled they give out-of-fold AUC
0.504 and max |r| 2.5 (below the |r| > 3 threshold); on the worst slice, 0.787
and max |r| 3.8, with 32% of generated samples confidently fake. Find the bad
slice with `evaluate_by_condition`, *then* localize inside it.

Full numbers and calibration: `../pinnde_eval/DEVLOG.md` §10 (toy) and §13 (real).

## Next
The open modelling gap is **low-energy showers on real data** (AUC 0.787 in the
lowest quartile). Directions worth trying, cheapest first: a logit transform on
the bounded observables so `sparsity` cannot be pushed against its ceiling;
weighting the training loss toward low `c`; a finer condition embedding where
the density changes fastest. Then the architecture ablations — adaptive
collocation, richer Fourier features, a physics-informed continuity-equation
residual — each scored with `pinnde_eval` against the null floor.
