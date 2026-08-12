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

Per energy bin, the first configuration told a completely different story —
and fixing it is the most instructive result in the project:

| energy bin | auc, `h128 d3 12k` | auc, `h384 d5 30k` |
|---|---|---|
| E 0–25% | **0.787 ± 0.010** | 0.509 ± 0.020 |
| E 25–50% | 0.573 ± 0.016 | 0.510 ± 0.011 |
| E 50–75% | 0.498 ± 0.015 | 0.487 ± 0.020 |
| E 75–100% | 0.527 ± 0.018 | 0.496 ± 0.009 |

The baseline had a quarter of the data trivially separable while its pooled
AUC sat at the floor. Note the trend is the *opposite* of the toy, where AUC
rose with energy — the toy did not predict where the real model fails.

**The diagnosis mattered more than the fix.** Separation power in the bad bin
was barely worse than elsewhere, which proved the marginals were fine and the
*joint* was wrong: every one of the 7 observables was individually at chance
(AUC 0.48–0.51), a linear classifier on all 7 was at chance (0.477), and only
a nonlinear one separated them (0.788). At low energy `E_tot` and `sparsity`
are nearly deterministic (Geant4 correlation −0.951); the flow produced −0.818,
a fatter cloud around a thin manifold.

Two plausible fixes did nothing. `sparsity` really is discrete — exactly
`1 − k/6480` — so it gets dequantized during training and floored back when
sampling, which is correct physics and moved AUC by 0.002. More ODE integration
steps moved it by 0.005 over a 16× increase. **Capacity was the bottleneck**,
and extra ODE steps only started paying off once the field was sharp enough to
resolve (0.544 → 0.507 at 200 steps).

The local maps only see the failure once you condition: pooled they gave
out-of-fold AUC 0.504 and max |r| 2.5 (below the |r| > 3 threshold); on the bad
slice, 0.787 and max |r| 3.8 with 32% of generated samples confidently fake.
Find the bad slice with `evaluate_by_condition`, *then* localize inside it.

Full numbers and the full diagnostic chain: `../pinnde_eval/DEVLOG.md` §10
(toy), §13 (real data), §14 (diagnosis and fix).

## Next
The 7-observable conditional model now sits at the Geant4 null floor in every
energy bin, so the next step is **more of the shower**: the 180-column
per-layer observables (`per_layer_observables`), then the voxel space itself.
Both raise the same manifold question §14 answered here at d=7 — expect
capacity, not sampling resolution, to be the binding constraint again.

Architecture ablations still open: adaptive collocation (non-uniform `t`
sampling), richer Fourier features, and a physics-informed
continuity-equation residual, each scored against the null floor.
