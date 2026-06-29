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
- `model.py` — `VelocityField`: MLP with a Fourier-feature time embedding + GELU.
- `core.py` — `fm_loss` (the objective) and `sample` (Euler/Heun ODE solver).
- `train.py` — `train_flow_matching`: Adam + cosine LR decay; optional Tier-3
  (`pinnde_eval`) monitoring during training.
- `demo.py` — trains on a 2-D GMM toy and reports `pinnde_eval` metrics.
- `tests/` — edge-case + "it actually learns" tests.

## Run

```bash
cd Tina
python -m flow_matching.demo        # train on a GMM, score vs truth
pytest flow_matching/tests -q
```

```python
from flow_matching import train_flow_matching, sample
model, history = train_flow_matching(data, dim=2)   # data: (N, d) tensor
gen = sample(model, n=5000, dim=2)                  # integrate noise -> data
```

## Toy result (2-D GMM, k=6, 4000 steps)
Generated-vs-truth lands near the statistical null floor: **SWD 0.066** (floor
≈0.049 from two true draws), **MMD ~1e-4**, **AUC 0.54 ± 0.003** (0.5 = perfect).
Cosine LR decay was needed — without it the samples drifted off the data
manifold late in training (final SWD 0.166 → 0.066 with decay).

## Next
Ablate the architecture upgrades one at a time, scoring each with `pinnde_eval`:
adaptive collocation (non-uniform `t` sampling), richer Fourier features, then a
physics-informed continuity-equation residual. Real calorimeter data is deferred
(plugs in via `evaluate(..., features_fn=...)`).
