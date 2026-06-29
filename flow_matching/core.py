"""The conditional flow-matching objective and the ODE sampler.

Time convention (Lipman / flow-matching, used everywhere in this package):
``t=0`` is NOISE, ``t=1`` is DATA, and we sample by integrating ``0 -> 1``.
The score-based / diffusion track uses the opposite labeling (``t=0`` data ->
``t=1`` noise, sampling ``1 -> 0``); the two are identical under ``t -> 1-t``.

Training (simulation-free, O(batch)):
  * draw noise x0 ~ N(0, I) and data x1,
  * interpolate on a straight line  x_t = (1-t) x0 + t x1,
  * the target velocity is the constant  u = x1 - x0,
  * regress  v_theta(x_t, t)  onto u with an MSE loss.

This replaces the score-based track's O(N*M) Monte-Carlo score estimate: the
target here is exact, not a noisy kernel sum.

Generation: integrate  dx/dt = v_theta(x, t)  from t=0 (noise) to t=1 (data)
with a fixed-step solver (Euler or Heun). Straight-line paths keep the number of
steps small, which is the point for *fast* simulation.
"""

import torch


def fm_loss(model, x1, generator=None):
    """Conditional flow-matching loss on a batch of data ``x1`` (B, d)."""
    x0 = torch.randn(x1.shape, generator=generator, device=x1.device)
    t = torch.rand(x1.shape[0], 1, generator=generator, device=x1.device)
    xt = (1.0 - t) * x0 + t * x1
    target = x1 - x0
    v = model(xt, t)
    return ((v - target) ** 2).mean()


@torch.no_grad()
def sample(model, n, dim, steps=50, method="heun", device="cpu", seed=0):
    """Generate ``n`` samples by integrating the learned velocity field.

    ``method`` is "euler" (1 eval/step) or "heun" (2 evals/step, 2nd order).
    Returns an (n, dim) tensor.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, dim, generator=g, device=device)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        dt = ts[i + 1] - ts[i]
        t0 = ts[i].expand(n, 1)
        v0 = model(x, t0)
        if method == "euler":
            x = x + dt * v0
        else:  # Heun (trapezoidal predictor-corrector)
            t1 = ts[i + 1].expand(n, 1)
            x_pred = x + dt * v0
            v1 = model(x_pred, t1)
            x = x + dt * 0.5 * (v0 + v1)
    return x
