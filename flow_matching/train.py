"""Training loop for the flow-matching velocity field.

Plain Adam on the conditional flow-matching loss. Optionally logs the Tier-3
``pinnde_eval`` monitors (MMD/SWD vs. a held-out real sample) every
``monitor_every`` steps, so convergence can be watched with the same numbers
used to compare the two tracks.
"""

import torch

from ._utils import seed_all
from .core import fm_loss, sample
from .model import VelocityField


def train_flow_matching(data, dim, n_steps=4000, batch_size=256, lr=1e-3,
                        hidden=128, depth=3, time_dim=64, device="cpu", seed=0,
                        monitor_every=0, monitor_real=None, monitor_n=2000,
                        sample_steps=50):
    """Train a ``VelocityField`` on ``data`` (N, d). Returns (model, history).

    ``history`` is a list of (step, loss, mmd, swd) rows; the mmd/swd columns are
    filled only when ``monitor_every > 0`` and ``monitor_real`` is given.
    """
    seed_all(seed)
    model = VelocityField(dim, hidden=hidden, depth=depth, time_dim=time_dim,
                          seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay to ~0: stabilizes the late training so the final samples
    # don't drift back off the data manifold.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    data = data.to(device)
    g = torch.Generator(device=device).manual_seed(seed)

    history = []
    for step in range(n_steps):
        idx = torch.randint(0, len(data), (batch_size,), generator=g, device=device)
        loss = fm_loss(model, data[idx], generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if monitor_every and (step % monitor_every == 0 or step == n_steps - 1):
            row = [step, float(loss.detach()), None, None]
            if monitor_real is not None:
                from pinnde_eval import mmd, swd
                gen = sample(model, monitor_n, dim, steps=sample_steps,
                             device=device, seed=seed)
                real = monitor_real[:monitor_n]
                row[2] = mmd(real, gen, seed=seed)
                row[3] = swd(real, gen, seed=seed)
            history.append(tuple(row))

    return model, history
