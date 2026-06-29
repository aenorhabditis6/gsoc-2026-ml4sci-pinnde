"""Lightweight edge-case tests for the flow_matching package.

Run from the project folder:  pytest flow_matching/tests -q
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flow_matching import VelocityField, fm_loss, sample, train_flow_matching


# ---------- model + objective shapes ----------

def test_velocity_output_shape():
    model = VelocityField(dim=3, seed=0)
    x = torch.randn(16, 3)
    t = torch.rand(16)
    assert model(x, t).shape == (16, 3)


def test_fm_loss_is_finite_scalar():
    model = VelocityField(dim=2, seed=0)
    g = torch.Generator().manual_seed(0)
    loss = fm_loss(model, torch.randn(32, 2), generator=g)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_sample_shape_and_finite():
    model = VelocityField(dim=2, seed=0)
    g = sample(model, n=50, dim=2, steps=10, seed=0)
    assert g.shape == (50, 2) and torch.isfinite(g).all()


def test_sampling_is_reproducible():
    model = VelocityField(dim=2, seed=0)
    a = sample(model, n=40, dim=2, steps=10, seed=1)
    b = sample(model, n=40, dim=2, steps=10, seed=1)
    assert torch.allclose(a, b)


def test_euler_and_heun_both_run():
    model = VelocityField(dim=2, seed=0)
    e = sample(model, n=30, dim=2, steps=10, method="euler", seed=0)
    h = sample(model, n=30, dim=2, steps=10, method="heun", seed=0)
    assert torch.isfinite(e).all() and torch.isfinite(h).all()


# ---------- it actually learns ----------

def test_training_reduces_loss():
    rng = np.random.default_rng(0)
    data = torch.tensor(rng.normal(size=(4000, 2)) * 0.5 + 2.0, dtype=torch.float32)
    _, history = train_flow_matching(
        data, dim=2, n_steps=600, monitor_every=50, seed=0,
    )
    losses = [loss for _, loss, _, _ in history]
    assert np.mean(losses[-3:]) < np.mean(losses[:3])


def test_training_moves_samples_toward_data():
    # A shifted Gaussian: a trained model should match it far better than the
    # untrained (random-init) model does.
    rng = np.random.default_rng(1)
    real = torch.tensor(rng.normal(size=(3000, 2)) * 0.3 + 3.0, dtype=torch.float32)

    from pinnde_eval import swd
    untrained = VelocityField(dim=2, seed=0)
    swd_before = swd(real, sample(untrained, 3000, 2, steps=50, seed=0))

    model, _ = train_flow_matching(real, dim=2, n_steps=1500, seed=0)
    swd_after = swd(real, sample(model, 3000, 2, steps=50, seed=0))

    assert swd_after < 0.3 * swd_before
