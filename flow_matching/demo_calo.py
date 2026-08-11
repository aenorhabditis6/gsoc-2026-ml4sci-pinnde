"""Conditional flow matching on real CaloChallenge showers.

The toy demo (``demo_conditional``) learns p(3 toy observables | c) on a
synthetic calo-flavoured distribution. This does the same job on real Geant4
data: it learns **p(shower observables | E_inc)** from CaloChallenge ds2 and
scores the result against the measured Geant4-vs-Geant4 null floor.

Train and evaluation data are kept strictly apart: the model trains on
``dataset_2_1.hdf5`` and is scored against ``dataset_2_2.hdf5``, which it never
sees. The floor row it is compared to comes from ds2_1-vs-ds2_2 (DEVLOG
section 11) and is what a *perfect* generator scores -- the point of the
stability study was that comparing to zero is meaningless.

Feature space is the 7 core observables. ``E_tot`` spans three decades
(1 GeV to 1 TeV incident), so it is modelled in log space; everything is then
z-scored with the training statistics and inverted after sampling.

Run from the ``Tina`` folder with both ds2 files present:

    python -m flow_matching.demo_calo
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from pinnde_eval import (binned_residual_map, classifier_discrepancy, evaluate,
                         evaluate_by_condition, report, separation_power)
from pinnde_eval.observables import observables_from_file

from .core import sample
from .train import train_flow_matching

# Measured Geant4-vs-Geant4 floor at N=8000, standardized (DEVLOG section 11).
# Any model number is only meaningful next to this row.
NULL_FLOOR = {"auc": 0.4971, "chi2_mean": 1.096, "swd": 0.0222,
              "w1_mean": 0.0221, "sep_mean": 0.0030}

BIN_EDGES = (0.25, 0.5, 0.75)
BIN_NAMES = ("E 0-25%", "E 25-50%", "E 50-75%", "E 75-100%")
LOG_FEATURES = ("E_tot",)


def energy_bin_labels(c):
    return np.array(BIN_NAMES)[np.digitize(np.asarray(c).ravel(), BIN_EDGES)]


class FeatureTransform:
    """Log the wide-dynamic-range columns, then z-score. Invertible.

    Fit on the training set only, so the evaluation set gets no say in the
    normalization -- the same discipline as ``evaluate(standardize=True)``.
    """

    def __init__(self, feats, names):
        self.names = list(names)
        self.log_idx = [i for i, n in enumerate(names) if n in LOG_FEATURES]
        x = self._to_log(np.asarray(feats, dtype=np.float64))
        self.mean = x.mean(axis=0)
        self.std = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)

    def _to_log(self, x):
        x = x.copy()
        for i in self.log_idx:
            x[:, i] = np.log(np.clip(x[:, i], 1e-8, None))
        return x

    def forward(self, feats):
        return (self._to_log(np.asarray(feats, dtype=np.float64))
                - self.mean) / self.std

    def inverse(self, z):
        x = np.asarray(z, dtype=np.float64) * self.std + self.mean
        for i in self.log_idx:
            x[:, i] = np.exp(np.clip(x[:, i], -50, 50))
        return x


def normalized_log_energy(e_inc, lo=None, hi=None):
    """Map incident energy to c in [0, 1] via log, the ds2 sampling variable."""
    log_e = np.log(np.asarray(e_inc, dtype=np.float64).ravel())
    lo = np.min(log_e) if lo is None else lo
    hi = np.max(log_e) if hi is None else hi
    return ((log_e - lo) / (hi - lo)).reshape(-1, 1), lo, hi


def compare_to_floor(res, title):
    """Print model numbers beside the measured perfect-generator floor.

    Caveat worth keeping in view: the floor was measured between two
    *independent* Geant4 draws, whose incident energies were sampled
    separately. The model generates at the evaluation set's own energies, so
    it does not pay that extra sampling variance. The floor is therefore
    mildly generous here, and a model landing slightly below it means
    "indistinguishable at this sample size", not "better than Geant4".
    """
    print(f"\n{title}")
    print(f"{'metric':>10} | {'model':>10} | {'null floor':>10} | {'vs floor':>9}")
    got = {"auc": res["auc"][0], "chi2_mean": res["chi2_mean"],
           "swd": res["swd"], "w1_mean": res["w1_mean"],
           "sep_mean": res["sep_mean"]}
    for key, floor in NULL_FLOOR.items():
        val = got[key]
        if key == "auc":
            note = f"{val - 0.5:+.4f}"
        else:
            note = f"{val / floor:.1f}x"
        print(f"{key:>10} | {val:10.4f} | {floor:10.4f} | {note:>9}")


def main(n_train=100000, n_eval=8000, n_steps=12000, seed=0, plot_path=None,
         data_dir=None):
    data_dir = data_dir or os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(data_dir, "dataset_2_1.hdf5")
    eval_path = os.path.join(data_dir, "dataset_2_2.hdf5")
    for p in (train_path, eval_path):
        if not os.path.exists(p):
            raise SystemExit(
                f"missing {p}\nDownload ds2 from "
                f"https://zenodo.org/records/6366271 into the Tina/ folder.")

    print(f"extracting observables: {n_train} train / {n_eval} eval showers")
    x_train, names, e_train = observables_from_file(train_path, n_train)
    x_eval, _, e_eval = observables_from_file(eval_path, n_eval)
    print(f"  {len(names)} observables: {', '.join(names)}")

    tf = FeatureTransform(x_train, names)
    z_train = tf.forward(x_train)
    c_train, lo, hi = normalized_log_energy(e_train)
    c_eval, _, _ = normalized_log_energy(e_eval, lo, hi)

    print(f"\ntraining conditional flow matching: d={len(names)}, cond=1, "
          f"n={n_train}, steps={n_steps}")
    model, history = train_flow_matching(
        torch.tensor(z_train, dtype=torch.float32), dim=len(names),
        cond=torch.tensor(c_train, dtype=torch.float32),
        n_steps=n_steps, seed=seed, monitor_every=max(1, n_steps // 8),
        monitor_real=torch.tensor(tf.forward(x_eval), dtype=torch.float32),
        monitor_cond=torch.tensor(c_eval, dtype=torch.float32),
    )
    print("\nstep |      loss |        mmd |        swd")
    for step, loss, m, s in history:
        print(f"{step:5d} | {loss:9.4f} | {m:10.4e} | {s:10.4f}")

    # Generate at the evaluation set's own conditions, then invert to physical
    # units so every number below is in the observables' real scale.
    z_gen = sample(model, n_eval, len(names),
                   cond=torch.tensor(c_eval, dtype=torch.float32),
                   steps=50, seed=seed)
    gen = tf.inverse(z_gen.detach().cpu().numpy())

    print()
    res = evaluate(x_eval, gen, tier="full", seed=seed, standardize=True)
    report(res, title="conditional FM vs Geant4 (ds2_2, pooled over energy)")
    compare_to_floor(res, "against the measured perfect-generator floor")

    print("\nper observable:")
    print(f"{'observable':>12} | {'chi2':>7} | {'sep':>8} | {'floor sep':>9}")
    for i, nm in enumerate(names):
        print(f"{nm:>12} | {res['chi2_per_feature'][i]:7.2f} | "
              f"{res['sep_per_feature'][i]:8.5f} | {NULL_FLOOR['sep_mean']:9.4f}")

    # Per energy bin -- the pooled number hides a bad slice.
    labels = energy_bin_labels(c_eval)
    by_bin = evaluate_by_condition(x_eval, gen, labels, tier="full", seed=seed,
                                   standardize=True)
    print(f"\n{'energy bin':>12} | {'n':>5} | {'swd':>7} | {'auc':>15} | "
          f"{'sep':>8}")
    for nm in BIN_NAMES:
        if nm not in by_bin:
            continue
        r = by_bin[nm]
        print(f"{nm:>12} | {r['n_real']:5d} | {r['swd']:7.4f} | "
              f"{r['auc'][0]:5.3f} +/- {r['auc'][1]:5.3f} | {r['sep_mean']:8.5f}")

    # Does the generator respect the physical range of each observable?
    # sparsity and f_samp are bounded; an unconstrained flow can leave the
    # support, which is a concrete defect rather than a distributional one.
    print(f"\n{'observable':>12} | {'Geant4 range':>22} | {'gen out of range':>16}")
    for i, nm in enumerate(names):
        lo, hi = x_eval[:, i].min(), x_eval[:, i].max()
        out = ((gen[:, i] < lo) | (gen[:, i] > hi)).mean()
        print(f"{nm:>12} | {lo:10.4g} .. {hi:9.4g} | {out:15.2%}")

    # Where does it fail? Run the local maps on the *worst energy bin*, not
    # pooled. The failure here is conditional: pooled AUC sits at the floor
    # while one quartile is far from it, so pooled maps see nothing.
    worst_bin = max((nm for nm in BIN_NAMES if nm in by_bin),
                    key=lambda nm: by_bin[nm]["auc"][0])
    mask = labels == worst_bin
    print(f"\nlocal maps on the worst slice ({worst_bin}, "
          f"auc {by_bin[worst_bin]['auc'][0]:.3f}, n={int(mask.sum())}):")

    for tag, r, g in (("pooled", x_eval, gen),
                      (worst_bin, x_eval[mask], gen[mask])):
        r_std, g_std = _standardized_pair(r, g)
        _, p_gen, oof_auc = classifier_discrepancy(r_std, g_std, seed=seed)
        worst_feats = tuple(np.argsort(separation_power(r, g))[-2:][::-1])
        resid, _ = binned_residual_map(r, g, features=worst_feats, bins=15)
        print(f"  {tag:>12}: oof AUC {oof_auc:.4f} | "
              f"confidently-fake gen {(p_gen < 0.1).mean():5.1%} | "
              f"max |r| {np.abs(resid).max():4.1f} on "
              f"({names[worst_feats[0]]}, {names[worst_feats[1]]})")

    if plot_path:
        _plot(x_eval, c_eval, gen, names, plot_path)
    return model, res


def _standardized_pair(real, gen):
    mean, std = real.mean(axis=0), real.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (real - mean) / std, (gen - mean) / std


def _plot(x_eval, c_eval, gen, names, path):
    """Conditional means vs energy (top) and marginals (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.asarray(c_eval).ravel()
    edges = np.linspace(0, 1, 11)
    mids = 0.5 * (edges[:-1] + edges[1:])
    show = [0, 2, 3, 4, 5, 6]      # skip f_samp, nearly flat in c

    fig, axes = plt.subplots(2, len(show), figsize=(3.1 * len(show), 6.6))
    for col, j in enumerate(show):
        ax = axes[0, col]
        for arr, label, color in ((x_eval, "Geant4", "tab:blue"),
                                  (gen, "flow matching", "tab:orange")):
            m = np.array([arr[(c >= a) & (c < b), j].mean()
                          for a, b in zip(edges, edges[1:])])
            s = np.array([arr[(c >= a) & (c < b), j].std()
                          for a, b in zip(edges, edges[1:])])
            ax.plot(mids, m, marker="o", ms=3, color=color, label=label)
            ax.fill_between(mids, m - s, m + s, color=color, alpha=0.2)
        ax.set_title(names[j], fontsize=10)
        ax.set_xlabel("normalized log $E_{inc}$", fontsize=8)
        if col == 0:
            ax.set_ylabel("conditional mean ± std")
            ax.legend(fontsize=8)
        if names[j] == "E_tot":
            ax.set_yscale("log")

        ax = axes[1, col]
        lo = min(x_eval[:, j].min(), gen[:, j].min())
        hi = max(x_eval[:, j].max(), gen[:, j].max())
        b = np.linspace(lo, hi, 60)
        ax.hist(x_eval[:, j], bins=b, density=True, histtype="step",
                color="tab:blue", label="Geant4")
        ax.hist(gen[:, j], bins=b, density=True, histtype="step",
                color="tab:orange", label="flow matching")
        ax.set_xlabel(names[j], fontsize=8)
        if names[j] == "E_tot":
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel("density")
    fig.suptitle("Conditional flow matching on real CaloChallenge ds2 showers")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    main(plot_path=os.path.join(out, "fm_calo.png") if os.path.isdir(out) else None)
