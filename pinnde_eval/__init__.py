"""pinnde_eval -- shared quantitative evaluation for the PINNDE tracks.

Three tiers of metrics behind one entry point:

* Tier 1 (CaloChallenge-standard): classifier two-sample AUC, per-feature
  histogram reduced chi-squared.
* Tier 2 (headline distances): FPD, KPD, per-feature Wasserstein-1.
* Tier 3 (cheap training-loop monitors): RBF-MMD, sliced Wasserstein.

    from pinnde_eval import evaluate, evaluate_by_condition, report
    results = evaluate(real, gen, tier="full")
    report(results)

Tier 3 metrics are also exposed directly (``mmd``, ``swd``) for use inside a
training loop.

Two study/diagnostic layers sit on top of the metrics:

* ``stability`` -- null floor, spread, and resolvability of every metric as a
  function of the sample size N (``python -m pinnde_eval.stability``).
* ``local`` -- local discrepancy maps (``mmd_witness``,
  ``classifier_discrepancy``, ``binned_residual_map``) that localize *where*
  two distributions disagree instead of returning one global number.
"""

from .evaluate import evaluate, evaluate_by_condition, report, plot_histograms
from .local import binned_residual_map, classifier_discrepancy, mmd_witness
from .stability import min_resolvable_n, separation_z, stability_study
from .tier1 import classifier_two_sample_test, histogram_chi2
from .tier2 import fpd, kpd, wasserstein_per_feature
from .tier3 import mmd, swd, median_bandwidth

__all__ = [
    "evaluate",
    "evaluate_by_condition",
    "report",
    "plot_histograms",
    "classifier_two_sample_test",
    "histogram_chi2",
    "fpd",
    "kpd",
    "wasserstein_per_feature",
    "mmd",
    "swd",
    "median_bandwidth",
    "stability_study",
    "separation_z",
    "min_resolvable_n",
    "mmd_witness",
    "classifier_discrepancy",
    "binned_residual_map",
]
