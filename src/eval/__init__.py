"""Evaluation: metrics, cross-generator sweep, robustness sweep, spectral analysis."""

from .metrics import (  # noqa: F401
    accuracy,
    accuracy_at_fpr,
    auc,
    average_precision,
    best_threshold_accuracy,
    binary_metrics,
    confusion_counts,
)
from .spectrum import (  # noqa: F401
    descriptor_vector,
    hf_energy_ratio,
    log_power_spectrum,
    pairwise_spectral_distance,
    radial_power_profile,
    spectral_distance,
    to_luma,
)
