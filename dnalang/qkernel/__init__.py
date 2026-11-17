"""Quantum fidelity-kernel SVMs on genotype features, benchmarked against classical models.

See docs/QKERNEL_GENOMICS.md. Run: ``python -m dnalang.qkernel --help``.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .baselines import KNN, GradientBoosting, LogisticRegression, RandomForest
    from .feature_maps import FAMILIES, feature_map, fidelity_circuit
    from .genotypes import (
        Preprocessor,
        genomic_relationship_matrix,
        kinship_clusters,
        ld_prune_r2,
        relatedness_aware_folds,
    )
    from .kernels import fidelity_kernel, psd_repair, sample_kernel
    from .statevector import simulate
    from .svm import PrecomputedSVC

_EXPORTS: dict[str, str] = {
    "KNN": ".baselines",
    "GradientBoosting": ".baselines",
    "LogisticRegression": ".baselines",
    "RandomForest": ".baselines",
    "FAMILIES": ".feature_maps",
    "feature_map": ".feature_maps",
    "fidelity_circuit": ".feature_maps",
    "Preprocessor": ".genotypes",
    "genomic_relationship_matrix": ".genotypes",
    "kinship_clusters": ".genotypes",
    "ld_prune_r2": ".genotypes",
    "relatedness_aware_folds": ".genotypes",
    "fidelity_kernel": ".kernels",
    "psd_repair": ".kernels",
    "sample_kernel": ".kernels",
    "simulate": ".statevector",
    "PrecomputedSVC": ".svm",
}

__all__ = [
    "FAMILIES",
    "KNN",
    "GradientBoosting",
    "LogisticRegression",
    "PrecomputedSVC",
    "Preprocessor",
    "RandomForest",
    "feature_map",
    "fidelity_circuit",
    "fidelity_kernel",
    "genomic_relationship_matrix",
    "kinship_clusters",
    "ld_prune_r2",
    "psd_repair",
    "relatedness_aware_folds",
    "sample_kernel",
    "simulate",
]


def __getattr__(name: str) -> Any:
    mod_name = _EXPORTS.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(mod_name, __name__)
    val = getattr(mod, name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
