"""Quantum fidelity-kernel SVMs on genotype features, benchmarked against classical models.

See docs/QKERNEL_GENOMICS.md. Run: ``python -m dnalang.qkernel --help``.
"""
from .feature_maps import FAMILIES, feature_map, fidelity_circuit
from .kernels import fidelity_kernel, psd_repair, sample_kernel
from .statevector import simulate
from .svm import PrecomputedSVC

__all__ = ["FAMILIES", "feature_map", "fidelity_circuit", "fidelity_kernel", "psd_repair",
           "sample_kernel", "simulate", "PrecomputedSVC"]
