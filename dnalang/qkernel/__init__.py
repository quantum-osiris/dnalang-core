"""Pre-registered quantum-kernel vs classical benchmark on real biallelic genotypes.

See docs/QKERNEL_GENOMICS.md. Run: ``python -m dnalang.qkernel --help``.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .feature_maps import FAMILIES, feature_map, fidelity_circuit
    from .kernels import fidelity_kernel, psd_repair, sample_kernel
    from .statevector import simulate
    from .svm import PrecomputedSVC

_EXPORTS: dict[str, str] = {
    "FAMILIES": ".feature_maps",
    "feature_map": ".feature_maps",
    "fidelity_circuit": ".feature_maps",
    "fidelity_kernel": ".kernels",
    "psd_repair": ".kernels",
    "sample_kernel": ".kernels",
    "simulate": ".statevector",
    "PrecomputedSVC": ".svm",
}

__all__ = [
    "FAMILIES",
    "feature_map",
    "fidelity_circuit",
    "fidelity_kernel",
    "psd_repair",
    "sample_kernel",
    "simulate",
    "PrecomputedSVC",
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
