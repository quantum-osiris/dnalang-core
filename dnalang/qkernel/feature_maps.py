"""Quantum feature maps as parametric templates that instantiate dnalang Circuit IR.

A template is a list of (gate, qubits, param) where `param` is None or a function
X -> angles over a batch X of shape (B, d). The same template drives

* `instantiate(x)`: a concrete `dnalang.qc_ir.Circuit` (exportable to Qiskit / OpenQASM 3)
* `dnalang.qkernel.statevector.simulate`: a batched exact state-vector simulation.

Four families, chosen so that each one is a control for the next:

* ``product``  - per-qubit rotations only; no entanglement (negative control for entanglement)
* ``nn``       - product layers plus a CZ nearest-neighbour chain
* ``zz``       - the Havlicek et al. (2019) ZZ map, linear pairs
* the matched classical control (random Fourier features) lives in `kernels.py`
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..qc_ir import Circuit, Op

Param = Optional[Callable[[np.ndarray], np.ndarray]]
SELF_INVERSE = {"h", "x", "y", "z", "cx", "cz", "swap"}
ROTATIONS = {"rx", "ry", "rz"}
FAMILIES = ("product", "nn", "zz")


@dataclass
class Template:
    family: str
    n_qubits: int
    reps: int
    ops: List[Tuple[str, Tuple[int, ...], Param]]

    def inverse(self) -> "Template":
        inv = []
        for g, q, p in reversed(self.ops):
            if g in SELF_INVERSE:
                inv.append((g, q, None))
            elif g in ROTATIONS:
                inv.append((g, q, (lambda X, f=p: -f(X))))
            else:
                raise ValueError(f"no inverse rule for {g}")
        return Template(self.family + "_dg", self.n_qubits, self.reps, inv)

    def instantiate(self, x: np.ndarray, name: Optional[str] = None) -> Circuit:
        X = np.asarray(x, dtype=float)[None, :]
        c = Circuit(n_qubits=self.n_qubits, n_cbits=0, name=name or f"fmap_{self.family}")
        for g, q, p in self.ops:
            c.ops.append(Op(g, q, () if p is None else (float(p(X)[0]),)))
        return c


def _lin(i: int, coeff: float = 1.0):
    return lambda X: coeff * X[:, i]


def _zz(i: int, j: int):
    return lambda X: 2.0 * (math.pi - X[:, i]) * (math.pi - X[:, j])


def feature_map(family: str, n_qubits: int, reps: int = 2) -> Template:
    """Build a feature-map template on `n_qubits` qubits for d = n_qubits features in [0, pi]."""
    if family not in FAMILIES:
        raise ValueError(f"unknown feature map {family!r}; choose from {FAMILIES}")
    n = n_qubits
    ops: List[Tuple[str, Tuple[int, ...], Param]] = []
    for _ in range(reps):
        for q in range(n):
            ops.append(("h", (q,), None))
            ops.append(("rz", (q,), _lin(q, 2.0)))
        if family == "product":
            for q in range(n):
                ops.append(("ry", (q,), _lin(q, 1.0)))
        elif family == "nn":
            for q in range(n - 1):
                ops.append(("cz", (q, q + 1), None))
            for q in range(n):
                ops.append(("ry", (q,), _lin(q, 1.0)))
        elif family == "zz":
            for q in range(n - 1):
                ops.append(("cx", (q, q + 1), None))
                ops.append(("rz", (q + 1,), _zz(q, q + 1)))
                ops.append(("cx", (q, q + 1), None))
    return Template(family, n, reps, ops)


def fidelity_circuit(t: Template, x: np.ndarray, y: np.ndarray) -> Circuit:
    """Compute-uncompute circuit U(y)^dagger U(x)|0>, measured; P(0...0) = |<phi(y)|phi(x)>|^2."""
    a = t.instantiate(x)
    b = t.inverse().instantiate(y)
    c = Circuit(n_qubits=t.n_qubits, n_cbits=t.n_qubits, name=f"fidelity_{t.family}")
    c.ops = a.ops + [Op("barrier", tuple(range(t.n_qubits)))] + b.ops
    c.ops += [Op("measure", (q,), cbit=q) for q in range(t.n_qubits)]
    return c
