"""Batched exact state-vector simulation of a feature-map template.

Qubit q is bit q of the basis index (Qiskit's little-endian convention), so
amplitudes agree with `qiskit.quantum_info.Statevector` of `Template.instantiate`.
"""
from __future__ import annotations

import numpy as np

from .feature_maps import Template

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _rot(g: str, th: np.ndarray) -> np.ndarray:
    c, s = np.cos(th / 2), np.sin(th / 2)
    M = np.zeros((th.size, 2, 2), dtype=complex)
    if g == "rz":
        M[:, 0, 0] = np.exp(-0.5j * th); M[:, 1, 1] = np.exp(0.5j * th)
    elif g == "ry":
        M[:, 0, 0] = c; M[:, 0, 1] = -s; M[:, 1, 0] = s; M[:, 1, 1] = c
    elif g == "rx":
        M[:, 0, 0] = c; M[:, 0, 1] = -1j * s; M[:, 1, 0] = -1j * s; M[:, 1, 1] = c
    return M


def simulate(t: Template, X: np.ndarray) -> np.ndarray:
    """Return states of shape (B, 2**n) for the batch X of shape (B, n)."""
    X = np.asarray(X, dtype=float)
    B, n = X.shape[0], t.n_qubits
    psi = np.zeros((B,) + (2,) * n, dtype=complex)
    psi[(slice(None),) + (0,) * n] = 1.0
    def ax(q: int) -> int:
        return 1 + (n - 1 - q)  # tensor axis of qubit q
    for g, qs, p in t.ops:
        if g == "h":
            psi = np.moveaxis(np.tensordot(psi, _H, axes=([ax(qs[0])], [1])), -1, ax(qs[0]))
        elif g in ("rx", "ry", "rz"):
            M = _rot(g, p(X))
            a = ax(qs[0])
            v = np.moveaxis(psi, a, -1)
            v = np.einsum("b...j,bij->b...i", v, M)
            psi = np.moveaxis(v, -1, a)
        elif g == "cz":
            idx = [slice(None)] * (n + 1)
            idx[ax(qs[0])] = 1; idx[ax(qs[1])] = 1
            psi[tuple(idx)] *= -1
        elif g == "cx":
            c, tq = ax(qs[0]), ax(qs[1])
            idx0 = [slice(None)] * (n + 1); idx0[c] = 1; idx0[tq] = 0
            idx1 = [slice(None)] * (n + 1); idx1[c] = 1; idx1[tq] = 1
            a0 = psi[tuple(idx0)].copy()
            psi[tuple(idx0)] = psi[tuple(idx1)]
            psi[tuple(idx1)] = a0
        else:
            raise ValueError(f"simulator has no rule for {g}")
    # flatten so that index = sum_q bit_q 2^q (axis order is q_{n-1} ... q_0)
    return psi.reshape(B, 2 ** n)
