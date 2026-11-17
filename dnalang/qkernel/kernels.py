"""Kernel matrices (quantum fidelity and classical), finite-shot sampling, PSD repair, diagnostics."""
from __future__ import annotations

import hashlib
from typing import Dict, Optional

import numpy as np

from .feature_maps import feature_map
from .statevector import simulate


# --- quantum ------------------------------------------------------------------
def fidelity_kernel(SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
    """K[a, b] = |<phi(a)|phi(b)>|^2 from state matrices (rows are states)."""
    return np.abs(SA.conj() @ SB.T) ** 2


def quantum_kernels(family: str, A: np.ndarray, B: np.ndarray, reps: int = 2):
    """Exact (noiseless, infinite-shot) train/train and test/train fidelity kernels."""
    t = feature_map(family, A.shape[1], reps)
    SA, SB = simulate(t, A), simulate(t, B)
    return fidelity_kernel(SA, SA), fidelity_kernel(SB, SA)


def sample_kernel(F: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool,
                  depolarizing: float = 0.0, n_qubits: Optional[int] = None) -> np.ndarray:
    """Finite-shot estimate: each entry is k/shots with k ~ Binomial(shots, P(0^n)).

    This is the ideal-sampling model of running one compute-uncompute circuit per entry.
    `depolarizing` applies a *global* depolarizing model P = (1-p) F + p / 2^n; it is a
    model, not a description of any device.
    """
    P = F
    if depolarizing:
        P = (1 - depolarizing) * F + depolarizing / 2 ** n_qubits
    P = np.clip(P, 0.0, 1.0)
    if not symmetric:
        return rng.binomial(shots, P) / shots
    iu = np.triu_indices(F.shape[0])            # diagonal is measured too, like on hardware
    K = np.zeros_like(F)
    K[iu] = rng.binomial(shots, P[iu]) / shots
    return K + np.triu(K, 1).T


def psd_repair(K: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues of a symmetric matrix to zero (same rule as Qiskit's enforce_psd)."""
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.clip(w, 0, None)) @ V.T


# --- classical ----------------------------------------------------------------
def linear_kernel(A, B):
    return B @ A.T if B is not A else A @ A.T


def rbf_kernel(A, B, gamma):
    d = (B ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :] - 2 * B @ A.T
    return np.exp(-gamma * np.clip(d, 0, None))


def poly_kernel(A, B, gamma, degree=3):
    return (gamma * (B @ A.T) + 1.0) ** degree


def random_fourier_kernel(A, B, dim: int, scale: float, seed: int):
    """Matched classical control: an explicit random nonlinear map of dimension `dim` (= 2^n)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=scale, size=(A.shape[1], dim))
    b = rng.uniform(0, 2 * np.pi, size=dim)
    fa = np.sqrt(2.0 / dim) * np.cos(A @ W + b)
    fb = np.sqrt(2.0 / dim) * np.cos(B @ W + b)
    return fb @ fa.T


# --- diagnostics --------------------------------------------------------------
def diagnostics(K: np.ndarray, y: np.ndarray, K_exact: Optional[np.ndarray] = None) -> Dict[str, float]:
    n = K.shape[0]
    off = K[~np.eye(n, dtype=bool)]
    w = np.linalg.eigvalsh((K + K.T) / 2)
    wp = np.clip(w, 0, None)
    p = wp / wp.sum() if wp.sum() > 0 else wp
    eff_rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())) if wp.sum() > 0 else 0.0
    k = len(np.unique(y))                       # multiclass ideal kernel: 1 same, -1/(k-1) different
    Y = np.where(y[:, None] == y[None, :], 1.0, -1.0 / max(k - 1, 1))
    kta = float((K * Y).sum() / (np.linalg.norm(K) * np.linalg.norm(Y)))
    same = (y[:, None] == y[None, :]) & ~np.eye(n, dtype=bool)
    out = {
        "offdiag_mean": float(off.mean()), "offdiag_std": float(off.std()),
        "within_class_mean": float(K[same].mean()),
        "between_class_mean": float(K[(y[:, None] != y[None, :])].mean()),
        "min_eigenvalue": float(w.min()), "negative_eigenvalues": int((w < -1e-12).sum()),
        "effective_rank": eff_rank, "kernel_target_alignment": kta,
    }
    if K_exact is not None:
        out["frobenius_error_vs_exact"] = float(np.linalg.norm(K - K_exact) / np.linalg.norm(K_exact))
    return out


def matrix_sha256(K: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(K, dtype=np.float64).tobytes()).hexdigest()
