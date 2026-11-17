"""C-SVM on a precomputed kernel, solved with SMO (first-order working-set selection,
the LIBSVM update and clipping rules; Fan, Chen & Lin 2005). Multiclass is one-vs-one
with majority vote, ties broken by summed decision values.

Written here so the experiment needs nothing beyond numpy/scipy; it is checked against
scipy's constrained optimiser in tests/test_qkernel.py.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import numpy as np


def smo(K: np.ndarray, y: np.ndarray, C: float, tol: float = 1e-3, max_iter: int = 100000):
    """Binary dual: min 1/2 a'Qa - e'a, 0 <= a <= C, y'a = 0, Q = yy' * K. Returns (alpha, rho)."""
    n = len(y)
    y = y.astype(float)
    Q = (y[:, None] * y[None, :]) * K
    a = np.zeros(n)
    G = -np.ones(n)
    tau = 1e-12
    for _ in range(max_iter):
        up = ((y > 0) & (a < C)) | ((y < 0) & (a > 0))
        low = ((y > 0) & (a > 0)) | ((y < 0) & (a < C))
        v = -y * G
        if not up.any() or not low.any():
            break
        i = np.where(up)[0][np.argmax(v[up])]
        j = np.where(low)[0][np.argmin(v[low])]
        if v[i] - v[j] < tol:
            break
        ai, aj = a[i], a[j]
        if y[i] != y[j]:
            quad = max(Q[i, i] + Q[j, j] + 2 * Q[i, j], tau)
            delta = (-G[i] - G[j]) / quad
            diff = ai - aj
            a[i] += delta; a[j] += delta
            if diff > 0:
                if a[j] < 0: a[j] = 0; a[i] = diff
            else:
                if a[i] < 0: a[i] = 0; a[j] = -diff
            if diff > 0:
                if a[i] > C: a[i] = C; a[j] = C - diff
            else:
                if a[j] > C: a[j] = C; a[i] = C + diff
        else:
            quad = max(Q[i, i] + Q[j, j] - 2 * Q[i, j], tau)
            delta = (G[i] - G[j]) / quad
            s = ai + aj
            a[i] -= delta; a[j] += delta
            if s > C:
                if a[i] > C: a[i] = C; a[j] = s - C
            else:
                if a[j] < 0: a[j] = 0; a[i] = s
            if s > C:
                if a[j] > C: a[j] = C; a[i] = s - C
            else:
                if a[i] < 0: a[i] = 0; a[j] = s
        G += Q[:, i] * (a[i] - ai) + Q[:, j] * (a[j] - aj)
    yG = y * G
    free = (a > 1e-9) & (a < C - 1e-9)
    if free.any():
        rho = yG[free].mean()
    else:
        up = ((y > 0) & (a < C)) | ((y < 0) & (a > 0))
        low = ((y > 0) & (a > 0)) | ((y < 0) & (a < C))
        ub = yG[up].min() if up.any() else np.inf     # LIBSVM bounds on rho
        lb = yG[low].max() if low.any() else -np.inf
        rho = (ub + lb) / 2 if np.isfinite(ub) and np.isfinite(lb) else (ub if np.isfinite(ub) else lb)
    return a, float(rho)


def dual_objective(K, y, a):
    y = y.astype(float)
    return 0.5 * (a * y) @ K @ (a * y) - a.sum()


class PrecomputedSVC:
    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, K: np.ndarray, y: np.ndarray) -> "PrecomputedSVC":
        self.classes_ = np.unique(y)
        self.models_: List[Tuple[int, int, np.ndarray, np.ndarray, float]] = []
        for ci, cj in combinations(range(len(self.classes_)), 2):
            idx = np.where((y == self.classes_[ci]) | (y == self.classes_[cj]))[0]
            yy = np.where(y[idx] == self.classes_[ci], 1.0, -1.0)
            a, rho = smo(K[np.ix_(idx, idx)], yy, self.C)
            self.models_.append((ci, cj, idx, a * yy, rho))
        return self

    def decision_votes(self, Kt: np.ndarray):
        m = len(self.classes_)
        votes = np.zeros((Kt.shape[0], m)); score = np.zeros((Kt.shape[0], m))
        for ci, cj, idx, ay, rho in self.models_:
            f = Kt[:, idx] @ ay - rho
            votes[:, ci] += f > 0; votes[:, cj] += f <= 0
            score[:, ci] += f; score[:, cj] -= f
        return votes, score

    def predict(self, Kt: np.ndarray) -> np.ndarray:
        votes, score = self.decision_votes(Kt)
        return self.classes_[np.argmax(votes + 1e-6 * np.tanh(score), axis=1)]
