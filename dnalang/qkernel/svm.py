"""C-SVM on a precomputed kernel, solved with SMO (first-order working-set selection,
the LIBSVM update and clipping rules; Fan, Chen & Lin 2005). Multiclass is one-vs-one
with majority vote, ties broken by summed decision values.

Probability calibration (`predict_proba`) fits cross-validated multiclass Platt /
temperature-and-bias scaling on out-of-fold training decision scores so training-kernel
memorisation (K_ii = 1) cannot over-sharpen predicted probabilities.

Written here so the experiment needs nothing beyond numpy/scipy; it is checked against
scipy's constrained optimiser in tests/test_qkernel.py.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


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
            a[i] += delta
            a[j] += delta
            if diff > 0:
                if a[j] < 0:
                    a[j] = 0
                    a[i] = diff
            else:
                if a[i] < 0:
                    a[i] = 0
                    a[j] = -diff
            if diff > 0:
                if a[i] > C:
                    a[i] = C
                    a[j] = C - diff
            else:
                if a[j] > C:
                    a[j] = C
                    a[i] = C + diff
        else:
            quad = max(Q[i, i] + Q[j, j] - 2 * Q[i, j], tau)
            delta = (G[i] - G[j]) / quad
            s = ai + aj
            a[i] -= delta
            a[j] += delta
            if s > C:
                if a[i] > C:
                    a[i] = C
                    a[j] = s - C
            else:
                if a[j] < 0:
                    a[j] = 0
                    a[i] = s
            if s > C:
                if a[j] > C:
                    a[j] = C
                    a[i] = s - C
            else:
                if a[i] < 0:
                    a[i] = 0
                    a[j] = s
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
    def __init__(self, C: float = 1.0, *, calibrate: bool = False, cal_folds: int = 3, seed: int = 0):
        self.C = C
        self.calibrate = calibrate
        self.cal_folds = cal_folds
        self.seed = seed
        self.cal_params_: Optional[Tuple[float, np.ndarray]] = None

    def _fit_ovo(self, K: np.ndarray, y: np.ndarray):
        models: List[Tuple[int, int, np.ndarray, np.ndarray, float]] = []
        for ci, cj in combinations(range(len(self.classes_)), 2):
            idx = np.where((y == self.classes_[ci]) | (y == self.classes_[cj]))[0]
            yy = np.where(y[idx] == self.classes_[ci], 1.0, -1.0)
            a, rho = smo(K[np.ix_(idx, idx)], yy, self.C)
            models.append((ci, cj, idx, a * yy, rho))
        return models

    @staticmethod
    def _scores_from_models(Kt: np.ndarray, models, n_classes: int):
        votes = np.zeros((Kt.shape[0], n_classes))
        score = np.zeros((Kt.shape[0], n_classes))
        for ci, cj, idx, ay, rho in models:
            f = Kt[:, idx] @ ay - rho
            votes[:, ci] += f > 0
            votes[:, cj] += f <= 0
            score[:, ci] += f
            score[:, cj] -= f
        return votes, score

    def fit(self, K: np.ndarray, y: np.ndarray) -> "PrecomputedSVC":
        self.classes_ = np.unique(y)
        self._K_train = K
        self._y_train = y
        self.models_ = self._fit_ovo(K, y)
        self.cal_params_ = None
        if self.calibrate:
            self._fit_calibration()
        return self

    def _fit_calibration(self) -> None:
        K, y = self._K_train, self._y_train
        m = len(self.classes_)
        _, y_idx = np.unique(y, return_inverse=True)
        min_class = min(int(np.sum(y == c)) for c in self.classes_)
        k_folds = min(max(self.cal_folds, 2), min_class) if min_class >= 2 else 1

        if k_folds >= 2:
            rng = np.random.default_rng(self.seed)
            fold = np.empty(len(y), dtype=int)
            for c in self.classes_:
                idx = rng.permutation(np.where(y == c)[0])
                fold[idx] = np.arange(len(idx)) % k_folds
            oof_scores = np.zeros((len(y), m), dtype=np.float64)
            for f in range(k_folds):
                tr = np.where(fold != f)[0]
                va = np.where(fold == f)[0]
                if len(np.unique(y[tr])) < m:
                    _, oof_scores[va] = self._scores_from_models(K[va], self.models_, m)
                else:
                    sub_models = self._fit_ovo(K[np.ix_(tr, tr)], y[tr])
                    _, oof_scores[va] = self._scores_from_models(K[np.ix_(va, tr)], sub_models, m)
        else:
            _, oof_scores = self.decision_votes(K)

        scale_divisor = max(float(m - 1), 1.0)
        S = oof_scores / scale_divisor
        Y = np.eye(m, dtype=np.float64)[y_idx]

        def nll(theta: np.ndarray):
            log_temp = float(np.clip(theta[0], -4.0, 4.0))
            temp = float(np.exp(log_temp))
            bias = theta[1:] - np.mean(theta[1:])
            logits = temp * S + bias[None, :]
            lse = logsumexp(logits, axis=1, keepdims=True)
            P = np.exp(logits - lse)
            loss = float(np.sum(lse[:, 0] - np.sum(logits * Y, axis=1))) + 1e-3 * (
                log_temp * log_temp + float(np.sum(bias * bias))
            )
            diff = P - Y
            g_log_temp = temp * float(np.sum(diff * S)) + 2e-3 * log_temp
            g_bias = np.sum(diff, axis=0) + 2e-3 * bias
            return loss, np.r_[g_log_temp, g_bias]

        res = minimize(nll, np.zeros(m + 1), jac=True, method="L-BFGS-B")
        temp = float(np.exp(np.clip(res.x[0], -4.0, 4.0)))
        bias = res.x[1:] - np.mean(res.x[1:])
        self.cal_params_ = (temp / scale_divisor, bias)

    def decision_votes(self, Kt: np.ndarray):
        return self._scores_from_models(Kt, self.models_, len(self.classes_))

    def predict_proba(self, Kt: np.ndarray) -> np.ndarray:
        if self.cal_params_ is None:
            self._fit_calibration()
        assert self.cal_params_ is not None
        alpha, bias = self.cal_params_
        _, score = self.decision_votes(Kt)
        logits = alpha * score + bias[None, :]
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    def predict(self, Kt: np.ndarray) -> np.ndarray:
        votes, score = self.decision_votes(Kt)
        return self.classes_[np.argmax(votes + 1e-6 * np.tanh(score), axis=1)]
