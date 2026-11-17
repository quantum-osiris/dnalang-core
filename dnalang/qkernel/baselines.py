"""Non-kernel classical baselines: multinomial logistic regression, k-nearest neighbours,
Random Forest (vectorised CART classification trees), and multiclass Gradient Boosting
(Friedman log-loss Newton-step regression trees). Pure NumPy/SciPy; zero extra dependencies.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


class LogisticRegression:
    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        k, d = len(self.classes_), X.shape[1]
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        Xb = np.hstack([X, np.ones((len(X), 1))])

        def f(w):
            W = w.reshape(d + 1, k)
            Z = Xb @ W
            lse = logsumexp(Z, axis=1)
            P = np.exp(Z - lse[:, None])
            reg = 0.5 / self.C * (W[:-1] ** 2).sum()
            g = Xb.T @ (P - Y)
            g[:-1] += W[:-1] / self.C
            return (lse - (Z * Y).sum(1)).sum() + reg, g.ravel()

        r = minimize(f, np.zeros((d + 1) * k), jac=True, method="L-BFGS-B")
        self.W_ = r.x.reshape(d + 1, k)
        return self

    def predict_proba(self, X):
        Z = np.hstack([X, np.ones((len(X), 1))]) @ self.W_
        return np.exp(Z - logsumexp(Z, axis=1)[:, None])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KNN:
    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X, y):
        self.X_, self.y_, self.classes_ = X, y, np.unique(y)
        return self

    def _neighbor_counts(self, X):
        k_eff = min(self.k, len(self.X_))
        d = ((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(-1)
        nn = np.argsort(d, axis=1)[:, :k_eff]
        lab = self.y_[nn]
        counts = (lab[:, :, None] == self.classes_[None, None, :]).sum(1)
        return counts, lab, k_eff

    def predict_proba(self, X):
        counts, _, k_eff = self._neighbor_counts(X)
        m = len(self.classes_)
        # Symmetric Dirichlet(1/m) smoothing avoids zero-probability log-loss singularities
        return (counts + 1.0 / m) / (float(k_eff) + 1.0)

    def predict(self, X):
        counts, lab, _ = self._neighbor_counts(X)
        # tie-break by the nearest neighbour's label
        counts = counts + 1e-3 * (lab[:, :1] == self.classes_[None, :])
        return self.classes_[np.argmax(counts, axis=1)]


class _CartClassificationTree:
    """Vectorised Gini classification tree for RandomForest."""

    def __init__(self, n_classes: int, max_depth: int, min_samples_leaf: int, max_features: int):
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree_ = None

    def fit(self, X: np.ndarray, y_idx: np.ndarray, rng: np.random.Generator):
        self.tree_ = self._build(X, y_idx, depth=0, rng=rng)
        return self

    def _build(self, X: np.ndarray, y_idx: np.ndarray, depth: int, rng: np.random.Generator):
        n, d = X.shape
        counts = np.bincount(y_idx, minlength=self.n_classes).astype(np.float64)
        probs = counts / max(float(n), 1.0)
        parent_gini = 1.0 - float(np.sum(probs * probs))
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or parent_gini < 1e-12:
            return ("leaf", probs)

        feat_choices = rng.choice(d, size=min(self.max_features, d), replace=False)
        best_gain = 0.0
        best_split = None
        Y_oh = np.eye(self.n_classes, dtype=np.float64)[y_idx]

        for feat in feat_choices:
            col = X[:, feat]
            order = np.argsort(col, kind="mergesort")
            xs = col[order]
            ys = Y_oh[order]
            # Candidate split positions i where xs[i-1] < xs[i] and both sides >= min_samples_leaf
            lo = self.min_samples_leaf
            hi = n - self.min_samples_leaf
            if hi < lo:
                continue
            valid = xs[lo:hi + 1] > xs[lo - 1:hi] + 1e-12
            if not np.any(valid):
                continue
            c_left = np.cumsum(ys, axis=0)[lo - 1:hi][valid]
            n_left = np.arange(lo, hi + 1, dtype=np.float64)[valid][:, None]
            c_right = counts[None, :] - c_left
            n_right = float(n) - n_left
            g_left = 1.0 - np.sum((c_left / n_left) ** 2, axis=1)
            g_right = 1.0 - np.sum((c_right / n_right) ** 2, axis=1)
            weighted = (n_left[:, 0] * g_left + n_right[:, 0] * g_right) / float(n)
            gains = parent_gini - weighted
            idx_local = int(np.argmax(gains))
            if gains[idx_local] > best_gain + 1e-12:
                best_gain = float(gains[idx_local])
                split_pos = np.where(valid)[0][idx_local] + lo
                thresh = 0.5 * float(xs[split_pos - 1] + xs[split_pos])
                best_split = (int(feat), thresh)

        if best_split is None:
            return ("leaf", probs)
        feat, thresh = best_split
        left_mask = X[:, feat] <= thresh
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return ("leaf", probs)
        return (
            "node",
            feat,
            thresh,
            self._build(X[left_mask], y_idx[left_mask], depth + 1, rng),
            self._build(X[~left_mask], y_idx[~left_mask], depth + 1, rng),
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((len(X), self.n_classes), dtype=np.float64)
        self._eval(self.tree_, X, np.arange(len(X)), out)
        return out

    def _eval(self, node, X: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
        if len(rows) == 0:
            return
        if node[0] == "leaf":
            out[rows] = node[1]
            return
        _, feat, thresh, left, right = node
        m = X[rows, feat] <= thresh
        self._eval(left, X, rows[m], out)
        self._eval(right, X, rows[~m], out)


class RandomForest:
    """Pure-NumPy Random Forest classifier with bootstrap aggregation and random feature subsets."""

    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 5,
        min_samples_leaf: int = 2,
        seed: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        max_f = max(1, int(math.sqrt(d)))
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        for _ in range(self.n_estimators):
            boot = rng.integers(0, n, size=n)
            tree = _CartClassificationTree(len(self.classes_), self.max_depth, self.min_samples_leaf, max_f)
            tree.fit(X[boot], y_idx[boot], rng)
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.mean([t.predict_proba(X) for t in self.trees_], axis=0)
        # Mild Laplace floor keeps log-loss finite on unseen extreme outliers
        eps = 1e-6
        probs = (1.0 - eps) * probs + eps / len(self.classes_)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _NewtonRegressionTree:
    """Shallow least-squares regression tree with Friedman multiclass Newton leaf values."""

    def __init__(self, max_depth: int, min_samples_leaf: int, n_classes: int):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_classes = n_classes
        self.tree_ = None

    def fit(self, X: np.ndarray, resid: np.ndarray, hess: np.ndarray):
        self.tree_ = self._build(X, resid, hess, depth=0)
        return self

    def _leaf_value(self, resid: np.ndarray, hess: np.ndarray) -> float:
        scale = (self.n_classes - 1.0) / max(float(self.n_classes), 1.0)
        denom = float(np.sum(hess))
        if denom < 1e-10:
            return 0.0
        return float(np.clip(scale * float(np.sum(resid)) / denom, -4.0, 4.0))

    def _build(self, X: np.ndarray, resid: np.ndarray, hess: np.ndarray, depth: int):
        n, d = X.shape
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return ("leaf", self._leaf_value(resid, hess))
        total_sum = float(np.sum(resid))
        parent_score = (total_sum * total_sum) / float(n)
        best_gain = 0.0
        best_split: Optional[tuple[int, float]] = None

        for feat in range(d):
            col = X[:, feat]
            order = np.argsort(col, kind="mergesort")
            xs = col[order]
            rs = resid[order]
            lo = self.min_samples_leaf
            hi = n - self.min_samples_leaf
            if hi < lo:
                continue
            valid = xs[lo:hi + 1] > xs[lo - 1:hi] + 1e-12
            if not np.any(valid):
                continue
            s_left = np.cumsum(rs)[lo - 1:hi][valid]
            n_left = np.arange(lo, hi + 1, dtype=np.float64)[valid]
            s_right = total_sum - s_left
            n_right = float(n) - n_left
            gains = (s_left * s_left) / n_left + (s_right * s_right) / n_right - parent_score
            idx_local = int(np.argmax(gains))
            if gains[idx_local] > best_gain + 1e-12:
                best_gain = float(gains[idx_local])
                split_pos = np.where(valid)[0][idx_local] + lo
                thresh = 0.5 * float(xs[split_pos - 1] + xs[split_pos])
                best_split = (feat, thresh)

        if best_split is None:
            return ("leaf", self._leaf_value(resid, hess))
        feat, thresh = best_split
        left_mask = X[:, feat] <= thresh
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return ("leaf", self._leaf_value(resid, hess))
        return (
            "node",
            feat,
            thresh,
            self._build(X[left_mask], resid[left_mask], hess[left_mask], depth + 1),
            self._build(X[~left_mask], resid[~left_mask], hess[~left_mask], depth + 1),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X), dtype=np.float64)
        self._eval(self.tree_, X, np.arange(len(X)), out)
        return out

    def _eval(self, node, X: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
        if len(rows) == 0:
            return
        if node[0] == "leaf":
            out[rows] = node[1]
            return
        _, feat, thresh, left, right = node
        m = X[rows, feat] <= thresh
        self._eval(left, X, rows[m], out)
        self._eval(right, X, rows[~m], out)


class GradientBoosting:
    """Friedman multiclass log-loss Gradient Boosted Decision Trees."""

    def __init__(
        self,
        n_estimators: int = 30,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        min_samples_leaf: int = 2,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        Y = np.eye(k, dtype=np.float64)[y_idx]
        priors = np.clip(Y.mean(axis=0), 1e-6, 1.0)
        self.init_logits_ = np.log(priors)
        F = np.tile(self.init_logits_, (len(X), 1))
        self.stages_ = []

        for _ in range(self.n_estimators):
            P = np.exp(F - logsumexp(F, axis=1, keepdims=True))
            resid = Y - P
            hess = np.abs(resid) * (1.0 - np.abs(resid))
            stage_trees = []
            for c in range(k):
                tree = _NewtonRegressionTree(self.max_depth, self.min_samples_leaf, k)
                tree.fit(X, resid[:, c], hess[:, c])
                F[:, c] += self.learning_rate * tree.predict(X)
                stage_trees.append(tree)
            self.stages_.append(stage_trees)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = np.tile(self.init_logits_, (len(X), 1))
        for stage_trees in self.stages_:
            for c, tree in enumerate(stage_trees):
                F[:, c] += self.learning_rate * tree.predict(X)
        return np.exp(F - logsumexp(F, axis=1, keepdims=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
