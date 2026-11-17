"""Non-kernel classical baselines: multinomial logistic regression and k-nearest neighbours."""
from __future__ import annotations

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

    def predict(self, X):
        d = ((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(-1)
        nn = np.argsort(d, axis=1)[:, : self.k]
        lab = self.y_[nn]
        counts = (lab[:, :, None] == self.classes_[None, None, :]).sum(1)
        # tie-break by the nearest neighbour's label
        counts = counts + 1e-3 * (lab[:, :1] == self.classes_[None, :])
        return self.classes_[np.argmax(counts, axis=1)]
