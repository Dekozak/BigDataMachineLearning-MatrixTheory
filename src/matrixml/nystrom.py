"""Nystrom approximation: kernel machines without the n x n matrix.

Kernel methods replace inner products by k(x, y) = <phi(x), phi(y)> and
require the n x n Gram matrix K -- quadratic memory, cubic solves, and
therefore hopeless at big-data scale.  The Nystrom method (Williams &
Seeger 2001) picks m << n landmark points, forms the n x m block C and
the m x m block W of K, and approximates

    K  ~  C W^+ C^T,

a rank-m positive semidefinite approximation computable in O(n m^2).
The quality is governed by the spectral decay of K: for kernels with
rapidly decaying eigenvalues (e.g. the RBF kernel on clustered data)
a few hundred landmarks capture the whole Gram matrix.  This is the
low-rank philosophy of Eckart-Young transplanted to kernel space.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rbf_kernel", "nystrom", "nystrom_error_curve"]


def rbf_kernel(X: np.ndarray, Y: np.ndarray | None = None, gamma: float = 1.0):
    """Gaussian RBF kernel matrix K[i,j] = exp(-gamma ||x_i - y_j||^2)."""
    Y = X if Y is None else Y
    XX = (X**2).sum(1)[:, None]
    YY = (Y**2).sum(1)[None, :]
    D2 = np.maximum(XX + YY - 2 * X @ Y.T, 0.0)
    return np.exp(-gamma * D2)


def nystrom(X: np.ndarray, m: int, gamma: float = 1.0, rng=None):
    """Rank-m Nystrom approximation of the RBF Gram matrix of X.

    Returns (K_approx, landmark_indices)."""
    rng = np.random.default_rng(rng)
    n = X.shape[0]
    idx = rng.choice(n, size=m, replace=False)
    C = rbf_kernel(X, X[idx], gamma)          # n x m
    W = C[idx]                                # m x m
    K_approx = C @ np.linalg.pinv(W, rcond=1e-10) @ C.T
    return K_approx, idx


def nystrom_error_curve(X: np.ndarray, ms, gamma: float = 1.0, rng=None):
    """Relative Frobenius error ||K - K_m||_F / ||K||_F versus landmarks m."""
    K = rbf_kernel(X, gamma=gamma)
    nK = np.linalg.norm(K, "fro")
    errs = []
    for m in ms:
        Km, _ = nystrom(X, m, gamma, rng)
        errs.append(float(np.linalg.norm(K - Km, "fro") / nK))
    return np.array(errs)
