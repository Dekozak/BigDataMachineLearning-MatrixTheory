"""Power iteration and PageRank: eigenvector computation at web scale.

Power iteration -- repeatedly apply A and normalize -- converges to the
dominant eigenvector at the geometric rate |lambda_2 / lambda_1|^t.
The whole analysis is one line of matrix theory: expand the start
vector in the eigenbasis and watch the subdominant terms decay.

PageRank (Brin-Page 1998) is power iteration on the Google matrix

    G = alpha * P + (1 - alpha) * (1/n) * J,

where P is the column-stochastic link matrix and alpha ~ 0.85 is the
damping factor.  Perron-Frobenius theory guarantees a unique positive
stationary vector, and -- crucially for big data -- the damping forces
|lambda_2(G)| <= alpha, so convergence takes O(log(1/tol) / log(1/alpha))
iterations *independently of the size of the web graph*.  This is
matrix analysis dictating the runtime of one of the largest
computations ever performed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["power_iteration", "pagerank"]


def power_iteration(A: np.ndarray, n_iter: int = 100, rng=None):
    """Dominant eigenpair by power iteration, with error history.

    Returns (eigenvalue, eigenvector, errors) where errors[t] is the
    sine of the angle between the iterate and the true dominant
    eigenvector -- theory says errors[t] ~ C * |lambda_2/lambda_1|^t.
    """
    rng = np.random.default_rng(rng)
    n = A.shape[0]
    # ground truth for the error history (small examples only)
    vals, vecs = np.linalg.eigh((A + A.T) / 2)
    v_true = vecs[:, np.argmax(np.abs(vals))]

    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)
    errors = []
    for _ in range(n_iter):
        x = A @ x
        x /= np.linalg.norm(x)
        errors.append(float(np.sqrt(max(0.0, 1.0 - (x @ v_true) ** 2))))
    lam = float(x @ A @ x)
    return lam, x, np.array(errors)


def pagerank(W: np.ndarray, alpha: float = 0.85, tol: float = 1e-10):
    """PageRank of a directed graph given by adjacency W (W[i,j]=1: i->j).

    Returns (rank_vector, n_iterations).  Dangling nodes are patched to
    uniform, as in the original formulation.
    """
    n = W.shape[0]
    out = W.sum(axis=1)
    P = np.where(out[:, None] > 0, W / np.maximum(out[:, None], 1e-12), 1.0 / n)
    r = np.full(n, 1.0 / n)
    for it in range(1, 10_000):
        r_new = alpha * (P.T @ r) + (1 - alpha) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new, it
        r = r_new
    return r, it
