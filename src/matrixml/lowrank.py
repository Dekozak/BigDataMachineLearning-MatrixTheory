"""Low-rank approximation: the Eckart-Young theorem and randomized SVD.

The mathematical bedrock of large-scale machine learning is the fact
that data matrices are approximately low rank, and that the *optimal*
low-rank approximation is read off the singular value decomposition:

Eckart-Young-Mirsky (1936):  if A = U diag(sigma) V^T, then for every
rank-k matrix B,

    ||A - B||_2  >=  sigma_{k+1},
    ||A - B||_F  >=  sqrt(sigma_{k+1}^2 + ... + sigma_r^2),

with equality for the truncated SVD A_k = U_k diag(sigma_1..k) V_k^T.

For big data the full SVD (O(m n min(m,n)) time) is unaffordable.  The
randomized SVD of Halko, Martinsson & Tropp (2011) computes a near-
optimal rank-k factorization from a *sketch* A @ Omega with a random
Gaussian test matrix Omega, in time O(m n (k+p)) plus an O((k+p)^2)-
sized dense SVD.  Their average-case bound (with oversampling p >= 2
and q power iterations):

    E ||A - Q Q^T A||_2  <=  [1 + sqrt(k/(p-1)) + e*sqrt(k+p)/p *
                              sqrt(min(m,n)-k)]^(1/(2q+1)) * sigma_{k+1}

so a couple of power iterations drive the error to essentially the
optimal sigma_{k+1}.  This module implements both and lets you verify
the bound numerically.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "truncated_svd",
    "eckart_young_errors",
    "randomized_svd",
    "randomized_range_finder",
]


def truncated_svd(A: np.ndarray, k: int):
    """Optimal rank-k approximation via the deterministic SVD.

    Returns (U_k, s_k, Vt_k, A_k)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]
    return U[:, :k], s[:k], Vt[:k], Ak


def eckart_young_errors(A: np.ndarray, k: int) -> dict:
    """The optimal rank-k errors predicted by Eckart-Young-Mirsky.

    Returns spectral and Frobenius optimal errors together with the
    achieved errors of the truncated SVD (they must coincide)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]
    return {
        "spectral_optimal": float(s[k]) if k < s.size else 0.0,
        "spectral_achieved": float(np.linalg.norm(A - Ak, 2)),
        "frobenius_optimal": float(np.sqrt((s[k:] ** 2).sum())),
        "frobenius_achieved": float(np.linalg.norm(A - Ak, "fro")),
    }


def randomized_range_finder(
    A: np.ndarray, size: int, n_power: int = 0, rng=None
) -> np.ndarray:
    """Orthonormal Q approximating the range of A (HMT Algorithm 4.3/4.4).

    Sketch Y = (A A^T)^q A Omega with Gaussian Omega, orthonormalize.
    Power iterations (q = n_power) sharpen the spectrum: singular values
    of the sketched operator are sigma_i^(2q+1), crushing the tail.
    """
    rng = np.random.default_rng(rng)
    Omega = rng.standard_normal((A.shape[1], size))
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    for _ in range(n_power):
        # subspace iteration with re-orthonormalization for stability
        Q, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Q)
    return Q


def randomized_svd(
    A: np.ndarray, k: int, oversample: int = 10, n_power: int = 2, rng=None
):
    """Randomized SVD (Halko-Martinsson-Tropp 2011).

    Returns (U, s, Vt) with k columns/values.  Cost: two passes over A
    plus dense work on (k + oversample)-sized matrices.
    """
    Q = randomized_range_finder(A, k + oversample, n_power, rng)
    B = Q.T @ A                     # small (k+p) x n matrix
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], s[:k], Vt[:k]
