"""Random projections: Johnson-Lindenstrauss and sketch-and-solve regression.

Johnson-Lindenstrauss lemma (1984): for any n points in R^d and any
0 < eps < 1, a random linear map S : R^d -> R^m with

    m  >=  8 ln(n) / eps^2

preserves all pairwise distances up to a factor (1 +- eps) with high
probability.  The remarkable point is that m is independent of d: a
million-dimensional dataset embeds into a few hundred dimensions with
small distortion.  A Gaussian matrix S with entries N(0, 1/m) works;
so do sparse constructions such as CountSketch (Clarkson-Woodruff 2013),
which apply in input-sparsity time.

Sketch-and-solve least squares: to solve min_x ||Ax - b||_2 with
A of size n x d, n >> d, draw a subspace embedding S in R^{m x n} with
m = O(d/eps^2) and solve the small problem min_x ||S A x - S b||_2.
The solution x~ satisfies

    ||A x~ - b||  <=  (1 + eps) min_x ||A x - b||,

turning a tall regression over big data into a small one.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gaussian_sketch",
    "count_sketch",
    "jl_max_distortion",
    "jl_dimension",
    "sketched_lstsq",
]


def gaussian_sketch(m: int, n: int, rng=None) -> np.ndarray:
    """Dense JL transform: m x n Gaussian with variance 1/m."""
    rng = np.random.default_rng(rng)
    return rng.standard_normal((m, n)) / np.sqrt(m)


def count_sketch(m: int, n: int, rng=None) -> np.ndarray:
    """CountSketch: one random +-1 per column in a random row.

    Applies to a vector in O(nnz) time; here materialized as a dense
    matrix for clarity (the mathematics, not the engineering).
    """
    rng = np.random.default_rng(rng)
    S = np.zeros((m, n))
    rows = rng.integers(0, m, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)
    S[rows, np.arange(n)] = signs
    return S


def jl_dimension(n_points: int, eps: float) -> int:
    """The JL lemma's sufficient target dimension m = ceil(8 ln n / eps^2)."""
    return int(np.ceil(8.0 * np.log(n_points) / eps**2))


def jl_max_distortion(X: np.ndarray, S: np.ndarray) -> float:
    """Worst-case relative distortion of pairwise distances under x -> Sx.

    Returns max over pairs i<j of | ||S(xi-xj)||^2 / ||xi-xj||^2 - 1 |.
    The JL lemma promises this is <= eps for a suitable random S.
    """
    Y = X @ S.T
    d2_orig = _pdist2(X)
    d2_proj = _pdist2(Y)
    mask = d2_orig > 0
    return float(np.max(np.abs(d2_proj[mask] / d2_orig[mask] - 1.0)))


def _pdist2(X: np.ndarray) -> np.ndarray:
    """Condensed squared pairwise distances."""
    G = X @ X.T
    sq = np.diag(G)
    D2 = sq[:, None] + sq[None, :] - 2 * G
    iu = np.triu_indices_from(D2, k=1)
    return np.maximum(D2[iu], 0.0)


def sketched_lstsq(A: np.ndarray, b: np.ndarray, m: int, kind="gaussian", rng=None):
    """Sketch-and-solve least squares.

    Returns (x_sketch, x_exact, residual_sketch, residual_exact) so the
    (1 + eps)-approximation guarantee can be checked directly.
    """
    n = A.shape[0]
    S = (gaussian_sketch if kind == "gaussian" else count_sketch)(m, n, rng)
    x_sk, *_ = np.linalg.lstsq(S @ A, S @ b, rcond=None)
    x_ex, *_ = np.linalg.lstsq(A, b, rcond=None)
    r_sk = float(np.linalg.norm(A @ x_sk - b))
    r_ex = float(np.linalg.norm(A @ x_ex - b))
    return x_sk, x_ex, r_sk, r_ex
