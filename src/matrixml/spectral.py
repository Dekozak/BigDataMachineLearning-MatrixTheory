"""Spectral graph theory for machine learning: Laplacians and clustering.

For a graph with adjacency matrix W and degree matrix D, the (symmetric
normalized) Laplacian is

    L_sym = I - D^{-1/2} W D^{-1/2},

a positive semidefinite matrix whose spectrum encodes cluster structure:
the multiplicity of the eigenvalue 0 equals the number of connected
components, and the second-smallest eigenvalue (the *algebraic
connectivity* or Fiedler value) measures how well connected the graph
is.  Cheeger's inequality makes this quantitative:

    lambda_2 / 2  <=  h(G)  <=  sqrt(2 * lambda_2),

where h(G) is the conductance of the best sparse cut.  Spectral
clustering (Shi-Malik 2000, Ng-Jordan-Weiss 2001) turns this theory
into an algorithm: embed vertices by the bottom k eigenvectors of
L_sym and run k-means in that embedding.

The stochastic block model (SBM) is the standard testbed: n vertices in
hidden communities, edge probability p inside and q < p across.  For
two balanced communities the Fiedler vector's signs recover the
partition exactly once (p - q) * n is large enough compared to the
noise level -- a fact this module lets you observe directly.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sbm",
    "normalized_laplacian",
    "fiedler_vector",
    "spectral_bipartition",
    "clustering_accuracy",
]


def sbm(n: int, k: int, p_in: float, p_out: float, rng=None):
    """Balanced stochastic block model.

    Returns (adjacency, labels) for n vertices in k equal communities."""
    rng = np.random.default_rng(rng)
    labels = np.repeat(np.arange(k), n // k)
    same = labels[:, None] == labels[None, :]
    P = np.where(same, p_in, p_out)
    U = rng.random((n, n))
    W = np.triu((U < P).astype(float), 1)
    W = W + W.T
    return W, labels


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    """L_sym = I - D^{-1/2} W D^{-1/2}, with isolated vertices handled."""
    d = W.sum(axis=1)
    d_isqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    return np.eye(W.shape[0]) - (d_isqrt[:, None] * W) * d_isqrt[None, :]


def fiedler_vector(W: np.ndarray):
    """Second-smallest eigenpair of the normalized Laplacian.

    Returns (lambda_2, v_2).  lambda_2 is the algebraic connectivity."""
    L = normalized_laplacian(W)
    vals, vecs = np.linalg.eigh(L)
    return float(vals[1]), vecs[:, 1]


def spectral_bipartition(W: np.ndarray) -> np.ndarray:
    """Two-way spectral clustering: signs of the Fiedler vector."""
    _, v = fiedler_vector(W)
    return (v >= 0).astype(int)


def clustering_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    """Accuracy of a 2-way clustering up to label swap."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    a = float((pred == truth).mean())
    return max(a, 1.0 - a)
