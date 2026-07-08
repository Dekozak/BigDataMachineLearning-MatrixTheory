"""Principal component analysis as pure matrix theory.

PCA is the Eckart-Young theorem applied to a centered data matrix.
If X (n x d) has centered rows, the covariance is C = X^T X / (n-1),
and the SVD X = U diag(s) V^T diagonalizes it: C = V diag(s^2/(n-1)) V^T.
The columns of V are the principal directions; projecting onto the top
k of them is the best rank-k reconstruction of the data in Frobenius
norm.  The 'explained variance ratio' of component i is

    s_i^2 / sum_j s_j^2,

so the singular value spectrum *is* the story of how compressible the
dataset is -- the quantity all of machine-learning dimensionality
reduction turns on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PCA"]


@dataclass
class PCA:
    """Principal component analysis via the SVD of the centered data."""

    n_components: int

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = self.n_components
        self.components_ = Vt[:k]                    # principal directions
        self.singular_values_ = s[:k]
        var = s**2 / (X.shape[0] - 1)
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / var.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Relative Frobenius reconstruction error -- by Eckart-Young the
        minimum achievable by *any* k-dimensional affine projection."""
        Xh = self.inverse_transform(self.transform(X))
        Xc = np.asarray(X, dtype=float) - self.mean_
        return float(np.linalg.norm(X - Xh, "fro") / np.linalg.norm(Xc, "fro"))
