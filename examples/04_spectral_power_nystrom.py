"""Spectral methods: clustering by Laplacian eigenvectors, power
iteration convergence, PageRank, and Nystrom kernel compression.
"""

import numpy as np

from matrixml import (
    clustering_accuracy,
    fiedler_vector,
    nystrom_error_curve,
    pagerank,
    power_iteration,
    sbm,
    spectral_bipartition,
)

rng = np.random.default_rng(0)

# --- spectral clustering of a stochastic block model -------------------
W, labels = sbm(600, 2, p_in=0.10, p_out=0.02, rng=1)
lam2, _ = fiedler_vector(W)
pred = spectral_bipartition(W)
acc = clustering_accuracy(pred, labels)
print(f"SBM n=600, p_in=0.10, p_out=0.02:")
print(f"  algebraic connectivity lambda_2 = {lam2:.4f}")
print(f"  Fiedler-sign clustering accuracy = {acc:.3%}")

# --- power iteration convergence rate -----------------------------------
D = np.diag([1.0, 0.9] + [0.5] * 48)
Q, _ = np.linalg.qr(rng.standard_normal((50, 50)))
A = Q @ D @ Q.T
_, _, errs = power_iteration(A, n_iter=200, rng=2)
emp_rate = (errs[100] / errs[50]) ** (1 / 50)
print(f"\npower iteration on spectrum (1, 0.9, 0.5...):")
print(f"  empirical convergence rate {emp_rate:.4f}  "
      f"(theory: |lambda_2/lambda_1| = 0.9000)")

# --- PageRank: iteration count independent of size ----------------------
for n in (100, 1000, 3000):
    Wd = (np.random.default_rng(3).random((n, n)) < 3.0 / n).astype(float)
    np.fill_diagonal(Wd, 0)
    r, iters = pagerank(Wd, alpha=0.85, tol=1e-10)
    print(f"PageRank n = {n:>5}: converged in {iters} iterations"
          f" (bound ~ log(1e-10)/log(0.85) = {np.log(1e-10)/np.log(0.85):.0f})")

# --- Nystrom: kernel compression driven by spectral decay ---------------
X = np.concatenate([rng.standard_normal((150, 20)) + 4 * rng.standard_normal(20)
                    for _ in range(4)])
ms = [10, 25, 50, 100, 200]
errs = nystrom_error_curve(X, ms, gamma=0.02, rng=4)
print("\nNystrom RBF Gram approximation (n = 600):")
for m, e in zip(ms, errs):
    print(f"  m = {m:>3} landmarks: relative Frobenius error {e:.4f}")
