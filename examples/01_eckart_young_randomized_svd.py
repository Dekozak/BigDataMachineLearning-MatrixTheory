"""Eckart-Young optimality and the randomized SVD on a big-ish matrix.

We build a 3000 x 1500 matrix with polynomially decaying spectrum,
verify that the truncated SVD achieves exactly the Eckart-Young optimal
errors, then show the randomized SVD matches it to several digits at a
fraction of the cost.
"""

import time

import numpy as np

from matrixml import eckart_young_errors, randomized_svd, truncated_svd

rng = np.random.default_rng(0)
m, n, k = 3000, 1500, 20

# synthetic data matrix with spectrum sigma_i = i^{-1.5}
U, _ = np.linalg.qr(rng.standard_normal((m, 200)))
V, _ = np.linalg.qr(rng.standard_normal((n, 200)))
s = (np.arange(1, 201, dtype=float)) ** -1.5
A = (U * s) @ V.T

ey = eckart_young_errors(A, k)
print(f"Eckart-Young, k = {k}:")
print(f"  spectral : optimal {ey['spectral_optimal']:.6e}  "
      f"achieved {ey['spectral_achieved']:.6e}")
print(f"  Frobenius: optimal {ey['frobenius_optimal']:.6e}  "
      f"achieved {ey['frobenius_achieved']:.6e}")

t0 = time.perf_counter()
_, s_full, _, Ak = truncated_svd(A, k)
t_full = time.perf_counter() - t0

t0 = time.perf_counter()
Ur, sr, Vtr = randomized_svd(A, k, oversample=10, n_power=2, rng=1)
t_rand = time.perf_counter() - t0
Ak_rand = (Ur * sr) @ Vtr

err_opt = np.linalg.norm(A - Ak, "fro")
err_rand = np.linalg.norm(A - Ak_rand, "fro")
print(f"\nfull SVD      : {t_full:.3f} s, rel err {err_opt/np.linalg.norm(A,'fro'):.3e}")
print(f"randomized SVD: {t_rand:.3f} s, rel err {err_rand/np.linalg.norm(A,'fro'):.3e}")
print(f"speedup x{t_full/t_rand:.1f}, singular value agreement: "
      f"max |sigma_i - sigma_i^rand| = {np.max(np.abs(s_full - sr)):.2e}")
