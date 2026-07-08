"""Johnson-Lindenstrauss in action, then sketch-and-solve regression.

Part 1: 200 points in R^10000 are projected to the JL dimension for
eps = 0.25; the worst pairwise distortion stays below eps.

Part 2: a 100000 x 50 least-squares problem is solved from a sketch of
2000 rows; the residual is within a fraction of a percent of optimal.
"""

import numpy as np

from matrixml import gaussian_sketch, jl_dimension, jl_max_distortion, sketched_lstsq

rng = np.random.default_rng(0)

# --- Part 1: JL distortion --------------------------------------------
n_pts, d, eps = 200, 10_000, 0.25
X = rng.standard_normal((n_pts, d))
m = jl_dimension(n_pts, eps)
S = gaussian_sketch(m, d, rng=1)
dist = jl_max_distortion(X, S)
print(f"JL: {n_pts} points, R^{d} -> R^{m} (eps = {eps})")
print(f"    worst pairwise distortion = {dist:.4f}  "
      f"(lemma: <= {eps} with high probability)")

# --- Part 2: sketched least squares ------------------------------------
n, p = 100_000, 50
A = rng.standard_normal((n, p))
x_true = rng.standard_normal(p)
b = A @ x_true + 0.5 * rng.standard_normal(n)

for m_sk in (500, 1000, 2000):
    _, _, r_sk, r_ex = sketched_lstsq(A, b, m_sk, rng=2)
    print(f"sketch m = {m_sk:>5}: residual ratio "
          f"||Ax~-b|| / ||Ax*-b|| = {r_sk / r_ex:.6f}")
