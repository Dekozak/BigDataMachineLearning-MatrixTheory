"""PCA = Eckart-Young on data: compression of an intrinsically
low-dimensional dataset embedded in high dimension.

Data: 2000 points on a noisy 5-dimensional subspace of R^300.
PCA with k = 5 recovers essentially all the variance; the
reconstruction error matches the Eckart-Young optimum by construction.
"""

import numpy as np

from matrixml import PCA

rng = np.random.default_rng(0)
n, d, r = 2000, 300, 5

# latent 5-d signal, embedded and noised
Z = rng.standard_normal((n, r)) @ np.diag([10, 8, 6, 4, 2])
B = np.linalg.qr(rng.standard_normal((d, r)))[0]
X = Z @ B.T + 0.3 * rng.standard_normal((n, d))

for k in (2, 5, 10):
    pca = PCA(k).fit(X)
    evr = pca.explained_variance_ratio_.sum()
    err = pca.reconstruction_error(X)
    print(f"k = {k:>2}: explained variance {evr:6.2%},  "
          f"relative reconstruction error {err:.4f}")

pca5 = PCA(5).fit(X)
print("\ntop singular values:", np.round(pca5.singular_values_, 1).tolist())
print("(five dominant values, then noise floor: the spectrum tells you"
      " the intrinsic dimension.)")
