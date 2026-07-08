# Matrix Theory → Machine Learning

**Linear algebra and matrix theory as the engine of big-data machine learning — made computational.**

Nearly every algorithm that makes machine learning work at scale is a theorem of matrix analysis wearing an engineering costume. This repository walks the chain, with executable, tested Python versions of each link:

```
Eckart–Young–Mirsky optimal low rank  (1936)
   └── randomized SVD: sketch the range, then decompose  (Halko–Martinsson–Tropp 2011)
        └── Johnson–Lindenstrauss random projection  (1984)
             └── sketch-and-solve least squares  (Sarlós 2006, Clarkson–Woodruff 2013)
                  └── PCA = Eckart–Young applied to data
                       └── spectral clustering: Laplacian eigenvectors + Cheeger's inequality
                            └── power iteration / PageRank: Perron–Frobenius at web scale
                                 └── Nyström: low rank transplanted into kernel space
```

Each module states the theorem in its docstring and implements the finite computation that the theorem governs, so you can watch the bounds hold — often with equality — on concrete matrices.

## The dictionary

| Matrix theorem | Big-data algorithm it powers |
|---|---|
| Eckart–Young: truncated SVD is the optimal rank-k approximation | data compression, latent factor models, recommender systems |
| HMT bound: a Gaussian sketch captures the range up to ≈ σ_{k+1} | randomized SVD — two passes over data, small dense work |
| Johnson–Lindenstrauss: m ≈ 8 ln(n)/ε² dimensions preserve all distances | dimensionality reduction independent of ambient dimension |
| subspace embeddings distort every ‖Ax − b‖ by at most (1 ± ε) | sketched regression: solve tall least squares from a small sketch |
| spectral theorem on X^T X | PCA, explained variance, intrinsic dimension |
| Cheeger: λ₂/2 ≤ h(G) ≤ √(2λ₂) | spectral clustering via the Fiedler vector |
| Perron–Frobenius + damping forces \|λ₂\| ≤ α | PageRank converges in O(log(1/tol)/log(1/α)) iterations, at any scale |
| spectral decay of the Gram matrix | Nyström kernel approximation: K ≈ C W⁺ Cᵀ from m ≪ n landmarks |

## What's in the package

`matrixml.lowrank` implements the truncated SVD, verifies the Eckart–Young optimal errors exactly, and provides the Halko–Martinsson–Tropp randomized SVD with oversampling and power iterations. `matrixml.sketching` provides Gaussian and CountSketch transforms, the JL target dimension and worst-pairwise-distortion check, and sketch-and-solve least squares with its (1+ε) residual guarantee. `matrixml.pca` is PCA via the SVD of centered data, with explained variance and the reconstruction error that Eckart–Young certifies as optimal. `matrixml.spectral` builds stochastic block models, normalized Laplacians, Fiedler vectors, and two-way spectral clustering. `matrixml.power` implements power iteration with a tracked error history (so the geometric rate |λ₂/λ₁|ᵗ is visible) and PageRank with its size-independent iteration count. `matrixml.nystrom` approximates RBF Gram matrices from landmark columns. Twelve unit tests bind the code to the theorems — e.g. that no random rank-k matrix ever beats the truncated SVD, that a disconnected graph has λ₂ = 0, and that Nyström with all points as landmarks is exact.

## Quick start

```bash
git clone https://github.com/you/matrix-ml
cd matrix-ml
pip install -e ".[dev]"
pytest                       # ~1 s: Eckart-Young, JL, Cheeger, Perron-Frobenius, ...

python examples/01_eckart_young_randomized_svd.py
python examples/02_jl_sketched_regression.py
python examples/03_pca_intrinsic_dimension.py
python examples/04_spectral_power_nystrom.py
```

A taste, from example 1 (a 3000 × 1500 matrix with polynomially decaying spectrum):

```
full SVD      : 2.662 s, rel err 3.129e-02
randomized SVD: 0.102 s, rel err 3.130e-02
speedup x26.0, singular value agreement: max |sigma_i - sigma_i^rand| = 1.77e-05
```

and from example 4: power iteration on a spectrum (1, 0.9, 0.5, …) converges at empirical rate 0.9000 against the theoretical |λ₂/λ₁| = 0.9, and PageRank converges in 34–37 iterations whether the graph has 100 or 3000 nodes — the damping factor, not the data size, sets the runtime.

## Reading list

Halko, Martinsson & Tropp, *Finding structure with randomness*, SIAM Review 53 (2011). Woodruff, *Sketching as a Tool for Numerical Linear Algebra*, Found. Trends TCS (2014). Trefethen & Bau, *Numerical Linear Algebra*, SIAM (1997). von Luxburg, *A tutorial on spectral clustering*, Stat. Comput. 17 (2007). Vershynin, *High-Dimensional Probability*, CUP (2018).

## License

MIT
