"""matrixml: matrix theory as the engine of large-scale machine learning.

The package walks the chain of ideas

    Eckart-Young optimal low rank
        -> randomized SVD (sketch the range, then decompose)
        -> Johnson-Lindenstrauss random projection
        -> sketch-and-solve least squares
        -> PCA (Eckart-Young applied to data)
        -> spectral clustering (Laplacian eigenvectors, Cheeger)
        -> power iteration / PageRank (Perron-Frobenius at web scale)
        -> Nystrom kernel approximation (low rank in feature space)

with executable, testable versions of each step.
"""

from .lowrank import (
    eckart_young_errors,
    randomized_range_finder,
    randomized_svd,
    truncated_svd,
)
from .sketching import (
    count_sketch,
    gaussian_sketch,
    jl_dimension,
    jl_max_distortion,
    sketched_lstsq,
)
from .pca import PCA
from .spectral import (
    clustering_accuracy,
    fiedler_vector,
    normalized_laplacian,
    sbm,
    spectral_bipartition,
)
from .power import pagerank, power_iteration
from .nystrom import nystrom, nystrom_error_curve, rbf_kernel

__version__ = "0.1.0"
