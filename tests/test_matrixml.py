import numpy as np
import pytest

from matrixml import (
    PCA,
    clustering_accuracy,
    eckart_young_errors,
    fiedler_vector,
    gaussian_sketch,
    jl_dimension,
    jl_max_distortion,
    normalized_laplacian,
    nystrom,
    pagerank,
    power_iteration,
    randomized_svd,
    rbf_kernel,
    sbm,
    sketched_lstsq,
    spectral_bipartition,
    truncated_svd,
)

RNG = np.random.default_rng(0)


def _lowrank_matrix(m=300, n=200, r=50, decay=1.5, rng=RNG):
    U, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))
    s = np.arange(1, r + 1, dtype=float) ** -decay
    return (U * s) @ V.T, s


# ---------- Eckart-Young / randomized SVD ----------------------------------

def test_eckart_young_truncated_svd_is_optimal():
    A, s = _lowrank_matrix()
    ey = eckart_young_errors(A, 10)
    assert ey["spectral_achieved"] == pytest.approx(ey["spectral_optimal"], rel=1e-8)
    assert ey["frobenius_achieved"] == pytest.approx(ey["frobenius_optimal"], rel=1e-8)
    assert ey["spectral_optimal"] == pytest.approx(s[10], rel=1e-8)


def test_random_rank_k_is_never_better_than_svd():
    A, _ = _lowrank_matrix()
    *_, Ak = truncated_svd(A, 5)
    opt = np.linalg.norm(A - Ak, "fro")
    rng = np.random.default_rng(1)
    for _ in range(5):
        B = rng.standard_normal((300, 5)) @ rng.standard_normal((5, 200)) * 0.01
        assert np.linalg.norm(A - B, "fro") >= opt


def test_randomized_svd_matches_optimal():
    A, s = _lowrank_matrix()
    k = 10
    U, sr, Vt = randomized_svd(A, k, oversample=10, n_power=2, rng=2)
    assert np.allclose(sr, s[:k], rtol=1e-3)
    Ak = (U * sr) @ Vt
    opt = np.sqrt((s[k:] ** 2).sum())
    assert np.linalg.norm(A - Ak, "fro") <= 1.01 * opt
    # factors are orthonormal
    assert np.allclose(U.T @ U, np.eye(k), atol=1e-10)


# ---------- JL / sketching ---------------------------------------------------

def test_jl_distortion_within_eps():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 2000))
    eps = 0.3
    m = jl_dimension(50, eps)
    S = gaussian_sketch(m, 2000, rng=4)
    assert jl_max_distortion(X, S) <= eps


def test_sketched_lstsq_near_optimal():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((20_000, 30))
    b = A @ rng.standard_normal(30) + rng.standard_normal(20_000)
    _, _, r_sk, r_ex = sketched_lstsq(A, b, m=1500, rng=6)
    assert r_sk <= 1.05 * r_ex  # (1+eps)-guarantee with headroom


# ---------- PCA --------------------------------------------------------------

def test_pca_matches_eckart_young():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 40)) @ np.diag(np.arange(40, 0, -1.0) ** 0.5)
    k = 5
    pca = PCA(k).fit(X)
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    achieved = np.linalg.norm(Xc - (pca.transform(X) @ pca.components_), "fro")
    optimal = np.sqrt((s[k:] ** 2).sum())
    assert achieved == pytest.approx(optimal, rel=1e-8)
    assert pca.explained_variance_ratio_.sum() <= 1.0


# ---------- spectral ---------------------------------------------------------

def test_laplacian_psd_and_zero_eigenvalue():
    W, _ = sbm(100, 2, 0.3, 0.05, rng=8)
    L = normalized_laplacian(W)
    vals = np.linalg.eigvalsh(L)
    assert vals[0] == pytest.approx(0.0, abs=1e-8)
    assert vals.min() >= -1e-8


def test_spectral_clustering_recovers_sbm():
    W, labels = sbm(400, 2, 0.15, 0.02, rng=9)
    pred = spectral_bipartition(W)
    assert clustering_accuracy(pred, labels) >= 0.95


def test_disconnected_graph_has_lambda2_zero():
    W = np.zeros((6, 6))
    W[0, 1] = W[1, 0] = W[2, 3] = W[3, 2] = W[4, 5] = W[5, 4] = 1
    lam2, _ = fiedler_vector(W)
    assert lam2 == pytest.approx(0.0, abs=1e-10)


# ---------- power iteration / PageRank ---------------------------------------

def test_power_iteration_rate():
    rng = np.random.default_rng(10)
    Q, _ = np.linalg.qr(rng.standard_normal((30, 30)))
    A = Q @ np.diag([1.0, 0.8] + [0.3] * 28) @ Q.T
    lam, v, errs = power_iteration(A, n_iter=60, rng=11)
    assert lam == pytest.approx(1.0, abs=1e-6)
    rate = (errs[40] / errs[10]) ** (1 / 30)
    assert rate == pytest.approx(0.8, abs=0.05)


def test_pagerank_is_stochastic_and_fast():
    rng = np.random.default_rng(12)
    W = (rng.random((500, 500)) < 0.01).astype(float)
    np.fill_diagonal(W, 0)
    r, iters = pagerank(W, alpha=0.85)
    assert r.sum() == pytest.approx(1.0, abs=1e-8)
    assert (r > 0).all()
    assert iters < 200  # size-independent bound ~ log(tol)/log(alpha)


# ---------- Nystrom -----------------------------------------------------------

def test_nystrom_error_decreases_and_full_rank_exact():
    rng = np.random.default_rng(13)
    X = rng.standard_normal((120, 5))
    K = rbf_kernel(X, gamma=0.5)
    K50, _ = nystrom(X, 50, gamma=0.5, rng=14)
    K120, _ = nystrom(X, 120, gamma=0.5, rng=15)
    e50 = np.linalg.norm(K - K50, "fro") / np.linalg.norm(K, "fro")
    e120 = np.linalg.norm(K - K120, "fro") / np.linalg.norm(K, "fro")
    assert e120 <= e50
    assert e120 <= 1e-6  # all points as landmarks: exact
