"""
Divisive Normalization Module
=============================

Implements population-level divisive normalization (Eq. 6/14 of the paper):

    r^{post}_i(S, θ) = γ · r^{pre}_i(S, θ) / D(S, θ)
    SUM(r^{post}_i(S, θ)) = GAMMA/D(S, THETA) * SUM(r^{pre}_i(S, θ))
    SUM(r^{post}_i(S, θ)) = GAMMA*N (TOTAL FIRING RATE)
    GAMMA = SUM(r^{post}_i(S, θ))/N = MEAN POPULATION RATE (WHEN SIGMA = 0)
    
    ####### COMPARATIVE ANALYSIS
    In Bays = GAMMA is TFR, Ours is MFR
where:
    r^{pre}_i(S, θ) = ∏_{k ∈ S} exp(f_{i,k}(θ_k))      [Eq. 13]
    D(S, θ) = σ² + N⁻¹ Σ_{j=1}^{N} r^{pre}_j(S, θ)     [Eq. 14]

Activity Cap Theorem (σ² → 0):
    Σ_i r^{post}_i(S, θ) = γN   for all (S, θ)           [Eq. 15]

Three computation paths, all implementing the same equation:
    1. pointwise  — exact DN at a single (S, θ) configuration  [used by decoder]
    2. tensor     — exact DN over the full n_θ^l grid          [small l only]
    3. montecarlo — sampled DN over random configurations       [verification]
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# =============================================================================
# CORE: The one equation, three entry points
# =============================================================================

def dn_pointwise(
    r_pre: np.ndarray,
    gamma: float,
    sigma_sq: float = 1e-6
) -> np.ndarray:
    """
    Apply divisive normalization to a vector of pre-normalized rates.

    This is the atomic operation — Eq. 6 evaluated at a single (S, θ):

        r^{post}_i = γ · r^{pre}_i / (σ² + N⁻¹ Σ_j r^{pre}_j)

    Parameters
    ----------
    r_pre : np.ndarray, shape (N,)
        Pre-normalized firing rates for all N neurons at one configuration.
    gamma : float
        Gain constant (Hz).
    sigma_sq : float
        Semi-saturation constant.

    Returns
    -------
    r_post : np.ndarray, shape (N,)
        Post-normalized firing rates.
    """
    D = sigma_sq + np.mean(r_pre)
    return gamma * r_pre / D


def dn_tensor(
    R_pre_stack: np.ndarray,
    gamma: float,
    sigma_sq: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply DN over a full configuration tensor (exact, materialised).

    Parameters
    ----------
    R_pre_stack : np.ndarray, shape (N, *config_dims)
        Pre-normalized rates for all N neurons over a grid of configurations.
        E.g. for l=2 with n_θ=10: shape (N, 10, 10).
    gamma : float
        Gain constant.
    sigma_sq : float
        Semi-saturation constant.

    Returns
    -------
    R_post_stack : np.ndarray, same shape as input
        Post-normalized rates.
    D : np.ndarray, shape (*config_dims)
        Denominator at each configuration.
    """
    D = sigma_sq + np.mean(R_pre_stack, axis=0)  # average over neurons
    R_post_stack = gamma * R_pre_stack / D[np.newaxis, ...]
    return R_post_stack, D


def dn_montecarlo(
    G: np.ndarray,
    subset: Tuple[int, ...],
    gamma: float,
    sigma_sq: float = 1e-6,
    n_samples: int = 10000,
    rng: Optional[np.random.RandomState] = None
) -> Dict:
    """
    Monte Carlo DN over random (S, θ) configurations.

    Samples n_samples orientation configurations uniformly, computes exact DN
    at each sample, and returns summary statistics.

    Memory: O(N × n_samples), independent of set size l.

    Parameters
    ----------
    G : np.ndarray, shape (N, n_locations, n_orientations)
        Exponentiated tuning functions: G[n, k, j] = exp(f_{n,k}(θ_j)).
    subset : Tuple[int, ...]
        Active location indices, |S| = l.
    gamma : float
        Gain constant.
    sigma_sq : float
        Semi-saturation constant.
    n_samples : int
        Number of Monte Carlo samples.
    rng : np.random.RandomState, optional
        Random state for reproducibility.

    Returns
    -------
    dict with keys:
        'R_pre'  : np.ndarray (n_samples, N) — pre-normalised rates
        'R_post' : np.ndarray (n_samples, N) — post-normalised rates
        'D'      : np.ndarray (n_samples,)   — denominator per sample
        'total_post_mean' : float — mean total post-DN activity (≈ γN)
    """
    if rng is None:
        rng = np.random.RandomState()

    N, _, n_orientations = G.shape
    l = len(subset)
    subset_arr = np.array(subset)

    # Sample random orientation indices: (n_samples, l)
    configs = rng.randint(0, n_orientations, size=(n_samples, l))

    # Build r^{pre}_{n}(S, θ) = ∏_k G[n, S[k], θ_k]
    R_pre = np.ones((n_samples, N))
    for k in range(l):
        loc = subset_arr[k]
        # G[:, loc, configs[:, k]] has shape (N, n_samples) → transpose
        R_pre *= G[:, loc, configs[:, k]].T

    # Apply DN (vectorised over samples)
    D = sigma_sq + np.mean(R_pre, axis=1)          # (n_samples,)
    R_post = gamma * R_pre / D[:, np.newaxis]       # (n_samples, N)

    return {
        'R_pre': R_pre,
        'R_post': R_post,
        'D': D,
        'total_post_mean': np.mean(np.sum(R_post, axis=1)),
    }


# =============================================================================
# PRE-NORMALISED RESPONSE HELPERS
# =============================================================================

def compute_r_pre_at_config(
    f_population: np.ndarray,
    subset: Tuple[int, ...],
    theta_indices: np.ndarray
) -> np.ndarray:
    """
    Compute r^{pre}_n(S, θ) = exp(Σ_{k ∈ S} f_{n,k}(θ_k)) for all N neurons
    at a single configuration.

    Parameters
    ----------
    f_population : np.ndarray, shape (N, n_locations, n_orientations)
        Log-rate tuning functions for the full population.
    subset : Tuple[int, ...]
        Active location indices.
    theta_indices : array-like of int, length l
        Orientation grid index at each active location.

    Returns
    -------
    r_pre : np.ndarray, shape (N,)
    """
    log_r = np.zeros(f_population.shape[0])
    for k, loc in enumerate(subset):
        log_r += f_population[:, loc, theta_indices[k]]
    return np.exp(log_r)


def compute_mean_r_pre_analytical(
    f_single_neuron: np.ndarray,
    subset: Tuple[int, ...]
) -> float:
    """
    E_{θ}[r^{pre}_n(S, θ)] via factorisation (no tensor needed).

    Since orientations are sampled independently across locations:
        E[∏_k g_{n,k}(θ_k)] = ∏_k E_θ[g_{n,k}(θ)]

    Complexity: O(l × n_θ) instead of O(n_θ^l).

    Parameters
    ----------
    f_single_neuron : np.ndarray, shape (n_locations, n_orientations)
        Log-rate tuning functions for one neuron.
    subset : Tuple[int, ...]
        Active location indices.

    Returns
    -------
    mean_r_pre : float
    """
    g = np.exp(f_single_neuron)                     # (n_locations, n_θ)
    g_bar = np.mean(g[list(subset), :], axis=1)     # (l,)
    return float(np.prod(g_bar))


# =============================================================================
# ACTIVITY CAP UTILITIES
# =============================================================================

def total_post_activity(gamma: float, N: int) -> float:
    """
    Analytical total post-DN activity (Activity Cap Theorem, Eq. 15).

    In the limit σ² → 0:  Σ_i r^{post}_i(S, θ) = γ · N
    """
    return gamma * N


def per_item_activity(gamma: float, N: int, l: int) -> float:
    """
    Per-item share of the activity budget: γN / l.

    This is the population-level quantity driving SNR ∝ 1/√l.
    """
    return gamma * N / l


def verify_activity_cap(
    observed_total: float,
    gamma: float,
    N: int
) -> Dict:
    """
    Compare an observed total post-DN activity against the theorem.

    Parameters
    ----------
    observed_total : float
        Empirically measured Σ_i r^{post}_i (from tensor or MC).
    gamma : float
    N : int

    Returns
    -------
    dict with theoretical value, absolute and relative error.
    """
    theoretical = gamma * N
    abs_err = abs(observed_total - theoretical)
    return {
        'observed': observed_total,
        'theoretical': theoretical,
        'absolute_error': abs_err,
        'relative_error': abs_err / theoretical if theoretical > 0 else float('inf'),
    }