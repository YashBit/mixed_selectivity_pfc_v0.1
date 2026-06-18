"""
Core Gaussian Process Module for Mixed Selectivity Framework

This module contains all GP-related functions for generating neural tuning curves
with location-dependent lengthscales (the source of mixed selectivity).

The key innovation is that different spatial locations have different tuning widths,
creating non-separable tuning: R(θ, L) ≠ f(θ) · g(L)

Author: Mixed Selectivity Project
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
# KERNEL FUNCTIONS
# ============================================================================

def periodic_rbf_kernel(
    orientations: np.ndarray,
    lengthscale: float,
    jitter: float = 1e-4
) -> np.ndarray:
    """
    Compute periodic RBF (squared exponential) kernel for circular orientation space.

    The kernel is: k(θ_i, θ_j) = exp(-d²/(2λ²))
    where d is the circular distance: d = min(|θ_i - θ_j|, 2π - |θ_i - θ_j|)

    Parameters
    ----------
    orientations : np.ndarray
        Array of orientation values in radians, shape (n_theta,)
    lengthscale : float
        Lengthscale λ controlling tuning width (smaller = sharper tuning)
    jitter : float
        Small value added to diagonal for numerical stability

    Returns
    -------
    K : np.ndarray
        Covariance matrix, shape (n_theta, n_theta)
    """
    n_theta = len(orientations)

    # Vectorized computation
    theta_i, theta_j = np.meshgrid(orientations, orientations, indexing='ij')
    dist = np.abs(theta_i - theta_j)
    dist = np.minimum(dist, 2 * np.pi - dist)
    K = np.exp(-dist**2 / (2 * lengthscale**2))

    # Add jitter for numerical stability
    K += jitter * np.eye(n_theta)

    return K


def sample_gp_function(
    K: np.ndarray,
    random_state: np.random.RandomState
) -> np.ndarray:
    """
    Sample a function from a Gaussian Process with covariance K.

    Uses Cholesky decomposition with eigenvalue fallback for numerical stability:
    - Primary: f = L @ z where L = chol(K), z ~ N(0, I)
    - Fallback: f = V @ (sqrt(λ) * z) using eigendecomposition

    Parameters
    ----------
    K : np.ndarray
        Covariance matrix, shape (n_theta, n_theta)
    random_state : np.random.RandomState
        Random state for reproducibility

    Returns
    -------
    f : np.ndarray
        GP sample, shape (n_theta,)
    """
    n = K.shape[0]
    z = random_state.randn(n)

    try:
        # Try Cholesky first (faster, more stable when it works)
        L = np.linalg.cholesky(K)
        return L @ z
    except np.linalg.LinAlgError:
        # Fallback to eigendecomposition (always works for symmetric matrices)
        eigvals, eigvecs = np.linalg.eigh(K)
        # Clamp negative eigenvalues to small positive value
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ (np.sqrt(eigvals) * z)


# ============================================================================
# LENGTHSCALE GENERATION
# ============================================================================

# Canonical method names accepted by `generate_location_dependent_lengthscales`.
# `random_vector` is retained as a backward-compatible alias for `sparse_broad`.
_VALID_METHODS = {"folded_normal", "gamma", "sparse_broad"}
_METHOD_ALIASES = {"random_vector": "sparse_broad"}


def generate_location_dependent_lengthscales(
    n_locations: int,
    base_lengthscale: float,
    variability: float,
    random_state: np.random.RandomState,
    use_gamma: bool = False,
    method: Optional[str] = None,
    n_high: int = 1,
    high_multiplier: float = 3.0,
) -> np.ndarray:
    """
    Generate location-dependent lengthscales (source of mixed selectivity).

    Three distribution options, selectable via `method`:

    1. Folded-Normal (`method="folded_normal"`, default):
        λ_i = λ_base × |1 + σ_λ × z_i| where z_i ~ N(0, 1)
        A Normal distribution folded at zero via abs(), then floored at
        0.1 for numerical stability. I.i.d. across locations.

    2. Gamma (`method="gamma"`):
        λ_i ~ Gamma(shape=k, scale=θ)
        with parameters chosen so that:
            mean(λ_i) = base_lengthscale
            CV(λ_i)  = std/mean = variability
        which gives:
            k = 1 / variability² (shape)
            θ = base_lengthscale × variability² (scale)
        Natively defined on (0, ∞) — no folding required — giving a
        smooth, principled prior. I.i.d. across locations.

    3. Sparse-Broad (`method="sparse_broad"`, alias: `"random_vector"`):
        Non-i.i.d. two-component scheme. A sparse subset of locations
        (`n_high`, default 1) are designated "broad" and drawn from a
        Gaussian centred at `base_lengthscale * high_multiplier`; the
        remaining locations are "sharp" and drawn from a Gaussian
        centred at `base_lengthscale`. Both Gaussians use `variability`
        as their std (in folded-Normal style: |1 + σ_λ · z|). The broad
        location(s) are chosen uniformly at random.

        When `n_high == 1` this reduces to a one-hot indicator over
        locations selecting the unique broad-tuning slot; for
        `n_high > 1` it is a k-hot generalisation. The result is a
        mixed-selective neuron with a sparse set of dominant
        broad-tuning locations — structured asymmetry instead of the
        symmetric heterogeneity of methods 1–2.

    All three options break separability and create conjunctive (mixed)
    selectivity, but `sparse_broad` produces a qualitatively different
    population structure: asymmetric instead of symmetric heterogeneity.

    Parameters
    ----------
    n_locations : int
        Number of spatial locations.
    base_lengthscale : float
        Base lengthscale λ_base. Interpreted as the (sharp-component)
        mean in all three methods.
    variability : float
        Variability parameter σ_λ controlling heterogeneity.
        - Folded-Normal: std of the underlying Normal before folding.
        - Gamma: target coefficient of variation (std/mean).
        - Sparse-Broad: std of both component Gaussians (folded style).
    random_state : np.random.RandomState
        Random state for reproducibility.
    use_gamma : bool, optional
        Deprecated. If True and `method` is None, behaves as if
        `method="gamma"`. Ignored when `method` is explicitly set.
        Retained for backward compatibility.
    method : str or None, optional
        Sampling method. One of `"folded_normal"`, `"gamma"`,
        `"sparse_broad"` (alias `"random_vector"`). If None, falls back
        to `use_gamma` for backward compatibility.
    n_high : int, optional
        Sparse-broad mode only. Number of "broad" locations per neuron.
        Default 1. Clamped to [0, n_locations].
    high_multiplier : float, optional
        Sparse-broad mode only. Multiplier on `base_lengthscale` that
        sets the mean of the broad-component Gaussian. Default 3.0.

    Returns
    -------
    lengthscales : np.ndarray
        Location-dependent lengthscales, shape (n_locations,).
    """
    # Resolve method (preferred API) with use_gamma fallback for back-compat.
    if method is None:
        method = "gamma" if use_gamma else "folded_normal"
    method = method.lower()
    # Apply alias resolution (e.g. legacy "random_vector" → "sparse_broad").
    method = _METHOD_ALIASES.get(method, method)

    if method == "folded_normal":
        # I.i.d. folded-Normal: λ_i = λ_base · |1 + σ_λ · z_i|.
        random_factors = 1.0 + variability * random_state.randn(n_locations)
        random_factors = np.abs(random_factors)
        lengthscales = base_lengthscale * random_factors

    elif method == "gamma":
        # I.i.d. Gamma parameterised by mean and CV.
        # mean = k·θ, var = k·θ², so CV = 1/√k.
        # ⇒ k = 1/CV², θ = mean·CV².
        if variability <= 0:
            lengthscales = np.full(n_locations, base_lengthscale)
        else:
            shape = 1.0 / (variability ** 2)
            scale = base_lengthscale * (variability ** 2)
            lengthscales = random_state.gamma(
                shape=shape, scale=scale, size=n_locations
            )

    elif method == "sparse_broad":
        # Non-i.i.d. two-component scheme.
        # A sparse `n_high`-subset of locations gets the broad-component mean;
        # the rest get the sharp-component mean. Both use `variability` as
        # std (folded-Normal style for positivity).
        n_high_eff = int(np.clip(n_high, 0, n_locations))

        # Choose which locations are "broad" uniformly at random (no replacement).
        broad_indices = random_state.choice(
            n_locations, size=n_high_eff, replace=False
        )
        is_broad = np.zeros(n_locations, dtype=bool)
        is_broad[broad_indices] = True

        # Sample folded-Normal factors and scale by component mean.
        z = random_state.randn(n_locations)
        factors = np.abs(1.0 + variability * z)

        sharp_mean = base_lengthscale
        broad_mean = base_lengthscale * high_multiplier

        lengthscales = np.where(is_broad, broad_mean, sharp_mean) * factors

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Expected one of: 'folded_normal', 'gamma', 'sparse_broad' "
            f"(alias 'random_vector')."
        )

    # Floor at 0.1 for numerical stability of the GP kernel.
    lengthscales = np.maximum(lengthscales, 0.1)
    return lengthscales


# ============================================================================
# NEURON GENERATION
# ============================================================================

def generate_neuron_tuning_curves(
    n_orientations: int,
    n_locations: int,
    base_lengthscale: float,
    lengthscale_variability: float,
    random_state: np.random.RandomState,
    gain_variability: float = 0.2,
    use_gamma: bool = False,
    method: Optional[str] = None,
    n_high: int = 1,
    high_multiplier: float = 3.0,
) -> Dict:
    """
    Generate tuning curves for a single neuron across all locations.

    For each location, we:
    1. Generate a location-specific lengthscale
    2. Build the covariance matrix with that lengthscale
    3. Sample a GP function (log-rate)
    4. Apply random gain modulation

    Parameters
    ----------
    n_orientations : int
        Number of orientation bins (n_θ)
    n_locations : int
        Number of spatial locations (L)
    base_lengthscale : float
        Base lengthscale λ_base
    lengthscale_variability : float
        Variability σ_λ for heterogeneous tuning
    random_state : np.random.RandomState
        Random state for reproducibility
    gain_variability : float
        Variability in gain modulation across locations
    use_gamma : bool, optional
        Deprecated. Kept for backward compatibility; see
        `generate_location_dependent_lengthscales`.
    method : str or None, optional
        Lengthscale sampling method: `"folded_normal"`, `"gamma"`, or
        `"sparse_broad"` (alias `"random_vector"`). If None, falls back
        to `use_gamma`.
    n_high : int, optional
        Sparse-broad method only. Number of "broad" locations.
        Default 1.
    high_multiplier : float, optional
        Sparse-broad method only. Multiplier on `base_lengthscale`
        setting the broad-component mean. Default 3.0.

    Returns
    -------
    dict with:
        - 'f_samples': np.ndarray, shape (n_locations, n_orientations)
            Log-rate tuning functions for each location
        - 'lengthscales': np.ndarray, shape (n_locations,)
            Location-dependent lengthscales
        - 'orientations': np.ndarray, shape (n_orientations,)
            Orientation values in radians
        - 'gains': np.ndarray, shape (n_locations,)
            Gain factors applied to each location
    """
    # Create orientation grid
    orientations = np.linspace(-np.pi, np.pi, n_orientations, endpoint=False)

    # Generate location-dependent lengthscales
    lengthscales = generate_location_dependent_lengthscales(
        n_locations, base_lengthscale, lengthscale_variability, random_state,
        use_gamma=use_gamma,
        method=method,
        n_high=n_high,
        high_multiplier=high_multiplier,
    )

    # Sample GP functions for each location
    f_samples = np.zeros((n_locations, n_orientations))
    gains = np.zeros(n_locations)

    for loc in range(n_locations):
        # Build kernel with this location's lengthscale
        K = periodic_rbf_kernel(orientations, lengthscales[loc])

        # Sample GP function (with robust fallback)
        f_loc = sample_gp_function(K, random_state)

        # Apply gain modulation
        gain = np.abs(1.0 + gain_variability * random_state.randn())
        gains[loc] = gain
        f_samples[loc, :] = f_loc * gain

    return {
        'f_samples': f_samples,
        'lengthscales': lengthscales,
        'orientations': orientations,
        'gains': gains
    }


def generate_neuron_population(
    n_neurons: int,
    n_orientations: int,
    n_locations: int,
    base_lengthscale: float,
    lengthscale_variability: float,
    seed: int,
    gain_variability: float = 0.2,
    use_gamma: bool = False,
    method: Optional[str] = None,
    n_high: int = 1,
    high_multiplier: float = 3.0,
) -> List[Dict]:
    """
    Generate a population of neurons with mixed selectivity.

    Each neuron gets its own random lengthscales, creating population
    heterogeneity that is characteristic of prefrontal cortex.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to generate
    n_orientations : int
        Number of orientation bins
    n_locations : int
        Number of spatial locations
    base_lengthscale : float
        Base lengthscale
    lengthscale_variability : float
        Lengthscale variability
    seed : int
        Master random seed
    gain_variability : float
        Gain variability
    use_gamma : bool, optional
        Deprecated. Kept for backward compatibility; see
        `generate_location_dependent_lengthscales`.
    method : str or None, optional
        Lengthscale sampling method: `"folded_normal"`, `"gamma"`, or
        `"sparse_broad"` (alias `"random_vector"`). If None, falls back
        to `use_gamma`.
    n_high : int, optional
        Sparse-broad method only. Number of "broad" locations per
        neuron. Default 1.
    high_multiplier : float, optional
        Sparse-broad method only. Multiplier on `base_lengthscale`
        setting the broad-component mean. Default 3.0.

    Returns
    -------
    population : List[Dict]
        List of neuron dictionaries, each containing tuning curve data
    """
    master_rng = np.random.RandomState(seed)
    population = []

    for neuron_idx in range(n_neurons):
        # Each neuron gets its own random state (derived from master)
        neuron_seed = master_rng.randint(0, 2**31)
        neuron_rng = np.random.RandomState(neuron_seed)

        neuron_data = generate_neuron_tuning_curves(
            n_orientations=n_orientations,
            n_locations=n_locations,
            base_lengthscale=base_lengthscale,
            lengthscale_variability=lengthscale_variability,
            random_state=neuron_rng,
            gain_variability=gain_variability,
            use_gamma=use_gamma,
            method=method,
            n_high=n_high,
            high_multiplier=high_multiplier,
        )
        neuron_data['neuron_idx'] = neuron_idx
        neuron_data['seed'] = neuron_seed

        population.append(neuron_data)

    return population


# ============================================================================
# RESPONSE COMPUTATION
# ============================================================================

def compute_log_rate_tensor(
    f_samples: np.ndarray,
    subset: Tuple[int, ...]
) -> np.ndarray:
    """
    Compute the log-rate tensor G for a subset of locations.

    G(θ_1, ..., θ_l) = Σ_k f_k(θ_k)

    This is the additive combination in log-space, which becomes
    multiplicative in rate-space after exponentiation.

    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions, shape (n_locations, n_orientations)
    subset : Tuple[int, ...]
        Indices of active locations

    Returns
    -------
    G : np.ndarray
        Log-rate tensor, shape (n_theta,) * l where l = len(subset)
    """
    n_theta = f_samples.shape[1]
    l = len(subset)

    # Initialize l-dimensional tensor
    G = np.zeros([n_theta] * l)

    # Add each location's contribution along its dimension
    for dim_idx, loc in enumerate(subset):
        shape = [1] * l
        shape[dim_idx] = n_theta
        G = G + f_samples[loc, :].reshape(shape)

    return G


def compute_pre_normalized_response(
    f_samples: np.ndarray,
    subset: Tuple[int, ...]
) -> np.ndarray:
    """
    Compute pre-normalized response R = exp(G) for a subset.

    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions
    subset : Tuple[int, ...]
        Active location indices

    Returns
    -------
    R_pre : np.ndarray
        Pre-normalized response tensor
    """
    G = compute_log_rate_tensor(f_samples, subset)
    return np.exp(G)


def compute_driving_input(f_samples: np.ndarray) -> np.ndarray:
    """
    Compute driving input g = exp(f) for all locations.

    This is the quantity that enters the DN denominator.

    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions, shape (n_locations, n_orientations)

    Returns
    -------
    g : np.ndarray
        Driving inputs, shape (n_locations, n_orientations)
    """
    return np.exp(f_samples)


def compute_mean_driving_input(f_samples: np.ndarray) -> np.ndarray:
    """
    Compute mean driving input ḡ_j = mean_θ[exp(f_j(θ))] for each location.

    This is what enters the DN denominator in Bays (2014).

    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions, shape (n_locations, n_orientations)

    Returns
    -------
    g_bar : np.ndarray
        Mean driving inputs, shape (n_locations,)
    """
    g = compute_driving_input(f_samples)
    return np.mean(g, axis=1)