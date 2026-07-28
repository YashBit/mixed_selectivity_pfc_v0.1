"""
Core Gaussian Process Module for Mixed Selectivity Framework

This module contains all tuning-curve generators for the framework:

    * 'gp'   -- Gaussian Process tuning with location-dependent lengthscales
                (the source of mixed selectivity via heterogeneous *widths*).
    * 'vm'   -- Von Mises tuning whose width at each location is drawn from the
                SAME three distributions used for the GP lengthscales
                (folded_normal / gamma / sparse_sharp), with a uniformly drawn
                preferred orientation per (neuron, location).
    * 'bays' -- Homogeneous Bays (2014, Eq. 1) tuning: a single fixed width
                omega, evenly spaced preferred orientations across the
                population, identical curve at every location. This is the
                parametric reference model; it lives here (not in population.py)
                so that all tuning-curve generation has a single home.

The key GP innovation is that different spatial locations have different tuning
widths, creating non-separable tuning: R(theta, L) != f(theta) * g(L).

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

    The kernel is: k(theta_i, theta_j) = exp(-d^2/(2*lambda^2))
    where d is the circular distance: d = min(|theta_i - theta_j|, 2pi - |theta_i - theta_j|)
    """
    n_theta = len(orientations)

    theta_i, theta_j = np.meshgrid(orientations, orientations, indexing='ij')
    dist = np.abs(theta_i - theta_j)
    dist = np.minimum(dist, 2 * np.pi - dist)
    K = np.exp(-dist**2 / (2 * lengthscale**2))

    K += jitter * np.eye(n_theta)
    return K


def sample_gp_function(
    K: np.ndarray,
    random_state: np.random.RandomState
) -> np.ndarray:
    """
    Sample a function from a Gaussian Process with covariance K.

    Uses Cholesky with an eigendecomposition fallback for numerical stability.
    """
    n = K.shape[0]
    z = random_state.randn(n)

    try:
        L = np.linalg.cholesky(K)
        return L @ z
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ (np.sqrt(eigvals) * z)


# ============================================================================
# VON MISES TUNING
# ============================================================================

def sample_vm_function(
    orientations: np.ndarray,
    omega: float,
    random_state: np.random.RandomState,
    preferred_orientation: Optional[float] = None,
) -> np.ndarray:
    """
    Sample a Von Mises (Bays 2014, Eq. 1) log-tuning function at a given width.

        f(theta) = (1/omega) * (cos(theta - phi) - 1)

    so the driving input g(theta) = exp(f(theta)) is a Von Mises bump that peaks
    at 1 at the preferred orientation phi and falls off with a width controlled
    by omega (larger omega => broader tuning).

    Parameters
    ----------
    orientations : np.ndarray, shape (n_theta,)
    omega : float
        Von Mises width at this location. NOTE: this is omega itself, not a GP
        lengthscale and not sqrt(omega).
    random_state : np.random.RandomState
    preferred_orientation : float, optional
        Preferred orientation phi in radians. If None, drawn from Uniform[-pi, pi).
    """
    if preferred_orientation is None:
        preferred_orientation = random_state.uniform(-np.pi, np.pi)
    return (1.0 / omega) * (np.cos(orientations - preferred_orientation) - 1.0)


# ============================================================================
# LENGTHSCALE / WIDTH GENERATION
# ============================================================================

_VALID_METHODS = {"folded_normal", "gamma", "sparse_sharp"}
_METHOD_ALIASES = {"random_vector": "sparse_sharp", "sparse_broad": "sparse_sharp"}


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
    Generate location-dependent lengthscales (or Von Mises widths).

    Three distribution options, selectable via `method`:

    1. Folded-Normal (`"folded_normal"`, default):
        lambda_i = lambda_base * |1 + sigma_lambda * z_i|, z_i ~ N(0, 1)

    2. Gamma (`"gamma"`):
        lambda_i ~ Gamma(mean=base_lengthscale, CV=variability)
        with shape k = 1/variability^2, scale = base_lengthscale * variability^2.

    3. Sparse-Sharp (`"sparse_sharp"`, aliases `"sparse_broad"`, `"random_vector"`):
        A sparse `n_high`-subset of locations is drawn from a Gaussian centred at
        base_lengthscale / high_multiplier (sharper), the rest from a Gaussian
        centred at base_lengthscale (broader background). Both use `variability`
        as std in folded-Normal style.

    All floored at 0.1 for numerical stability.
    """
    if method is None:
        method = "gamma" if use_gamma else "folded_normal"
    method = method.lower()
    method = _METHOD_ALIASES.get(method, method)

    if method == "folded_normal":
        random_factors = 1.0 + variability * random_state.randn(n_locations)
        random_factors = np.abs(random_factors)
        lengthscales = base_lengthscale * random_factors

    elif method == "gamma":
        if variability <= 0:
            lengthscales = np.full(n_locations, base_lengthscale)
        else:
            shape = 1.0 / (variability ** 2)
            scale = base_lengthscale * (variability ** 2)
            lengthscales = random_state.gamma(
                shape=shape, scale=scale, size=n_locations
            )

    elif method == "sparse_sharp":
        n_sharp_eff = int(np.clip(n_high, 0, n_locations))
        sharp_indices = random_state.choice(
            n_locations, size=n_sharp_eff, replace=False
        )
        is_sharp = np.zeros(n_locations, dtype=bool)
        is_sharp[sharp_indices] = True

        z = random_state.randn(n_locations)
        factors = np.abs(1.0 + variability * z)

        sharp_mean = base_lengthscale / high_multiplier
        base_mean = base_lengthscale
        lengthscales = np.where(is_sharp, sharp_mean, base_mean) * factors

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Expected one of: 'folded_normal', 'gamma', 'sparse_sharp' "
            f"(aliases 'sparse_broad', 'random_vector')."
        )

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
    gain_variability: float = 1,
    use_gamma: bool = False,
    method: Optional[str] = None,
    n_high: int = 1,
    high_multiplier: float = 3.0,
    model: str = "gp",
    omega: Optional[float] = None,
    gain_hi: float = 1.0,
    gain_low: float = 0.0,
    preferred_orientation: Optional[float] = None,
) -> Dict:
    """
    Generate tuning curves for a single neuron across all locations.

    Three generative models are supported via `model`:

    'gp'   -- Gaussian Process tuning with location-dependent lengthscales.
              Mixed selectivity arises from heterogeneous tuning *widths*.

    'vm'   -- Von Mises tuning. The width at EACH location is drawn from the
              same three-distribution sampler as the GP lengthscales
              (`method`), centred on the base Von Mises width omega. Each
              location draws an independent uniform preferred orientation phi,
              so the theta-profile shifts across locations (mixed selectivity)
              while the tuning family stays Von Mises.

    'bays' -- Homogeneous Bays (2014, Eq. 1). A single fixed width omega and one
              preferred orientation phi (passed in via `preferred_orientation`
              from the population-level even spacing), giving the SAME curve at
              every location.

    Returns
    -------
    dict with 'f_samples' (n_locations, n_orientations), 'lengthscales',
    'orientations', 'gains', and 'preferred_orientations'.
    """
    model = model.lower()

    orientations = np.linspace(-np.pi, np.pi, n_orientations, endpoint=False)

    f_samples = np.zeros((n_locations, n_orientations))
    gains = np.zeros(n_locations)
    preferred_orientations = None

    if model == "gp":
        # ---- Gaussian Process model (heterogeneous widths) ----
        lengthscales = generate_location_dependent_lengthscales(
            n_locations, base_lengthscale, lengthscale_variability, random_state,
            use_gamma=use_gamma,
            method=method,
            n_high=n_high,
            high_multiplier=high_multiplier,
        )

        for loc in range(n_locations):
            K = periodic_rbf_kernel(orientations, lengthscales[loc])
            f_loc = sample_gp_function(K, random_state)
            gain = np.abs(1.0 + gain_variability * random_state.randn())
            gains[loc] = gain
            f_samples[loc, :] = f_loc * gain

    elif model == "vm":
        # ---- Von Mises model: per-location width from the SAME distributions ----
        vm_omega = base_lengthscale if omega is None else omega

        # Draw a Von Mises width per location from folded_normal / gamma /
        # sparse_sharp (the same sampler used for GP lengthscales).
        widths = generate_location_dependent_lengthscales(
            n_locations, vm_omega, lengthscale_variability, random_state,
            use_gamma=use_gamma,
            method=method,
            n_high=n_high,
            high_multiplier=high_multiplier,
        )

        gains = np.ones(n_locations)
        preferred_orientations = np.zeros(n_locations)
        for loc in range(n_locations):
            # Uniformly assigned "true"/preferred value per location.
            phi = random_state.uniform(-np.pi, np.pi)
            preferred_orientations[loc] = phi
            f_samples[loc, :] = sample_vm_function(
                orientations, widths[loc], random_state,
                preferred_orientation=phi,
            )
        lengthscales = widths

    elif model == "bays":
        # ---- Homogeneous Bays (2014, Eq. 1) ----
        bays_omega = base_lengthscale if omega is None else omega
        phi = (random_state.uniform(-np.pi, np.pi)
               if preferred_orientation is None else float(preferred_orientation))
        curve = (1.0 / bays_omega) * (np.cos(orientations - phi) - 1.0)
        for loc in range(n_locations):
            f_samples[loc, :] = curve
        gains = np.ones(n_locations)
        preferred_orientations = np.full(n_locations, phi)
        lengthscales = np.full(n_locations, bays_omega)

    else:
        raise ValueError(
            f"Unknown model '{model}'. Expected 'gp', 'vm', or 'bays'."
        )

    return {
        'f_samples': f_samples,
        'lengthscales': lengthscales,
        'orientations': orientations,
        'gains': gains,
        'preferred_orientations': preferred_orientations,
    }


def generate_neuron_population(
    n_neurons: int,
    n_orientations: int,
    n_locations: int,
    base_lengthscale: float,
    lengthscale_variability: float,
    seed: int,
    gain_variability: float = 0,
    use_gamma: bool = False,
    method: Optional[str] = None,
    n_high: int = 1,
    high_multiplier: float = 3.0,
    model: str = "gp",
    omega: Optional[float] = None,
    gain_hi: float = 1.0,
    gain_low: float = 0.0,
) -> List[Dict]:
    """
    Generate a population of neurons under the chosen `model`.

    For 'gp' and 'vm', each neuron gets its own random widths / preferred
    orientations (population heterogeneity). For 'bays', preferred orientations
    are assigned deterministically and evenly across the population (Bays 2014,
    Eq. 1), giving a homogeneous reference population.
    """
    model = model.lower()
    master_rng = np.random.RandomState(seed)
    population = []

    # Bays: evenly spaced preferred orientations across the population.
    bays_phis = (np.linspace(-np.pi, np.pi, n_neurons, endpoint=False)
                 if model == "bays" else None)

    for neuron_idx in range(n_neurons):
        neuron_seed = master_rng.randint(0, 2**31)
        neuron_rng = np.random.RandomState(neuron_seed)

        extra = {}
        if model == "bays":
            extra['preferred_orientation'] = bays_phis[neuron_idx]

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
            model=model,
            omega=omega,
            gain_hi=gain_hi,
            gain_low=gain_low,
            **extra,
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
    """Log-rate tensor G(theta_1, ..., theta_l) = sum_k f_k(theta_k)."""
    n_theta = f_samples.shape[1]
    l = len(subset)
    G = np.zeros([n_theta] * l)
    for dim_idx, loc in enumerate(subset):
        shape = [1] * l
        shape[dim_idx] = n_theta
        G = G + f_samples[loc, :].reshape(shape)
    return G


def compute_pre_normalized_response(
    f_samples: np.ndarray,
    subset: Tuple[int, ...]
) -> np.ndarray:
    """Pre-normalized response R = exp(G) for a subset."""
    G = compute_log_rate_tensor(f_samples, subset)
    return np.exp(G)


def compute_driving_input(f_samples: np.ndarray) -> np.ndarray:
    """Driving input g = exp(f) for all locations."""
    return np.exp(f_samples)


def compute_mean_driving_input(f_samples: np.ndarray) -> np.ndarray:
    """Mean driving input g_bar_j = mean_theta[exp(f_j(theta))] per location."""
    g = compute_driving_input(f_samples)
    return np.mean(g, axis=1)