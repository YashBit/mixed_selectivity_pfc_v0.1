"""
Maximum Likelihood Decoding with Factorised Marginalisation
===========================================================

Implements the efficient decoder derived in §5 of the paper.

The full Poisson log-likelihood (Eq. 16) simplifies in three steps:

    Step 2  Activity Cap eliminates the rate-sum term:
            L(θ_S) = Σ_i n_i log r^{pre}_i(S,θ)  −  (Σ_i n_i) log D(S,θ)  +  const

    Step 4  Denominator concentrates → penalty absorbed as θ-independent constant.

    Step 5  Multiplicative separability (Assumptions 2.1 & 2.2) decomposes:
            L(θ_S) = Σ_{k∈S} L_k(θ_k)   where  L_k(θ_k) = Σ_i n_i f_{i,k}(θ_k)   [Eq. 23]

    Step 6  Marginalisation over non-cued locations factorises:
            L_M(θ_c) = L_c(θ_c) + Σ_{k≠c} logsumexp(L_k)                          [Eq. 26]

    Estimate:  θ̂_c = argmax_{θ_c} L_M(θ_c)                                         [Eq. 28]

Complexity: O(N · l · n_θ)  vs.  naive O(N · n_θ^l).

Note: Since Σ_{k≠c} logsumexp(L_k) is constant w.r.t. θ_c, the ML point
estimate depends only on L_c(θ_c) = Σ_i n_i f_{i,c}(θ_c).  The non-cued
items affect the estimate *only* through their impact on spike counts via
divisive normalization (reduced rates → noisier data), not through the
decoding computation itself.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.special import logsumexp

from core.encoder.divisive_normalization import dn_pointwise, compute_r_pre_at_config


# =============================================================================
# CORE DECODER  (§5, Eqs. 23 → 26 → 28)
# =============================================================================

def decode(
    spike_counts: np.ndarray,
    f_per_location: List[np.ndarray],
    theta_grid: np.ndarray,
    cued_location: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficient ML decoding with factorised marginalisation.
    Vectorised over a batch of trials.

    The full marginal log-likelihood (Eq. 26) is:

        L_M(θ_c) = L_c(θ_c)  +  Σ_{k≠c} logsumexp(L_k)

    The non-cued logsumexp terms are scalars w.r.t. θ_c, so they vanish
    under argmax. We therefore compute only L_c(θ_c) — one matmul against
    the cued location's tuning matrix — instead of doing l matmuls and
    l−1 logsumexps. The returned L_marginal is L_c(θ_c) alone; it differs
    from the full L_M by a θ-independent additive constant.

    Complexity drops from O(N · l · n_θ) to O(N · n_θ).

    Parameters
    ----------
    spike_counts : np.ndarray
        Observed spike counts. Either shape (N,) for a single trial or
        (n_trials, N) for a batch.
    f_per_location : list of np.ndarray, each shape (N, n_θ)
        f_per_location[k][i, :] = f_{i,k}(θ) — log-rate tuning of neuron i
        at the k-th active location, evaluated on the orientation grid.
        Only f_per_location[cued_location] is read.
    theta_grid : np.ndarray, shape (n_θ,)
        Orientation grid values.
    cued_location : int
        Index (into f_per_location) of the cued location.

    Returns
    -------
    theta_hat : float (single trial) or np.ndarray of shape (n_trials,)
        ML orientation estimate(s) at the cued location.
    L_marginal : np.ndarray
        Shape (n_θ,) for single trial; (n_trials, n_θ) for a batch.
        Contains L_c(θ_c) — see note above on the dropped constant.
    """
    f_k = f_per_location[cued_location]                    # (N, n_θ)

    # Promote 1-D input to (1, N) for the batched matmul, but remember
    # whether to squeeze on return so single-trial callers get a scalar.
    counts_2d = np.atleast_2d(spike_counts)                # (n_trials, N)
    is_single_trial = (spike_counts.ndim == 1)
    n_trials = counts_2d.shape[0]

    # Eq. 23:  L_c(θ_c) = Σ_i n_i · f_{i,c}(θ_c)
    L_marginal = counts_2d @ f_k                           # (n_trials, n_θ)

    # Eq. 28
    theta_hat_idx = np.argmax(L_marginal, axis=1)          # (n_trials,)
    theta_hat = theta_grid[theta_hat_idx]                  # (n_trials,)

    if is_single_trial:
        return float(theta_hat[0]), L_marginal[0]
    return theta_hat, L_marginal


# =============================================================================
# GRID HELPERS
# =============================================================================

def snap_to_grid(theta: float, grid: np.ndarray) -> int:
    """
    Return the grid index closest to theta under circular distance.

    This is used to discretise continuous orientations for both encoding
    (evaluating tuning curves) and ground-truth comparison.
    """
    d = np.abs(grid - theta)
    return int(np.argmin(np.minimum(d, 2 * np.pi - d)))


# =============================================================================
# SINGLE-TRIAL PIPELINE
# =============================================================================

def run_trial(
    f_population: np.ndarray,
    theta_grid: np.ndarray,
    active_locations: Tuple[int, ...],
    true_orientations: np.ndarray,
    cued_index: int,
    gamma: float,
    sigma_sq: float,
    T_d: float,
    rng: np.random.RandomState
) -> Dict:
    """
    Run one encode → spike → decode trial.

    Pipeline
    --------
    1. Snap true orientations to the grid (discrete simulation).
    2. Compute pre-normalised rates at the snapped configuration (Eq. 13).
    3. Apply divisive normalisation (Eq. 6 via dn_pointwise).
    4. Generate Poisson spikes (Def. 4.5).
    5. Decode cued orientation via factorised ML (Eqs. 23–28).
    6. Compute circular error relative to the *snapped* true value,
       so that reported error reflects decoding noise only, not grid
       quantisation.

    Parameters
    ----------
    f_population : np.ndarray, shape (N, n_locations, n_θ)
        Log-rate tuning functions for the full population.
    theta_grid : np.ndarray, shape (n_θ,)
        Orientation grid.
    active_locations : tuple of int, length l
        Which spatial locations carry items.
    true_orientations : np.ndarray, length l
        Continuous true orientations (will be snapped).
    cued_index : int
        Index (into active_locations) of the probed item.
    gamma, sigma_sq, T_d : float
        DN gain, semi-saturation, decoding window.
    rng : np.random.RandomState

    Returns
    -------
    dict with keys: theta_true, theta_est, error, spike_counts, L_marginal
    """
    l = len(active_locations)
    N = f_population.shape[0]
    n_theta = len(theta_grid)

    # --- 1. Build per-location tuning matrices & snap orientations ---
    f_per_loc = [f_population[:, loc, :] for loc in active_locations]   # l × (N, n_θ)
    theta_idx = [snap_to_grid(t, theta_grid) for t in true_orientations]
    theta_true_snapped = theta_grid[theta_idx[cued_index]]

    # --- 2–3. Encode: r_pre → DN → r_post ---
    r_pre = compute_r_pre_at_config(f_population, active_locations, theta_idx)
    r_post = dn_pointwise(r_pre, gamma, sigma_sq)

    # --- 4. Spike ---
    spike_counts = rng.poisson(r_post * T_d)

    # --- 5. Decode ---
    theta_hat, L_marginal = decode(spike_counts, f_per_loc, theta_grid, cued_index)

    # --- 6. Error (against snapped ground truth) ---
    error = circular_error(theta_true_snapped, theta_hat)

    return {
        'theta_true': theta_true_snapped,
        'theta_est': theta_hat,
        'error': error,
        'spike_counts': spike_counts,
        'L_marginal': L_marginal,
    }


# =============================================================================
# BATCH EXPERIMENT
# =============================================================================

def run_experiment(
    f_population: np.ndarray,
    theta_grid: np.ndarray,
    set_sizes: Tuple[int, ...],
    n_locations: int,
    gamma: float,
    sigma_sq: float,
    T_d: float,
    n_trials: int,
    seed: int = 42
) -> Dict[int, Dict]:
    """
    Run decoding experiment across set sizes.

    Parameters
    ----------
    f_population : np.ndarray, shape (N, n_locations, n_θ)
    theta_grid   : np.ndarray, shape (n_θ,)
    set_sizes    : tuple of int
    n_locations  : int — total spatial locations available
    gamma, sigma_sq, T_d : float
    n_trials     : int — trials per set size
    seed         : int

    Returns
    -------
    results : dict[int, dict]
        Keyed by set size.  Each value contains 'errors' (array),
        'circular_std', 'circular_std_deg'.
    """
    rng = np.random.RandomState(seed)
    results = {}

    for l in set_sizes:
        errors = np.empty(n_trials)

        for t in range(n_trials):
            locs = tuple(rng.choice(n_locations, size=l, replace=False))
            orientations = rng.uniform(-np.pi, np.pi, size=l)
            cued = rng.randint(l)

            trial = run_trial(
                f_population, theta_grid, locs, orientations,
                cued, gamma, sigma_sq, T_d, rng
            )
            errors[t] = trial['error']

        std = circular_std(errors)
        results[l] = {
            'errors': errors,
            'circular_std': std,
            'circular_std_deg': np.degrees(std),
        }

    return results