"""
Poisson Spiking Noise - Core Module

=============================================================================
THEORETICAL FOUNDATION
=============================================================================

THE POISSON DISTRIBUTION
------------------------
For a neuron with firing rate r (Hz) observed for time T_d (seconds):
    - Expected spike count: λ = r × T_d
    - Actual count: n ~ Poisson(λ)
    - P(n | λ) = (λⁿ × e^{-λ}) / n!

KEY PROPERTIES:
    - E[n] = λ              (mean)
    - Var[n] = λ            (variance = mean)
    - SNR = √λ              (signal-to-noise ratio)
    - CV = 1/√λ             (coefficient of variation)
    - Fano Factor = 1       (variance/mean ratio)

THE CAPACITY LIMIT CONNECTION:
    Under DN: rate ∝ 1/l  →  λ ∝ 1/l  →  SNR ∝ 1/√l  →  Error ∝ √l

=============================================================================
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PoissonStats:
    """Statistics from Poisson spike generation."""
    mean: float
    variance: float
    fano_factor: float
    snr: float
    cv: float  # coefficient of variation


# =============================================================================
# CORE SPIKE GENERATION
# =============================================================================

def generate_spikes(
    rates: np.ndarray,
    T_d: float,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Generate spike counts from Poisson distribution.
    
    Parameters
    ----------
    rates : np.ndarray
        Firing rates in Hz, shape (N,)
    T_d : float
        Decoding time window in seconds
    rng : np.random.RandomState, optional
        Random number generator for reproducibility
        
    Returns
    -------
    counts : np.ndarray
        Spike counts, shape (N,), dtype int
    """
    if rng is None:
        rng = np.random.RandomState()
    
    rates = np.maximum(np.asarray(rates, dtype=np.float64), 0.0)
    expected_counts = rates * T_d
    
    return rng.poisson(expected_counts)


def generate_spikes_multi_trial(
    rates: np.ndarray,
    T_d: float,
    n_trials: int,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Generate spike counts for multiple trials (vectorized).
    
    Parameters
    ----------
    rates : np.ndarray
        Firing rates in Hz, shape (N,)
    T_d : float
        Decoding time window in seconds
    n_trials : int
        Number of independent trials
    rng : np.random.RandomState, optional
        Random number generator
        
    Returns
    -------
    counts : np.ndarray
        Spike counts, shape (n_trials, N), dtype int
    """
    if rng is None:
        rng = np.random.RandomState()
    
    rates = np.maximum(np.asarray(rates, dtype=np.float64), 0.0)
    expected_counts = rates * T_d
    
    return rng.poisson(expected_counts, size=(n_trials, len(rates)))


# =============================================================================
# THEORETICAL COMPUTATIONS
# =============================================================================

def compute_theoretical_snr(lambda_expected: float) -> float:
    """
    Compute theoretical SNR for Poisson process.
    
    SNR = E[n] / Std[n] = λ / √λ = √λ
    """
    return np.sqrt(max(lambda_expected, 1e-10))


def compute_theoretical_cv(lambda_expected: float) -> float:
    """
    Compute theoretical coefficient of variation.
    
    CV = Std[n] / E[n] = √λ / λ = 1/√λ
    """
    return 1.0 / np.sqrt(max(lambda_expected, 1e-10))


def compute_expected_lambda(
    gamma: float,
    N: int,
    l: int,
    T_d: float
) -> float:
    """
    Compute expected spike count under DN.
    
    Under DN: total rate = γN, per-item rate = γN/l
    Expected spikes per item = (γN/l) × T_d
    
    Parameters
    ----------
    gamma : float
        Gain constant (Hz per neuron)
    N : int
        Number of neurons
    l : int
        Set size (number of items)
    T_d : float
        Decoding time window (seconds)
        
    Returns
    -------
    lambda_expected : float
        Expected spike count
    """
    per_item_rate = gamma * N / l
    return per_item_rate * T_d


# =============================================================================
# EMPIRICAL STATISTICS
# =============================================================================

def compute_empirical_stats(spike_counts: np.ndarray) -> PoissonStats:
    """
    Compute empirical statistics from spike count array.
    
    Parameters
    ----------
    spike_counts : np.ndarray
        Spike counts, shape (n_trials,) or (n_trials, N)
        
    Returns
    -------
    stats : PoissonStats
        Empirical statistics
    """
    # Flatten if multi-neuron
    counts = spike_counts.flatten() if spike_counts.ndim > 1 else spike_counts
    
    mean = np.mean(counts)
    variance = np.var(counts, ddof=1)
    
    # Avoid division by zero
    mean_safe = max(mean, 1e-10)
    
    return PoissonStats(
        mean=mean,
        variance=variance,
        fano_factor=variance / mean_safe,
        snr=mean / np.sqrt(variance) if variance > 0 else 0.0,
        cv=np.sqrt(variance) / mean_safe
    )


def compute_population_stats(
    spike_counts: np.ndarray,
    axis: int = 0
) -> Dict[str, np.ndarray]:
    """
    Compute statistics across trials for each neuron.
    
    Parameters
    ----------
    spike_counts : np.ndarray
        Shape (n_trials, N)
    axis : int
        Axis over which to compute (0 = across trials)
        
    Returns
    -------
    stats : dict
        Dictionary with per-neuron statistics
    """
    means = np.mean(spike_counts, axis=axis)
    variances = np.var(spike_counts, axis=axis, ddof=1)
    
    means_safe = np.maximum(means, 1e-10)
    
    return {
        'means': means,
        'variances': variances,
        'fano_factors': variances / means_safe,
        'snr': means / np.sqrt(np.maximum(variances, 1e-10)),
        'cv': np.sqrt(variances) / means_safe
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_heterogeneous_rates(
    total_rate: float,
    n_neurons: int,
    heterogeneity: float = 2.0,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Create heterogeneous firing rates that sum to total_rate.
    
    Uses gamma distribution to create realistic rate heterogeneity.
    
    Parameters
    ----------
    total_rate : float
        Total firing rate (sum across neurons)
    n_neurons : int
        Number of neurons
    heterogeneity : float
        Shape parameter (higher = more homogeneous)
    rng : np.random.RandomState, optional
        Random number generator
        
    Returns
    -------
    rates : np.ndarray
        Firing rates, shape (n_neurons,), sum = total_rate
    """
    if rng is None:
        rng = np.random.RandomState()
    
    mean_rate = total_rate / n_neurons
    scale = mean_rate / heterogeneity
    
    rates = rng.gamma(heterogeneity, scale, size=n_neurons)
    
    # Normalize to enforce exact total
    rates = rates * (total_rate / np.sum(rates))
    
    return rates