"""
loader.py
=========

Single point of access for the Bays (2014) reference cache built by
build_bays_cache.py.

Notebooks should never hardcode paths. Instead:

    from experiments.bays_cache import load_bays_cache
    errors, meta = load_bays_cache(M=100)

    # errors is dict[int, np.ndarray]:  {1: errs_N1, 2: errs_N2, ...}
    # meta   is dict:                   the parameters the cache was built at

If the requested cache doesn't exist on disk, the loader raises a clear
FileNotFoundError telling you exactly how to build it.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# Keep these in sync with build_bays_cache.py. If you change the canonical
# Bays parameters, change them in BOTH files.
_DEFAULT_OMEGA       = 0.52
_DEFAULT_GAMMA_TOTAL = 119.0
_DEFAULT_SEED        = 12345
_DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent.parent / 'cached_data'
)

def _cache_filename(M, N_trials, omega, gamma_total, seed):
    """Must match the schema in build_bays_cache.cache_filename exactly."""
    return (
        f'bays_cache_omega{omega:g}_gamma{gamma_total:g}'
        f'_M{M}_N{N_trials}_seed{seed}.npz'
    )


def _list_available_caches(cache_dir):
    """
    Scan cache_dir for files matching the canonical schema and return
    a sorted list of (M, N_trials, filename) tuples.

    Used to make the FileNotFoundError message informative.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []

    found = []
    for path in cache_dir.glob('bays_cache_*.npz'):
        # Parse "bays_cache_omega0.52_gamma119_M100_N10000_seed12345.npz"
        try:
            stem = path.stem  # strip .npz
            parts = stem.split('_')
            tokens = {}
            for p in parts:
                # token e.g. "omega0.52", "M100", "N10000", "seed12345"
                for key in ('omega', 'gamma', 'M', 'N', 'seed'):
                    if p.startswith(key) and p != 'bays' and p != 'cache':
                        try:
                            tokens[key] = p[len(key):]
                        except ValueError:
                            pass
                        break
            if 'M' in tokens and 'N' in tokens:
                M_i = int(tokens['M'])
                N_i = int(tokens['N'])
                found.append((M_i, N_i, path.name))
        except (ValueError, KeyError, IndexError):
            # Filename doesn't match schema — ignore.
            continue
    return sorted(found)


def load_bays_cache(
    M: int,
    N_trials: int = 10000,
    omega: float = _DEFAULT_OMEGA,
    gamma_total: float = _DEFAULT_GAMMA_TOTAL,
    seed: int = _DEFAULT_SEED,
    cache_dir=None,
) -> Tuple[Dict[int, np.ndarray], dict]:
    """
    Load a Bays cache built by experiments/bays_cache/build_bays_cache.py.

    Parameters
    ----------
    M : int
        Number of neurons the cache was built at. Must match an existing
        cache file on disk.
    N_trials : int, default 10000
        Trials per set size.
    omega, gamma_total, seed : float, float, int
        Bays generating parameters. Defaults match build_bays_cache.py.
    cache_dir : Path or str, optional
        Where to look. Default is the project's shared cache directory.

    Returns
    -------
    errors_by_N : dict[int, np.ndarray]
        Mapping from set size to a 1-D array of signed circular errors.
    meta : dict
        The metadata dict that was stored alongside the errors. Use this
        to assert-check that the cache matches your expectations.

    Raises
    ------
    FileNotFoundError
        If no cache at (M, N_trials, omega, gamma_total, seed) exists.
        The error message lists what build_bays_cache.py command to run
        and what caches DO exist in the directory.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)

    fname = _cache_filename(M, N_trials, omega, gamma_total, seed)
    cache_path = cache_dir / fname

    if not cache_path.exists():
        # Build a helpful error message rather than a bare FileNotFoundError.
        available = _list_available_caches(cache_dir)
        if available:
            avail_str = '\n  '.join(
                f'M={m}, N={n}  ({fn})' for m, n, fn in available
            )
            avail_block = f'\nAvailable caches in this directory:\n  {avail_str}'
        else:
            avail_block = (
                f'\nNo Bays caches found in {cache_dir}. '
                f'Run the builder to create some.'
            )

        raise FileNotFoundError(
            f'No Bays cache at M={M}, N_trials={N_trials}, '
            f'omega={omega}, gamma_total={gamma_total}, seed={seed}.\n'
            f'Expected at: {cache_path}\n\n'
            f'To build it, run:\n'
            f'    python -m experiments.bays_cache.build_bays_cache '
            f'--M {M} --N {N_trials}\n'
            f'{avail_block}'
        )

    # Load the .npz
    with np.load(cache_path, allow_pickle=False) as npz:
        # Reconstruct the meta dict from its pickled bytes
        if '_meta_pickled' in npz.files:
            meta = pickle.loads(npz['_meta_pickled'].tobytes())
        else:
            # Older caches without embedded meta — synthesize from filename
            meta = {
                'omega':       omega,
                'gamma_total': gamma_total,
                'M':           M,
                'N_trials':    N_trials,
                'seed':        seed,
                'set_sizes':   None,  # caller must determine from error keys
            }

        # Pull out every errors_N{k} entry
        errors_by_N = {}
        for key in npz.files:
            if key.startswith('errors_N'):
                N = int(key[len('errors_N'):])
                errors_by_N[N] = npz[key].copy()

    if not errors_by_N:
        raise RuntimeError(
            f'Cache at {cache_path} contained no errors_N* arrays. '
            f'It may be corrupt or built by an incompatible version.'
        )

    return errors_by_N, meta


def list_available_caches(cache_dir=None):
    """
    Convenience function for notebooks that want to know what Bays caches
    are currently on disk. Returns a list of (M, N_trials, filename).
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    return _list_available_caches(cache_dir)