#!/usr/bin/env python3
"""
build_bays_cache.py
===================

Standalone builder for the Bays (2014) reference cache.

This script is the canonical regeneration tool for the cached Bays
datasets that every figure-2 / figure-3 / ... notebook reads from. It
implements the Bays parametric population coding model directly from
the equations in Bays (2014) Materials and Methods — Eqs. 1, 2/3, 4,
5–8 — with no dependencies on the GP framework in core.encoder.* or
core.decoder.*. The only external library is NumPy.

Why this exists
---------------

Previously each notebook built its own Bays reference inline, which
meant (a) the reference parameters drifted across files, (b) the cache
filename did not encode M, so different-M runs would silently overwrite
each other, and (c) the comparison reference was being regenerated on
every fresh kernel rather than loaded from disk. This script + the
companion loader.py centralise that work: one canonical Bays
implementation, one canonical filename schema, one canonical location
on disk.

Filename schema
---------------

    bays_cache_omega{OMEGA}_gamma{GAMMA_TOTAL}_M{M}_N{N_TRIALS}_seed{SEED}.npz

    e.g. bays_cache_omega0.52_gamma119_M100_N10000_seed12345.npz

Cache content
-------------

Each .npz contains:
    errors_N{set_size}  ndarray of signed circular errors, shape (N_TRIALS,)
                        one such array per entry in SET_SIZES
    meta                pickled dict carrying every parameter used so the
                        notebook can validate at load time

Usage
-----

    # Build every (M, N) configuration in CONFIGURATIONS, skipping any
    # cache that already exists on disk:
    python build_bays_cache.py

    # Force regeneration even if files exist:
    python build_bays_cache.py --force

    # Build just one configuration:
    python build_bays_cache.py --M 100 --N 10000
    python build_bays_cache.py --M 10000           # uses default N

    # List configurations without building anything:
    python build_bays_cache.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np


# =============================================================================
## PARAMETERS — edit here to change the canonical Bays reference
# =============================================================================

# --- Bays model parameters (group-mean ML fit from Bays 2014 Fig 2b) ---
OMEGA          = 0.52                  # tuning width  (Bays Eq. 1)
GAMMA_TOTAL    = 119.0                 # total population gain, Hz  (Bays Eq. 3)
T_D            = 0.1                   # decoding window, seconds  (Bays Eq. 4)
SET_SIZES      = list(range(1, 9))     # set sizes to sample: 1,2,3,4,5,6,7,8
                                       # (was [1, 2, 4, 8]; widened to the full
                                       #  1-8 sweep. NOTE: the filename schema does
                                       #  NOT encode set sizes, so rebuild existing
                                       #  caches with --force to overwrite the old
                                       #  4-size file with this 8-size one.)
SEED           = 12345                 # base RNG seed
N_DECODE_GRID  = 1000                  # ML decoder evaluation grid points

# --- Configurations to build by default ---
# Trials are held fixed at 10000 across all M values. Sampling noise on
# kurtosis at this trial count is well under 0.1 units, comfortably below
# the effect sizes we care about. M is what varies.
CONFIGURATIONS = [
    {'M': 100,   'N_trials': 10000},   # primary reference for every fit / comparison
    {'M': 1000,  'N_trials': 10000},   # M-scaling diagnostic
    {'M': 10000, 'N_trials': 10000},   # large-M asymptotic reference (~10 minutes to build)
]

# --- Output directory ---
# Hard-coded to the shared project location so every notebook can find
# the same caches without any path negotiation. If you reorganise the
# project layout, this is the one line that needs updating here.
DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent.parent / 'cached_data'
)

# =============================================================================
## BAYS (2014) PARAMETRIC MODEL — Eqs. 1, 2/3, 4, 5–8
# =============================================================================
# This is a verbatim NumPy port of the equations in Bays's Methods.
# No imports from core/. The script is intended to be readable side by
# side with the paper — every block is annotated with its equation number.

def bays_build_population(M):
    """
    M neurons with preferred orientations evenly spaced on [-pi, pi).

    Bays does not store tuning curves as a discrete grid: the parametric
    form (Eq. 1) lets us evaluate at any real-valued orientation, so we
    only need the M preferred orientations phi_i.
    """
    return np.linspace(-np.pi, np.pi, M, endpoint=False)


def bays_driving_input(phi, theta_per_loc, omega):
    """
    Bays Eq. 1 evaluated at continuous orientations.

        f_i(theta_k) = exp( (1/omega) * (cos(phi_i - theta_k) - 1) )

    phi             : (M,)  preferred orientations
    theta_per_loc   : (l,)  TRUE continuous orientations at each item
    omega           : float tuning width

    Returns f, shape (l, M).
    """
    diff = phi[None, :] - np.asarray(theta_per_loc)[:, None]
    return np.exp((1.0 / omega) * (np.cos(diff) - 1.0))


def bays_divisive_normalisation(f, gamma_total):
    """
    Bays Eq. 2 / Eq. 3 with alpha_j = 1 for every active item
    (Experiment 1 convention). The DN denominator is the sum over ALL
    M*l neurons in ALL l subpopulations — no semi-saturation term.

        r_ij = gamma_total * f_ij / sum(f)

    Total post-DN population activity therefore equals gamma_total.

    f             : (l, M)
    gamma_total   : float (Hz, total population gain)

    Returns rates, shape (l, M).
    """
    return gamma_total * f / f.sum()


def bays_poisson_spikes(rates, T_d, rng):
    """Bays Eq. 4: independent Poisson, mean = rates * T_d."""
    return rng.poisson(rates * T_d)


def bays_run_trials_vectorised(n_trials, set_size, omega, gamma_total, T_d,
                               M=100, rng=None, n_decode_grid=1000):
    """
    Vectorised batch trial runner — Bays's full Experiment 1 pipeline,
    n_trials of (Eqs. 1, 3, 4, 8) in a single NumPy broadcast.

    Per-trial pipeline (mathematically unchanged):
        1. Sample continuous true orientations theta_k ~ Uniform[-pi, pi)
           for each of l = set_size items.
        2. Sample cued item uniformly at random from {0, ..., l-1}.
        3. Eq. 1: f_ij(theta) = exp((1/omega)(cos(phi_i - theta) - 1))
        4. Eq. 3: rates = gamma_total * f / sum(f) over the (l, M) matrix.
        5. Eq. 4: independent Poisson spikes.
        6. Eq. 8: ML decoder over the cued subpopulation only, on an
           n_decode_grid-point grid with random tie-breaking.

    The decoder picks the argmax of  Σ_i n_i cos(phi_i - theta)  over
    n_decode_grid evenly-spaced candidate orientations. Ties are broken
    uniformly at random per Bays's explicit specification.

    Returns
    -------
    errors : np.ndarray, shape (n_trials,)
        Signed circular errors in [-pi, pi).
    """
    if rng is None:
        rng = np.random.RandomState()

    l = set_size
    phi = bays_build_population(M)                                  # (M,)

    # ---- 1-2. Sample stimuli and cued indices ----
    theta_true = rng.uniform(-np.pi, np.pi, size=(n_trials, l))     # (T, l)
    cued = rng.randint(l, size=n_trials)                            # (T,)

    # ---- 3. Eq. 1: driving inputs at the continuous true orientations ----
    diff = phi[None, None, :] - theta_true[:, :, None]              # (T, l, M)
    f = np.exp((1.0 / omega) * (np.cos(diff) - 1.0))                # (T, l, M)

    # ---- 4. Eq. 3: DN with denominator summed over the (l, M) matrix ----
    denom = f.sum(axis=(1, 2))                                      # (T,)
    rates = gamma_total * f / denom[:, None, None]                  # (T, l, M)

    # ---- 5. Eq. 4: independent Poisson spike counts ----
    counts = rng.poisson(rates * T_d)                               # (T, l, M)

    # ---- 6. Pull out the cued subpopulation ----
    counts_probed = counts[np.arange(n_trials), cued, :]            # (T, M)

    # ---- 7. Eq. 8: ML decoder on an n_decode_grid-point evaluation grid ----
    theta_eval = np.linspace(-np.pi, np.pi, n_decode_grid, endpoint=False)
    cos_grid = np.cos(phi[:, None] - theta_eval[None, :])           # (M, n_grid)
    objective = counts_probed @ cos_grid                            # (T, n_grid)

    # Random tie-breaking across tied argmax indices (Bays explicit spec).
    max_vals = objective.max(axis=1, keepdims=True)                 # (T, 1)
    tied_mask = objective >= max_vals - 1e-12                       # (T, n_grid)
    keys = rng.random(objective.shape) * tied_mask
    chosen_idx = keys.argmax(axis=1)                                # (T,)
    theta_hat = theta_eval[chosen_idx]                              # (T,)

    # ---- 8. Signed circular error in [-pi, pi) ----
    d = theta_hat - theta_true[np.arange(n_trials), cued]
    return (d + np.pi) % (2.0 * np.pi) - np.pi


# =============================================================================
## CACHE I/O — filename schema and write helpers
# =============================================================================

def cache_filename(M, N_trials, omega=OMEGA, gamma_total=GAMMA_TOTAL, seed=SEED):
    """
    Canonical cache filename. Every parameter that affects the content
    is encoded in the filename so different-parameter caches cannot
    collide on disk.
    """
    # int-ify the floats so the filename is stable across float formatting:
    # omega 0.52 → '0.52', gamma_total 119.0 → '119'
    return (
        f'bays_cache_omega{omega:g}_gamma{gamma_total:g}'
        f'_M{M}_N{N_trials}_seed{seed}.npz'
    )


def write_cache(cache_path, errors_by_N, meta):
    """
    Save errors and metadata to a single .npz file.

    Layout:
        errors_N1, errors_N2, errors_N4, errors_N8, ...   (one per set size)
        meta  (a pickled dict; use load_bays_cache to unpack)
    """
    # Pickle the meta dict to a bytes array so np.savez can carry it
    meta_bytes = np.frombuffer(pickle.dumps(meta), dtype=np.uint8)

    save_kwargs = {f'errors_N{N}': errors_by_N[N] for N in errors_by_N}
    save_kwargs['_meta_pickled'] = meta_bytes
    np.savez(cache_path, **save_kwargs)


def quick_summary(errors):
    """One-line summary stats per set size — for the build log only."""
    z = np.exp(1j * errors)
    m1, m2 = np.mean(z), np.mean(z * z)
    rho1, rho2 = np.abs(m1), np.abs(m2)
    variance = -2.0 * np.log(max(rho1, 1e-15))
    V = 1.0 - rho1
    if V > 1e-10:
        kurt = (rho2 * np.cos(np.angle(m2) - 2 * np.angle(m1)) - rho1**4) / (V ** 2)
    else:
        kurt = 0.0
    return variance, kurt


# =============================================================================
## BUILD ORCHESTRATOR
# =============================================================================

def build_one_cache(M, N_trials, cache_dir, force=False):
    """
    Build (or skip) a single Bays cache at the given M and N_trials.

    Returns
    -------
    status : str
        'built', 'skipped' (file exists, --force not set), or 'failed'.
    cache_path : Path
        Where the file lives (or would live).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_filename(M, N_trials)

    if cache_path.exists() and not force:
        return 'skipped', cache_path

    print(f'\n--- Building M={M}, N_trials={N_trials} ---')
    print(f'    Path: {cache_path}')
    t_total = time.time()

    errors_by_N = {}
    for N in SET_SIZES:
        # Per-set-size seed offset so each (M, N_trials, N) is independent
        # but reproducible. We anchor on SEED, add M*100 to separate caches
        # at different M, and add N to separate set sizes.
        rng_seed = SEED + M * 100 + N
        rng = np.random.RandomState(rng_seed)

        t0 = time.time()
        errors = bays_run_trials_vectorised(
            n_trials=N_trials,
            set_size=N,
            omega=OMEGA,
            gamma_total=GAMMA_TOTAL,
            T_d=T_D,
            M=M,
            rng=rng,
            n_decode_grid=N_DECODE_GRID,
        )
        var, kurt = quick_summary(errors)
        print(f'    N={N}: var={var:.4f}  kurt={kurt:>7.3f}  '
              f'({time.time() - t0:.1f}s)')
        errors_by_N[N] = errors

    meta = {
        'omega':            OMEGA,
        'gamma_total':      GAMMA_TOTAL,
        'T_d':              T_D,
        'M':                M,
        'N_trials':         N_trials,
        'set_sizes':        list(SET_SIZES),
        'seed':             SEED,
        'n_decode_grid':    N_DECODE_GRID,
        'cache_filename':   cache_path.name,
        'builder_version':  '2.0',
    }
    write_cache(cache_path, errors_by_N, meta)

    print(f'    Total time: {time.time() - t_total:.1f}s')
    return 'built', cache_path


def main():
    parser = argparse.ArgumentParser(
        description='Build the Bays (2014) reference cache.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--M', type=int, default=None,
                        help='Build only this M (default: all in CONFIGURATIONS).')
    parser.add_argument('--N', type=int, default=10000,
                        help='Trials per set size when --M is given (default: 10000).')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild caches that already exist on disk.')
    parser.add_argument('--list', action='store_true',
                        help='List configurations and existing caches, then exit.')
    parser.add_argument('--out-dir', type=str, default=str(DEFAULT_CACHE_DIR),
                        help=f'Output directory (default: {DEFAULT_CACHE_DIR}).')
    args = parser.parse_args()

    cache_dir = Path(args.out_dir).expanduser().resolve()

    # ---- --list path: show what's planned and what exists ----
    if args.list:
        print(f'Cache directory: {cache_dir}')
        print(f'Cache directory exists: {cache_dir.exists()}\n')
        print('Configurations in this script:')
        for cfg in CONFIGURATIONS:
            fn = cache_filename(cfg['M'], cfg['N_trials'])
            on_disk = (cache_dir / fn).exists()
            tag = '[exists]' if on_disk else '[missing]'
            print(f'  {tag}  M={cfg["M"]:>5}, N={cfg["N_trials"]:>5}   {fn}')

        print('\nOther Bays caches on disk (any M, N):')
        if cache_dir.exists():
            found = sorted(cache_dir.glob('bays_cache_*.npz'))
            cfg_filenames = {cache_filename(c['M'], c['N_trials'])
                             for c in CONFIGURATIONS}
            extras = [p for p in found if p.name not in cfg_filenames]
            if extras:
                for p in extras:
                    print(f'  {p.name}')
            else:
                print('  (none beyond the configured set)')
        return

    # ---- Build path ----
    print(f'Bays cache builder')
    print(f'Output directory: {cache_dir}')
    print(f'Force rebuild:    {args.force}')

    if args.M is not None:
        to_build = [{'M': args.M, 'N_trials': args.N}]
        print(f'Single config:    M={args.M}, N={args.N}')
    else:
        to_build = CONFIGURATIONS
        print(f'Configurations:   {len(to_build)}')

    t_grand = time.time()
    built, skipped = [], []
    for cfg in to_build:
        status, path = build_one_cache(
            cfg['M'], cfg['N_trials'], cache_dir, force=args.force
        )
        if status == 'built':
            built.append(path)
        elif status == 'skipped':
            print(f'\n--- Skipping M={cfg["M"]}, N={cfg["N_trials"]} '
                  f'(cache exists, use --force to rebuild) ---')
            skipped.append(path)

    print(f'\n=== Summary ===')
    print(f'Built:   {len(built)}')
    for p in built:
        print(f'  {p.name}')
    print(f'Skipped: {len(skipped)}')
    for p in skipped:
        print(f'  {p.name}')
    print(f'Total wall time: {time.time() - t_grand:.1f}s')


if __name__ == '__main__':
    main()