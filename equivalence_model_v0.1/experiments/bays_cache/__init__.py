"""
experiments.bays_cache
======================

Centralised generation and loading of the Bays (2014) reference cache.

Building the cache (from the command line):

    python -m experiments.bays_cache.build_bays_cache
    python -m experiments.bays_cache.build_bays_cache --M 100 --N 10000
    python -m experiments.bays_cache.build_bays_cache --list

Loading the cache (from a notebook or script):

    from experiments.bays_cache import load_bays_cache
    errors, meta = load_bays_cache(M=100)

See build_bays_cache.py for the Bays model implementation
(Eqs. 1, 2/3, 4, 5–8 directly from the paper) and loader.py for the
filename schema and load-time validation.
"""

from .loader import load_bays_cache, list_available_caches

__all__ = ['load_bays_cache', 'list_available_caches']