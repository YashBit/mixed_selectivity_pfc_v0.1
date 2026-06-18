"""
Population Class
================

Thin OOP orchestration layer over the existing core modules:
    - core.encoder.gaussian_process        (tuning curve generation)
    - core.encoder.divisive_normalization  (DN encoding)
    - core.encoder.poisson_spiking         (spike generation)
    - core.decoder.ml_decoder              (ML decoding)

Usage:
    pop = Population(M=1000, n_theta=256, omega=0.5, tuning_type='bays')
    pop = Population(M=1000, n_theta=256, omega=0.5, tuning_type='gp', n_locations=8)
    pop = Population(M=1000, n_theta=256, omega=0.5, tuning_type='gp',
                     n_locations=8, lengthscale_variability=0.5, method='gamma')
    pop = Population(M=1000, n_theta=256, omega=0.5, tuning_type='gp',
                     n_locations=8, lengthscale_variability=0.3,
                     method='sparse_broad', n_high=1, high_multiplier=3.0)

    counts = pop.encode(active_locations, theta_indices, gamma, T_d)
    theta_hat, L_marg = pop.decode(counts, active_locations, cued_location)
    errors = pop.run_trials(n_trials, gamma, T_d, set_size=4)

    # Bays Fig 3 / cued recall:
    errors_cued = pop.run_trials(
        n_trials, gamma, T_d, set_size=4,
        active_locations=tuple(range(4)),        # deterministic ordering
        probe_cued=True,                         # always probe loc 0
        alpha=2.3,                               # recruitment boost
        recruited_mask=mask,                     # which neurons to boost
    )

encode() and decode() are the single source of truth for the per-trial
pipeline at discrete grid orientations. run_trials() is a vectorised
batch runner that draws CONTINUOUS true orientations from
Uniform[-pi, pi) and linearly interpolates the stored tuning curves at
those off-grid points -- this avoids the error-distribution quantization
artefact discussed in the bays_figure_2_notebook diagnostic block
(uniq_GP / zero_GP / min|e|_GP).
"""

import numpy as np
from typing import Tuple, Optional

from core.encoder.gaussian_process import generate_neuron_population
from core.encoder.divisive_normalization import dn_pointwise, compute_r_pre_at_config
from core.encoder.poisson_spike import generate_spikes, generate_spikes_multi_trial
from core.decoder.ml_decoder import decode as ml_decode, circular_error


class Population:
    """
    A population of N neurons with tuning curves over L spatial locations
    and n_theta orientation bins.

    Parameters
    ----------
    M : int
        Number of neurons.
    n_theta : int
        Number of orientation bins.
    omega : float
        Tuning width parameter (Bays's omega). For GP mode, converted
        to lengthscale via lambda = sqrt(omega).
    tuning_type : str, 'bays' or 'gp'
        'bays' -- Bays (2014) Eq. 1 parametric tuning, homogeneous,
                 evenly spaced preferred orientations, single location.
        'gp'   -- Gaussian Process tuning curves with location-dependent
                 lengthscales.
    n_locations : int
        Number of spatial locations (only used when tuning_type='gp').
    seed : int
        Random seed (only used when tuning_type='gp').
    lengthscale_variability : float
        sigma_lambda for heterogeneous tuning widths across locations
        (only used when tuning_type='gp', 0 = homogeneous).
    gain_variability : float
        Amplitude variability across locations
        (only used when tuning_type='gp', 0 = homogeneous).
    method : str, optional
        Lengthscale sampling method for GP mode:
            'folded_normal'  -- lambda_i = lambda_base * |1 + sigma_lambda * z_i|, z_i ~ N(0,1)
            'gamma'          -- lambda_i ~ Gamma with mean=lambda_base, CV=sigma_lambda
            'sparse_broad'   -- k-hot two-component scheme: `n_high` locations
                               drawn from a broad-mean Gaussian (mean
                               = lambda_base * high_multiplier), the rest from a
                               sharp-mean Gaussian (mean = lambda_base). Both use
                               sigma_lambda in folded-Normal style. Alias:
                               'random_vector' (legacy).
        Default 'folded_normal' preserves prior behaviour. Only used
        when tuning_type='gp'.
    n_high : int, optional
        Sparse-broad method only. Number of "broad" (wide-tuning)
        locations per neuron. Default 1 (one-hot indicator over
        locations). Ignored for other methods.
    high_multiplier : float, optional
        Sparse-broad method only. Multiplier on `base_lengthscale`
        setting the broad-component mean. Default 3.0. Ignored for
        other methods.

    Attributes
    ----------
    f : np.ndarray, shape (N, L, n_theta)
        Log-rate tuning functions.
    g : np.ndarray, shape (N, L, n_theta)
        Driving inputs exp(f).
    log_g : np.ndarray, shape (N, L, n_theta)
        log(max(g, eps)).
    theta_grid : np.ndarray, shape (n_theta,)
        Orientation grid in radians.
    N, L, n_theta : int
        Population size, number of locations, orientation resolution.
    tuning_type : str
        'bays' or 'gp'.
    omega : float
        Tuning width parameter used to construct the population.
    """

    def __init__(
        self,
        M: int,
        n_theta: int,
        omega: float,
        tuning_type: str = "bays",
        n_locations: int = 1,
        seed: int = 42,
        lengthscale_variability: float = 0.0,
        gain_variability: float = 0.0,
        method: str = "folded_normal",
        n_high: int = 1,
        high_multiplier: float = 3.0,
    ):
        if tuning_type not in ("bays", "gp"):
            raise ValueError(f"tuning_type must be 'bays' or 'gp', got '{tuning_type}'")

        self.tuning_type = tuning_type
        self.omega = omega

        if tuning_type == "bays":
            f, theta_grid = self._build_bays(M, n_theta, omega)
        else:
            f, theta_grid = self._build_gp(
                M, n_theta, omega, n_locations, seed,
                lengthscale_variability, gain_variability, method,
                n_high=n_high, high_multiplier=high_multiplier,
            )

        self.f = f
        self.theta_grid = theta_grid
        self.N, self.L, self.n_theta = f.shape

        self.g = np.exp(self.f)
        self.log_g = np.log(np.maximum(self.g, 1e-30))

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bays(M, n_theta, omega):
        """Bays (2014) Eq. 1: f_i(theta) = (1/omega)(cos(phi_i - theta) - 1)"""
        theta_grid = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        phi = np.linspace(-np.pi, np.pi, M, endpoint=False)
        diff = phi[:, None] - theta_grid[None, :]
        f = (1.0 / omega) * (np.cos(diff) - 1.0)
        return f[:, np.newaxis, :], theta_grid     # (M, 1, n_theta)

    @staticmethod
    def _build_gp(M, n_theta, omega, n_locations, seed,
                  lengthscale_variability, gain_variability,
                  method="folded_normal",
                  n_high=1, high_multiplier=3.0):
        """GP tuning curves with lambda = sqrt(omega).

        `n_high` and `high_multiplier` are forwarded to the GP sampler;
        they are only consulted when method='sparse_broad' (or its
        legacy alias 'random_vector').
        """
        lengthscale = np.sqrt(omega)
        raw = generate_neuron_population(
            n_neurons=M,
            n_orientations=n_theta,
            n_locations=n_locations,
            base_lengthscale=lengthscale,
            lengthscale_variability=lengthscale_variability,
            seed=seed,
            gain_variability=gain_variability,
            method=method,
            n_high=n_high,
            high_multiplier=high_multiplier,
        )
        theta_grid = raw[0]["orientations"]
        f = np.zeros((M, n_locations, n_theta))
        for n in range(M):
            f[n] = raw[n]["f_samples"]
        return f, theta_grid

    # ------------------------------------------------------------------
    # Helper: circular linear interpolation indices
    # ------------------------------------------------------------------

    @staticmethod
    def _circular_linear_interp_indices(theta_continuous, n_theta):
        """
        Bracketing-grid indices and weights for linear interpolation of a
        function tabulated on the grid

            grid = np.linspace(-pi, pi, n_theta, endpoint=False)

        at continuous theta values.

        For a function f tabulated on `grid`, the interpolated value at
        theta is

            f(theta) ~= (1 - w) * f[grid[i0]] + w * f[grid[i1]]

        with i1 = (i0 + 1) mod n_theta (so the interval wraps around the
        circle from grid[n_theta-1] to grid[0]).

        Parameters
        ----------
        theta_continuous : array-like
            Continuous orientations in [-pi, pi). Any shape is accepted;
            i0, i1, w are returned with the same shape.
        n_theta : int
            Number of grid points.

        Returns
        -------
        i0 : np.ndarray of int
            Left bracket grid indices.
        i1 : np.ndarray of int
            Right bracket grid indices, wrapped circularly.
        w : np.ndarray of float
            Interpolation weights in [0, 1).
        """
        theta_continuous = np.asarray(theta_continuous, dtype=float)
        spacing = 2.0 * np.pi / n_theta
        # grid[k] = -pi + k * spacing  =>  fractional index = (theta + pi) / spacing
        idx_f = (theta_continuous + np.pi) / spacing
        i0_float = np.floor(idx_f)
        i0 = (i0_float.astype(np.intp)) % n_theta
        i1 = (i0 + 1) % n_theta
        w = idx_f - i0_float
        return i0, i1, w

    # ------------------------------------------------------------------
    # Core methods -- the single source of truth (discrete-grid theta)
    # ------------------------------------------------------------------

    def encode(
        self,
        active_locations: Tuple[int, ...],
        theta_indices: np.ndarray,
        gamma: float,
        T_d: float,
        sigma_sq: float = 1e-6,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Encode a stimulus configuration into Poisson spike counts.

        Pipeline:  r_pre (Eq. 13) -> DN (Eq. 6) -> Poisson (Def. 4.5)

        This is the DISCRETE-grid path: orientations are passed in as
        integer indices into theta_grid. For the continuous-orientation
        path used by run_trials(), see _circular_linear_interp_indices.

        Parameters
        ----------
        active_locations : tuple of int, length l
            Which spatial locations carry items.
        theta_indices : array-like of int, length l
            Orientation grid index at each active location.
        gamma : float
            Mean per-neuron gain (Hz).
        T_d : float
            Decoding time window (seconds).
        sigma_sq : float
            Semi-saturation constant.
        rng : RandomState, optional

        Returns
        -------
        spike_counts : np.ndarray, shape (N,)
        """
        if rng is None:
            rng = np.random.RandomState()

        r_pre = compute_r_pre_at_config(self.f, active_locations, theta_indices)
        r_post = dn_pointwise(r_pre, gamma, sigma_sq)
        # Delegate to poisson_spiking so any downstream changes propagate here.
        return generate_spikes(r_post, T_d, rng=rng)

    def decode(
        self,
        spike_counts: np.ndarray,
        active_locations: Tuple[int, ...],
        cued_location: int,
    ) -> Tuple[float, np.ndarray]:
        """
        Decode spike counts via factorised ML (Eqs. 23-28).

        Parameters
        ----------
        spike_counts : np.ndarray, shape (N,)
        active_locations : tuple of int, length l
            Which spatial locations are active.
        cued_location : int
            Index *into active_locations* of the cued item.

        Returns
        -------
        theta_hat : float
            ML orientation estimate (lives on theta_grid).
        L_marginal : np.ndarray, shape (n_theta,)
            Marginal log-likelihood curve.
        """
        f_per_loc = [self.f[:, loc, :] for loc in active_locations]
        return ml_decode(spike_counts, f_per_loc, self.theta_grid, cued_location)

    # ------------------------------------------------------------------
    # Vectorised batch trial runner (continuous theta, interpolated tuning)
    # ------------------------------------------------------------------

    def run_trials(
        self,
        n_trials: int,
        gamma: float,
        T_d: float,
        set_size: int = 1,
        sigma_sq: float = 1e-6,
        rng: Optional[np.random.RandomState] = None,
        active_locations: Optional[Tuple[int, ...]] = None,
        probe_cued: Optional[bool] = None,
        alpha: float = 1.0,
        recruited_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run n_trials of encode -> decode, return circular errors.

        Continuous-theta trial engine. True orientations are sampled from
        Uniform[-pi, pi) and the stored tuning curves are linearly
        interpolated at those off-grid points. This eliminates the
        error-quantization artefact that occurs when both encoding and
        decoding share the same theta grid (see the uniq_GP / zero_GP /
        min|e|_GP diagnostic in the figure-2 notebook).

        The decoder grid itself stays at n_theta points -- theta_hat is
        still snapped to the grid -- but the truth is continuous, so the
        error theta_hat - theta_true is continuous and free of the
        delta-at-zero spike that previously inflated kurtosis at low set
        sizes.

        Encode (vectorised over trials):
            f(theta) ~= (1-w_k)*f[loc, i0_k] + w_k*f[loc, i1_k]   (linear interp)
            r_pre[n, t]  = exp( sum_k f_interp[n, locs[t,k], theta[t,k]] )  (Eq. 13)
            r_pre[recruited, t] *= alpha                             (Fig 3 cue)
            D[t]         = sigma^2 + mean_n( r_pre[n, t] )
            r_post[n, t] = gamma * r_pre[n, t] / D[t]               (Eq. 6)
            counts[n, t] ~ Poisson( r_post[n, t] * T_d )            (Def. 4.5)

        Decode (vectorised over trials):
            L_c[t, theta] = sum_n counts[n,t] * f[n, cued_loc[t], theta]  (Eq. 23)
            theta_hat_idx[t] = argmax_theta L_c[t, theta]                 (Eq. 28)
            theta_hat[t]     = theta_grid[theta_hat_idx[t]]  (on-grid)
            error[t] = wrap(theta_hat[t] - theta_true[t])   (continuous truth)

        Note: The ML point estimate depends only on L_c -- the logsumexp
        terms over non-cued locations are constant w.r.t. theta_c and do
        not affect the argmax (see ml_decoder.py docstring).

        Parameters
        ----------
        n_trials : int
        gamma : float
            Mean per-neuron gain (Hz).
        T_d : float
            Decoding time window (seconds).
        set_size : int
            Number of items (l).
        sigma_sq : float
        rng : RandomState, optional
        active_locations : tuple of int, optional
            Deterministic location set used on EVERY trial. When supplied,
            location 0 of this tuple is the conventional cued location
            (used together with ``probe_cued`` and ``recruited_mask``).
            When None (default), locations are sampled per trial as
            before -- backwards compatible with prior Figure 2 usage.
        probe_cued : bool, optional
            Only meaningful when ``active_locations`` is supplied.
              - True  -> every trial probes location 0 (the cued location).
              - False -> every trial probes a location uniformly sampled
                        from active_locations[1:] (an uncued location).
              - None  -> uniform sampling over all active_locations
                        (default trial-runner behaviour).
        alpha : float, default 1.0
            Multiplicative boost on the pre-DN drive of the neurons
            selected by ``recruited_mask``. 1.0 = no boost (no cue).
        recruited_mask : np.ndarray of bool, shape (N,), optional
            Boolean mask over neurons. The boost ``alpha`` is applied
            to r_pre[recruited_mask, :] before DN. Required when
            ``alpha != 1.0``.

        Returns
        -------
        errors : np.ndarray, shape (n_trials,)
            Signed circular errors in [-pi, pi).
        """
        if rng is None:
            rng = np.random.RandomState()

        l = set_size
        N, L, n_theta = self.N, self.L, self.n_theta

        # ---- Validate the recruitment-boost arguments ----
        if alpha != 1.0 and recruited_mask is None:
            raise ValueError(
                "alpha != 1.0 requires recruited_mask "
                "(boolean mask of shape (N,) marking which neurons "
                "receive the boost)."
            )
        if recruited_mask is not None:
            recruited_mask = np.asarray(recruited_mask, dtype=bool)
            if recruited_mask.shape != (N,):
                raise ValueError(
                    f"recruited_mask must have shape ({N},), "
                    f"got {recruited_mask.shape}"
                )

        # ---- 1. Sample all configurations upfront ----
        if active_locations is not None:
            active_arr = np.asarray(active_locations, dtype=int)
            if active_arr.size != l:
                raise ValueError(
                    f"active_locations has length {active_arr.size}, "
                    f"expected set_size={l}."
                )
            locs = np.tile(active_arr, (n_trials, 1))    # (n_trials, l)
        elif l == 1:
            locs = rng.randint(L, size=(n_trials, 1))
        elif l >= L:
            locs = np.tile(np.arange(L), (n_trials, 1))
        else:
            # Vectorised sampling without replacement via argsort trick.
            locs = rng.random((n_trials, L)).argsort(axis=1)[:, :l]

        # Continuous true orientations -- no grid quantization.
        theta_true = rng.uniform(-np.pi, np.pi, size=(n_trials, l))   # (n_trials, l)

        # Cued-index resolution
        if probe_cued is True:
            cued = np.zeros(n_trials, dtype=int)
        elif probe_cued is False:
            if l <= 1:
                raise ValueError(
                    "probe_cued=False is undefined for set_size <= 1 "
                    "(no uncued location to probe)."
                )
            cued = rng.randint(1, l, size=n_trials)
        else:
            cued = (np.zeros(n_trials, dtype=int) if l == 1
                    else rng.randint(l, size=n_trials))

        # ---- 2. Vectorised encode with linear interpolation ----
        i0, i1, w = self._circular_linear_interp_indices(theta_true, n_theta)
        # i0, i1: int (n_trials, l).  w: float (n_trials, l).

        log_r_pre = np.zeros((N, n_trials))
        for k in range(l):
            f_left  = self.f[:, locs[:, k], i0[:, k]]    # (N, n_trials)
            f_right = self.f[:, locs[:, k], i1[:, k]]    # (N, n_trials)
            log_r_pre += (1.0 - w[:, k]) * f_left + w[:, k] * f_right

        r_pre = np.exp(log_r_pre)                        # (N, n_trials)

        # Recruitment boost before DN.
        if recruited_mask is not None and alpha != 1.0:
            r_pre[recruited_mask, :] *= alpha

        # DN: r_post = gamma * r_pre / D,  D = sigma^2 + mean_n(r_pre)
        D = sigma_sq + r_pre.mean(axis=0)                # (n_trials,)
        r_post = gamma * r_pre / D[np.newaxis, :]        # (N, n_trials)

        # Poisson spikes -- delegate to core.encoder.poisson_spiking.
        #
        # generate_spikes(rates, T_d, rng) calls rng.poisson on whatever
        # array is passed as `rates`, so passing the full (N, n_trials)
        # r_post matrix produces counts of shape (N, n_trials) directly,
        # matching the layout expected by the decoder below.
        counts = generate_spikes(r_post, T_d, rng=rng)  # (N, n_trials)

        # ---- 3. Vectorised decode (point estimate only) ----
        # L_c[t, theta] = sum_n counts[n,t] * f[n, cued_loc[t], theta]
        cued_locs = locs[np.arange(n_trials), cued]      # (n_trials,)

        L_c = np.empty((n_trials, n_theta))
        for loc in range(L):
            mask = (cued_locs == loc)
            if mask.any():
                # (n_mask, N) @ (N, n_theta) -> (n_mask, n_theta)
                L_c[mask] = counts[:, mask].T @ self.f[:, loc, :]

        theta_hat_idx = np.argmax(L_c, axis=1)           # (n_trials,)

        # ---- 4. Circular errors (continuous truth vs grid-snapped estimate) ----
        theta_true_probed = theta_true[np.arange(n_trials), cued]
        theta_hat = self.theta_grid[theta_hat_idx]

        d = theta_hat - theta_true_probed
        errors = (d + np.pi) % (2.0 * np.pi) - np.pi

        return errors

    # ------------------------------------------------------------------
    # Sequential reference implementation (for validation / debugging)
    # ------------------------------------------------------------------

    def _run_trials_sequential(
        self,
        n_trials: int,
        gamma: float,
        T_d: float,
        set_size: int = 1,
        sigma_sq: float = 1e-6,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Sequential (non-vectorised) trial loop -- kept for validation.

        Uses the continuous-theta + interpolation pipeline to match
        run_trials() (statistically; individual draws differ because
        RNG consumption order differs). For the discrete-grid encode/decode
        pipeline, use encode() and decode() directly.
        """
        if rng is None:
            rng = np.random.RandomState()

        errors = np.empty(n_trials)

        for t in range(n_trials):
            locs = tuple(rng.choice(self.L, size=set_size, replace=False))
            theta_true_per_loc = rng.uniform(-np.pi, np.pi, size=set_size)
            cued = rng.randint(set_size)
            theta_true = theta_true_per_loc[cued]

            # Continuous-theta encoding: interpolate f at each location.
            i0, i1, w = self._circular_linear_interp_indices(
                theta_true_per_loc, self.n_theta
            )
            log_r_pre = np.zeros(self.N)
            for k, loc in enumerate(locs):
                f_left  = self.f[:, loc, i0[k]]
                f_right = self.f[:, loc, i1[k]]
                log_r_pre += (1.0 - w[k]) * f_left + w[k] * f_right
            r_pre = np.exp(log_r_pre)

            # DN + Poisson -- delegate to poisson_spiking module.
            D = sigma_sq + r_pre.mean()
            r_post = gamma * r_pre / D
            counts = generate_spikes(r_post, T_d, rng=rng)

            # ML decode (factorised marginalisation).
            theta_hat, _ = self.decode(counts, locs, cued)

            errors[t] = circular_error(theta_true, theta_hat)

        return errors

    def __repr__(self) -> str:
        return (
            f"Population(N={self.N}, L={self.L}, n_theta={self.n_theta}, "
            f"tuning='{self.tuning_type}', omega={self.omega})"
        )