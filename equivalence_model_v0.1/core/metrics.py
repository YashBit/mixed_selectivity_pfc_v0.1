"""
analysis/metrics.py
===================

Central home for the circular error metrics and the Bays-equivalence
comparison helpers used by the Figure-1 / Figure-2 (Bays 2014) notebooks.

Previously these lived inline in each notebook, which meant the definitions
drifted between files. Everything a notebook needs to (a) turn a pair of
angles into a signed per-trial error, (b) turn a vector of signed circular
errors into Fisher (1995) circular variance / kurtosis, and (c) run the
"l items at gamma == 1 item at gamma/l" equivalence test, now lives here.

Only depends on numpy and scipy.

Public API
----------
circular_error(theta_true, theta_hat)     -> float | np.ndarray
circular_variance_fisher(errors)          -> float
circular_kurtosis_fisher(errors)          -> float
circular_moments(errors)                  -> dict
vonmises_kde(data, eval_points, kappa)    -> np.ndarray
interp_row_at_gain(grid, target, gammas)  -> (N_GRID,) vector over omega
run_equivalence_test(heatmaps, gammas_total, gamma_ref_idx, ...) -> (table, gamma_ref)
format_equivalence_table(results_table, gamma_ref)               -> str
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0, i1, logsumexp
from scipy.optimize import brentq
from scipy.stats import vonmises

# numpy 2.0 renamed np.trapz -> np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# Canonical (display name, dict-key) pairs for the three metrics compared
# across the notebook. Import this so the notebook and module never disagree
# on ordering / keys.
QUANTITIES = [
    ("variance", "var"),
    ("kurtosis", "kur"),
    ("exponent", "exp"),
]


# =============================================================================
# Per-trial signed error (produces the vectors the metrics below consume)
# =============================================================================

def circular_error(theta_true, theta_hat):
    """
    Signed circular error, wrapped to the half-open interval [-pi, pi).

    Returns the wrapped difference (theta_hat - theta_true): the decoded /
    estimated orientation minus the true orientation. The argument order is
    (theta_true, theta_hat) to match the historical trial-loop call site:

        errors[t] = circular_error(theta_true, theta_hat)

    The wrap is bit-identical to the inline form used by the vectorised trial
    engine, ``(d + np.pi) % (2*np.pi) - np.pi`` with ``d = theta_hat -
    theta_true``, so sequential and vectorised paths agree in sign and in
    boundary handling (theta = +pi maps to -pi).

    Parameters
    ----------
    theta_true : float or np.ndarray
        True orientation(s) in radians.
    theta_hat : float or np.ndarray
        Estimated orientation(s) in radians. Must broadcast against theta_true.

    Returns
    -------
    error : float or np.ndarray
        Signed circular error(s) in [-pi, pi). Scalar in -> scalar out;
        array in -> array out.
    """
    d = np.asarray(theta_hat, dtype=float) - np.asarray(theta_true, dtype=float)
    wrapped = (d + np.pi) % (2.0 * np.pi) - np.pi
    # Preserve scalar-in/scalar-out behaviour.
    if np.ndim(wrapped) == 0:
        return float(wrapped)
    return wrapped


# =============================================================================
# Circular error metrics (Fisher, 1995) — the quantities Bays (2014) plots
# =============================================================================

def circular_variance_fisher(errors):
    """
    Circular variance = squared circular standard deviation (Fisher, 1995).

    For angular errors (radians), R = |m1| is the mean resultant length,
    where m1 = <exp(i*errors)>. The circular SD is sqrt(-2 ln R), so

        sigma^2 = -2 ln R

    This is the quantity Bays (2014) plots as 'variance'. It is 0 for a
    perfectly concentrated set and grows without bound as the distribution
    approaches uniform (R -> 0).
    """
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return np.nan
    R = np.abs(np.mean(np.exp(1j * errors)))
    R = min(R, 1.0)              # guard tiny FP overshoot above 1
    if R < 1e-12:               # effectively uniform -> very large variance
        return -2.0 * np.log(1e-12)
    return -2.0 * np.log(R)


def circular_kurtosis_fisher(errors):
    """
    Circular kurtosis (Fisher, 1995), the excess-kurtosis analogue used by
    Bays (2014):

        k = [ rho2 * cos(mu2 - 2*mu1) - rho1**4 ] / (1 - rho1)**2

    where rho_p = |m_p|, mu_p = arg(m_p), and m_p = <exp(i*p*errors)> is the
    p-th uncentered trigonometric moment. A circular-normal (von Mises)
    distribution has k ~ 0; positive k = sharper peak / heavier tails.
    """
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return np.nan
    m1 = np.mean(np.exp(1j * errors))
    m2 = np.mean(np.exp(2j * errors))
    rho1, rho2 = np.abs(m1), np.abs(m2)
    mu1, mu2 = np.angle(m1), np.angle(m2)
    denom = (1.0 - rho1) ** 2
    if denom < 1e-15:
        return np.nan
    return (rho2 * np.cos(mu2 - 2.0 * mu1) - rho1 ** 4) / denom


def circular_moments(errors):
    """Variance (Fisher 1995 / Bays 2014), kurtosis, and mean resultant length.

    Both moments delegate to the functions above so nothing ever diverges
    numerically between callers.
    """
    errors = np.asarray(errors, dtype=float)
    rho1 = float(np.abs(np.mean(np.exp(1j * errors))))
    return {
        "variance":       circular_variance_fisher(errors),
        "kurtosis":       circular_kurtosis_fisher(errors),
        "mean_resultant": rho1,
    }


def _direct_circular_variance(errors):
    """Independent reimplementation of sigma^2 = -2 log R (consistency check)."""
    errors = np.asarray(errors, dtype=float)
    R = np.abs(np.mean(np.exp(1j * errors)))
    return float(-2.0 * np.log(max(R, 1e-15)))


def _kde_circular_variance(theta_grid, density):
    """Circular variance read off a KDE density curve sampled on theta_grid.

        R = | integral e^{i*theta} p(theta) dtheta | / integral p(theta) dtheta
        sigma^2 = -2 log R

    Diverges from the raw-sample variance when smoothing flattens long tails,
    which is exactly what the diagnostic is meant to surface.
    """
    theta_grid = np.asarray(theta_grid, dtype=float)
    density = np.asarray(density, dtype=float)
    Z = _trapz(density, theta_grid)
    if Z <= 0:
        return np.nan
    m1 = _trapz(density * np.exp(1j * theta_grid), theta_grid) / Z
    R = np.abs(m1)
    return float(-2.0 * np.log(max(R, 1e-15)))


def _estimate_von_mises_kappa(rho1):
    """Invert mean-resultant-length -> concentration for a von Mises."""
    if rho1 < 1e-6:
        return 0.0
    if rho1 > 0.9999:
        return 700.0
    return brentq(lambda k: float(i1(k) / i0(k)) - rho1, 1e-4, 700.0)


def compute_deviation_from_normal(errors, n_bins=50):
    """Empirical error histogram minus the variance-matched von Mises pdf."""
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    emp, _ = np.histogram(errors, bins=bin_edges, density=True)
    rho1 = np.abs(np.mean(np.exp(1j * errors)))
    kappa_fit = _estimate_von_mises_kappa(rho1)
    vm_pdf = vonmises.pdf(centers, kappa_fit)
    return {"bin_centers": centers, "empirical": emp,
            "normal_fit": vm_pdf, "deviation": emp - vm_pdf}


# =============================================================================
# Von Mises circular KDE
# =============================================================================

def vonmises_kde(data, eval_points, kappa):
    """
    Circular KDE using von Mises kernels.

    Parameters
    ----------
    data : array-like, shape (n,)
        Sample of angles in [-pi, pi).
    eval_points : array-like, shape (m,)
        Points at which to evaluate the density.
    kappa : float
        Concentration of the von Mises kernel. Higher kappa -> narrower
        kernel (less smoothing); kappa ~ 1/h^2 for Gaussian bandwidth h.

    Returns
    -------
    density : ndarray, shape (m,)
    """
    data = np.asarray(data)
    eval_points = np.asarray(eval_points)
    n = len(data)
    diff = eval_points[:, None] - data[None, :]          # (m, n)
    log_norm = np.log(2.0 * np.pi * float(i0(kappa)))
    log_kernels = kappa * np.cos(diff) - log_norm        # (m, n)
    density = np.exp(logsumexp(log_kernels, axis=1) - np.log(n))
    return density


# =============================================================================
# Bays-equivalence comparison helpers
# =============================================================================

def interp_row_at_gain(grid, target_gain, gammas_total):
    """
    Linearly interpolate a heatmap (gamma x omega) along the log2-gain axis
    at `target_gain` Hz. Returns a length-N_GRID vector over omega.

    Linear-in-log2(gamma) matches how the gain grid was constructed. Values
    outside the grid range clamp to the nearest edge row.

    Parameters
    ----------
    grid : (N_GAMMA, N_OMEGA) ndarray
        A metric heatmap indexed [gamma_idx, omega_idx].
    target_gain : float
        Total population gain (Hz) at which to sample the row.
    gammas_total : (N_GAMMA,) ndarray
        The gamma_total axis the grid was built on.
    """
    log2_gammas = np.log2(gammas_total)
    target_log = np.log2(target_gain)
    if target_log <= log2_gammas[0]:
        return grid[0, :].copy()
    if target_log >= log2_gammas[-1]:
        return grid[-1, :].copy()
    j_hi = np.searchsorted(log2_gammas, target_log)
    j_lo = j_hi - 1
    w = (target_log - log2_gammas[j_lo]) / (log2_gammas[j_hi] - log2_gammas[j_lo])
    return (1 - w) * grid[j_lo, :] + w * grid[j_hi, :]


def run_equivalence_test(
    heatmaps,
    gammas_total,
    gamma_ref_idx,
    set_sizes=(2, 4, 8),
    quantities=QUANTITIES,
):
    """
    Quantitative equivalence test (Candidate A, interpolated).

    Bays equivalence claims: l items at gamma_ref behaves like 1 item at
    gamma_ref / l. For each set size and metric we build:

        prediction  (Bays):  heatmaps[1] interpolated at gamma_ref / l
        measurement (GP):    heatmaps[l] row at gamma_ref (no interpolation)

    Both are vectors over omega (length N_OMEGA). They are compared with
    Pearson correlation and (relative) mean absolute error.

    NOTE: this is an *evaluation*, not a fit. Nothing here is optimised;
    no likelihood is maximised and no parameters are estimated. Bays's
    (omega, gamma) are ML fits imported upstream as constants.

    Returns
    -------
    results_table : list[dict]
        One dict per set size, carrying corr/pval/mae/rmae and the raw
        pred/meas vectors for each metric.
    gamma_ref : float
        The reference total gain, gammas_total[gamma_ref_idx].
    """
    from scipy.stats import pearsonr

    gamma_ref = gammas_total[gamma_ref_idx]
    results_table = []

    for l in set_sizes:
        g_eff = gamma_ref / l
        row_result = {"l": l, "g_eff": g_eff}

        for q_name, q_key in quantities:
            # PREDICTION: l=1 heatmap interpolated at gamma_eff = gamma_ref / l
            vec_pred = interp_row_at_gain(heatmaps[1][q_key], g_eff, gammas_total)
            # MEASUREMENT: l-th heatmap at gamma_ref
            vec_meas = heatmaps[l][q_key][gamma_ref_idx, :]

            valid = np.isfinite(vec_pred) & np.isfinite(vec_meas)
            vp, vm = vec_pred[valid], vec_meas[valid]

            if len(vp) >= 3:
                corr, p_val = pearsonr(vp, vm)
                mae = np.mean(np.abs(vp - vm))
                rmae = mae / (np.mean(np.abs(vp)) + 1e-15)
            else:
                corr, p_val, mae, rmae = np.nan, np.nan, np.nan, np.nan

            row_result[f"{q_name}_corr"] = corr
            row_result[f"{q_name}_pval"] = p_val
            row_result[f"{q_name}_mae"] = mae
            row_result[f"{q_name}_rmae"] = rmae
            row_result[f"{q_name}_pred"] = vec_pred
            row_result[f"{q_name}_meas"] = vec_meas

        results_table.append(row_result)

    return results_table, gamma_ref


def format_equivalence_table(results_table, gamma_ref):
    """Render the results_table from run_equivalence_test as a printable string."""
    header = (
        f"{'l':>3s}  {'g_eff':>8s}  |  "
        f"{'Var rho':>7s} {'Var rMAE':>9s}  |  "
        f"{'Kur rho':>7s} {'Kur rMAE':>9s}  |  "
        f"{'Exp rho':>7s} {'Exp rMAE':>9s}"
    )
    lines = [header, "-" * len(header)]
    for r in results_table:
        lines.append(
            f"{r['l']:3d}  {r['g_eff']:8.3f}  |  "
            f"{r['variance_corr']:7.4f} {r['variance_rmae']:9.2%}  |  "
            f"{r['kurtosis_corr']:7.4f} {r['kurtosis_rmae']:9.2%}  |  "
            f"{r['exponent_corr']:7.4f} {r['exponent_rmae']:9.2%}"
        )
    lines += [
        "",
        "rho = Pearson correlation, rMAE = mean|pred - meas| / mean|pred|",
        "Prediction  = heatmaps[1] interpolated at gamma_ref/l   (Bays)",
        "Measurement = heatmaps[l] at gamma_ref                  (GP)",
        f"gamma_ref = {gamma_ref:.2f} Hz",
    ]
    return "\n".join(lines)