from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import invgamma

from .fit import GIGG_P0, PRIORS
ROOT = Path(__file__).resolve().parents[2]
PRIOR_METHODS = (('HORSESHOE', '$HS$'), ('RHS', '$RHS$'), ('GR_RHS', '$GR-RHS$'), ('GHS', '$GHS$'), ('GIGG', '$GIGG$'), ('R2D2', '$R2D2$'), ('GROUP_R2', '$Group-R^2$'))

SEED_ORDER = tuple((method for method, _ in PRIOR_METHODS))

def effective_contribution(variance: np.ndarray, n: int, sigma: float) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        contribution = 1.0 / (1.0 + sigma ** 2 / (n * variance))
    return np.nan_to_num(contribution, nan=0.0, posinf=1.0, neginf=0.0)

def effective_size(variance: np.ndarray, n: int, sigma: float) -> np.ndarray:
    return effective_contribution(variance, n, sigma).sum(axis=-1)

def groupwise_effective_size(variance: np.ndarray, n: int, sigma: float, groups: int) -> np.ndarray:
    group_size = variance.shape[-1] // groups
    return effective_contribution(variance, n, sigma).reshape(-1, groups, group_size).sum(axis=-1)

def harmonic_variance(q: np.ndarray, slab_variance: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        variance = 1.0 / (1.0 / q + 1.0 / slab_variance)
    return np.nan_to_num(variance, nan=np.inf, posinf=np.inf, neginf=0.0)

def sample_prior_meff(*, draws: int, batch_size: int, seed: int, n: int, p: int, groups: int, p0: float, sigma: float, return_groupwise: bool=False, selected_methods: tuple[str, ...] | None=None) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    requested_methods = set(SEED_ORDER if selected_methods is None else selected_methods)
    unknown_methods = requested_methods.difference(SEED_ORDER)
    if unknown_methods:
        raise ValueError(f'Unknown prior methods: {sorted(unknown_methods)}')
    output_methods = tuple((method for method, _ in PRIOR_METHODS if method in requested_methods))
    group_size = p // groups
    tau0 = p0 / (p - p0) * sigma / np.sqrt(n)
    gigg_tau0 = GIGG_P0 / (p - GIGG_P0) * sigma / np.sqrt(n)
    child_seeds = np.random.SeedSequence(seed).spawn(len(SEED_ORDER))
    rngs = {method: np.random.default_rng(child_seed) for method, child_seed in zip(SEED_ORDER, child_seeds) if method in requested_methods}
    samples = {method: np.empty(draws) for method in output_methods}
    group_samples = {method: np.empty((draws, groups)) for method in output_methods} if return_groupwise else None

    def record(method: str, start: int, stop: int, variance: np.ndarray) -> None:
        groupwise = groupwise_effective_size(variance, n=n, sigma=sigma, groups=groups)
        samples[method][start:stop] = groupwise.sum(axis=1)
        if group_samples is not None:
            group_samples[method][start:stop] = groupwise
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        size = stop - start
        for method in ('HORSESHOE', 'RHS', 'GR_RHS'):
            if method not in requested_methods:
                continue
            rng = rngs[method]
            tau = tau0 * np.abs(rng.standard_cauchy(size))
            local = np.abs(rng.standard_cauchy((size, p)))
            q = np.square(tau[:, None] * local)
            if method == 'HORSESHOE':
                variance = q
            elif method == 'RHS':
                prior = PRIORS['GR_RHS']
                c2 = prior['slab_variance_scale'] * (rng.gamma(prior['alpha_b'], 1.0, size) / rng.gamma(prior['beta_b'], 1.0, size))
                variance = harmonic_variance(q, c2[:, None])
            else:
                prior = PRIORS['GR_RHS']
                alpha_b = prior['alpha_b']
                beta_b = prior['beta_b']
                c2_group = prior['slab_variance_scale'] * (rng.gamma(alpha_b, 1.0, (size, groups)) / rng.gamma(beta_b, 1.0, (size, groups)))
                c2 = np.repeat(c2_group, group_size, axis=1)
                variance = harmonic_variance(q, c2)
            record(method, start, stop, variance)
        for method in ('R2D2', 'GROUP_R2'):
            if method not in requested_methods:
                continue
            rng = rngs[method]
            prior = PRIORS[method]
            r2 = rng.beta(prior['r2_a'], prior['r2_b'], size)
            with np.errstate(divide='ignore', over='ignore'):
                total_variance = sigma ** 2 * r2 / (1.0 - r2)
            if method == 'R2D2':
                concentration = PRIORS['R2D2']['phi_concentration']
                allocation = rng.gamma(concentration, 1.0, (size, p))
                allocation /= allocation.sum(axis=1, keepdims=True)
                variance = total_variance[:, None] * allocation
            else:
                prior = PRIORS['GROUP_R2']
                group_allocation = rng.gamma(prior['phi_group_concentration'], 1.0, (size, groups))
                group_allocation /= group_allocation.sum(axis=1, keepdims=True)
                within_allocation = rng.gamma(prior['phi_within_concentration'], 1.0 / prior['phi_within_rate'], (size, groups, group_size))
                within_allocation /= within_allocation.sum(axis=2, keepdims=True)
                allocation = (group_allocation[:, :, None] * within_allocation).reshape(size, p)
                variance = total_variance[:, None] * allocation
            record(method, start, stop, variance)
        if 'GHS' in requested_methods:
            rng = rngs['GHS']
            tau = tau0 * np.abs(rng.standard_cauchy(size))
            group_scale = np.abs(rng.standard_cauchy((size, groups)))
            local_scale = np.abs(rng.standard_cauchy((size, p)))
            delta = np.repeat(group_scale, group_size, axis=1)
            with np.errstate(over='ignore', invalid='ignore'):
                variance = np.square(tau[:, None] * delta * local_scale)
            record('GHS', start, stop, variance)
        if 'GIGG' in requested_methods:
            rng = rngs['GIGG']
            prior = PRIORS['GIGG']
            tau = gigg_tau0 * np.abs(rng.standard_cauchy(size))
            gamma2_group = rng.gamma(prior['gamma2_shape'], 1.0 / prior['gamma2_rate'], (size, groups))
            gamma2 = np.repeat(gamma2_group, group_size, axis=1)
            lambda2 = invgamma.rvs(prior['lambda2_shape'], scale=prior['lambda2_scale'], size=(size, p), random_state=rng)
            with np.errstate(over='ignore', invalid='ignore'):
                variance = np.square(tau[:, None]) * gamma2 * lambda2
            record('GIGG', start, stop, variance)
    if group_samples is not None:
        return (samples, group_samples)
    return samples


"""Calibrate independent R2 prior means for R2D2 and Group-R2 against p0.

R2D2 and Group-R2 have no analogue of the horseshoe family's global scale
tau, so their prior cannot be calibrated to p0 by solving
``tau0 = p0 / (p - p0) * sigma / sqrt(n)`` the way HS, RHS, GR-RHS, GHS and
GIGG are (see ``analysis/prior_simulation.py``). Each method's
``r2_mean`` (holding ``r2_precision`` fixed) is instead found by bisection so
its own prior-predictive *median* effective model size ``m_eff`` matches the
horseshoe family's own simulated median under the same ``n, p, groups, p0``.

R2D2 and Group-R2 are calibrated separately rather than sharing one r2_mean:
Group-R2 splits the total prior variance through an extra group/within
Dirichlet layer that dilutes each coefficient's share, and how much dilution
that causes depends on ``groups``, so a shared value calibrated at one
``groups`` can drift outside the "broadly calibrated" band (median within
+/-20% of the HS median) at another. Re-run this and update the paper configuration prior means if p0, n, p, or groups change.

Usage::

    python -m analysis.calibrate_r2_prior --p0 100 --n 200 --p 500 --groups 5
"""
import numpy as np
import pandas as pd

def summarize(samples, hs_median, r2_means):
    rows = []
    for method, label in PRIOR_METHODS:
        values = samples[method]
        q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
        relative_to_hs = median / hs_median
        rows.append({'method': method, 'label': label.replace('$', ''), 'r2_mean': r2_means.get(method, float('nan')), 'mean': float(values.mean()), 'median': float(median), 'q25': float(q25), 'q75': float(q75), 'iqr': float(q75 - q25), 'prob_meff_lt_10': float(np.mean(values < 10.0)), 'prob_25_lt_meff_lt_100': float(np.mean((values > 25.0) & (values < 100.0))), 'median_relative_to_hs': float(relative_to_hs), 'classification': 'broadly calibrated' if 0.8 <= relative_to_hs <= 1.2 else 'under-regularized relative to HS' if relative_to_hs > 1.2 else 'over-regularized relative to HS'})
    return pd.DataFrame(rows)

def method_median_meff(method, r2_mean, r2_precision, *, n, p, groups, seed, draws, batch_size):
    PRIORS[method]['r2_a'] = r2_mean * r2_precision
    PRIORS[method]['r2_b'] = (1.0 - r2_mean) * r2_precision
    samples = sample_prior_meff(draws=draws, batch_size=batch_size, seed=seed, n=n, p=p, groups=groups, p0=0.0, sigma=1.0, selected_methods=(method,))
    return float(np.median(samples[method]))

def calibrate_r2_mean(method, hs_median, r2_precision, *, n, p, groups, seed, draws, tol, max_iter):
    """Bisection search on r2_mean in (0, 1) matching one method's median m_eff to HS."""
    lo, hi = (0.001, 1.0 - 0.001)
    mid = value = None
    for iteration in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = method_median_meff(method, mid, r2_precision, n=n, p=p, groups=groups, seed=seed + iteration, draws=draws, batch_size=min(draws, 1000))
        if abs(value - hs_median) < tol:
            return (mid, value, iteration + 1)
        if value < hs_median:
            lo = mid
        else:
            hi = mid
    return (mid, value, max_iter)

def half_cauchy(rng, shape):
    return np.abs(rng.standard_cauchy(shape))

def regression_shrinkage_weight(variance, n, sigma):
    """Map prior variance to orthogonal-regression shrinkage."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        kappa = sigma ** 2 / (sigma ** 2 + n * variance)
    return np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=1.0)

def sample_coefficients(method, rng, size, n, p, groups, p0, sigma, calibrated=True):
    pg = p // groups
    tau0 = p0 / (p - p0) * sigma / np.sqrt(n) if calibrated else sigma
    if method in {'HORSESHOE', 'RHS', 'GR_RHS'}:
        tau = tau0 * half_cauchy(rng, size)
        local = half_cauchy(rng, (size, p))
        q = (tau[:, None] * local) ** 2
        if method == 'HORSESHOE':
            variance = q
        elif method == 'RHS':
            prior = PRIORS['GR_RHS']
            c2 = prior['slab_variance_scale'] * (rng.gamma(prior['alpha_b'], 1.0, size) / rng.gamma(prior['beta_b'], 1.0, size))
            variance = 1.0 / (1.0 / q + 1.0 / c2[:, None])
        else:
            prior = PRIORS['GR_RHS']
            numerator = rng.gamma(prior['alpha_b'], 1.0, (size, groups))
            denominator = rng.gamma(prior['beta_b'], 1.0, (size, groups))
            c2 = prior['slab_variance_scale'] * np.repeat(numerator / denominator, pg, axis=1)
            variance = 1.0 / (1.0 / q + 1.0 / c2)
    elif method == 'GHS':
        tau = tau0 * half_cauchy(rng, size)
        delta = np.repeat(half_cauchy(rng, (size, groups)), pg, axis=1)
        variance = (tau[:, None] * delta * half_cauchy(rng, (size, p))) ** 2
    elif method == 'GIGG':
        prior = PRIORS['GIGG']
        gigg_tau0 = GIGG_P0 / (p - GIGG_P0) * sigma / np.sqrt(n) if calibrated else sigma
        tau = gigg_tau0 * half_cauchy(rng, size)
        gamma2 = np.repeat(rng.gamma(prior['gamma2_shape'], 1.0 / prior['gamma2_rate'], (size, groups)), pg, axis=1)
        lambda2 = invgamma.rvs(prior['lambda2_shape'], scale=prior['lambda2_scale'], size=(size, p), random_state=rng)
        variance = tau[:, None] ** 2 * gamma2 * lambda2
    else:
        prior = PRIORS[method]
        if calibrated:
            r2 = rng.beta(prior['r2_a'], prior['r2_b'], size)
        else:
            r2 = rng.beta(1.0, 1.0, size)
        total = sigma ** 2 * r2 / (1.0 - r2)
        if method == 'R2D2':
            alloc = rng.dirichlet(np.full(p, prior['phi_concentration']), size=size)
        else:
            group = rng.dirichlet(np.full(groups, prior['phi_group_concentration']), size=size)
            within = rng.gamma(prior['phi_within_concentration'], 1.0 / prior['phi_within_rate'], (size, groups, pg))
            within /= within.sum(axis=2, keepdims=True)
            alloc = (group[:, :, None] * within).reshape(size, p)
        variance = total[:, None] * alloc
    beta = rng.normal(size=variance.shape) * np.sqrt(variance)
    kappa = regression_shrinkage_weight(variance, n, sigma)
    return (beta, kappa, variance)

def simulate_priors(draws, batch_size, seed, n, p, groups, p0, sigma, calibrated=True):
    rngs = {method: np.random.default_rng(child) for method, child in zip(SEED_ORDER, np.random.SeedSequence(seed).spawn(len(SEED_ORDER)))}
    kappas = {m: [] for m, _ in PRIOR_METHODS}
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        for method, _ in PRIOR_METHODS:
            beta, kappa, _ = sample_coefficients(method, rngs[method], size, n, p, groups, p0, sigma, calibrated)
            flat_beta, flat_kappa = (beta.ravel(), kappa.ravel())
            take = min(40 * size, flat_beta.size)
            indices = rngs[method].choice(flat_beta.size, take, replace=False)
            kappas[method].append(flat_kappa[indices])
    return (None, kappas)
