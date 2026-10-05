from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import logsumexp
import arviz as az

from .design import Dataset, generate_group_scale_dgp
from .fit import Fit, METHODS, validate_method_ids
from .fit import RHS_OUTPUT_METHOD

from scipy.stats import spearmanr as spearmanr

def _predictive(fit: Fit, dataset: Dataset):
    prediction = dataset.X_test @ fit.beta_mean + np.mean(fit.intercept)
    log_lik = fit.stan_fit.stan_variable('log_lik_test')
    elpd = logsumexp(log_lik, axis=0) - np.log(log_lik.shape[0])
    return (prediction, elpd)

def prediction_metrics(fit: Fit, dataset: Dataset) -> dict[str, float]:
    """Held-out predictive performance."""
    prediction, elpd = _predictive(fit, dataset)
    return {'elpd': float(elpd.sum()), 'test_rmse': float(np.sqrt(np.mean((dataset.y_test - prediction) ** 2)))}

def slab_recovery_metrics(c2_draws: np.ndarray, true_group_slab_scales: np.ndarray) -> tuple[dict, list[dict]]:
    """Evaluate recovery of the ordering, not equality, of group tail scales."""
    c_draws = np.sqrt(np.asarray(c2_draws))
    posterior_c = np.median(c_draws, axis=0)
    correlation = spearmanr(true_group_slab_scales, posterior_c).statistic if np.unique(true_group_slab_scales).size > 1 and np.unique(posterior_c).size > 1 else float('nan')
    comparable = 0
    correctly_ordered = 0
    for left in range(true_group_slab_scales.size):
        for right in range(left + 1, true_group_slab_scales.size):
            true_difference = true_group_slab_scales[left] - true_group_slab_scales[right]
            if true_difference == 0:
                continue
            comparable += 1
            estimated_difference = posterior_c[left] - posterior_c[right]
            correctly_ordered += int(np.sign(estimated_difference) == np.sign(true_difference))
    rows = [{'group': group, 'true_c_g': float(true_group_slab_scales[group]), 'posterior_c_median': float(posterior_c[group]), 'posterior_c_mean': float(np.mean(c_draws[:, group])), 'posterior_c_q025': float(np.quantile(c_draws[:, group], 0.025)), 'posterior_c_q975': float(np.quantile(c_draws[:, group], 0.975))} for group in range(true_group_slab_scales.size)]
    return ({'c_spearman': float(correlation), 'c_pairwise_order_accuracy': float(correctly_ordered / comparable) if comparable else float('nan')}, rows)

def coefficient_metrics(fit: Fit, dataset: Dataset) -> tuple[dict, list[dict]]:
    """Retain the mechanism study's mean squared errors without effect classes."""
    squared_error = np.square(fit.beta_mean - dataset.beta)
    active = dataset.beta != 0
    metrics = {'mse_overall': float(squared_error.mean()), 'mse_null': float(squared_error[~active].mean()), 'mse_nonnull': float(squared_error[active].mean())}
    groups = []
    for group in np.unique(dataset.groups):
        selected = dataset.groups == group
        groups.append({'group': int(group), 'mse_nonnull': float(squared_error[selected & active].mean())})
    return (metrics, groups)

def fit_result(fit, dataset, identifiers, sigma):
    from .fit import sampler_diagnostics
    metrics, groups = coefficient_metrics(fit, dataset)
    return ({**identifiers, 'sigma': sigma, **metrics, **prediction_metrics(fit, dataset), **sampler_diagnostics(fit)}, groups)


def flat_draws(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 0:
        return value.reshape(1)
    if value.ndim == 1:
        return value
    return value.reshape((-1,) + value.shape[-1:])

def scalar_summary(draws: np.ndarray, prefix: str) -> dict[str, float]:
    draws = np.asarray(draws, dtype=float).reshape(-1)
    return {f'{prefix}_mean': float(np.mean(draws)), f'{prefix}_q025': float(np.quantile(draws, 0.025)), f'{prefix}_q975': float(np.quantile(draws, 0.975)), f'{prefix}_sd': float(np.std(draws, ddof=1))}

def coverage_summary(draws: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    draws = np.asarray(draws, dtype=float)
    truth = np.asarray(truth, dtype=float)
    lower, upper = np.quantile(draws, (0.025, 0.975), axis=0)
    covered = (truth >= lower) & (truth <= upper)
    return (float(np.mean(covered)), float(np.mean(upper - lower)))

def load_scientific_manifest(root: Path) -> dict:
    paths = sorted(root.glob('manifest_replication_*.json'))
    full_run_manifest = root / 'manifest.json'
    if full_run_manifest.exists():
        paths = [full_run_manifest, *paths]
    if not paths:
        raise FileNotFoundError(f'No scientific manifest in {root}')
    manifests = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    config = manifests[0]['config']
    if any((item.get('config') != config for item in manifests[1:])):
        raise ValueError(f'Replication manifests disagree on scientific configuration: {root}')
    return manifests[0]

def analyze_posterior(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = load_scientific_manifest(root)
    config = manifest['config']
    dgp_config = config['dgp']
    rows: list[dict[str, float | int | str]] = []
    allowed_methods = set(validate_method_ids(config['methods']))
    if not allowed_methods:
        raise ValueError('Manifest has no enabled methods')
    for result_path in sorted(root.glob('fits/k*/**/result.json')):
        result = json.loads(result_path.read_text(encoding='utf-8'))
        method = result['method']
        if method not in allowed_methods:
            raise ValueError(f'Unexpected method ID in fresh-run results: {method}')
        replication = int(result['replication'])
        information = int(result['signals_per_group'])
        archive_path = result_path.with_name('posterior_core.npz')
        archive = np.load(archive_path)
        beta_draws = flat_draws(archive['beta'])
        intercept_draws = flat_draws(archive['intercept'])
        sigma_draws = flat_draws(archive['sigma'])
        data = generate_group_scale_dgp(seed=int(config['base_seed']) + replication - 1, condition=result['condition'], signals_per_group=information, **dgp_config)
        beta_truth = data.dataset.beta
        active = beta_truth != 0
        beta_mean = np.mean(beta_draws, axis=0)
        beta_lower, beta_upper = np.quantile(beta_draws, (0.025, 0.975), axis=0)
        posterior_rmse_by_coefficient = np.sqrt(np.mean((beta_draws - beta_truth[None, :]) ** 2, axis=0))
        beta_coverage = (beta_truth >= beta_lower) & (beta_truth <= beta_upper)
        prediction = data.dataset.X_test @ beta_mean + np.mean(intercept_draws)
        residual = data.dataset.y_test - prediction
        row = {'target_r2': result['target_r2'], 'condition': result['condition'], 'method': method, 'replication': replication, 'test_rmse': float(np.sqrt(np.mean(residual ** 2))), 'test_mae': float(np.mean(np.abs(residual))), 'test_r2': float(1.0 - residual @ residual / np.sum((data.dataset.y_test - data.dataset.y_test.mean()) ** 2)), 'test_bias': float(np.mean(residual)), 'posterior_rmse_overall': float(np.mean(posterior_rmse_by_coefficient)), 'posterior_rmse_nonnull': float(np.mean(posterior_rmse_by_coefficient[active])), 'posterior_rmse_null': float(np.mean(posterior_rmse_by_coefficient[~active])), 'beta_coverage_nonnull': float(np.mean(beta_coverage[active])), 'beta_coverage_null': float(np.mean(beta_coverage[~active])), 'beta_interval_width_overall': float(np.mean(beta_upper - beta_lower)), 'beta_interval_width_nonnull': float(np.mean((beta_upper - beta_lower)[active])), 'beta_interval_width_null': float(np.mean((beta_upper - beta_lower)[~active])), 'sigma_coverage': coverage_summary(sigma_draws, np.asarray(data.sigma))[0], 'sigma_interval_width': coverage_summary(sigma_draws, np.asarray(data.sigma))[1], **scalar_summary(sigma_draws, 'posterior_sigma')}
        for key in ('tau', 'R2'):
            if key in archive:
                row.update(scalar_summary(flat_draws(archive[key]), f'posterior_{key}'))
        if 'c2_g' in archive:
            c_draws = np.sqrt(flat_draws(archive['c2_g']))
            c_coverage, c_width = coverage_summary(c_draws, data.true_group_slab_scales)
            row.update({'c_coverage': c_coverage, 'c_interval_width': c_width, 'posterior_c_mean': float(np.mean(c_draws))})
        rows.append(row)
    observations = pd.DataFrame(rows)
    metric_columns = [c for c in observations.columns if c not in {'target_r2', 'condition', 'method', 'replication'}]
    summary_rows = []
    for keys, group in observations.groupby(['target_r2', 'condition', 'method'], sort=True):
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors='coerce').dropna()
            if values.empty:
                continue
            sd = values.std(ddof=1) if len(values) > 1 else float('nan')
            se = sd / math.sqrt(len(values)) if len(values) > 1 else float('nan')
            mean = values.mean()
            summary_rows.append({'target_r2': keys[0], 'condition': keys[1], 'method': keys[2], 'metric': metric, 'n': len(values), 'mean': mean, 'sd': sd, 'se': se, 'q25': values.quantile(0.25), 'median': values.median(), 'q75': values.quantile(0.75), 'q025': values.quantile(0.025), 'q975': values.quantile(0.975), 'ci95_low': mean - 1.96 * se if len(values) > 1 else float('nan'), 'ci95_high': mean + 1.96 * se if len(values) > 1 else float('nan')})
    return (observations, pd.DataFrame(summary_rows))

"""Prior-predictive effective-size samplers shared by calibration analyses."""
SCENARIOS = [(r, c) for r in (0.3, 0.7) for c in ('hom', 'het_tail')]
MAIN = Path(__file__).resolve().parents[2] / 'results/regression/confirmatory'
RHS_BP = Path(__file__).resolve().parents[2] / 'results/regression' / RHS_OUTPUT_METHOD / 'confirmatory'
MAIN_POST = MAIN / 'analysis_posterior'
RHS_BP_POST = RHS_BP.parent / 'analysis_posterior'
def read_fits(root, r2, condition, methods):
    folder = root / f'r2_{r2}_{condition}'
    files = sorted(folder.glob('results_replication_*.csv'))
    if not files:
        raise FileNotFoundError(f'No replication files found in {folder}')
    data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    return data[data.method.isin(methods)].copy()

def read_posterior(root, r2, condition, methods):
    path = root / f'r2_{r2}_{condition}' / 'posterior_observations.csv'
    data = pd.read_csv(path)
    return data[data.method.isin(methods)].copy()

def load():
    base = ['GR_RHS', 'HORSESHOE', 'R2D2', 'GROUP_R2', 'GHS', 'GIGG']
    fits, posts = ([], [])
    for r2, condition in SCENARIOS:
        fits.append(read_fits(MAIN, r2, condition, base))
        fits.append(read_fits(RHS_BP, r2, condition, ['RHS']))
        posts.append(read_posterior(MAIN_POST, r2, condition, base))
        posts.append(read_posterior(RHS_BP_POST, r2, condition, ['RHS']))
    keys = ['target_r2', 'condition', 'method', 'replication']
    fits = pd.concat(fits, ignore_index=True)
    posts = pd.concat(posts, ignore_index=True)
    assert not fits.duplicated(keys).any() and (not posts.duplicated(keys).any())
    assert (fits.groupby(keys[:-1]).size() == 100).all()
    assert (posts.groupby(keys[:-1]).size() == 100).all()
    return (fits, posts)

def load_low_signal_fits():
    """Load only the fit metrics used by the R-squared 0.3 figure."""
    base = ['GR_RHS', 'HORSESHOE', 'R2D2', 'GROUP_R2', 'GHS', 'GIGG']
    pieces = []
    expected_methods = set(base + ['RHS'])
    for condition in ('hom', 'het_tail'):
        pieces.append(read_fits(MAIN, 0.3, condition, base))
        pieces.append(read_fits(RHS_BP, 0.3, condition, ['RHS']))
    fits = pd.concat(pieces, ignore_index=True)
    keys = ['target_r2', 'condition', 'method', 'replication']
    if fits.duplicated(keys).any():
        raise ValueError('Duplicate low-signal method/replication fit rows')
    for condition in ('hom', 'het_tail'):
        group = fits[(fits.target_r2 == 0.3) & (fits.condition == condition)]
        counts = group.groupby('method').size().to_dict()
        if counts != {method: 100 for method in expected_methods}:
            raise ValueError(f'Expected 100 low-signal replications for each method in {condition}; got {counts}')
        if group.groupby('replication').method.apply(set).map(lambda methods: methods == expected_methods).eq(False).any():
            raise ValueError(f'Low-signal methods are not paired by replication in {condition}')
        if group.groupby('replication').sigma.nunique().ne(1).any():
            raise ValueError(f'Low-signal methods do not share sigma within replications in {condition}')
    return fits

def low_signal_paired_contrasts(fits):
    """Compute only the four fit-metric contrasts plotted at R-squared 0.3."""
    rows = []
    metrics = ('elpd', 'mse_overall', 'mse_null', 'mse_nonnull')
    for condition in ('hom', 'het_tail'):
        group = fits[(fits.target_r2 == 0.3) & (fits.condition == condition)]
        wide = group.pivot(index='replication', columns='method', values=list(metrics))
        methods = ['GR_RHS'] + SUPP_METHODS
        required = [(metric, method) for metric in metrics for method in methods]
        if wide.index.nunique() != 100 or wide.loc[:, required].isna().any().any():
            raise ValueError(f'Incomplete paired low-signal contrasts in {condition}')
        for metric in metrics:
            for method in SUPP_METHODS:
                delta = wide[metric, 'GR_RHS'] - wide[metric, method]
                if metric not in HIGHER_IS_BETTER:
                    delta = -delta
                mean = delta.mean()
                margin = t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
                rows.append({'target_r2': 0.3, 'condition': condition, 'metric': metric, 'comparator': method, 'n': len(delta), 'mean': mean, 'ci_low': mean - margin, 'ci_high': mean + margin})
    return pd.DataFrame(rows).set_index(['target_r2', 'condition', 'metric', 'comparator']).sort_index()
def paired_contrasts(fits, posts):
    rows = []
    for source, metrics in ((fits, FIT_METRICS), (posts, POST_METRICS)):
        for r2, condition in SCENARIOS:
            group = source[(source.target_r2 == r2) & (source.condition == condition)]
            for metric in metrics:
                wide = group.pivot(index='replication', columns='method', values=metric)
                assert wide.index.nunique() == 100 and (not wide[SUPP_METHODS + ['GR_RHS']].isna().any().any())
                for method in SUPP_METHODS:
                    delta = wide['GR_RHS'] - wide[method]
                    if metric not in HIGHER_IS_BETTER:
                        delta = -delta
                    mean = delta.mean()
                    margin = t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
                    rows.append(dict(target_r2=r2, condition=condition, metric=metric, comparator=method, mean=mean, ci_low=mean - margin, ci_high=mean + margin))
    return pd.DataFrame(rows).set_index(['target_r2', 'condition', 'metric', 'comparator']).sort_index()
