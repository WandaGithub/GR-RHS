from __future__ import annotations

from pathlib import Path
from .design import Dataset

ROOT = Path(__file__).resolve().parents[2]

"""Stan data builders and common sampling path."""
from dataclasses import dataclass as dataclass
from pathlib import Path as Path
from typing import Any as Any
import numpy as np
import json
import arviz as az
import pandas as pd
METHODS = ('GR_RHS', 'RHS', 'HORSESHOE', 'R2D2', 'GROUP_R2', 'GHS', 'GIGG')
R2D2_R2_MEAN = 0.48
GROUP_R2_R2_MEAN = 0.57
R2D2_R2_PRECISION = 3.0
GIGG_P0 = 100.0

def validate_method_ids(methods, *, allowed=None):
    """Validate exact current method keys without historical aliases."""
    values = tuple(methods)
    invalid = sorted(set(values).difference(METHODS))
    if invalid:
        raise ValueError(f'Unsupported current method IDs: {invalid}')
    if allowed is not None:
        disallowed = sorted(set(values).difference(allowed))
        if disallowed:
            raise ValueError(f'Methods are not enabled by this experiment: {disallowed}')
    return values
COMMON_PRIOR = {'intercept_mean': 0.0, 'intercept_sd': 5.0, 'sigma_df': 3.0, 'sigma_location': 0.0, 'sigma_scale': None, 'z_mean': 0.0, 'z_sd': 1.0, 'tau_scale_multiplier': 1.0, 'slab_variance_scale': 1.0}
PRIORS = {'GR_RHS': dict(**COMMON_PRIOR, tau_calibrated=True, lambda_normal_mean=0.0, lambda_normal_sd=1.0, lambda_ig_shape=0.5, lambda_ig_scale=0.5, alpha_b=1.0, beta_b=1.0), 'HORSESHOE': dict(**COMMON_PRIOR, tau_calibrated=True, tau_location=0.0, lambda_location=0.0, lambda_scale=1.0), 'R2D2': dict(**COMMON_PRIOR, r2_a=R2D2_R2_MEAN * R2D2_R2_PRECISION, r2_b=(1.0 - R2D2_R2_MEAN) * R2D2_R2_PRECISION, phi_concentration=0.5), 'GROUP_R2': dict(**COMMON_PRIOR, r2_a=GROUP_R2_R2_MEAN * R2D2_R2_PRECISION, r2_b=(1.0 - GROUP_R2_R2_MEAN) * R2D2_R2_PRECISION, phi_group_concentration=0.5, phi_within_concentration=0.5, phi_within_rate=1.0), 'GHS': dict(**COMMON_PRIOR, tau_calibrated=True), 'GIGG': dict(**COMMON_PRIOR, tau_calibrated=True, tau_location=0.0, gamma2_shape=0.5, gamma2_rate=1.0, lambda2_shape=0.5, lambda2_scale=1.0)}
PRIORS['RHS'] = PRIORS['GR_RHS'].copy()

def prior_for(method_id, prior_config):
    prior = {**PRIORS[method_id], 'p0': float(prior_config['p0'])}
    if method_id in ('R2D2', 'GROUP_R2'):
        mean_key = 'r2d2_prior_mean' if method_id == 'R2D2' else 'group_r2_prior_mean'
        mean = float(prior_config[mean_key])
        precision = float(prior_config['r2_prior_precision'])
        prior.update(r2_a=mean * precision, r2_b=(1.0 - mean) * precision)
    if method_id in ('GR_RHS', 'RHS'):
        prior.update(slab_variance_scale=float(prior_config['slab_variance_scale']), alpha_b=float(prior_config['beta_prime_alpha']), beta_b=float(prior_config['beta_prime_beta']))
    return prior
MODEL_DIR = ROOT / 'stan'
MODEL_NAMES = {'GR_RHS': 'grrhs', 'RHS': 'rhs', 'HORSESHOE': 'horseshoe', 'R2D2': 'r2d2', 'GROUP_R2': 'group_r2', 'GHS': 'ghs', 'GIGG': 'gigg'}
POSTERIOR_VARIABLES = {'GR_RHS': {'beta': 'beta', 'intercept': 'b_intercept', 'sigma': 'sigma', 'tau': 'tau', 'lambda': 'lambda', 'c2_g': 'c2_g_out'}, 'RHS': {'beta': 'beta', 'intercept': 'b_intercept', 'sigma': 'sigma', 'tau': 'tau', 'lambda': 'lambda', 'c2': 'c2_g_out'}, 'HORSESHOE': {'beta': 'beta', 'intercept': 'b_intercept', 'sigma': 'sigma', 'tau': 'tau', 'lambda': 'lambda'}, 'GHS': {'beta': 'beta', 'intercept': 'b_intercept', 'sigma': 'sigma', 'tau': 'tau', 'lambda': 'lambda', 'delta': 'delta'}, 'GIGG': {'beta': 'beta', 'intercept': 'b_intercept', 'sigma': 'sigma', 'tau': 'tau', 'gamma2': 'gamma2', 'lambda2': 'lambda2'}, 'R2D2': {'beta': 'beta', 'intercept': 'b_intercept', 'sigma': 'sigma', 'R2': 'R2', 'phi': 'phi', 'tau2': 'tau2', 'lambda2': 'lambda2'}, 'GROUP_R2': {'beta': 'beta', 'intercept': 'b_intercept', 'sigma': 'sigma', 'R2': 'R2', 'phi_group': 'phi_group', 'lambdas': 'lambdas', 'tau2': 'tau2', 'lambda2_groups': 'lambda2_groups'}}

@dataclass
class Fit:
    beta: np.ndarray
    intercept: np.ndarray
    stan_fit: Any
    seed: int

    @property
    def beta_mean(self) -> np.ndarray:
        return self.beta.reshape(-1, self.beta.shape[-1]).mean(axis=0)

def _common_data(dataset: Dataset, prior: dict[str, Any]) -> dict[str, Any]:
    X = np.column_stack((np.ones(dataset.X.shape[0]), dataset.X))
    X_test = np.column_stack((np.ones(dataset.X_test.shape[0]), dataset.X_test))
    sigma_scale = dataset.y.std(ddof=1) if prior['sigma_scale'] is None else prior['sigma_scale']
    return {'N': dataset.X.shape[0], 'y': dataset.y, 'p': X.shape[1], 'X': X, 'Ntest': dataset.X_test.shape[0], 'ytest': dataset.y_test, 'Xtest': X_test, 'intercept_mean': prior['intercept_mean'], 'intercept_sd': prior['intercept_sd'], 'sigma_df': prior['sigma_df'], 'sigma_location': prior['sigma_location'], 'sigma_scale': sigma_scale, 'z_mean': prior['z_mean'], 'z_sd': prior['z_sd']}

def build_grrhs_common_data(dataset: Dataset, prior: dict[str, Any]):
    group_count = dataset.groups.max() + 1
    slab_variance_scale = prior['slab_variance_scale']
    float(slab_variance_scale)
    return {**_common_data(dataset, prior), 'G': group_count, 'group_id': dataset.groups + 1, 'p0': prior['p0'], 'use_tau_calibration': int(prior['tau_calibrated']), 'tau_scale_multiplier': prior['tau_scale_multiplier'], 'use_tau_fixed': int(prior.get('tau_fixed', None) is not None), 'tau_fixed': float(prior.get('tau_fixed', 1.0) or 1.0), 'alpha_b': float(prior.get('alpha_b', 1.0)), 'beta_b': float(prior.get('beta_b', 1.0)), 'slab_variance_scale': slab_variance_scale, 'lambda_normal_mean': prior['lambda_normal_mean'], 'lambda_normal_sd': prior['lambda_normal_sd'], 'lambda_ig_shape': prior['lambda_ig_shape'], 'lambda_ig_scale': prior['lambda_ig_scale']}

def build_grrhs_data(dataset: Dataset, prior: dict[str, Any]):
    return (build_grrhs_common_data(dataset, prior), None)

def build_rhs_data(dataset: Dataset, prior: dict[str, Any]):
    """Use one BetaPrime slab for all coefficients while retaining DGP groups.

    The original group labels remain on ``dataset`` for group-level metrics;
    only the Stan slab partition is collapsed to one shared slab.
    """
    data = build_grrhs_common_data(dataset, prior)
    data['G'] = 1
    data['group_id'] = np.ones(dataset.X.shape[1], dtype=int)
    return (data, None)

def build_horseshoe_data(dataset: Dataset, prior: dict[str, Any]):
    return ({**_common_data(dataset, prior), 'p0': prior['p0'], 'use_tau_calibration': int(prior['tau_calibrated']), 'tau_location': prior['tau_location'], 'lambda_location': prior['lambda_location'], 'lambda_scale': prior['lambda_scale']}, None)

def build_r2d2_data(dataset: Dataset, prior: dict[str, Any]):
    shape1 = prior['r2_a']
    shape2 = prior['r2_b']
    return ({**_common_data(dataset, prior), 'R2_a': shape1 / (shape1 + shape2), 'R2_b': shape1 + shape2, 'phi_concentration': prior['phi_concentration']}, None)

def build_group_r2_data(dataset: Dataset, prior: dict[str, Any]):
    groups = [np.flatnonzero(dataset.groups == group) for group in np.unique(dataset.groups)]
    order = np.concatenate(groups)
    restore = np.argsort(order)
    ordered = Dataset(dataset.X[:, order], dataset.y, dataset.X_test[:, order], dataset.y_test, dataset.groups[order], dataset.beta)
    shape1, shape2 = (prior['r2_a'], prior['r2_b'])
    return ({**_common_data(ordered, prior), 'G': len(groups), 'pg': [group.size for group in groups], 'phi_group_concentration': prior['phi_group_concentration'], 'phi_within_concentration': prior['phi_within_concentration'], 'phi_within_rate': prior['phi_within_rate'], 'R2_a': shape1 / (shape1 + shape2), 'R2_b': shape1 + shape2}, restore)

def build_ghs_data(dataset: Dataset, prior: dict[str, Any]):
    return ({**_common_data(dataset, prior), 'G': dataset.groups.max() + 1, 'group_id': dataset.groups + 1, 'p0': prior['p0'], 'use_tau_calibration': int(prior['tau_calibrated'])}, None)

def build_gigg_data(dataset: Dataset, prior: dict[str, Any]):
    group_count = dataset.groups.max() + 1
    return ({**_common_data(dataset, prior), 'G': group_count, 'group_id': dataset.groups + 1, 'tau_location': prior['tau_location'], 'p0': prior['p0'], 'use_tau_calibration': int(prior['tau_calibrated']), 'gamma2_shape': np.full(group_count, prior['gamma2_shape']), 'gamma2_rate': prior['gamma2_rate'], 'lambda2_shape': np.full(dataset.X.shape[1], prior['lambda2_shape']), 'lambda2_scale': prior['lambda2_scale']}, None)
BUILDERS = {'GR_RHS': build_grrhs_data, 'RHS': build_rhs_data, 'HORSESHOE': build_horseshoe_data, 'R2D2': build_r2d2_data, 'GROUP_R2': build_group_r2_data, 'GHS': build_ghs_data, 'GIGG': build_gigg_data}

def build_model(method_id: str):
    from cmdstanpy import CmdStanModel
    if method_id not in MODEL_NAMES:
        raise ValueError(f'Unknown current method: {method_id}')
    name = MODEL_NAMES[method_id]
    stan_file = MODEL_DIR / f'{name}.stan'
    return CmdStanModel(stan_file=str(stan_file))

def compile_models(method_ids: list[str]) -> None:
    for method_id in method_ids:
        build_model(method_id)

def fit_method(method_id: str, dataset: Dataset, *, seed: int, sampler: dict, prior: dict[str, Any], output_dir: Path | None=None) -> Fit:
    stan_data, restore = BUILDERS[method_id](dataset, prior)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    result = build_model(method_id).sample(data=stan_data, seed=seed, chains=sampler['chains'], parallel_chains=sampler['chains'], iter_warmup=sampler['warmup'], iter_sampling=sampler['draws'], adapt_delta=sampler['adapt_delta'], max_treedepth=sampler['max_treedepth'], show_progress=False, output_dir=str(output_dir) if output_dir is not None else None)
    beta = result.stan_variable('beta')
    if restore is not None:
        beta = beta[:, restore]
    return Fit(beta=beta, intercept=result.stan_variable('b_intercept'), stan_fit=result, seed=seed)

def save_posterior_core(fit: Fit, method_id: str, output_dir: Path) -> Path:
    """Persist the small primitive posterior archive used after CSV cleanup."""
    arrays = {name: np.asarray(fit.stan_fit.stan_variable(variable)) for name, variable in POSTERIOR_VARIABLES[method_id].items()}
    path = output_dir / 'posterior_core.npz'
    save_posterior_archive(arrays, path)
    return path

def save_posterior_archive(arrays: dict[str, np.ndarray], path: Path) -> None:
    """Write the compact posterior archive used by the publication analysis."""
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    np.savez_compressed(path, **payload)

def cleanup_raw_csv(fit: Fit) -> None:
    """Remove only the raw CmdStan CSVs belonging to this completed fit."""
    for csv_path in fit.stan_fit.runset.csv_files:
        Path(csv_path).unlink()

def sampler_diagnostics(fit: Fit) -> dict[str, float]:
    posterior = az.from_cmdstanpy(posterior=fit.stan_fit)
    # The inferential target for the regression study is beta.  Restrict
    # parameter-wise convergence summaries to beta so that auxiliary scales,
    # slab variables, and other nuisance quantities cannot determine the
    # reported R-hat/ESS extrema.  Divergences remain global HMC diagnostics
    # and are therefore computed over all post-warmup transitions below.
    summary = az.summary(posterior, var_names=['beta'], kind='diagnostics')
    r_hat = pd.to_numeric(summary['r_hat'], errors='coerce')
    ess_bulk = pd.to_numeric(summary['ess_bulk'], errors='coerce')
    ess_tail = pd.to_numeric(summary['ess_tail'], errors='coerce')
    return {'rhat_max': float(r_hat.max()), 'ess_bulk_min': float(ess_bulk.min()), 'ess_tail_min': float(ess_tail.min()), 'divergence_rate': float(posterior.sample_stats['diverging'].mean())}

def persist_fit(fit, method_id, output_dir, fit_result, identifiers, group_rows, true_group_slab_scales=None, recovery=None, slab_rows=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if recovery is not None:
        fit_result.update(recovery)
        pd.DataFrame([{**identifiers, **row} for row in slab_rows]).to_csv(output_dir / 'slab_recovery.csv', index=False)
    pd.DataFrame([{**identifiers, 'true_c_g': float(true_group_slab_scales[row['group']]), **row} for row in group_rows]).to_csv(output_dir / 'group_metrics.csv', index=False)
    save_posterior_core(fit, method_id, output_dir)
    cleanup_raw_csv(fit)
    (output_dir / 'result.json').write_text(json.dumps(fit_result, indent=2, sort_keys=True), encoding='utf-8')




RHS_OUTPUT_METHOD = 'RHS-BP'

def output_method(method_id: str) -> str:
    return RHS_OUTPUT_METHOD if method_id == 'RHS' else method_id

def fit_one_method(method_id, dataset, *, seed, sampler, prior, output_dir=None):
    return fit_method(method_id, dataset, seed=seed, sampler=sampler, prior=prior, output_dir=output_dir)
