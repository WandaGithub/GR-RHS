"""CmdStan fitting, posterior extraction, and fit diagnostics."""

from pathlib import Path

import numpy as np

from .design import build_model_data, generate_dataset, write_json

ROOT = Path(__file__).resolve().parents[2]
STAN_MODELS = {
    'rhs': ROOT / 'stan/normal_means_rhs.stan',
    'grrhs': ROOT / 'stan/normal_means_grrhs.stan',
}


def build_model(model_name):
    from cmdstanpy import CmdStanModel
    return CmdStanModel(stan_file=str(STAN_MODELS[model_name]))


def compute_diagnostics(fit, config, case):
    import arviz as az

    names = ['log_lambda', 'log_c2'] + (['log_tau_free'] if case['arm'] == 'learned' else [])
    posterior = fit.draws_xr(vars=names)
    pars = az.summary(posterior, kind='diagnostics', round_to='none')
    method = fit.method_variables()
    rhats = pars['r_hat'].to_numpy()
    ess = pars['ess_bulk'].to_numpy()
    diagnostics = dict(divergences=int(method['divergent__'].sum()), treedepth_hits=int((method['treedepth__'] >= config['max_treedepth']).sum()), max_rhat=float(np.max(rhats)) if np.isfinite(rhats).all() else None, min_bulk_ess=float(np.min(ess)) if np.isfinite(ess).all() else None)
    diagnostics['needs_review'] = bool(diagnostics['divergences'] or diagnostics['treedepth_hits'] or diagnostics['max_rhat'] is None or (diagnostics['max_rhat'] >= 1.01) or (diagnostics['min_bulk_ess'] is None) or (diagnostics['min_bulk_ess'] < 400))
    return diagnostics


def extract_result(fit, config, dataset, case):
    beta = np.asarray(dataset['beta'])
    estimate = fit.stan_variable('beta_conditional_mean').mean(axis=0)
    kappa = fit.stan_variable('kappa').mean(axis=0)
    ratio = fit.stan_variable('attenuation_ratio').mean(axis=0)
    floor = fit.stan_variable('slab_floor').mean(axis=0)
    logc = np.median(fit.stan_variable('log_c2'), axis=0)
    tau_med = float(np.median(fit.stan_variable('tau')))
    groups = np.asarray(dataset['group_id'])
    reference_start = sum(config['group_sizes'][:config['reference_group']])
    varying_start = sum(config['group_sizes'][:config['varying_group']])
    count = config['signals_per_active_group']
    masks = {'target': np.isin(np.arange(config['p']), np.arange(varying_start, varying_start + count)), 'reference': np.isin(np.arange(config['p']), np.arange(reference_start, reference_start + count)), 'target_within_null': np.isin(np.arange(config['p']), np.arange(varying_start + count, varying_start + config['group_sizes'][config['varying_group']])), 'reference_within_null': np.isin(np.arange(config['p']), np.arange(reference_start + count, reference_start + config['group_sizes'][config['reference_group']])), 'null': beta == 0, 'external_null': (beta == 0) & (groups != 2), 'all': np.ones(config['p'], dtype=bool)}
    rows = []
    for name, mask in masks.items():
        rows.append(dict(stratum=name, mse=float(np.mean((estimate[mask] - beta[mask]) ** 2)), kappa=float(kappa[mask].mean()), ratio=float(ratio[mask].mean()), count=int(mask.sum())))
    for group in range(1, len(config['group_sizes']) + 1):
        slab_index = 0 if case['model'] == 'rhs' else group - 1
        rows.append(dict(stratum=f'group{group}', slab_floor=float(floor[slab_index]), log_c2_median=float(logc[slab_index]), ratio=float(ratio[groups == group].mean())))
    return dict(case=case, tau_median=tau_med, rows=rows, diagnostics=compute_diagnostics(fit, config, case), beta_posterior_mean=estimate.tolist())


def fit_one_case(model, config, output_dir, case):
    fit_dir = output_dir / 'fits' / case['name']
    dataset = generate_dataset(config, case['rep'], case['amplitude'])
    fit_dir.mkdir(parents=True, exist_ok=True)
    write_json(fit_dir / 'input.json', build_model_data(config, dataset, case['model'], case['arm']))
    print('Running', case['name'], flush=True)
    fit = model.sample(data=str(fit_dir / 'input.json'), seed=case['fit_seed'], chains=config['chains'], parallel_chains=config['parallel_chains'], iter_warmup=config['iter_warmup'], iter_sampling=config['iter_sampling'], adapt_delta=config['adapt_delta'], max_treedepth=config['max_treedepth'], output_dir=str(fit_dir), show_progress=False, refresh=100)
    result = extract_result(fit, config, dataset, case)
    write_json(fit_dir / 'result.json', result)
    print('Completed', case['name'], result['diagnostics'], flush=True)
    return result
