"""Design and deterministic inputs for the normal-means experiment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if sum(config['group_sizes']) != config['p'] or config['replications'] < 1 or (not 0 < config['p0'] < config['p']):
        raise ValueError('Invalid replications or p0.')
    if config['models'] != ['rhs', 'grrhs'] or config['arms'] != ['fixed', 'learned']:
        raise ValueError('The paper experiment requires both models and both tau arms.')
    if config['experiment_id'] == 'normal_means_paper' and (config['p'] != 50 or config['group_sizes'] != [10] * 5 or config['replications'] != 50 or (len(config['amplitudes']) != 7)):
        raise ValueError('Paper normal-means configuration does not match the manuscript design.')
    if config['experiment_id'] == 'normal_means_paper' and config['amplitudes'] != [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        raise ValueError('Paper normal-means amplitude grid differs from the manuscript design.')
    if len(set(config['amplitudes'])) != len(config['amplitudes']) or any((a <= 0 for a in config['amplitudes'])):
        raise ValueError('Amplitudes must be positive and distinct.')
    if config['reference_signal'] <= 0:
        raise ValueError('The reference signal must be positive.')
    for key in ('slab_variance_scale', 'alpha_b', 'beta_b', 'noise_scale'):
        if config[key] <= 0:
            raise ValueError(f'{key} must be positive.')
    return config


def generate_dataset(config, replication_id, amplitude):
    seed = int(np.random.SeedSequence([config['seed'], replication_id, 0]).generate_state(1)[0])
    group_starts = np.cumsum([0, *config['group_sizes'][:-1]])
    reference_start = int(group_starts[config['reference_group']])
    varying_start = int(group_starts[config['varying_group']])
    beta = np.zeros(config['p'])
    count = config['signals_per_active_group']
    beta[reference_start:reference_start + count] = config['reference_signal']
    beta[varying_start:varying_start + count] = amplitude
    sigma = float(config['noise_scale'])
    noise = np.random.default_rng(seed).normal(0, sigma, config['p'])
    return dict(beta=beta.tolist(), noise=noise.tolist(), sigma=sigma, y=(beta + noise).tolist(), group_id=np.repeat(np.arange(1, len(config['group_sizes']) + 1), config['group_sizes']).tolist(), seed=seed)


def build_model_data(config, dataset, model, arm):
    model_data = dict(p=config['p'], y=dataset['y'], sigma=dataset['sigma'], learn_tau=int(arm == 'learned'), tau_reference=config['p0'] / (config['p'] - config['p0']) * dataset['sigma'], slab_variance_scale=config['slab_variance_scale'], alpha_b=config['alpha_b'], beta_b=config['beta_b'])
    if model == 'grrhs':
        model_data.update(K=len(config['group_sizes']), slab_id=dataset['group_id'])
    return model_data


def iter_cases(config):
    for replication_id in range(config['replications']):
        for amplitude_idx, amplitude in enumerate(config['amplitudes']):
            for model_idx, model in enumerate(config['models']):
                for tau_arm_idx, arm in enumerate(config['arms']):
                    seed = int(np.random.SeedSequence([config['seed'], replication_id, amplitude_idx, model_idx, tau_arm_idx, 1]).generate_state(1)[0])
                    yield dict(rep=replication_id, amplitude=amplitude, amplitude_index=amplitude_idx, model=model, arm=arm, fit_seed=seed, name=f'r{replication_id:03d}_a{amplitude_idx:02d}_{model}_{arm}')


def initialize_design(config, output_dir):
    for replication_id in range(config['replications']):
        for amplitude_idx, amplitude in enumerate(config['amplitudes']):
            write_json(output_dir / 'data' / f'r{replication_id:03d}_a{amplitude_idx:02d}.json', generate_dataset(config, replication_id, amplitude))
    pd.DataFrame(iter_cases(config)).to_csv(output_dir / 'jobs.csv', index=False)


def check_design(config):
    low, high = (generate_dataset(config, 0, config['amplitudes'][0]), generate_dataset(config, 0, config['amplitudes'][-1]))
    np.testing.assert_allclose(np.asarray(low['noise']) / low['sigma'], np.asarray(high['noise']) / high['sigma'])
    beta = np.asarray(low['beta'])
    assert np.count_nonzero(beta) == 10 and np.all(beta >= 0)
    rng = np.random.default_rng(42)
    c2, q = np.exp(rng.normal(size=(2, 100)))
    lv = -np.logaddexp(-np.log(c2), -np.log(q))
    np.testing.assert_allclose(np.exp(lv), c2 * q / (c2 + q))
    sigma = low['sigma']
    np.testing.assert_allclose(expit(2 * np.log(sigma) - lv), sigma ** 2 / (sigma ** 2 + np.exp(lv)))
    jobs = list(iter_cases(config))
    assert len({case['fit_seed'] for case in jobs}) == len(jobs)
    assert 'K' not in build_model_data(config, low, 'rhs', 'fixed')
    assert build_model_data(config, low, 'grrhs', 'learned')['slab_id'] == low['group_id']
    print(f'Checks passed: positive signals, paired noise, stable formulas, unique seeds; {len(jobs)} planned fits.')
