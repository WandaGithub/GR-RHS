"""Public orchestration entry point for the correlated-regression study."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd

from . import calibration, mechanism, plot, summarize
from .design import generate_group_scale_dgp
from .fit import METHODS, compile_models, fit_method, fit_one_method, output_method, persist_fit, prior_for, RHS_OUTPUT_METHOD
from .design import seed_for

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'config/regression_paper.json'


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')


def run_replication(task):
    config, scale_scenario, information, replication, root = task
    data = generate_group_scale_dgp(
        seed=config['base_seed'] + replication - 1,
        condition=scale_scenario,
        signals_per_group=information,
        **config['dgp'],
    )
    relative = Path(f'k{information}') / scale_scenario / f'replication_{replication:03d}'
    rows = []
    prior_config = {
        'p0': config['prior_p0'],
        'r2d2_prior_mean': config['r2d2_prior_mean'],
        'group_r2_prior_mean': config['group_r2_prior_mean'],
        'r2_prior_precision': config['r2_prior_precision'],
        'slab_variance_scale': config['slab_variance_scale'],
        'beta_prime_alpha': config['beta_prime_alpha'],
        'beta_prime_beta': config['beta_prime_beta'],
    }
    for method_id in config['methods']:
        output = root / 'fits' / relative / output_method(method_id)
        prior = prior_for(method_id, prior_config)
        sampler = {**config['sampler'], **config.get('sampler_overrides', {}).get(method_id, {})}
        seed_method = 'RHS' if method_id == 'RHS' else method_id
        fitted = fit_one_method(
            method_id,
            data.dataset,
            seed=seed_for(config['fit_seed_namespace'], replication, f'{seed_method}-{information}-{scale_scenario}'),
            sampler=sampler,
            prior=prior,
            output_dir=output,
        )
        assert fitted.stan_fit.chains == sampler['chains']
        assert fitted.stan_fit.num_draws_sampling == sampler['draws']
        assert fitted.beta.shape[0] == sampler['chains'] * sampler['draws']
        identifiers = dict(replication=replication, signals_per_group=information, target_r2=config['dgp']['target_r2'], condition=scale_scenario, method=method_id)
        result, groups = summarize.fit_result(fitted, data.dataset, identifiers, data.sigma)
        recovery = slab_rows = None
        if method_id == 'GR_RHS':
            recovery, slab_rows = summarize.slab_recovery_metrics(fitted.stan_fit.stan_variable('c2_g_out'), data.true_group_slab_scales)
        persist_fit(fitted, method_id, output, result, identifiers, groups, data.true_group_slab_scales, recovery, slab_rows)
        rows.append(result)
        print(f'Completed {scale_scenario} k={information} rep={replication} {method_id}', flush=True)
    return rows


def run_study(config_path=DEFAULT_CONFIG, output=None, replication=None, methods=None, condition=None, target_r2=None, jobs=1, settings=None):
    config = json.loads(Path(config_path).read_text(encoding='utf-8')) if settings is None else json.loads(json.dumps(settings))
    configured_methods = tuple(config['methods'])
    if methods is not None:
        config['methods'] = list(methods)
    if condition is not None:
        config['conditions'] = [condition]
    if target_r2 is not None:
        config['target_r2_levels'] = [target_r2]
        config['dgp']['target_r2'] = target_r2
    elif len(config['target_r2_levels']) == 1:
        config['dgp']['target_r2'] = config['target_r2_levels'][0]
    else:
        raise ValueError('Select one target with --target-r2; use the paper launcher for all targets.')
    if len(config['conditions']) != 1:
        raise ValueError('Select one condition with --condition; use the paper launcher for all conditions.')
    if any(method not in METHODS for method in config['methods']):
        raise ValueError(f"Configuration contains a non-current method ID: {config['methods']}")
    if methods is not None:
        unknown = set(methods).difference(METHODS)
        outside_config = set(methods).difference(configured_methods)
        if unknown or outside_config:
            raise ValueError(f'Methods must be current IDs included in the config; invalid={sorted(unknown | outside_config)}')
    if output is not None:
        config['output'] = str(output)
    condition_name = config['conditions'][0]
    target = config['dgp']['target_r2']
    root = ROOT / (Path(output) if output is not None else Path(config['output']) / f'r2_{target}_{condition_name}')
    root.mkdir(parents=True, exist_ok=True)
    suffix = '' if replication is None else f'_replication_{replication:03d}'
    _write_json(root / f'manifest{suffix}.json', {'config': config, 'replication': replication})
    compile_models(config['methods'])
    replications = [replication] if replication is not None else range(1, config['n_replications'] + 1)
    tasks = [(config, condition_name, information, rep, root) for information in config['information_levels'] for rep in replications]

    def save_rows(batches):
        with (root / f'results{suffix}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = None
            for rows in batches:
                for row in rows:
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
    if jobs == 1:
        save_rows(map(run_replication, tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            save_rows(pool.map(run_replication, tasks))
    print(f'Wrote results under {root}')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('run', 'posterior', 'figure3', 'low-signal', 'figure4', 'supplement', 'prior-shape', 'variance', 'calibrate-r2', 'tables'))
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.stage == 'run':
        p = argparse.ArgumentParser()
        p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        p.add_argument('--target', choices=('all', 'baseline', 'rhs-bp'), default='all')
        p.add_argument('--target-r2', nargs='+', type=float)
        p.add_argument('--condition', nargs='+', choices=('hom', 'het_tail'))
        p.add_argument('--replication', type=int)
        p.add_argument('--jobs', type=int, default=1)
        x = p.parse_args(args.args)
        config = json.loads(x.config.read_text(encoding='utf-8'))
        targets = x.target_r2 or config['target_r2_levels']
        conditions = x.condition or config['conditions']
        baseline = [method for method in config['methods'] if method != 'RHS']
        selections = []
        if x.target in ('all', 'baseline') and baseline:
            selections.append((baseline, ROOT / 'results/regression/confirmatory'))
        if x.target in ('all', 'rhs-bp') and 'RHS' in config['methods']:
            selections.append((['RHS'], ROOT / 'results/regression' / RHS_OUTPUT_METHOD / 'confirmatory'))
        for target in targets:
            for cond in conditions:
                for selected, archive in selections:
                    output = archive / f'r2_{target}_{cond}'
                    run_study(x.config, output=output, replication=x.replication, methods=selected, condition=cond, target_r2=target, jobs=x.jobs)
                    observations, summary = summarize.analyze_posterior(output)
                    destination = archive / 'analysis_posterior' / f'r2_{target}_{cond}'
                    destination.mkdir(parents=True, exist_ok=True)
                    observations.to_csv(destination / 'posterior_observations.csv', index=False)
                    summary.to_csv(destination / 'posterior_summary_statistics.csv', index=False)
    elif args.stage == 'posterior':
        p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
        x = p.parse_args(args.args); x.output.mkdir(parents=True, exist_ok=True)
        observations, summary = summarize.analyze_posterior(x.root)
        observations.to_csv(x.output / 'posterior_observations.csv', index=False); summary.to_csv(x.output / 'posterior_summary_statistics.csv', index=False)
    elif args.stage in ('figure3', 'low-signal'):
        plot.build_figure3(low_signal=args.stage == 'low-signal')
    elif args.stage == 'figure4':
        mechanism.build_slab_summary(); mechanism.build_mechanism_data(); plot.build_mechanism_figure()
    elif args.stage == 'prior-shape':
        kappas = calibration.simulate_priors(200000, 1000, 20260903, 200, 500, 5, 100.0, 1.0)[1]
        plot.build_prior_shape_figure(kappas, kappas, 'prior_shape_pair')
    elif args.stage == 'variance':
        plot.build_variance_figure()
    elif args.stage == 'calibrate-r2':
        p = argparse.ArgumentParser()
        p.add_argument('--n', type=int, default=200); p.add_argument('--p', type=int, default=500)
        p.add_argument('--groups', type=int, default=5); p.add_argument('--p0', type=float, default=100.0)
        p.add_argument('--r2-precision', type=float, default=3.0); p.add_argument('--draws', type=int, default=50000)
        p.add_argument('--seed', type=int, default=20260903); p.add_argument('--tol', type=float, default=0.05)
        p.add_argument('--max-iter', type=int, default=40)
        p.add_argument('--output', type=Path, default=ROOT / 'results/figure_data/diagnostics/r2_prior_calibration')
        x = p.parse_args(args.args)
        hs_probe = calibration.sample_prior_meff(draws=x.draws, batch_size=min(x.draws, 1000), seed=x.seed, n=x.n, p=x.p, groups=x.groups, p0=x.p0, sigma=1.0, selected_methods=('HORSESHOE',))
        hs_median = float(np.median(hs_probe['HORSESHOE']))
        print(f'HS median m_eff at p0={x.p0}: {hs_median:.2f}')
        r2_means = {}
        for method, seed_offset in (('R2D2', 0), ('GROUP_R2', 10000)):
            r2_mean, search_median, iterations = calibration.calibrate_r2_mean(method, hs_median, x.r2_precision, n=x.n, p=x.p, groups=x.groups, seed=x.seed + seed_offset, draws=x.draws, tol=x.tol, max_iter=x.max_iter)
            r2_means[method] = r2_mean
            print(f'{method} r2_mean={r2_mean:.4f} (r2_precision={x.r2_precision}), search median={search_median:.2f}, {iterations} iterations')
            calibration.PRIORS[method]['r2_a'] = r2_mean * x.r2_precision
            calibration.PRIORS[method]['r2_b'] = (1.0 - r2_mean) * x.r2_precision
        samples = calibration.sample_prior_meff(draws=200000, batch_size=1000, seed=x.seed + 1000, n=x.n, p=x.p, groups=x.groups, p0=x.p0, sigma=1.0)
        summary = calibration.summarize(samples, hs_median, r2_means)
        summary.insert(1, 'r2_precision', x.r2_precision)
        x.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(x.output.with_name(x.output.name + '_summary.csv'), index=False)
        print(summary.to_string(index=False, float_format=lambda value: f'{value:.3f}'))
        print(f"Wrote {x.output.with_name(x.output.name + '_summary.csv')}")
    elif args.stage == 'tables':
        mechanism.build_null_summary(); mechanism.build_mechanism_tables()
    elif args.stage == 'supplement':
        main_args = ['low-signal']; main_args.extend([]); plot.build_figure3(low_signal=True)
        kappas = calibration.simulate_priors(200000, 1000, 20260903, 200, 500, 5, 100.0, 1.0)[1]; plot.build_prior_shape_figure(kappas, kappas, 'prior_shape_pair')
        plot.build_variance_figure(); mechanism.build_slab_summary(); mechanism.build_mechanism_data(); plot.build_mechanism_figure(); mechanism.build_null_summary(); mechanism.build_mechanism_tables()


if __name__ == '__main__':
    main()
