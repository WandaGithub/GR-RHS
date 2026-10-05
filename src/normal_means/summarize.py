"""Fit-level and paired summaries for the normal-means experiment."""

import json

import numpy as np
import pandas as pd
from scipy.stats import t

from .design import iter_cases, write_json


def aggregate_with_intervals(summary_df, keys, value):
    rows = []
    for key, group in summary_df.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        values = group[value].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        mean = float(np.mean(values))
        half = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
        rows.append(dict(zip(keys, key), n=len(values), mean=mean, lower=mean - half, upper=mean + half))
    return pd.DataFrame(rows)


def load_fit_results(config, output_dir):
    rows, diagnostics, taus = [], [], []
    for case in iter_cases(config):
        result_path = output_dir / 'fits' / case['name'] / 'result.json'
        fit_result = json.loads(result_path.read_text())
        key = {name: case[name] for name in ('rep', 'amplitude', 'model', 'arm')}
        rows.extend(dict(key, **row) for row in fit_result['rows'])
        diagnostics.append(dict(key, **fit_result['diagnostics']))
        taus.append(dict(key, tau=fit_result['tau_median']))
    return rows, diagnostics, taus


def summarize_experiment(config, output_dir):
    rows, diagnostics, taus = load_fit_results(config, output_dir)
    dest = output_dir / 'summary'
    dest.mkdir(exist_ok=True)
    fit_metrics_df, diagnostics_df, tau_summary_df = (pd.DataFrame(rows), pd.DataFrame(diagnostics), pd.DataFrame(taus))
    fit_metrics_df.to_csv(dest / 'fit_metrics.csv', index=False)
    diagnostics_df.to_csv(dest / 'diagnostics.csv', index=False)
    tau_summary_df.to_csv(dest / 'tau_by_fit.csv', index=False)
    paired_mse_df = fit_metrics_df.dropna(subset=['mse']).pivot(index=['rep', 'amplitude', 'arm', 'stratum'], columns='model', values='mse')
    if set(config['models']).issubset(paired_mse_df.columns):
        paired_mse_df = paired_mse_df.dropna(subset=list(config['models']))
        paired_mse_df['difference'] = paired_mse_df['rhs'] - paired_mse_df['grrhs']
        paired_mse_df.reset_index().to_csv(dest / 'paired_mse.csv', index=False)
        contrasts = aggregate_with_intervals(paired_mse_df.reset_index(), ['amplitude', 'arm', 'stratum'], 'difference')
    else:
        contrasts = pd.DataFrame()
    contrasts.to_csv(dest / 'paired_mse_intervals.csv', index=False)
    n_expected = len(list(iter_cases(config)))
    status = dict(completed=len(diagnostics_df), expected=n_expected, complete=len(diagnostics_df) == n_expected, fits_needing_review=int(diagnostics_df.needs_review.sum()), experiment_id=config['experiment_id'])
    write_json(dest / 'status.json', status)
    print(json.dumps(status, indent=2))
