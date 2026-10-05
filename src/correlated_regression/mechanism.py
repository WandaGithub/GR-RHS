from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .design import generate_group_scale_dgp
from .fit import RHS_OUTPUT_METHOD

ROOT = Path(__file__).resolve().parents[2]

MECHANISM_MAIN = ROOT / 'results/regression/confirmatory'
MECHANISM_RHS_BP = ROOT / 'results/regression' / RHS_OUTPUT_METHOD / 'confirmatory'
MECHANISM_FIGURE_DATA = ROOT / 'results/figure_data'
MECHANISM_METHODS = (('GR-RHS', MECHANISM_MAIN, 'GR_RHS'), ('RHS', MECHANISM_RHS_BP, 'RHS'))

def mechanism_fit_path(root, r2, condition, rep, fit):
    return root / f'r2_{r2}_{condition}' / 'fits/k20' / condition / f'replication_{rep:03d}' / fit / 'posterior_core.npz'

def build_mechanism_data():
    MECHANISM_FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    tau_rows, mse_rows = ([], [])
    for r2 in (0.3, 0.7):
        for condition in ('hom', 'het_tail'):
            scenario = f'r2_{r2}_{condition}'
            config = json.loads((MECHANISM_MAIN / scenario / 'manifest_replication_001.json').read_text())['config']
            G = int(config['dgp']['group_count'])
            for rep in range(1, 101):
                if condition == 'het_tail' and r2 == 0.7:
                    dgp = generate_group_scale_dgp(seed=int(config['base_seed']) + rep - 1, condition=condition, signals_per_group=20, **config['dgp'])
                    truth = np.asarray(dgp.dataset.beta)
                    groups = np.arange(truth.size) * G // truth.size
                    active = truth != 0
                for method, root, fit in MECHANISM_METHODS:
                    with np.load(mechanism_fit_path(root, r2, condition, rep, fit)) as a:
                        tau = a['tau']
                        tau_rows.append(dict(r2=r2, condition=condition, rep=rep, method=method, tau_median=float(np.median(tau)), tau_mean=float(np.mean(tau))))
                        if condition == 'het_tail' and r2 == 0.7:
                            beta_mean = a['beta'].mean(axis=0)
                    if condition == 'het_tail' and r2 == 0.7:
                        err = np.square(beta_mean - truth)
                        for g in range(G):
                            sel = active & (groups == g)
                            mse_rows.append(dict(r2=r2, rep=rep, method=method, scale=float(dgp.true_group_slab_scales[g]), mse_nonnull=float(err[sel].mean()), mse_null=float(err[~active & (groups == g)].mean())))
                if rep % 25 == 0:
                    print(f'{scenario}: {rep}/100', flush=True)
    tau = pd.DataFrame(tau_rows)
    tau.to_csv(MECHANISM_FIGURE_DATA / 'mechanism_tau_by_replication.csv', index=False)
    mse = pd.DataFrame(mse_rows).groupby(['r2', 'rep', 'method', 'scale'], as_index=False).mean(numeric_only=True)
    slab = pd.read_csv(MECHANISM_FIGURE_DATA / 'slab_mechanism_summary_by_replication.csv')
    slab = slab.groupby(['r2', 'rep', 'method', 'scale'], as_index=False).mean(numeric_only=True)[['r2', 'rep', 'method', 'scale', 'q', 'v', 'T', 'c_g', 'P_bind']]
    panel = mse.merge(slab, on=['r2', 'rep', 'method', 'scale'], how='left')
    panel = panel.merge(tau[tau.condition == 'het_tail'][['r2', 'rep', 'method', 'tau_median']], on=['r2', 'rep', 'method'], how='left')
    assert not panel.isna().any().any(), panel.isna().sum()
    panel.to_csv(MECHANISM_FIGURE_DATA / 'mechanism_panel_data.csv', index=False)
    print(tau.groupby(['r2', 'condition', 'method']).tau_median.median().unstack())
    print(panel.groupby(['method', 'scale'])[['mse_nonnull', 'T', 'v', 'c_g', 'tau_median']].median())


"""Main-text figures for the matched comparison and the posterior mechanism.

``main_matched_contrasts.pdf``
    GR-RHS minus matched RHS across the four scenarios (ELPD gain and the
    overall / null / non-null MSE reductions), read from
``figures/confirmatory_paired_contrasts.csv`` (rows ``RHS``).

``main_mechanism.pdf``
    HET, R2* = 0.7.  (a) variance ratio and (b) non-null MSE reduction, both
    by generating scale, for GR-RHS and RHS, read from
    ``figures/mechanism_panel_data.csv`` (built by ``build_mechanism_data.py``).
    A single-column figure, included at ``\\columnwidth``.  The conditional
    prior variance v_{j,g} and the posterior global scale tau are reported as
    numbers in the text.

Colors follow the other contrast figures: blue for GR-RHS, orange for the
matched RHS specification.
"""
from pathlib import Path as Path
import matplotlib as matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
MECHANISM_FIGURE_DATA = ROOT / 'results/figure_data'
GR_COLOR, RHS_COLOR = ('#24658a', '#d17a2a')
MECHANISM_WIDTH = 7.56
MECHANISM_GR_LABEL = 'GR-RHS'
MECHANISM_SCENARIOS = [(0.3, 'hom'), (0.7, 'hom'), (0.3, 'het_tail'), (0.7, 'het_tail')]
SCENARIO_LABELS = ['HOM, $R^2_\\star=0.3$', 'HOM, $R^2_\\star=0.7$', 'HET, $R^2_\\star=0.3$', 'HET, $R^2_\\star=0.7$']
MECHANISM_SCALES = [0.5, 1.0, 2.0, 3.0]

def matched_contrasts():
    table = pd.read_csv(MECHANISM_FIGURE_DATA / 'confirmatory_paired_contrasts.csv')
    table = table[table.comparator == 'RHS'].set_index(['target_r2', 'condition', 'metric'])
    panels = [('elpd', 'ELPD gain'), ('mse_overall', 'Overall MSE reduction'), ('mse_null', 'Null MSE reduction'), ('mse_nonnull', 'Non-null MSE reduction')]
    fig, axes = plt.subplots(1, 4, figsize=(MECHANISM_WIDTH, 1.85))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.2, top=0.86, wspace=0.32)
    for k, (ax, (metric, title)) in enumerate(zip(axes, panels)):
        for i, (r2, condition) in enumerate(MECHANISM_SCENARIOS):
            row = table.loc[r2, condition, metric]
            ax.errorbar(row['mean'], i, xerr=[[row['mean'] - row['ci_low']], [row['ci_high'] - row['mean']]], fmt='o', color=GR_COLOR, capsize=3, markersize=4.5, linewidth=1.2)
        ax.axvline(0, color='0.55', linewidth=0.8)
        ax.set_yticks(range(4), SCENARIO_LABELS if k == 0 else [''] * 4)
        ax.set_ylim(3.6, -0.6)
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.15)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4 if metric == 'elpd' else 3))
    axes[0].set_xlabel(f'ELPD({MECHANISM_GR_LABEL}) - ELPD(RHS)', fontsize=11)
    for ax in axes[1:]:
        ax.set_xlabel(f'MSE(RHS) - MSE({GR_LABEL})', fontsize=11)
    fig.savefig(MECHANISM_FIGURE_DATA / 'main_matched_contrasts.pdf', bbox_inches='tight')
    fig.savefig(MECHANISM_FIGURE_DATA / 'main_matched_contrasts.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
BENCH_METHODS = ['RHS', 'HORSESHOE', 'R2D2', 'GROUP_R2', 'GHS', 'GIGG']
BENCH_LABELS = {'RHS': 'RHS', 'HORSESHOE': 'HS', 'R2D2': 'R2D2', 'GROUP_R2': 'Group-$R^2$', 'GHS': 'GHS', 'GIGG': 'GIGG'}

def benchmark_contrasts():
    """2 x 4 grid at R2* = 0.7: rows HOM / HET, columns ELPD and the three MSE
    reductions, all six main-text comparators on the y axis (RHS in orange)."""
    table = pd.read_csv(MECHANISM_FIGURE_DATA / 'confirmatory_paired_contrasts.csv')
    table = table[table.target_r2 == 0.7].set_index(['condition', 'metric', 'comparator'])
    panels = [('elpd', 'ELPD'), ('mse_overall', 'Overall MSE'), ('mse_null', 'Null MSE'), ('mse_nonnull', 'Non-null MSE')]
    fig, axes = plt.subplots(2, 4, figsize=(MECHANISM_WIDTH, 3.47))
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.11, top=0.93, wspace=0.3, hspace=0.45)
    for r, condition in enumerate(('hom', 'het_tail')):
        for k, (metric, title) in enumerate(panels):
            ax = axes[r, k]
            for j, method in enumerate(BENCH_METHODS):
                row = table.loc[condition, metric, method]
                color = RHS_COLOR if method == 'RHS' else GR_COLOR
                ax.errorbar(row['mean'], j, xerr=[[row['mean'] - row['ci_low']], [row['ci_high'] - row['mean']]], fmt='o', color=color, capsize=3, markersize=4, linewidth=1.1)
            ax.axvline(0, color='0.55', linewidth=0.8)
            ax.set_yticks(range(len(BENCH_METHODS)), [BENCH_LABELS[m] for m in BENCH_METHODS] if k == 0 else [''] * len(BENCH_METHODS))
            ax.set_ylim(len(BENCH_METHODS) - 0.5, -0.5)
            ax.set_title(('HOM / ' if condition == 'hom' else 'HET / ') + title, fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='x', alpha=0.15)
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4 if metric == 'elpd' else 3))
            if r == 1:
                ax.set_xlabel('ELPD gain' if metric == 'elpd' else 'MSE reduction', fontsize=11)
    fig.savefig(MECHANISM_FIGURE_DATA / 'main_benchmark_contrasts.pdf', bbox_inches='tight')
    fig.savefig(MECHANISM_FIGURE_DATA / 'main_benchmark_contrasts.png', dpi=600, bbox_inches='tight')
    plt.close(fig)

def paired_by_scale(panel, column):
    wide = panel.pivot_table(index=['rep', 'scale'], columns='method', values=column)
    out = []
    for scale in MECHANISM_SCALES:
        d = wide.xs(scale, level='scale')
        delta = d['RHS'] - d['GR-RHS']
        half = student_t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
        out.append((delta.mean(), half))
    return out
"""Render or verify the posterior-mechanism tables in Appendix C.

The input summaries are created without refitting by
``slab_mechanism_summary.py`` and ``summarize_null_mechanism.py``.
"""
from pathlib import Path as Path
import re as re
import pandas as pd
FIGURE_DATA = ROOT / 'results/figure_data'

FIGURE_DATA = ROOT / 'results/figure_data'
MANUSCRIPT = ROOT / 'grrhs/content/C-simulation-diagnostics.tex'
SCALES = (0.5, 1.0, 2.0, 3.0)
SCENARIOS = (0.3, 0.7)

def read_inputs():
    slab = pd.read_csv(FIGURE_DATA / 'slab_mechanism_summary.csv')
    slab = slab[slab.method.isin(('GR-RHS', 'RHS'))]
    null = pd.read_csv(FIGURE_DATA / 'null_mechanism_summary.csv')
    paired = pd.read_csv(FIGURE_DATA / 'null_mechanism_paired_differences.csv')
    expected_slab = {(r2, method, scale) for r2 in SCENARIOS for method in ('GR-RHS', 'RHS') for scale in SCALES}
    observed_slab = set(zip(slab.r2, slab.method, slab.scale))
    if len(slab) != 16 or observed_slab != expected_slab:
        raise ValueError(f'Slab summary must contain the 16 expected method/scenario/scale rows; found {len(slab)}')
    expected_null = {(r2, method, scale) for r2 in SCENARIOS for method in ('GR--RHS', 'RHS') for scale in SCALES}
    observed_null = set(zip(null.r2, null.method, null.scale))
    if len(null) != 16 or observed_null != expected_null:
        raise ValueError(f'Null summary must contain the 16 expected method/scenario/scale rows; found {len(null)}')
    expected_paired = {(r2, scale) for r2 in SCENARIOS for scale in SCALES}
    differences = paired[(paired.quantity == 'mse_null') & (paired.n_replications == 100)]
    if len(differences) != 8 or set(zip(differences.r2, differences.scale)) != expected_paired:
        raise ValueError('Null-MSE paired summary must contain eight 100-replication scenario/scale rows')
    return (slab, null, differences)

def find_row(frame, r2, method, scale):
    selected = frame[(frame.r2 == r2) & (frame.method == method) & (frame.scale == scale)]
    if len(selected) != 1:
        raise ValueError(f'Expected one summary row for r2={r2}, method={method}, scale={scale}')
    return selected.iloc[0]

def slab_table(slab):
    lines = ['\\begin{table}[!htbp]', '\\centering', '\\begin{threeparttable}', '\\caption{Posterior scale summaries segmented by generating scale within the HET scenario.}', '\\label{tab:slab-mechanism-summary}', '\\footnotesize', '\\setlength{\\tabcolsep}{5pt}', '\\begin{tabular}{llcccc}', '\\toprule Method & $s_g^*$ & Slab $c_g$ & $\\tau^2\\lambda_j^2$', '& Variance ratio & $v_{j,g}$\\\\']
    for r2 in SCENARIOS:
        lines.append('\\midrule \\multicolumn{6}{l}{\\textit{$R^2_\\star=' + f'{r2:.1f}' + '$}}\\\\[2pt]')
        for method in ('GR-RHS', 'RHS'):
            for scale in SCALES:
                row = find_row(slab, r2, method, scale)
                label = method if method == 'GR-RHS' else 'RHS'
                values = (f'{row.c_g:.2f}', f'{row.q:.3f}', f"{row['T']:.3f}", f'{row.v:.3f}')
                lines.append(label + ' & ' + f'{scale:g}' + ' & ' + ' & '.join(values) + '\\\\')
            if method == 'GR-RHS':
                lines.append('\\addlinespace[2pt]')
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\begin{tablenotes}[flushleft]', '\\item \\textit{Note.} All values are medians over 100 replications of the within-group medians among the true nonzero coefficients. The Slab column gives the posterior median of $c_g$ under GR-RHS and of the single $c$ under RHS. This scale bounds the prior variance of coefficients with large local scales and therefore determines the lower bound $\\underline{\\kappa}_g$ on their shrinkage weights. Each column is summarized separately; the identity $v_{j,g}=\\{c_g^2/(c_g^2+\\tau^2\\lambda_j^2)\\}\\tau^2\\lambda_j^2$ applies at the draw level before column-specific aggregation.', '\\end{tablenotes}', '\\end{threeparttable}', '\\end{table}'])
    return '\n'.join(lines)

def null_table(null, differences):
    lines = ['\\begin{table}[!htbp]', '\\centering', '\\begin{threeparttable}', '\\caption{Posterior scale contrasts and paired null MSE differences at the true zero coefficients in the HET scenario.}', '\\label{tab:null-mechanism}', '\\footnotesize', '\\setlength{\\tabcolsep}{5pt}', '\\begin{tabular}{lcccc}', '\\toprule $s_g^*$ & \\shortstack{$\\tau^2\\lambda_j^2$\\\\(GR-RHS/RHS)} & \\shortstack{Variance ratio\\\\(GR-RHS/RHS)} & \\shortstack{$v_{j,g}$\\\\(GR-RHS/RHS)}', '& \\shortstack{$\\Delta$ null MSE\\\\(95\\% CI)}\\\\']
    for r2 in SCENARIOS:
        lines.append('\\midrule \\multicolumn{5}{l}{\\textit{$R^2_\\star=' + f'{r2:.1f}' + '$}}\\\\[2pt]')
        for scale in SCALES:
            gr = find_row(null, r2, 'GR--RHS', scale)
            rhs = find_row(null, r2, 'RHS', scale)
            delta_rows = differences[(differences.r2 == r2) & (differences.scale == scale)]
            if len(delta_rows) != 1:
                raise ValueError(f'Expected one paired null-MSE row for r2={r2}, scale={scale}')
            delta = delta_rows.iloc[0]
            mean = f'{delta.mean_difference_grrhs_minus_rhs:.3f}'
            if delta.mean_difference_grrhs_minus_rhs >= 0:
                mean = '\\phantom{-}' + mean
            contrast = '$' + mean + '\\;[' + f'{delta.ci_low:.3f},{delta.ci_high:.3f}' + ']$'
            values = (f"{gr['q']:.3f}/{rhs['q']:.3f}", f"{gr['T']:.3f}/{rhs['T']:.3f}", f"{gr['v']:.3f}/{rhs['v']:.3f}", contrast)
            lines.append(f'{scale:g} & ' + ' & '.join(values[:3]) + '\n& ' + values[3] + '\\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\begin{tablenotes}[flushleft]', '\\item \\textit{Note.} Entries before and after the slash are GR-RHS and RHS medians, respectively, over 100 replications of the within-group summaries among the true zero coefficients. The $\\Delta$ column reports the mean paired difference in null MSE, GR-RHS minus RHS, with a 95\\% Student-$t$ interval; negative values favor GR-RHS. Each quantity is summarized separately.', '\\end{tablenotes}', '\\end{threeparttable}', '\\end{table}'])
    return '\n'.join(lines)

def replace_block(text, name, replacement):
    begin = f'% BEGIN GENERATED {name}'
    end = f'% END GENERATED {name}'
    pattern = re.compile(re.escape(begin) + '.*?' + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f'Missing generated-table markers for {name} in {MANUSCRIPT}')
    value = begin + '\n' + replacement + '\n' + end
    return pattern.sub(lambda _: value, text, count=1)

def build_mechanism_tables():
    slab, null, differences = read_inputs()
    generated = {'SLAB-MECHANISM TABLE': slab_table(slab), 'NULL-MECHANISM TABLE': null_table(null, differences)}
    original = MANUSCRIPT.read_text(encoding='utf-8')
    updated = original
    for name, table in generated.items():
        updated = replace_block(updated, name, table)
    if updated != original:
        raise SystemExit('Mechanism tables are stale relative to the retained summaries.')
    print('Both manuscript mechanism tables match the derived 100-replication summaries.')


"""Posterior scale diagnostics for true-zero coefficients in HET simulations.

The aggregation mirrors ``slab_mechanism_summary.py``: posterior summaries are
computed coefficient by coefficient, then aggregated within each generating
group, the two scale-0.5 groups are averaged within replication, and finally
the 100 replication-level summaries are aggregated.  No model is refit.
"""
import json as json
import sys as sys
from pathlib import Path as Path

NULL_MAIN = ROOT / 'results/regression/confirmatory'
NULL_RHS_BP = ROOT / 'results/regression' / RHS_OUTPUT_METHOD / 'confirmatory'
NULL_OUTPUT = ROOT / 'results/figure_data'
NULL_METHODS = (('GR--RHS', NULL_MAIN, 'GR_RHS'), ('RHS', NULL_RHS_BP, 'RHS'))
NULL_THIN = 4
NULL_N = 200
NULL_QUANTITIES = ('q', 'v', 'T', 'kappa', 'P_bind', 'c', 'mse_null')

def posterior_coefficient_summaries(path, truth, groups):
    with np.load(path) as draws:
        tau = draws['tau'][::NULL_THIN]
        lam = draws['lambda'][::NULL_THIN]
        sigma = draws['sigma'][::NULL_THIN]
        c2 = draws['c2'] if 'c2' in draws.files else draws['c2_g']
        c2 = c2[::NULL_THIN]
        c2j = np.broadcast_to(c2, lam.shape) if c2.shape[1] == 1 else c2[:, groups]
        q = np.square(tau[:, None] * lam)
        v = c2j * q / (c2j + q)
        retained = c2j / (c2j + q)
        sigma2 = np.square(sigma)[:, None]
        kappa = sigma2 / (sigma2 + NULL_N * v)
        beta_mean = draws['beta'].mean(axis=0)
        return {'q': np.median(q, axis=0), 'v': np.median(v, axis=0), 'T': np.median(retained, axis=0), 'kappa': np.median(kappa, axis=0), 'P_bind': (q > c2j).mean(axis=0), 'c': np.sqrt(np.median(c2, axis=0)), 'sq_error': np.square(beta_mean - truth)}

def null_fit_path(root, r2, rep, fit):
    return root / f'r2_{r2}_het_tail' / 'fits/k20/het_tail' / f'replication_{rep:03d}' / fit / 'posterior_core.npz'

def paired_summary(replication_rows):
    wide = replication_rows.pivot_table(index=['r2', 'rep', 'scale'], columns='method', values=NULL_QUANTITIES)
    rows = []
    for r2 in (0.3, 0.7):
        for scale in (0.5, 1.0, 2.0, 3.0):
            block = wide.xs((r2, scale), level=('r2', 'scale'))
            for quantity in NULL_QUANTITIES:
                diff = block[quantity, 'GR--RHS'] - block[quantity, 'RHS']
                half = student_t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
                rows.append({'r2': r2, 'scale': scale, 'quantity': quantity, 'mean_difference_grrhs_minus_rhs': diff.mean(), 'ci_low': diff.mean() - half, 'ci_high': diff.mean() + half, 'n_replications': len(diff)})
    return pd.DataFrame(rows)

def overall_summaries(replication_rows):
    """Pool equal-sized groups, accounting for the duplicated scale-0.5 group."""
    weighted = replication_rows.copy()
    weighted['weight'] = weighted['scale'].map({0.5: 2, 1.0: 1, 2.0: 1, 3.0: 1})

    def aggregate(block):
        return pd.Series({quantity: np.average(block[quantity], weights=block['weight']) for quantity in NULL_QUANTITIES})
    overall_replications = weighted.groupby(['r2', 'rep', 'method']).apply(aggregate, include_groups=False).reset_index()
    model_medians = overall_replications.groupby(['r2', 'method'], as_index=False)[list(NULL_QUANTITIES)].median()
    wide = overall_replications.pivot_table(index=['r2', 'rep'], columns='method', values=NULL_QUANTITIES)
    paired_rows = []
    for r2 in (0.3, 0.7):
        block = wide.xs(r2)
        for quantity in NULL_QUANTITIES:
            diff = block[quantity, 'GR--RHS'] - block[quantity, 'RHS']
            half = student_t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            paired_rows.append({'r2': r2, 'quantity': quantity, 'mean_difference_grrhs_minus_rhs': diff.mean(), 'ci_low': diff.mean() - half, 'ci_high': diff.mean() + half, 'n_replications': len(diff)})
    paired = pd.DataFrame(paired_rows)
    return (model_medians, paired)

def build_null_summary():
    global MAIN, RHS_BP, OUT, METHODS, THIN, N, QUANTITIES
    MAIN, RHS_BP, OUT, METHODS, THIN, N, QUANTITIES = NULL_MAIN, NULL_RHS_BP, NULL_OUTPUT, NULL_METHODS, NULL_THIN, NULL_N, NULL_QUANTITIES
    rows = []
    for r2 in (0.3, 0.7):
        scenario = f'r2_{r2}_het_tail'
        manifest = NULL_MAIN / scenario / 'manifest_replication_001.json'
        config = json.loads(manifest.read_text())['config']
        group_count = int(config['dgp']['group_count'])
        for rep in range(1, 101):
            dgp = generate_group_scale_dgp(seed=int(config['base_seed']) + rep - 1, condition='het_tail', signals_per_group=20, **config['dgp'])
            truth = np.asarray(dgp.dataset.beta)
            groups = np.arange(truth.size) * group_count // truth.size
            inactive = truth == 0
            for method, root, fit in NULL_METHODS:
                summaries = posterior_coefficient_summaries(null_fit_path(root, r2, rep, fit), truth, groups)
                for group in range(group_count):
                    selected = inactive & (groups == group)
                    c_index = group if summaries['c'].size > 1 else 0
                    rows.append({'r2': r2, 'rep': rep, 'method': method, 'scale': float(dgp.true_group_slab_scales[group]), 'q': np.median(summaries['q'][selected]), 'v': np.median(summaries['v'][selected]), 'T': np.median(summaries['T'][selected]), 'kappa': np.median(summaries['kappa'][selected]), 'P_bind': summaries['P_bind'][selected].mean(), 'c': summaries['c'][c_index], 'mse_null': summaries['sq_error'][selected].mean()})
            if rep % 20 == 0:
                print(f'{scenario}: {rep}/100', flush=True)
    group_rows = pd.DataFrame(rows)
    replication_rows = group_rows.groupby(['r2', 'rep', 'method', 'scale'], as_index=False).mean(numeric_only=True)
    model_summary = replication_rows.groupby(['r2', 'method', 'scale'], as_index=False)[list(NULL_QUANTITIES)].median()
    paired = paired_summary(replication_rows)
    overall_model, overall_paired = overall_summaries(replication_rows)
    NULL_OUTPUT.mkdir(parents=True, exist_ok=True)
    replication_rows.to_csv(NULL_OUTPUT / 'null_mechanism_by_replication.csv', index=False)
    model_summary.to_csv(NULL_OUTPUT / 'null_mechanism_summary.csv', index=False)
    paired.to_csv(NULL_OUTPUT / 'null_mechanism_paired_differences.csv', index=False)
    overall_model.to_csv(NULL_OUTPUT / 'null_mechanism_overall_model_medians.csv', index=False)
    overall_paired.to_csv(NULL_OUTPUT / 'null_mechanism_overall_paired_differences.csv', index=False)
    print('\nMODEL MEDIANS ACROSS 100 REPLICATIONS')
    print(model_summary.to_string(index=False, float_format=lambda x: f'{x:.6g}'))
    print('\nPAIRED MEAN DIFFERENCES (GR--RHS - RHS), WITH 95% t INTERVALS')
    print(paired.to_string(index=False, float_format=lambda x: f'{x:.6g}'))


SLAB_MAIN = ROOT / 'results/regression/confirmatory'
SLAB_RHS_BP = ROOT / 'results/regression' / RHS_OUTPUT_METHOD / 'confirmatory'
SLAB_FIGURE_DATA = ROOT / 'results/figure_data'
SLAB_THIN = 4
SLAB_N = 200
SLAB_METHODS = (('GR-RHS', SLAB_MAIN, 'GR_RHS'), ('RHS', SLAB_RHS_BP, 'RHS'))

def summarize_slab_posterior(path, truth, groups):
    with np.load(path) as draws:
        tau = draws['tau'][::SLAB_THIN]
        lam = draws['lambda'][::SLAB_THIN]
        sigma = draws['sigma'][::SLAB_THIN]
        c2 = draws['c2'] if 'c2' in draws.files else draws['c2_g']
        c2 = c2[::SLAB_THIN]
        c2j = np.broadcast_to(c2, lam.shape) if c2.shape[1] == 1 else c2[:, groups]
        q = np.square(tau[:, None] * lam)
        v = c2j * q / (c2j + q)
        retained = c2j / (c2j + q)
        kappa = np.square(sigma)[:, None] / (np.square(sigma)[:, None] + SLAB_N * v)
        beta = draws['beta'][::SLAB_THIN]
        return {'q': np.median(q, axis=0), 'v': np.median(v, axis=0), 'T': np.median(retained, axis=0), 'kappa': np.median(kappa, axis=0), 'P_bind': (q > c2j).mean(axis=0), 'c2': np.median(c2, axis=0), 'abs_bias': np.abs(beta.mean(axis=0) - truth), 'beta_ratio': np.abs(beta.mean(axis=0)) / np.maximum(np.abs(truth), 1e-12)}

def slab_fit_path(root, r2, rep, fit):
    return root / f'r2_{r2}_het_tail' / 'fits/k20/het_tail' / f'replication_{rep:03d}' / fit / 'posterior_core.npz'

def build_slab_summary():
    SLAB_FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for r2 in (0.3, 0.7):
        scenario = f'r2_{r2}_het_tail'
        config = json.loads((SLAB_MAIN / scenario / 'manifest_replication_001.json').read_text())['config']
        group_count = int(config['dgp']['group_count'])
        for rep in range(1, 101):
            dgp = generate_group_scale_dgp(seed=int(config['base_seed']) + rep - 1, condition='het_tail', signals_per_group=20, **config['dgp'])
            truth = np.asarray(dgp.dataset.beta)
            groups = np.arange(truth.size) * group_count // truth.size
            active = truth != 0
            for method, root, fit in SLAB_METHODS:
                values = summarize_slab_posterior(slab_fit_path(root, r2, rep, fit), truth, groups)
                for group in range(group_count):
                    selected = active & (groups == group)
                    c_index = group if values['c2'].size > 1 else 0
                    rows.append({'r2': r2, 'rep': rep, 'method': method, 'scale': float(dgp.true_group_slab_scales[group]), 'q': np.median(values['q'][selected]), 'v': np.median(values['v'][selected]), 'T': np.median(values['T'][selected]), 'kappa': np.median(values['kappa'][selected]), 'P_bind': values['P_bind'][selected].mean(), 'c_g': float(np.sqrt(values['c2'][c_index])), 'abs_bias': values['abs_bias'][selected].mean(), 'beta_ratio': np.median(values['beta_ratio'][selected])})
            if rep % 20 == 0:
                print(f'{scenario}: {rep}/100', flush=True)
    by_replication = pd.DataFrame(rows)
    by_replication.to_csv(SLAB_FIGURE_DATA / 'slab_mechanism_summary_by_replication.csv', index=False)
    per_replication = by_replication.groupby(['r2', 'rep', 'method', 'scale'], as_index=False).mean(numeric_only=True)
    summary = per_replication.groupby(['r2', 'method', 'scale'], as_index=False).agg(q=('q', 'median'), v=('v', 'median'), T=('T', 'median'), kappa=('kappa', 'median'), P_bind=('P_bind', 'mean'), c_g=('c_g', 'median'), abs_bias=('abs_bias', 'mean'), beta_ratio=('beta_ratio', 'median'))
    summary.to_csv(SLAB_FIGURE_DATA / 'slab_mechanism_summary.csv', index=False)
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.4g}'))


"""Build the second Introduction mechanism figure: the conditional-variance
response v = c^2 * (tau^2 lambda_j^2) / (c^2 + tau^2 lambda_j^2), on ordinary
linear-linear axes (not log-log).

No extra symbol (no chi_j) is introduced: the x-axis is the literal
expression tau^2*lambda_j^2, and each panel's y-axis is the literal
conditional-variance expression for that prior, so every axis is
self-explanatory without a name lookup.

Parameters are fixed at plain illustrative values, not tuned for effect:
    RHS:     c^2 = 2
    GR-RHS:  c_1^2, c_2^2, c_3^2 = 0.5, 1, 3

The x-range is pushed out far enough that the GR-RHS curves visibly close
in on their dashed asymptote lines without ever touching them: v(x) -> c_g^2
as x -> infinity, but for finite x, v(x) < c_g^2 strictly.

The HS line v = x is unbounded and is clipped at the top of the shared
y-range with a short arrow, rather than rescaling the axes to fit it.
"""
