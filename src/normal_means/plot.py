"""Figure 2 data preparation and plotting for the normal-means experiment."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
PAPER_STYLE = {'font.family': 'serif', 'font.serif': ['Latin Modern Roman', 'STIX Two Text', 'Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'cm', 'font.size': 10.0, 'axes.labelsize': 12, 'axes.titlesize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.titlesize': 11, 'axes.edgecolor': '#4A4A4A', 'axes.linewidth': 0.8, 'xtick.color': '#4A4A4A', 'ytick.color': '#4A4A4A', 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 300, 'savefig.dpi': 300}
FIG_DPI = 600
DEFAULT_SUMMARY_DIR = ROOT / 'results' / 'normal_means' / 'normal_means50_sigma1' / 'summary'
FIGURE_DIR = ROOT / 'results/figure_data'
FULL_W = 7.56
GR_LABEL = 'GR-RHS'
RHS_COLOR = '#d17a2a'
TARGET_COLOR = '#24658a'
REF_COLOR = '#199e70'
NULL_COLOR = '#898781'
INK = '#4a4a4a'
BOUND_CURVES = [('RHS', RHS_COLOR, '-', 's', 'full', 'RHS'), ('gr_target', TARGET_COLOR, '-', 'o', 'full', f'{GR_LABEL}, varying group'), ('gr_reference', REF_COLOR, '--', '^', 'full', f'{GR_LABEL}, reference group'), ('gr_null', NULL_COLOR, ':', 'D', 'full', f'{GR_LABEL}, null groups')]
SUBSETS = [('target', TARGET_COLOR, '-', 'o', 'full', 'Varying-group signals'), ('target_within_null', TARGET_COLOR, '--', 'o', 'none', 'Varying-group nulls'), ('external_null', NULL_COLOR, ':', '^', 'none', 'Nulls in other groups'), ('all', INK, '-', None, 'full', 'All coefficients')]
TAU_REF = 0.25
EQUAL_A = 1.0
GUIDE = '#a8a8a8'


def aggregate_with_intervals(frame, keys, value):
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        values = group[value].to_numpy(float)
        values = values[np.isfinite(values)]
        mean = float(values.mean())
        half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
        rows.append(dict(zip(keys, key), n=len(values), mean=mean, lower=mean - half, upper=mean + half))
    return pd.DataFrame(rows)


def prepare_bound_curves(summary_dir):
    frame = pd.read_csv(summary_dir / 'fit_metrics.csv').dropna(subset=['stratum'])
    frame = frame[frame.stratum.str.startswith('group')].copy()
    frame['gid'] = frame.stratum.str[-1].astype(int)
    cols = ['rep', 'amplitude', 'arm', 'curve', 'slab_floor']
    parts = [frame[(frame.model == 'rhs') & (frame.gid == 1)].assign(curve='RHS')[cols], frame[(frame.model == 'grrhs') & (frame.gid == 2)].assign(curve='gr_target')[cols], frame[(frame.model == 'grrhs') & (frame.gid == 1)].assign(curve='gr_reference')[cols]]
    parts.append(frame[(frame.model == 'grrhs') & frame.gid.isin([3, 4, 5])].groupby(['rep', 'amplitude', 'arm'], as_index=False).slab_floor.mean().assign(curve='gr_null')[cols])
    return pd.concat(parts, ignore_index=True)


def draw_lines(ax, table, specs, key):
    for name, color, dash, marker, fill, label in specs:
        group = table[table[key] == name].sort_values('amplitude')
        x = group.amplitude.to_numpy()
        ax.fill_between(x, group.lower, group.upper, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, group['mean'], color=color, linestyle=dash, linewidth=1.3, marker=marker, fillstyle=fill, markersize=4.2, markeredgecolor=color, markeredgewidth=1.0, label=label, clip_on=False, zorder=3)


def set_ticks(ax, axis, values, labelled):
    getattr(ax, f'set_{axis}ticks')(values, [f'{value:g}' if value in labelled else '' for value in values])


def style_axis(ax, amplitudes):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.15)
    ax.tick_params(labelsize=9)
    ax.axvline(EQUAL_A, color=GUIDE, linewidth=0.7, linestyle=(0, (2.5, 2.5)), zorder=1)
    ax.set_xlim(0.12, 3.13)
    set_ticks(ax, 'x', amplitudes, {0.25, 1.0, 2.0, 3.0})


def build_figure(summary_dir=DEFAULT_SUMMARY_DIR):
    bounds = prepare_bound_curves(summary_dir)
    tau = pd.read_csv(summary_dir / 'tau_by_fit.csv')
    paired = pd.read_csv(summary_dir / 'paired_mse.csv')
    amplitudes = sorted(bounds.amplitude.unique())
    bound_summary = aggregate_with_intervals(bounds, ['arm', 'curve', 'amplitude'], 'slab_floor')
    tau_summary = aggregate_with_intervals(tau[tau.arm == 'learned'], ['model', 'amplitude'], 'tau')
    subset_names = [subset[0] for subset in SUBSETS]
    mse_summary = aggregate_with_intervals(paired[(paired.arm == 'learned') & paired.stratum.isin(subset_names)], ['stratum', 'amplitude'], 'difference')
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.62))
    fig.subplots_adjust(left=0.042, right=0.995, bottom=0.175, top=0.905, wspace=0.3)
    ax = axes[0]
    draw_lines(ax, bound_summary[bound_summary.arm == 'fixed'], BOUND_CURVES, 'curve')
    ax.set_ylim(0, 0.7)
    ax.set_title('(a) Lower shrinkage bound, fixed $\\tau$', fontsize=10)
    style_axis(ax, amplitudes)
    set_ticks(ax, 'y', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], {0, 0.2, 0.4, 0.6})
    ax.legend(fontsize=7, frameon=False, loc='lower left', handlelength=1.4, labelspacing=0.3, borderpad=0.1, handletextpad=0.45, borderaxespad=0.2)
    ax = axes[1]
    ax.axhline(TAU_REF, color=GUIDE, linewidth=0.7, linestyle=(0, (1, 2.5)), zorder=1)
    for model, color, marker, label in (('grrhs', TARGET_COLOR, 'o', GR_LABEL), ('rhs', RHS_COLOR, 's', 'RHS')):
        group = tau_summary[tau_summary.model == model].sort_values('amplitude')
        x = group.amplitude.to_numpy()
        ax.fill_between(x, group.lower, group.upper, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, group['mean'], color=color, linewidth=1.3, marker=marker, markersize=3.8, label=label, clip_on=False, zorder=3)
        ax.annotate(label, xy=(x[-1], group['mean'].to_numpy()[-1]), xytext=(-2, 7 if model == 'grrhs' else -12), textcoords='offset points', fontsize=8, color=color, ha='right')
    ax.set_ylim(0, 1.72)
    ax.set_ylabel('Posterior median $\\tau$', fontsize=11)
    ax.set_title('(b) Posterior median of $\\tau$', fontsize=10)
    style_axis(ax, amplitudes)
    set_ticks(ax, 'y', [0, 0.5, 1, 1.5], {0, 0.5, 1, 1.5})
    ax = axes[2]
    ax.axhline(0, color='#1a1a1a', linewidth=0.7, linestyle=(0, (4, 2.5)), zorder=2)
    draw_lines(ax, mse_summary, SUBSETS, 'stratum')
    ax.set_ylim(-0.3, 1.12)
    ax.set_ylabel('Paired MSE difference', fontsize=11)
    ax.set_title('(c) MSE contrast', fontsize=10)
    style_axis(ax, amplitudes)
    set_ticks(ax, 'y', [-0.25, 0, 0.25, 0.5, 0.75, 1], {-0.25, 0, 0.5, 1})
    ax.legend(fontsize=7, frameon=False, loc='upper left', handlelength=1.4, labelspacing=0.3, borderpad=0.1, handletextpad=0.45, borderaxespad=0.2)
    fig.supxlabel('Signal magnitude $A$', fontsize=11, y=0.012)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / 'amplitude_probe.pdf', bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'amplitude_probe.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    plotted = pd.concat([bound_summary.assign(panel=lambda d: np.where(d.arm == 'fixed', 'a', 'text'), quantity='slab_shrinkage_bound').rename(columns={'curve': 'series'}), tau_summary.assign(panel='b', arm='learned', quantity='posterior_median_tau').rename(columns={'model': 'series'}), mse_summary.assign(panel='c', arm='learned', quantity='paired_mse_difference').rename(columns={'stratum': 'series'})], ignore_index=True)[['panel', 'quantity', 'arm', 'series', 'amplitude', 'n', 'mean', 'lower', 'upper']]
    plotted.to_csv(FIGURE_DIR / 'amplitude_probe_summary.csv', index=False)
    return plotted
