from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mechanism import paired_by_scale
from .summarize import load, load_low_signal_fits, low_signal_paired_contrasts, paired_contrasts
from .calibration import PRIOR_METHODS
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]

PAPER_STYLE = {'font.family': 'serif', 'font.serif': ['Latin Modern Roman', 'STIX Two Text', 'Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'cm', 'font.size': 10.0, 'axes.labelsize': 12, 'axes.titlesize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.titlesize': 11, 'axes.edgecolor': '#4A4A4A', 'axes.linewidth': 0.8, 'xtick.color': '#4A4A4A', 'ytick.color': '#4A4A4A', 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 300, 'savefig.dpi': 300}
FIG_DPI = 600


"""Build the paired-contrast figures for the main text and the supplement.

The matched beta-prime common-slab specification is reported as ``RHS``.

The confirmatory fits live in ``results/regression/confirmatory``.  The matched
Matched RHS fits use the dedicated output directory configured in ``fit.py``.
Posterior-summary metrics (posterior RMSE, coverage) come from the
``analysis_posterior`` folders of the same two runs.
"""
OUT = ROOT / 'results/figure_data'
LABELS = {'RHS': 'RHS', 'HORSESHOE': 'HS', 'R2D2': 'R2D2', 'GROUP_R2': 'Group-$R^2$', 'GHS': 'GHS', 'GIGG': 'GIGG'}
MAIN_METHODS = ['RHS', 'HORSESHOE', 'R2D2', 'GROUP_R2', 'GHS', 'GIGG']
SUPP_METHODS = ['RHS', 'HORSESHOE', 'R2D2', 'GROUP_R2', 'GHS', 'GIGG']
SCENARIOS = [(r, c) for r in (0.3, 0.7) for c in ('hom', 'het_tail')]
MATCHED_COLOR, OTHER_COLOR = ('#d17a2a', '#24658a')
FULL_W = 7.56
GR_LABEL = 'GR-RHS'

def dots(ax, table, methods):
    for j, method in enumerate(methods):
        row = table.loc[method]
        color = MATCHED_COLOR if method == 'RHS' else OTHER_COLOR
        ax.errorbar(row['mean'], j, xerr=[[row['mean'] - row['ci_low']], [row['ci_high'] - row['mean']]], fmt='o', color=color, capsize=3, markersize=4)
    ax.axvline(0, color='0.55', linewidth=0.8)
    ax.set_yticks(np.arange(len(methods)), [LABELS[m] for m in methods])
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.15)

def main_figure(contrasts, metrics, titles, filename, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(9.6, 3.25 if rows == 1 else 5.8), squeeze=False)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.16, top=0.88 if rows == 1 else 0.94, wspace=0.72, hspace=0.34)
    for i, (ax, metric, title) in enumerate(zip(axes, metrics, titles)):
        condition = 'hom' if rows == 1 and i == 0 or (rows == 2 and i < 3) else 'het_tail'
        dots(ax, contrasts.loc[0.7, condition, metric], MAIN_METHODS)
        if rows == 1 and i == 1 or (rows == 2 and i % 3 != 0):
            ax.tick_params(axis='y', labelleft=False)
        ax.set_title(title, fontsize=10)
    fig.supxlabel(f'Positive values favor {GR_LABEL}', y=0.025, fontsize=11)
    fig.savefig(OUT / f'{filename}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{filename}.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)

def supplement_figure(contrasts, metric, filename):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.15, top=0.93, wspace=0.55, hspace=0.42)
    for ax, (r2, condition) in zip(axes.flat, SCENARIOS):
        dots(ax, contrasts.loc[r2, condition, metric], SUPP_METHODS)
        ax.tick_params(axis='both', labelsize=7.5)
        ax.set_title(f'$R^2_\\star={r2}$, ' + ('HOM' if condition == 'hom' else 'HET'), fontsize=10)
    fig.supxlabel(f'Positive values favor {GR_LABEL}', y=0.025, fontsize=11)
    fig.savefig(OUT / f'{filename}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{filename}.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)

def low_signal_summary_figure(contrasts, filename):
    """Compactly report the four low-signal paired-contrast metrics."""
    metrics = ['elpd', 'mse_overall', 'mse_null', 'mse_nonnull']
    titles = ['ELPD', 'Overall MSE', 'Null MSE', 'Non-null MSE']
    fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 3.47), squeeze=False)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.11, top=0.93, wspace=0.3, hspace=0.45)
    for row, condition in enumerate(('hom', 'het_tail')):
        for col, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[row, col]
            dots(ax, contrasts.loc[0.3, condition, metric], SUPP_METHODS)
            ax.set_title(('HOM / ' if row == 0 else 'HET / ') + title, fontsize=10)
            ax.tick_params(axis='both', labelsize=9)
            if col:
                ax.tick_params(axis='y', labelleft=False)
            ax.set_xlabel('ELPD gain' if row == 1 and metric == 'elpd' else 'MSE reduction' if row == 1 else '', fontsize=11)
    fig.savefig(OUT / f'{filename}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{filename}.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)

def build_figure3(low_signal=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if low_signal:
        fits = load_low_signal_fits()
        contrasts = low_signal_paired_contrasts(fits)
        contrasts.to_csv(OUT / 'confirmatory_low_signal_paired_contrasts.csv')
        plt.rcParams.update(PAPER_STYLE)
        plt.rcParams.update({'font.size': 8})
        low_signal_summary_figure(contrasts, 'confirmatory_low_signal_contrasts')
        return
    fits, posts = load()
    contrasts = paired_contrasts(fits, posts)
    contrasts.to_csv(OUT / 'confirmatory_paired_contrasts.csv')
    plt.rcParams.update(PAPER_STYLE)
    main_figure(contrasts, ['elpd', 'elpd'], ['HOM / ELPD gain', 'HET / ELPD gain'], 'main_elpd_contrasts', 1, 2)
    main_figure(contrasts, ['mse_overall', 'mse_null', 'mse_nonnull'] * 2, ['HOM / Total MSE reduction', 'HOM / Null MSE reduction', 'HOM / Non-null MSE reduction', 'HET / Total MSE reduction', 'HET / Null MSE reduction', 'HET / Non-null MSE reduction'], 'main_mse_contrasts', 2, 3)
    plt.rcParams.update({'font.size': 8})
    for metric, stem in (('elpd', 'confirmatory_elpd_contrasts'), ('mse_overall', 'confirmatory_mse_overall_contrasts'), ('mse_nonnull', 'confirmatory_mse_nonnull_contrasts'), ('mse_null', 'confirmatory_mse_null_contrasts'), ('posterior_rmse_overall', 'confirmatory_posterior_rmse_overall_contrasts'), ('posterior_rmse_nonnull', 'confirmatory_posterior_rmse_nonnull_contrasts'), ('posterior_rmse_null', 'confirmatory_posterior_rmse_null_contrasts'), ('beta_coverage_nonnull', 'confirmatory_nonnull_coverage_contrasts')):
        supplement_figure(contrasts, metric, stem)
    low_signal_summary_figure(contrasts, 'confirmatory_low_signal_contrasts')

FIG = ROOT / 'results/figure_data'
GR_COLOR, RHS_COLOR = ('#24658a', '#d17a2a')
FULL_W = 7.56
GR_LABEL = 'GR-RHS'
SCENARIOS = [(0.3, 'hom'), (0.7, 'hom'), (0.3, 'het_tail'), (0.7, 'het_tail')]
SCENARIO_LABELS = ['HOM, $R^2_\\star=0.3$', 'HOM, $R^2_\\star=0.7$', 'HET, $R^2_\\star=0.3$', 'HET, $R^2_\\star=0.7$']
SCALES = [0.5, 1.0, 2.0, 3.0]

def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.15)
    ax.tick_params(labelsize=9)

COLUMN_W = 3.71

def boxes(ax, panel, column, log=False):
    width, offset = (0.32, 0.19)
    for method, color, sign in (('GR-RHS', GR_COLOR, -1), ('RHS', RHS_COLOR, 1)):
        data = [panel[(panel.method == method) & (panel.scale == s)][column].values for s in SCALES]
        pos = np.arange(len(SCALES)) + sign * offset
        ax.boxplot(data, positions=pos, widths=width, patch_artist=True, showfliers=False, medianprops=dict(color='white', linewidth=1.2), whiskerprops=dict(color=color, linewidth=1), capprops=dict(color=color, linewidth=1), boxprops=dict(facecolor=color, edgecolor=color, linewidth=0.8))
    ax.set_xticks(range(len(SCALES)), ['0.5', '1', '2', '3'])
    if log:
        ax.set_yscale('log')
    style(ax)

def mechanism():
    """Two panels in one column: the conditional variance ratio in the
    correlated design, then the non-null MSE reduction.  The generating scale
    labels both panels once from the figure, and each quantity is named by its
    panel title, so neither long axis label has to fit a 1.3in panel."""
    panel = pd.read_csv(FIG / 'mechanism_panel_data.csv')
    panel = panel[panel.r2 == 0.7]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_W, 2.55))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.215, top=0.885, wspace=0.52)
    boxes(axes[0], panel, 'T')
    axes[0].set_title('(a) Variance ratio', fontsize=10)
    axes[0].set_ylim(0.15, 1.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GR_COLOR), plt.Rectangle((0, 0), 1, 1, color=RHS_COLOR)]
    axes[0].legend(handles, [GR_LABEL, 'RHS'], fontsize=7.5, frameon=False, loc='lower right', handlelength=1.1, labelspacing=0.25, borderpad=0.1, handletextpad=0.45, borderaxespad=0.2)
    ax = axes[1]
    gains = paired_by_scale(panel, 'mse_nonnull')
    ax.axhline(0, color='#1a1a1a', linewidth=0.7, linestyle=(0, (4, 2.5)), zorder=2)
    ax.errorbar(range(len(SCALES)), [g[0] for g in gains], yerr=[g[1] for g in gains], fmt='o-', color=GR_COLOR, capsize=3, markersize=4.5, linewidth=1.2, zorder=3)
    ax.set_xticks(range(len(SCALES)), ['0.5', '1', '2', '3'])
    ax.set_title('(b) Non-null MSE reduction', fontsize=10)
    style(ax)
    fig.supxlabel('Generating scale $s_g^*$', fontsize=11, y=0.015)
    fig.savefig(FIG / 'main_mechanism.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'main_mechanism.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    summary = panel.groupby(['method', 'scale'])[['mse_nonnull', 'T', 'v', 'c_g', 'tau_median']].median()
    print(summary)
    print('Non-null MSE reduction by scale (mean, half-width):', gains)


METHOD_COLORS = {'HORSESHOE': '#333333', 'RHS': '#d17a2a', 'GR_RHS': '#24658a', 'GHS': '#3F8F6E', 'GIGG': '#A6473F', 'R2D2': '#7B6BA8', 'GROUP_R2': '#8A6D1F'}
METHOD_LINESTYLES = {'HORSESHOE': (0, (5, 1.6)), 'RHS': (0, (1.2, 1.4)), 'GR_RHS': '-', 'GHS': (0, (7, 1.6)), 'GIGG': (0, (4, 1.4, 1, 1.4)), 'R2D2': (0, (3, 1.4)), 'GROUP_R2': (0, (4, 1.4, 1, 1.4, 1, 1.4))}
FULL_W = 7.56

def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)
def save_figure(fig, stem):
    with plt.rc_context(PAPER_STYLE):
        fig.savefig(stem.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight')
        fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def _plot_kappa_cdf_panel(ax, kappas, title):
    order = sorted(PRIOR_METHODS, key=lambda item: item[0] == 'GR_RHS')
    for method, label in order:
        x = np.sort(np.concatenate(kappas[method]))
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, color=METHOD_COLORS[method], linestyle=METHOD_LINESTYLES[method], linewidth=1.5 if method == 'GR_RHS' else 1.3, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$\\kappa_j^{\\mathrm{reg}}$', fontsize=11)
    ax.set_ylabel('$F_{\\kappa^{\\mathrm{reg}}}(x)=\\Pr(\\kappa_j^{\\mathrm{reg}}\\leq x)$', fontsize=11)
    ax.set_title(title, fontsize=10, pad=5)
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8, length=3)
    style_axes(ax)

def plot_kappa_cdf_pair(kappas_uncalibrated, kappas_calibrated, stem):
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9), sharey=True)
        _plot_kappa_cdf_panel(axes[0], kappas_uncalibrated, 'Before calibration')
        _plot_kappa_cdf_panel(axes[1], kappas_calibrated, 'After calibration')
        axes[1].set_ylabel('')
        handles = [Line2D([], [], color=METHOD_COLORS[m], linestyle=METHOD_LINESTYLES[m], linewidth=1.5 if m == 'GR_RHS' else 1.3) for m, _ in PRIOR_METHODS]
        labels = [label for _, label in PRIOR_METHODS]
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=len(labels), frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.0, handletextpad=0.5)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9), w_pad=1.2)
    save_figure(fig, stem)

OUT_DIR = ROOT / 'results/figure_data'
X_MAX = 4.0
Y_MAX = 4.0
X_MARGIN = 0.85
COLUMN_W = 3.25 / 0.894
X = np.linspace(0, X_MAX, 800)
COMMON_C2 = 2.0
GROUP_C2 = [0.5, 1.0, 3.0]
GROUP_COLORS = ['#3B6FA0', '#3F8F6E', '#D17D33']
COMMON_COLOR = '#A6473F'
HS_COLOR = '#333333'
XLABEL = '$\\tau^2\\lambda_j^2$'

def v_of(x, c2):
    return c2 * x / (c2 + x)

def declutter(values, min_gap):
    """Push label y-positions apart when their data values sit too close
    together to keep both labels legible, without moving the dashed
    reference lines themselves (those stay at the exact data value)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    positions = [values[i] for i in order]
    for k in range(1, len(positions)):
        if positions[k] - positions[k - 1] < min_gap:
            positions[k] = positions[k - 1] + min_gap
    out = [0.0] * len(values)
    for rank, idx in enumerate(order):
        out[idx] = positions[rank]
    return out

def style_axis(ax, ylabel=None, margin=0.0):
    ax.set_xlim(0, X_MAX + margin)
    ax.set_ylim(0, Y_MAX)
    xticks = np.arange(0, X_MAX + 0.01, 1)
    yticks = np.arange(0, Y_MAX + 0.01, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, X_MAX)
    for y in yticks:
        ax.plot([0, X_MAX], [y, y], color='#E7E7E7', linewidth=0.5, zorder=0)
    for x in xticks:
        ax.plot([x, x], [0, Y_MAX], color='#E7E7E7', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(XLABEL)
    if ylabel:
        ax.set_ylabel(ylabel)

def build_variance_figure():
    style = dict(PAPER_STYLE)
    style.update({'font.size': 8.5, 'axes.labelsize': 11, 'pdf.fonttype': 42})
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(COLUMN_W, 2.1), sharey=True)
        ax = axes[0]
        chi_top = Y_MAX
        ax.plot([0, chi_top], [0, chi_top], color=HS_COLOR, linewidth=1.3, zorder=3)
        ax.annotate('', xy=(chi_top, Y_MAX), xytext=(chi_top - 0.4, Y_MAX - 0.4), arrowprops=dict(arrowstyle='-|>', color=HS_COLOR, linewidth=1.3, mutation_scale=10), zorder=3)
        ax.text(0.02, 0.96, 'A.  HS', transform=ax.transAxes, fontsize=9, fontweight='normal', va='top')
        style_axis(ax, 'Conditional prior variance', margin=X_MARGIN)
        ax = axes[1]
        ax.plot([0, X_MAX], [0, X_MAX], color='#B0B0B0', linewidth=0.6, linestyle=(0, (1, 1.6)), zorder=2)
        v_common = v_of(X, COMMON_C2)
        ax.plot(X, v_common, color=COMMON_COLOR, linewidth=1.3, zorder=3)
        ax.plot([0, X_MAX], [COMMON_C2, COMMON_C2], color=COMMON_COLOR, linewidth=0.6, linestyle='--', alpha=0.6, zorder=2)
        ax.text(X_MAX + 0.18, COMMON_C2, '$c^2$', color=COMMON_COLOR, fontsize=11, ha='left', va='center')
        ax.text(0.02, 0.96, 'B.  RHS', transform=ax.transAxes, fontsize=9, fontweight='normal', va='top')
        style_axis(ax, margin=X_MARGIN)
        ax = axes[2]
        ax.plot([0, X_MAX], [0, X_MAX], color='#B0B0B0', linewidth=0.6, linestyle=(0, (1, 1.6)), zorder=2)
        label_y = declutter(GROUP_C2, min_gap=0.7)
        for i, (c2, ly, col) in enumerate(zip(GROUP_C2, label_y, GROUP_COLORS), start=1):
            v = v_of(X, c2)
            ax.plot(X, v, color=col, linewidth=1.3, zorder=3)
            ax.plot([0, X_MAX], [c2, c2], color=col, linewidth=0.6, linestyle='--', alpha=0.6, zorder=2)
            ax.text(X_MAX + 0.18, ly, f'$c_{i}^2$', color=col, fontsize=11, ha='left', va='center')
        ax.text(0.02, 0.96, 'C.  GR-RHS', transform=ax.transAxes, fontsize=9, fontweight='normal', va='top')
        style_axis(ax, margin=X_MARGIN)
        fig.tight_layout(pad=0.35, w_pad=0.4)
        for ext in ('pdf', 'png'):
            fig.savefig(OUT_DIR / f'variance_response.{ext}', dpi=FIG_DPI, bbox_inches='tight')
        plt.close(fig)
    print('Wrote variance_response.pdf/.png')

build_mechanism_figure = mechanism
build_prior_shape_figure = plot_kappa_cdf_pair
