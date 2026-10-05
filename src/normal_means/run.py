"""Public command-line entry point for the normal-means experiment."""

import argparse
from pathlib import Path

import pandas as pd

from .design import check_design, initialize_design, iter_cases, load_config
from .fit import build_model, fit_one_case
from .plot import DEFAULT_SUMMARY_DIR, build_figure
from .summarize import summarize_experiment

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'config/normal_means_paper.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'prepare', 'run', 'summarize', 'figure'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--input', type=Path, default=DEFAULT_SUMMARY_DIR)
    args = parser.parse_args()
    if args.command == 'check':
        config = load_config(args.config)
        check_design(config)
        return
    if args.command == 'figure':
        table = build_figure(args.input)
        pd.set_option('display.width', 200)
        for panel, group in table.groupby('panel'):
            print(f'\n--- panel ({panel}) {group.quantity.iloc[0]} ---')
            print(group.pivot_table(index='amplitude', columns='series', values='mean').round(4).to_string())
        print('\nWrote results/figure_data/amplitude_probe.pdf and summary CSV.')
        return
    config = load_config(args.config)
    output_dir = (args.output or ROOT / config['output']).resolve()
    initialize_design(config, output_dir)
    if args.command == 'run':
        models = {model_name: build_model(model_name) for model_name in config['models']}
        for case in iter_cases(config):
            fit_one_case(models[case['model']], config, output_dir, case)
    elif args.command == 'summarize':
        summarize_experiment(config, output_dir)
    else:
        print(f'Prepared {len(list(iter_cases(config)))} fits in {output_dir}; no sampling performed.')
