"""Public entry point for the GSE40279 publication workflow."""
import argparse, json
from pathlib import Path
import numpy as np
from . import prepare, fit, summarize

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'config/gse40279_paper.json'
METHODS=('GR_RHS','RHS','HORSESHOE','R2D2','GROUP_R2','GHS','GIGG')

def derived_seed(base_seed: int, *indices: int) -> int:
    return int(np.random.SeedSequence([base_seed,*indices]).generate_state(1)[0])

def run_fit(cli_args):
    config=json.loads(cli_args.config.resolve().read_text(encoding='utf-8')); data_dir=Path(cli_args.data_dir or config['data_dir']); output=Path(cli_args.output or config['output']); replication_start=cli_args.replication_start or config['replication_start']; replication_end=cli_args.replication_end or config.get('replication_end');
    if not data_dir.is_absolute(): data_dir=ROOT/data_dir
    if not output.is_absolute(): output=ROOT/output
    configured_methods=tuple(config['methods']); unknown=set(configured_methods).difference(METHODS)
    if unknown: raise ValueError(f'Config contains unsupported paper methods: {sorted(unknown)}')
    methods=[cli_args.method] if cli_args.method else list(configured_methods)
    if cli_args.method is not None and cli_args.method not in configured_methods: raise ValueError(f'Method is not enabled by this configuration: {cli_args.method}')
    sampler={'chains':config['chains'],'warmup':config['warmup'],'draws':config['draws'],'adapt_delta':config['adapt_delta'],'max_treedepth':config['max_treedepth']}; output.mkdir(parents=True,exist_ok=True); methylation_matrix,age_years,cpg_group_ids,_=prepare.load_gse40279_data(data_dir/'runner_ready')
    if methylation_matrix.shape != (config['expected_n'],config['expected_p']): raise ValueError(f'Prepared GSE matrix has shape {methylation_matrix.shape}; expected ({config["expected_n"]}, {config["expected_p"]})')
    if np.unique(cpg_group_ids).size != config['expected_groups']: raise ValueError('Prepared GSE feature groups do not match the selected configuration')
    splits=fit.make_repeated_cv_splits(age_years,seed=config['seed'],fold_count=config['folds'],replications=config['replications']); fit.compile_models(methods); final_replication=replication_end or config['replications']
    for replication,fold,_,train,test in splits:
        if not (replication_start <= replication <= final_replication) or (cli_args.fold is not None and fold != cli_args.fold): continue
        dataset, y_center, y_scale = fit.build_fold_dataset(methylation_matrix, age_years, cpg_group_ids, train, test); fold_root=output/f'rep_{replication:03d}'/f'fold_{fold:02d}'
        for method_id in methods:
            print(f'rep={replication:03d} fold={fold:02d} method={method_id}',flush=True); method_index=METHODS.index(method_id); method_sampler={**sampler,**config.get('sampler_overrides',{}).get(method_id,{})}
            if config['paired_seed_methods'] and method_id in ('GR_RHS','RHS'): method_index=METHODS.index('GR_RHS')
            seed=derived_seed(config['seed'],replication,fold,method_index); fit.fit_one_fold(method_id,dataset,seed,method_sampler,fold_root/method_id,y_center,y_scale,{**config,'slab_variance_scale':config['slab_variance_scale']})

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=('prepare','fit','table')); args,rest=parser.parse_known_args(argv)
    if args.stage=='prepare':
        p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=prepare.DEFAULT_CONFIG); a=p.parse_args(rest); prepare.prepare_methylation_data(json.loads(a.config.read_text(encoding='utf-8')))
    elif args.stage=='fit':
        p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); p.add_argument('--data-dir',type=Path); p.add_argument('--output',type=Path); p.add_argument('--replication-start',type=int); p.add_argument('--replication-end',type=int); p.add_argument('--fold',type=int); p.add_argument('--method',choices=METHODS); a=p.parse_args(rest); run_fit(a)
    elif args.stage=='table':
        p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path); p.add_argument('--fold-metrics-csv',type=Path); p.add_argument('--rhs-bp-run-root',type=Path); p.add_argument('--replications',type=int,default=20); p.add_argument('--folds',type=int,default=5); p.add_argument('--csv',dest='csv_output',type=Path); p.add_argument('--tex',dest='tex_output',type=Path); p.add_argument('--fold-metrics-output',type=Path); a=p.parse_args(rest); table_csv=a.fold_metrics_csv if a.run_root or a.fold_metrics_csv else ROOT/'results/reproducibility/data/gse40279_paper_fold_metrics.csv'; summarize.summarize_table1(run_root=a.run_root,fold_metrics_csv=table_csv,rhs_bp_run_root=a.rhs_bp_run_root,replications=a.replications,folds=a.folds,csv_output=a.csv_output or ROOT/'results/figure_data/gse40279_paper_table_summary.csv',tex_output=a.tex_output or ROOT/'results/figure_data/gse40279_paper_table.tex',fold_metrics_output=a.fold_metrics_output)

if __name__=='__main__': main()
