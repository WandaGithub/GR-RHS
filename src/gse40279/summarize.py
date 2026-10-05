"""Aggregate completed GSE40279 fold results and render Table 1."""
import csv, json, statistics
from pathlib import Path
import numpy as np

PAPER_METHODS={'GR_RHS':'GR-RHS','RHS':'RHS','HORSESHOE':'HS','R2D2':'R2D2','GROUP_R2':'Group-$R^2$','GHS':'GHS','GIGG':'GIGG'}
METRICS=('test_rmse','test_mae','test_r2','elpd_test')
RHS_OUTPUT_DIR='RHS-BP'

def point_prediction_metrics(prediction, observed):
    residual=np.asarray(observed)-np.asarray(prediction)
    return {'test_rmse':float(np.sqrt(np.mean(residual ** 2))),'test_mae':float(np.mean(np.abs(residual))),'test_r2':float(1.0-residual @ residual / np.sum((observed-np.mean(observed)) ** 2)),'test_bias':float(np.mean(residual))}

def load_complete_metrics(run_root: Path, rhs_bp_run_root: Path, replications: int, folds: int):
    expected=replications*folds; rows={method:[] for method in PAPER_METHODS}; seen=set()
    for method in PAPER_METHODS:
        method_root=rhs_bp_run_root if method in ('GR_RHS','RHS') else run_root
        method_dirs=(method,RHS_OUTPUT_DIR) if method=='RHS' else (method,)
        paths=[path for method_dir in method_dirs for path in method_root.glob(f'rep_*/fold_*/{method_dir}/metrics.json')]
        for path in sorted(paths):
            try: replication=int(path.parents[2].name.removeprefix('rep_')); fold=int(path.parents[1].name.removeprefix('fold_'))
            except ValueError as exc: raise ValueError(f'Unexpected result path: {path}') from exc
            if not 1<=replication<=replications or not 1<=fold<=folds: continue
            key=(method,replication,fold)
            if key in seen: raise ValueError(f'Duplicate metric cell: method={method}, rep={replication}, fold={fold}')
            seen.add(key); row=json.loads(path.read_text(encoding='utf-8')); missing=[name for name in METRICS if name not in row]
            if missing: raise ValueError(f"{path} lacks required metrics: {', '.join(missing)}")
            rows[method].append({'source_set':'paired' if method in ('GR_RHS','RHS') else 'formal','replication':replication,'fold':fold,'method':method,**{name:float(row[name]) for name in METRICS}})
    incomplete={method:len(values) for method,values in rows.items() if len(values)!=expected}
    if incomplete: raise ValueError('Expected exactly completed fold metrics per paper method; found '+', '.join(f'{m}={n}/{expected}' for m,n in incomplete.items()))
    return rows

def load_fold_metrics_csv(path: Path, replications: int, folds: int):
    expected=replications*folds; rows={method:[] for method in PAPER_METHODS}; seen=set(); required={'source_set','replication','fold','method',*METRICS}
    with path.open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle); missing=required.difference(reader.fieldnames or ())
        if missing: raise ValueError(f'{path} lacks columns: {", ".join(sorted(missing))}')
        for row in reader:
            method=row['method']
            if method not in rows: raise ValueError(f'Unexpected paper method key in {path}: {method}')
            expected_source='paired' if method in ('GR_RHS','RHS') else 'formal'
            if row['source_set']!=expected_source: raise ValueError(f'Wrong source_set for {method}: expected {expected_source}, got {row["source_set"]}')
            replication,fold=int(row['replication']),int(row['fold']); key=(method,replication,fold)
            if not 1<=replication<=replications or not 1<=fold<=folds: raise ValueError(f'Out-of-range fold cell: {method} rep={replication} fold={fold}')
            if key in seen: raise ValueError(f'Duplicate fold cell: {method} rep={replication} fold={fold}')
            seen.add(key); rows[method].append({'source_set':row['source_set'],'replication':replication,'fold':fold,'method':method,**{name:float(row[name]) for name in METRICS}})
    incomplete={method:len(values) for method,values in rows.items() if len(values)!=expected}
    if incomplete: raise ValueError('Expected exactly completed fold metrics per paper method; found '+', '.join(f'{m}={n}/{expected}' for m,n in incomplete.items()))
    return rows

def summarize(rows):
    result=[]
    for method,label in PAPER_METHODS.items():
        values=rows[method]; summary={'method':method,'label':label,'completed_folds':len(values)}
        for metric in METRICS:
            observations=[row[metric] for row in values]; summary[f'{metric}_mean']=statistics.mean(observations)
            if metric in ('test_rmse','test_mae'): summary[f'{metric}_sd']=statistics.stdev(observations)
        result.append(summary)
    return result

def write_fold_metrics_csv(rows,path):
    fields=['source_set','replication','fold','method',*METRICS]; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for method in PAPER_METHODS:
            for row in sorted(rows[method],key=lambda item:(item['replication'],item['fold'])): writer.writerow({field:row[field] for field in fields})

def make_tex(summary):
    best_two={metric:{row['method'] for row in sorted(summary,key=lambda item:item[f'{metric}_mean'],reverse=reverse)[:2]} for metric,reverse in (('test_rmse',False),('test_mae',False),('test_r2',True),('elpd_test',True))}; lines=['\\begin{tabular}{lrrrr}','\\toprule','Method & RMSE (SD) $\\downarrow$ & MAE (SD) $\\downarrow$ & $R^2\\uparrow$ & ELPD $\\uparrow$ \\\\','\\midrule']
    for row in summary:
        rmse=f"{row['test_rmse_mean']:.3f} ({row['test_rmse_sd']:.3f})"; mae=f"{row['test_mae_mean']:.3f} ({row['test_mae_sd']:.3f})"; r2=f"{row['test_r2_mean']:.3f}"; elpd=f"{row['elpd_test_mean']:.3f}"
        if row['method'] in best_two['test_rmse']: rmse=f'\\textbf{{{rmse}}}'
        if row['method'] in best_two['test_mae']: mae=f'\\textbf{{{mae}}}'
        if row['method'] in best_two['test_r2']: r2=f'\\textbf{{{r2}}}'
        if row['method'] in best_two['elpd_test']: elpd=f'\\textbf{{{elpd}}}'
        lines.append(f"{row['label']} & {rmse} & {mae} & {r2} & {elpd} "+'\\\\')
    lines.extend(('\\bottomrule','\\end{tabular}')); return '\n'.join(lines)+'\n'

def summarize_table1(*, run_root=None, fold_metrics_csv=None, rhs_bp_run_root=None, replications=20, folds=5, csv_output=None, tex_output=None, fold_metrics_output=None):
    rows=load_fold_metrics_csv(fold_metrics_csv,replications,folds) if fold_metrics_csv else load_complete_metrics(run_root,rhs_bp_run_root or run_root,replications,folds)
    if fold_metrics_output: write_fold_metrics_csv(rows,fold_metrics_output)
    summary=summarize(rows); tex=make_tex(summary); print(tex,end='')
    if csv_output:
        csv_output.parent.mkdir(parents=True,exist_ok=True)
        with csv_output.open('w',newline='',encoding='utf-8') as handle: writer=csv.DictWriter(handle,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    if tex_output: tex_output.parent.mkdir(parents=True,exist_ok=True); tex_output.write_text(tex,encoding='utf-8')
    return summary
