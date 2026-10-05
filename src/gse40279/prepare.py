"""Prepare the fixed GSE40279 methylation representation."""
from collections import Counter
from datetime import datetime, timezone
import csv, gzip, hashlib, io, json, re, shutil, ssl, urllib.request
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'config' / 'gse40279_paper.json'
BETA_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/suppl/GSE40279_average_beta.txt.gz'
SERIES_MATRIX_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/GSE40279_series_matrix.txt.gz'
ANNOTATION_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/suppl/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz'
RAW_SOURCE_MANIFEST = ROOT / 'data/gse40279_data/manifest.json'
AGE_RE = re.compile(r'age \(y\):\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)

def load_runner_feature_data(directory: Path):
    X = np.load(directory / 'X.npy'); y = np.load(directory / 'y.npy')
    features = (directory / 'feature_names.txt').read_text(encoding='utf-8').splitlines()
    mapping = json.loads((directory / 'group_map.json').read_text(encoding='utf-8'))
    groups = np.asarray([mapping[name] for name in features], dtype=int)
    return X, y, groups, features

def load_gse40279_data(directory: Path):
    X, y, groups, features = load_runner_feature_data(directory)
    return X, y, groups, features

def verify_raw_file(path: Path, expected_sha256: str) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f'SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual}')
    return actual

def verify_raw_inputs(beta_path: Path, annotation_path: Path) -> dict[str, str]:
    manifest = json.loads(RAW_SOURCE_MANIFEST.read_text(encoding='utf-8'))
    expected = {item['role']: item['sha256'] for item in manifest['files']}
    return {'beta_matrix': verify_raw_file(beta_path, expected['methylation_beta_matrix']), 'annotation': verify_raw_file(annotation_path, expected['platform_annotation'])}

def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'GR-RHS-data-preparation/1.0'})
    with urllib.request.urlopen(request, context=ssl.create_default_context(), timeout=120) as response, destination.open('wb') as handle:
        shutil.copyfileobj(response, handle)

def _fetch_series_header() -> bytes:
    request = urllib.request.Request(SERIES_MATRIX_URL, headers={'User-Agent': 'GR-RHS-data-preparation/1.0', 'Range': 'bytes=0-1048575'})
    with urllib.request.urlopen(request, timeout=120) as response: return response.read()

def _partial_gzip_lines(raw: bytes) -> list[str]:
    lines = []
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
        text = io.TextIOWrapper(compressed, encoding='utf-8', errors='replace')
        try: lines.extend(text)
        except (EOFError, gzip.BadGzipFile): pass
    return lines

def _fields(line: str) -> list[str]:
    return [v.strip().strip('"') for v in next(csv.reader([line.rstrip('\n')], delimiter='\t'))]

def _sample_metadata() -> pd.DataFrame:
    names = accessions = ages = None
    for line in _partial_gzip_lines(_fetch_series_header()):
        if line.startswith('!Sample_source_name_ch1'): names = _fields(line)[1:]
        elif line.startswith('!Sample_geo_accession'): accessions = _fields(line)[1:]
        elif line.startswith('!Sample_characteristics_ch1'):
            values = _fields(line)[1:]
            if values and values[0].lower().startswith('age (y):'):
                parsed = [AGE_RE.search(value) for value in values]
                if any(match is None for match in parsed): raise ValueError('Could not parse every chronological age in the GEO series header')
                ages = [float(match.group(1)) for match in parsed]
        if line.startswith('!series_matrix_table_begin'): break
    if names is None or accessions is None or ages is None or not len(names) == len(accessions) == len(ages): raise RuntimeError('Incomplete or inconsistent sample metadata in the GEO series header')
    return pd.DataFrame({'sample_name': names, 'geo_accession': accessions, 'age_years': ages})

def _first_gene(value: Any) -> str:
    if pd.isna(value): return ''
    return next((token.strip() for token in str(value).split(';') if token.strip()), '')

def _block_anchor(labels: pd.Series) -> str:
    values = [text for value in labels if (text := str(value).strip())]
    if not values: return ''
    counts = Counter(values); best = max(counts.values())
    return sorted(label for label, count in counts.items() if count == best)[0]

def _local_block_annotation(annotation_path: Path, block_gap_bp: int):
    frame = pd.read_csv(annotation_path, compression='gzip', skiprows=7, usecols=['Name','CHR','MAPINFO','UCSC_RefGene_Name'], low_memory=False).rename(columns={'Name':'probe_id','CHR':'chromosome','MAPINFO':'position'})
    frame['probe_id'] = frame['probe_id'].astype(str).str.strip(); frame['chromosome'] = frame['chromosome'].astype(str).str.strip(); frame['position'] = pd.to_numeric(frame['position'], errors='coerce'); frame['anchor_gene'] = frame['UCSC_RefGene_Name'].map(_first_gene)
    valid = frame['probe_id'].str.startswith('cg') & frame['position'].notna() & ~frame['chromosome'].isin({'','nan','NA','0','X','Y','chrX','chrY'})
    probes = frame.loc[valid, ['probe_id','chromosome','position','anchor_gene']].copy(); probes['position'] = probes['position'].astype(int); probes = probes.sort_values(['chromosome','position','probe_id'], kind='stable').reset_index(drop=True)
    block_ids=[]; block_id=-1; previous=None
    for chrom, position in probes[['chromosome','position']].itertuples(index=False, name=None):
        current=(str(chrom),int(position))
        if previous is None or current[0] != previous[0] or current[1]-previous[1] > block_gap_bp: block_id += 1
        block_ids.append(block_id); previous=current
    probes['block_id']=np.asarray(block_ids,dtype=np.int32)
    catalog=probes.groupby('block_id',as_index=False).agg(chromosome=('chromosome','first'),block_start=('position','min'),block_end=('position','max'),block_probe_count=('probe_id','count'),anchor_gene=('anchor_gene',_block_anchor))
    catalog['group_label']=[f'{gene}@chr{chrom}:{start}-{end}' if gene else f'chr{chrom}:{start}-{end}' for chrom,start,end,gene in catalog[['chromosome','block_start','block_end','anchor_gene']].itertuples(index=False,name=None)]
    annotation=probes.merge(catalog[['block_id','block_start','block_end','block_probe_count','group_label']],on='block_id',how='left')
    return annotation,catalog

def _beta_columns(beta_path: Path) -> list[str]:
    with gzip.open(beta_path,'rt',encoding='utf-8',errors='replace') as handle: columns=handle.readline().rstrip('\n').split('\t')
    if not columns or columns[0] != 'ID_REF': raise RuntimeError('Unexpected GSE40279 beta-matrix header')
    return columns[1:]

def _rank_probes(beta_path, annotation, samples, chunk_size):
    indexed=annotation.set_index('probe_id'); results=[]
    for chunk_no, chunk in enumerate(pd.read_csv(beta_path,sep='\t',compression='gzip',chunksize=chunk_size,low_memory=False),start=1):
        probes=chunk['ID_REF'].astype(str).str.strip(); annot=indexed.reindex(probes); valid=annot['group_label'].notna().to_numpy()
        if valid.any():
            values=chunk.loc[valid,samples].to_numpy(dtype=np.float32,copy=False); finite=np.isfinite(values).all(axis=1); variances=values[finite].var(axis=1,dtype=np.float64); positive=variances>0; keep=np.flatnonzero(finite)[positive]; kept_probes=probes.loc[valid].to_numpy()[keep]; kept_annot=annot.loc[valid].iloc[keep]
            results.append(pd.DataFrame({'probe_id':kept_probes,'variance':variances[positive],'chromosome':kept_annot['chromosome'].astype(str).to_numpy(),'group_label':kept_annot['group_label'].astype(str).to_numpy()}))
        if chunk_no % 25 == 0: print(f'[gse40279] variance pass: {chunk_no} chunks')
    if not results: raise RuntimeError('No eligible finite, non-constant autosomal CpG probes')
    return pd.concat(results,ignore_index=True).sort_values(['variance','probe_id'],ascending=[False,True],kind='stable').reset_index(drop=True)

def _select(ranked, top_k, min_group_size, max_group_size):
    capped=ranked.groupby('group_label',sort=False,group_keys=False).head(max_group_size); counts=capped['group_label'].value_counts(); capped=capped.loc[capped['group_label'].isin(counts[counts>=min_group_size].index)].copy(); groups=[]
    for label, frame in capped.groupby('group_label',sort=False): groups.append((str(label),float(frame['variance'].head(10).sum()),frame.reset_index(drop=True)))
    groups.sort(key=lambda x:(-x[1],x[0])); chosen=[]; available=0
    for label,_,frame in groups:
        if available >= top_k: break
        chosen.append((label,frame)); available += len(frame)
    selected=pd.concat([frame.iloc[:min_group_size] for _,frame in chosen],ignore_index=True); remaining=top_k-len(selected); extras=[frame.iloc[min_group_size:] for _,frame in chosen if len(frame)>min_group_size]
    if extras and remaining>0: selected=pd.concat([selected,pd.concat(extras,ignore_index=True).sort_values(['variance','probe_id'],ascending=[False,True],kind='stable').head(remaining)],ignore_index=True)
    if len(selected)!=top_k: raise RuntimeError(f'Selection produced {len(selected)} features, expected {top_k}; relax group constraints')
    selected=selected.sort_values(['variance','probe_id'],ascending=[False,True],kind='stable').reset_index(drop=True); order=list(dict.fromkeys(selected['group_label'].astype(str))); selected['group_id']=selected['group_label'].map({label:i for i,label in enumerate(order)}).astype(int); return selected

def _extract_matrix(beta_path, probes, samples, chunk_size):
    wanted=set(probes); rows={}
    for chunk_no, chunk in enumerate(pd.read_csv(beta_path,sep='\t',compression='gzip',chunksize=chunk_size,low_memory=False),start=1):
        subset=chunk.loc[chunk['ID_REF'].astype(str).isin(wanted),['ID_REF',*samples]]
        for row in subset.itertuples(index=False): rows[str(row[0])] = np.asarray(row[1:],dtype=np.float32)
        if chunk_no % 25 == 0: print(f'[gse40279] extraction pass: {chunk_no} chunks')
    missing=[probe for probe in probes if probe not in rows]
    if missing: raise RuntimeError(f'Missing {len(missing)} selected probes; first is {missing[0]}')
    return np.column_stack([rows[probe] for probe in probes])

def _write_lines(path: Path, values: Iterable[str]) -> None: path.write_text('\n'.join(values)+'\n',encoding='utf-8')

def prepare_methylation_data(config):
    settings=config['preparation']; raw_dir=ROOT/config['data_root']/'raw'; output=ROOT/config['data_root']/'prepared'; chunk_size=settings['chunk_size']; top_k=settings['top_k']; block_gap_bp=settings['block_gap_bp']; min_group_size=settings['min_group_size']; max_group_size=settings['max_group_size']; raw_dir.mkdir(parents=True,exist_ok=True)
    beta_path=raw_dir/'GSE40279_average_beta.txt.gz'; annotation_path=raw_dir/'GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz'; _download(BETA_URL,beta_path); _download(ANNOTATION_URL,annotation_path); raw_sha256=verify_raw_inputs(beta_path,annotation_path); runner=output/'runner_ready'; runner.mkdir(parents=True,exist_ok=True); metadata=_sample_metadata(); samples=_beta_columns(beta_path)
    if metadata['sample_name'].tolist()!=samples: raise RuntimeError('GEO metadata sample order does not match beta-matrix columns')
    annotation,_=_local_block_annotation(annotation_path,block_gap_bp); ranked=_rank_probes(beta_path,annotation,samples,chunk_size); selected=_select(ranked,top_k,min_group_size,max_group_size); selected=selected.merge(annotation[['probe_id','position','block_id','block_start','block_end','block_probe_count']],on='probe_id',how='left',validate='one_to_one'); probes=selected['probe_id'].astype(str).tolist(); methylation_matrix=_extract_matrix(beta_path,probes,samples,chunk_size); age_years=metadata['age_years'].to_numpy(dtype=np.float32); np.save(runner/'X.npy',methylation_matrix); np.save(runner/'y.npy',age_years); _write_lines(runner/'feature_names.txt',probes); cpg_group_labels=selected[['group_id','group_label']].drop_duplicates().sort_values('group_id')['group_label'].tolist(); _write_lines(runner/'group_labels.txt',map(str,cpg_group_labels)); (runner/'group_map.json').write_text(json.dumps(dict(selected[['probe_id','group_id']].itertuples(index=False,name=None)),indent=2),encoding='utf-8'); selected.to_csv(output/'selected_features.tsv',sep='\t',index=False); group_sizes=selected['group_id'].value_counts(); summary={'dataset_id':'gse40279_age_local_blocks_p500_gap1000_cap30','generated_at':datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),'source':{'geo_accession':'GSE40279','beta_matrix_url':BETA_URL,'series_matrix_url':SERIES_MATRIX_URL,'annotation_url':ANNOTATION_URL,'raw_sha256':raw_sha256},'preprocessing':{'response_used_for_screening':False,'screening_rule':'Sample variance among finite, positive-variance autosomal CpGs with genomic coordinates.','grouping_rule':f'Coordinate-local blocks; split adjacent CpGs when distance exceeds {block_gap_bp} bp.','top_k':top_k,'min_group_size':min_group_size,'max_group_size':max_group_size},'benchmark':{'sample_count':int(methylation_matrix.shape[0]),'feature_count':int(methylation_matrix.shape[1]),'group_count':int(group_sizes.size),'group_size_summary':{'min':int(group_sizes.min()),'median':float(group_sizes.median()),'max':int(group_sizes.max())},'age_min':float(age_years.min()),'age_max':float(age_years.max())},'runner_ready':str(runner.relative_to(ROOT))}; (output/'dataset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary['benchmark'],indent=2)); return runner
