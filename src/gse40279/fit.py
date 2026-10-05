"""Cross-validation, Stan fitting, predictions, and fold persistence."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import arviz as az
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'stan'
MODEL_FILES = {'GR_RHS': MODEL_DIR/'grrhs.stan','RHS': MODEL_DIR/'rhs.stan','HORSESHOE': MODEL_DIR/'horseshoe.stan','R2D2': MODEL_DIR/'r2d2.stan','GROUP_R2': MODEL_DIR/'group_r2.stan','GHS': MODEL_DIR/'ghs.stan','GIGG': MODEL_DIR/'gigg.stan'}
COMMON_PRIOR = {'intercept_mean':0.0,'intercept_sd':5.0,'sigma_df':3.0,'sigma_location':0.0,'sigma_scale':None,'z_mean':0.0,'z_sd':1.0,'tau_scale_multiplier':1.0}
PRIORS = {'GR_RHS':dict(**COMMON_PRIOR,tau_calibrated=True,lambda_normal_mean=0.0,lambda_normal_sd=1.0,lambda_ig_shape=0.5,lambda_ig_scale=0.5,alpha_b=1.0,beta_b=1.0),'HORSESHOE':dict(**COMMON_PRIOR,tau_calibrated=True,tau_location=0.0,lambda_location=0.0,lambda_scale=1.0),'R2D2':dict(**COMMON_PRIOR,r2_a=0.0,r2_b=0.0,phi_concentration=0.5),'GROUP_R2':dict(**COMMON_PRIOR,r2_a=0.0,r2_b=0.0,phi_group_concentration=0.5,phi_within_concentration=0.5,phi_within_rate=1.0),'GHS':dict(**COMMON_PRIOR,tau_calibrated=True),'GIGG':dict(**COMMON_PRIOR,tau_calibrated=True,tau_location=0.0,gamma2_shape=0.5,gamma2_rate=1.0,lambda2_shape=0.5,lambda2_scale=1.0)}
PRIORS['RHS'] = PRIORS['GR_RHS'].copy()

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    groups: np.ndarray

@dataclass
class Fit:
    beta_draws: np.ndarray
    intercept_draws: np.ndarray
    stan_fit: Any

def make_cv_splits(y: np.ndarray, *, seed: int=12345, fold_count: int=5):
    y=np.asarray(y,dtype=float); bins=np.digitize(y,np.unique(np.quantile(y,[.2,.4,.6,.8]))); cv=StratifiedKFold(n_splits=fold_count,shuffle=True,random_state=seed); return list(cv.split(np.zeros(y.size),bins))

def make_repeated_cv_splits(y: np.ndarray, *, seed: int=12345, fold_count: int=5, replications: int=100):
    if fold_count<2 or replications<1: raise ValueError('fold_count >= 2 and replications >= 1 are required')
    for replication in range(1,replications+1):
        split_seed=seed+replication-1
        for fold,(train,test) in enumerate(make_cv_splits(y,seed=split_seed,fold_count=fold_count),start=1): yield replication,fold,split_seed,train,test

def standardize_from_training(X, train, test):
    center=X[train].mean(axis=0); scale=X[train].std(axis=0,ddof=0); scale=np.where(scale>0,scale,1.0); return (X[train]-center)/scale,(X[test]-center)/scale,center,scale

def build_fold_dataset(X, y, groups, train, test):
    X_train, X_test, _, _ = standardize_from_training(X, train, test)
    y_center = float(y[train].mean())
    y_scale = float(y[train].std(ddof=0))
    if not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError('Training response must have finite positive SD')
    dataset = Dataset(X_train, (y[train] - y_center) / y_scale, X_test, (y[test] - y_center) / y_scale, groups)
    return dataset, y_center, y_scale

def real_data_prior_config(method: str, *, p0: float, slab_variance_scale: float=1.0, r2d2_mean: float, group_r2_mean: float, r2_precision: float, beta_prime_alpha: float, beta_prime_beta: float):
    if slab_variance_scale<=0: raise ValueError('slab_variance_scale must be positive')
    if method=='R2D2': return {**PRIORS[method],'r2_a':r2d2_mean*r2_precision,'r2_b':(1-r2d2_mean)*r2_precision}
    if method=='GROUP_R2': return {**PRIORS[method],'r2_a':group_r2_mean*r2_precision,'r2_b':(1-group_r2_mean)*r2_precision}
    return {**PRIORS[method],'p0':float(p0),'tau_calibrated':True,'slab_variance_scale':float(slab_variance_scale),'alpha_b':float(beta_prime_alpha),'beta_b':float(beta_prime_beta)}

def _design_matrix(dataset, prior):
    X=np.column_stack((np.ones(dataset.X_train.shape[0]),dataset.X_train)); X_test=np.column_stack((np.ones(dataset.X_test.shape[0]),dataset.X_test)); sigma_scale=dataset.y_train.std(ddof=1) if prior['sigma_scale'] is None else prior['sigma_scale']
    return {'N':dataset.X_train.shape[0],'y':dataset.y_train,'p':X.shape[1],'X':X,'Ntest':dataset.X_test.shape[0],'ytest':dataset.y_test,'Xtest':X_test,'intercept_mean':prior['intercept_mean'],'intercept_sd':prior['intercept_sd'],'sigma_df':prior['sigma_df'],'sigma_location':prior['sigma_location'],'sigma_scale':sigma_scale,'z_mean':prior['z_mean'],'z_sd':prior['z_sd']}

def make_stan_data(method,dataset,prior):
    if method in ('GR_RHS','RHS'):
        return ({**_design_matrix(dataset,prior),'G':1 if method=='RHS' else dataset.groups.max()+1,'group_id':np.ones(dataset.X_train.shape[1],dtype=int) if method=='RHS' else dataset.groups+1,'p0':prior['p0'],'use_tau_calibration':int(prior['tau_calibrated']),'tau_scale_multiplier':prior['tau_scale_multiplier'],'use_tau_fixed':0,'tau_fixed':1.0,'lambda_normal_mean':prior['lambda_normal_mean'],'lambda_normal_sd':prior['lambda_normal_sd'],'lambda_ig_shape':prior['lambda_ig_shape'],'lambda_ig_scale':prior['lambda_ig_scale'],'alpha_b':prior['alpha_b'],'beta_b':prior['beta_b'],'slab_variance_scale':prior['slab_variance_scale']},None)
    if method=='HORSESHOE': return ({**_design_matrix(dataset,prior),'p0':prior['p0'],'use_tau_calibration':int(prior['tau_calibrated']),'tau_location':prior['tau_location'],'lambda_location':prior['lambda_location'],'lambda_scale':prior['lambda_scale']},None)
    if method=='R2D2':
        a,b=prior['r2_a'],prior['r2_b']; return ({**_design_matrix(dataset,prior),'R2_a':a/(a+b),'R2_b':a+b,'phi_concentration':prior['phi_concentration']},None)
    if method=='GROUP_R2':
        groups=[np.flatnonzero(dataset.groups==group) for group in np.unique(dataset.groups)]; order=np.concatenate(groups); restore=np.argsort(order); ordered=Dataset(dataset.X_train[:,order],dataset.y_train,dataset.X_test[:,order],dataset.y_test,dataset.groups[order]); a,b=prior['r2_a'],prior['r2_b']; return ({**_design_matrix(ordered,prior),'G':len(groups),'pg':[g.size for g in groups],'phi_group_concentration':prior['phi_group_concentration'],'phi_within_concentration':prior['phi_within_concentration'],'phi_within_rate':prior['phi_within_rate'],'R2_a':a/(a+b),'R2_b':a+b},restore)
    if method=='GHS': return ({**_design_matrix(dataset,prior),'G':dataset.groups.max()+1,'group_id':dataset.groups+1,'p0':prior['p0'],'use_tau_calibration':int(prior['tau_calibrated'])},None)
    if method=='GIGG':
        G=dataset.groups.max()+1; return ({**_design_matrix(dataset,prior),'G':G,'group_id':dataset.groups+1,'tau_location':prior['tau_location'],'p0':prior['p0'],'use_tau_calibration':int(prior['tau_calibrated']),'gamma2_shape':np.full(G,prior['gamma2_shape']),'gamma2_rate':prior['gamma2_rate'],'lambda2_shape':np.full(dataset.X_train.shape[1],prior['lambda2_shape']),'lambda2_scale':prior['lambda2_scale']},None)
    raise ValueError(f'Unknown method: {method}')

def _model(method):
    from cmdstanpy import CmdStanModel
    if method not in MODEL_FILES: raise ValueError(f'Unknown current method: {method}')
    return CmdStanModel(stan_file=str(MODEL_FILES[method]))

def compile_models(methods):
    for method in methods: _model(method)

def fit_model(method,dataset,*,seed,sampler,prior):
    stan_data,restore=make_stan_data(method,dataset,prior); result=_model(method).sample(data=stan_data,seed=seed,chains=sampler['chains'],parallel_chains=sampler['chains'],iter_warmup=sampler['warmup'],iter_sampling=sampler['draws'],adapt_delta=sampler['adapt_delta'],max_treedepth=sampler['max_treedepth'],show_progress=False); beta=result.stan_variable('beta');
    if restore is not None: beta=beta[:,restore]
    return Fit(beta_draws=beta,intercept_draws=result.stan_variable('b_intercept'),stan_fit=result)

def cleanup_raw_csv(fit):
    for csv_path in fit.stan_fit.runset.csv_files: Path(csv_path).unlink()

def back_transform_predictions(prediction,pointwise_elpd,y_center,y_scale):
    if not np.isfinite(y_scale) or y_scale<=0: raise ValueError('Response scale must be finite and positive')
    return y_center+y_scale*np.asarray(prediction),np.asarray(pointwise_elpd)-np.log(y_scale)

def sampler_diagnostics(fit):
    posterior=az.from_cmdstanpy(posterior=fit.stan_fit); summary=az.summary(posterior,kind='diagnostics'); r_hat=pd.to_numeric(summary['r_hat'],errors='coerce'); ess_bulk=pd.to_numeric(summary['ess_bulk'],errors='coerce'); ess_tail=pd.to_numeric(summary['ess_tail'],errors='coerce'); return {'rhat_max':float(r_hat.max()),'ess_bulk_min':float(ess_bulk.min()),'ess_tail_min':float(ess_tail.min()),'divergence_rate':float(posterior.sample_stats['diverging'].mean())}

def cv_fold_metrics(fit,dataset):
    beta_mean=np.asarray(fit.beta_draws,dtype=float).mean(axis=0); prediction=dataset.X_test@beta_mean+np.mean(fit.intercept_draws); log_lik_test=fit.stan_fit.stan_variable('log_lik_test'); pointwise=logsumexp(log_lik_test,axis=0)-np.log(log_lik_test.shape[0]); return {'observed':dataset.y_test,'prediction':prediction,'pointwise_elpd':pointwise,**sampler_diagnostics(fit)}

def fit_one_fold(method,dataset,seed,sampler,output,y_center,y_scale,config):
    output.mkdir(parents=True,exist_ok=True); prior=real_data_prior_config(method,p0=config['p0'],slab_variance_scale=config['slab_variance_scale'],r2d2_mean=config['r2d2_prior_mean'],group_r2_mean=config['group_r2_prior_mean'],r2_precision=config['r2_prior_precision'],beta_prime_alpha=config['beta_prime_alpha'],beta_prime_beta=config['beta_prime_beta']); fit=fit_model(method,dataset,seed=seed,sampler=sampler,prior=prior); fold=cv_fold_metrics(fit,dataset); observed=y_center+y_scale*dataset.y_test; prediction,elpd=back_transform_predictions(fold['prediction'],fold['pointwise_elpd'],y_center,y_scale); metrics={'method':method,'test_rmse':float(np.sqrt(np.mean((observed-prediction)**2))),'test_mae':float(np.mean(np.abs(observed-prediction))),'test_r2':float(1-(observed-prediction)@(observed-prediction)/np.sum((observed-np.mean(observed))**2)),'test_bias':float(np.mean(observed-prediction)),'elpd_test':float(np.sum(elpd)),'response_units':'years','rhat_max':fold['rhat_max'],'ess_bulk_min':fold['ess_bulk_min'],'ess_tail_min':fold['ess_tail_min'],'divergence_rate':fold['divergence_rate']}; (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); cleanup_raw_csv(fit)
