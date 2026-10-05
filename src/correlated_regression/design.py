from __future__ import annotations

"""Grouped Gaussian data generation for the HOM / HET-tail group-scale study."""
from dataclasses import dataclass as dataclass
import hashlib
import numpy as np

@dataclass(frozen=True)
class Dataset:
    """One simulated split, including truth needed by recovery metrics."""
    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    groups: np.ndarray
    beta: np.ndarray | None = None

def group_ids(n_predictors, n_groups):
    """Map consecutive predictors to equally sized groups."""
    return np.arange(n_predictors) * n_groups // n_predictors

def generate_ar1_predictors(rng, n_observations, predictor_group_ids, rho):
    """Draw grouped predictors from the corresponding AR(1) process."""
    n_groups = predictor_group_ids[-1] + 1
    predictors_per_group = predictor_group_ids.size // n_groups
    X = rng.normal(size=(n_groups, n_observations, predictors_per_group))
    sd = np.sqrt(1 - rho ** 2)
    for j in range(1, predictors_per_group):
        X[:, :, j] = rho * X[:, :, j - 1] + sd * X[:, :, j]
    return X.transpose(1, 0, 2).reshape(n_observations, -1)

@dataclass(frozen=True)
class GroupScaleData:
    dataset: Dataset
    true_group_slab_scales: np.ndarray
    sigma: float

def seed_stream(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))

def seed_for(namespace: str, replication: int, label: str) -> int:
    """Construct the stable fit seed used by the regression experiment."""
    payload = f'{namespace}:{replication}:{label}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')

def generate_group_scale_dgp(*, seed: int, condition: str, signals_per_group: int, n: int, n_test: int, p: int, group_count: int, within_correlation: float, target_r2: float, heterogeneous_tail_scales, homogeneous_tail_scale: float | None) -> GroupScaleData:
    """Generate one matched HOM or HET-tail group-scale replication.

    Each group draws ``signals_per_group`` nonzero coefficients i.i.d. from
    ``N(0, c_g^2)``. HOM shares one slab scale ``c_g`` across groups;
    HET-tail assigns a random permutation of ``heterogeneous_tail_scales``
    per group, calibrated to the same average squared scale as HOM.
    """
    heterogeneous_tail_scales = np.asarray(heterogeneous_tail_scales, float)
    if homogeneous_tail_scale is None:
        homogeneous_tail_scale = float(np.sqrt(np.mean(np.square(heterogeneous_tail_scales))))
    group_size = p // group_count
    groups = group_ids(p, group_count)
    x_rng = seed_stream(seed, 1)
    support_rng = seed_stream(seed, 2)
    coef_rng = seed_stream(seed, 3)
    order_rng = seed_stream(seed, 4)
    noise_rng = seed_stream(seed, 5)
    X = generate_ar1_predictors(x_rng, n, groups, within_correlation)
    X_test = generate_ar1_predictors(x_rng, n_test, groups, within_correlation)
    if condition == 'hom':
        true_group_slab_scales = np.full(group_count, homogeneous_tail_scale, dtype=float)
    else:
        true_group_slab_scales = order_rng.permutation(heterogeneous_tail_scales)
    beta = np.zeros(p, dtype=float)
    for group in range(group_count):
        group_columns = np.flatnonzero(groups == group)
        positions = support_rng.permutation(group_size)[:signals_per_group]
        selected = group_columns[positions]
        beta[selected] = coef_rng.normal(0.0, true_group_slab_scales[group], size=signals_per_group)
    signal = X @ beta
    signal_test = X_test @ beta
    signal_variance = float(np.var(signal))
    sigma = float(np.sqrt(signal_variance * (1.0 - target_r2) / target_r2))
    eps = noise_rng.standard_normal(n)
    eps_test = noise_rng.standard_normal(n_test)
    y = signal + sigma * eps
    y_test = signal_test + sigma * eps_test
    return GroupScaleData(dataset=Dataset(X, y, X_test, y_test, groups, beta), true_group_slab_scales=true_group_slab_scales, sigma=sigma)



def load_config(path):
    import json
    return json.loads(path.read_text(encoding='utf-8'))
