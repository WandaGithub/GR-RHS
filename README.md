# Group-Regularized Regularized Horseshoe

Code and Stan models for the Group-Regularized Regularized Horseshoe (GR-RHS) paper.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

Install CmdStan and make it available through `CMDSTAN`.

## Experiments

Run the three experiment families from the repository root:

```bash
python -m src.normal_means.run run --config config/normal_means_paper.json

python -m src.correlated_regression.run run \
  --config config/regression_paper.json --target all --jobs 4

python -m src.gse40279.run prepare --config config/gse40279_paper.json
python -m src.gse40279.run fit --config config/gse40279_paper.json
```

Use the corresponding `smoke` configuration for a short test run.

## Data and results

The GSE40279 raw files and derived participant-level arrays are not included.
Download the public source data and prepare the analysis inputs with:

```bash
python scripts/prepare_gse40279.py
```
