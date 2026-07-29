# HDACiAP

Related code for the paper: *HDACiAP: A Curated Database and Analytical Platform for
Histone Deacetylase Inhibitors*.

Web Server

https://hdac.kangsgo.cn

## Data Cleaning Pipeline

Scripts in `data_clean/` process raw molecular data into analysis-ready datasets.
Run them in order:

| Step | Script | Input → Output |
|------|--------|----------------|
| 1 | `step1_split.py` | `molecules.csv` → per-target subdirectories (`HDAC1/`, `SIRT2/`, …) |
| 2 | `step2_classification_dataset.py` | `*_molecules.csv` → `classic.csv` (binary labels: Active=1, Inactive=0) |
| 3 | `step3_regression_dataset.py` | `*_molecules.csv` → `re5.csv` (with $pIC_{50}$) |
| 4 | `step4_classification_scaffold_split.py` | `classic.csv` → `train_classic.csv` / `test_classic.csv` |
| 5 | `step5_regression_scaffold_split.py` | `re5.csv` → `reg_scaffold.csv` |
| 6 | `step6_makere_n_statusre.py` | `re5.csv` → distribution plots (`status_figures/`) |

## Usage

### install by Docker

```bash
cd docker

docker build .
```

### 1. Activate environment
```bash
docker run -it --gpus all --rm -v your_local_dir:/app pyg-image
```

### 2. Place molecules.csv in data_clean/ and run the pipeline
```
cd data_clean
python step1_split.py                         # Split by target
python step2_classification_dataset.py        # Build classification dataset
python step3_regression_dataset.py            # Build regression dataset ($pIC_{50}$)
python step4_classification_scaffold_split.py # Scaffold split for classification
python step5_regression_scaffold_split.py     # Scaffold split for regression
python step6_makere_n_statusre.py            # Generate distribution plots
```
