# Machine Learning-Based Disease Classification Using RNA-Seq Transcriptomic Data

This repository provides a Python-based workflow for building disease classifiers from RNA-Seq transcriptomic data. It includes utilities for loading and preprocessing expression matrices, aligning sample metadata, training machine learning models, evaluating performance, and generating visualizations.

## Project structure

- `src/` - Python package with data loading, preprocessing, and modeling utilities.
- `scripts/` - Command-line entrypoints and analysis notebooks for training, evaluation, and plotting.
- `data/` - Local dataset folders for raw inputs and processed outputs.
- `models/` - Serialized model artifacts for trained classifiers.
- `notebooks/` - Exploratory analysis and model development notebooks.

## Recommended dataset layout

- `data/raw/` - Raw input files such as expression matrices, sample metadata, or local GTF files.
- `data/processed/` - Processed matrices, label files, and generated outputs ready for modeling.

Example files:
- `data/raw/expression.csv` - rows are samples, columns are genes, values are raw counts or normalized expression.
- `data/raw/metadata.csv` - sample metadata with a `sample_id` column and a `label` column.

For the GSE52194 breast cancer dataset, the `scripts/01_download_gse_series_matrix.py` script parses local GTF files and writes:

- `data/processed/GSE52194_expression_matrix.csv`
- `data/processed/GSE52194_labels.csv`

## Setup

Recommended Python version: `3.10` or newer.

Using `pip`:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Using `conda`:

```bash
conda env create -f environment.yml
conda activate disease-classification-rnaseq
```

Optionally install the package in editable mode:

```bash
pip install -e .
```

## Usage

Train a model from an expression matrix and metadata file:

```bash
python scripts/02_run_training.py \
  --expression data/raw/expression.csv \
  --metadata data/raw/metadata.csv \
  --label-column label \
  --output models/disease_classifier.joblib
```

If using the GSE52194 local GTF workflow, run:

```bash
python scripts/01_download_gse_series_matrix.py
```

Then evaluate with the LOOCV script:

```bash
python scripts/scripts03_loocv_eval.py
```

Generate performance and biomarker plots:

```bash
python scripts/scripts04_generate_plots.py
```

## Extending the pipeline

- Add or replace RNA-Seq input data in `data/raw/`.
- Adapt preprocessing strategies in `src/data.py`.
- Experiment with classifiers in `src/model.py`.
- Add notebooks in `notebooks/` for visualization and analysis.
