# PlanetAI: Exoplanet Transit Classification Starter Repository

PlanetAI is a research-grade, reproducible Python starter project for classifying whether stellar light curves contain evidence of planetary transits.

The repository is designed for iterative research: you can run a deterministic
synthetic workflow immediately, then switch to real MAST-backed Kepler
ingestion when environment and network constraints allow.

The repository includes:
- NASA Exoplanet Archive metadata ingestion (TAP query)
- Supervised dataset construction (`synthetic`, `real`, and `real_stub` modes)
- Reusable preprocessing and feature extraction utilities
- Optional phase folding using KOI period and epoch metadata
- Baseline classifiers (Logistic Regression, Random Forest)
- Deep learning baseline (1D CNN in PyTorch)
- Hyperparameter tuning for baselines and CNN
- Experiment tracking with JSON/JSONL run logs
- Reusable evaluation and plotting utilities
- Independent stage CLIs and one end-to-end orchestrator
- Pytest suite for core preprocessing, feature, and evaluation logic

## Important Limitation

`real` mode uses MAST downloads through Lightkurve and can fail depending on
network access, target availability, and local environment. `real_stub` remains
available for deterministic dry runs when full real-data ingestion is not feasible.

## Requirements

- Python 3.11+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
PlanetAI/
  configs/
    default.yaml
  data/
    raw/
    interim/
    processed/
  notebooks/
  reports/
    figures/
  src/
    data/
      exoplanet_archive.py
      download_metadata.py
      dataset_builder.py
    features/
      preprocessing.py
      feature_engineering.py
    models/
      evaluation.py
      plotting.py
      train_baselines.py
      cnn_model.py
      train_cnn.py
    utils/
      paths.py
      io.py
      config.py
    run_pipeline.py
  tests/
    conftest.py
    test_preprocessing.py
    test_feature_engineering.py
    test_evaluation.py
```

## CLI Usage

Run all commands from project root.

### 1. Download metadata

Purpose:
- Query NASA Exoplanet Archive via TAP and save candidate metadata used for
  label generation and optional phase-fold parameters.

Primary input:
- Archive table/query arguments.

Primary output:
- CSV metadata file in `data/raw/`.

```bash
python -m src.data.download_metadata \
  --table cumulative \
  --output data/raw/kepler_metadata.csv \
  --limit 1000
```

### 2. Build dataset

Purpose:
- Convert metadata into supervised artifacts consumed by model training.

Modes:
- `synthetic`: generates simplified transit-like curves for fast reproducible tests.
- `real`: downloads/stitches real Kepler curves from MAST via Lightkurve.
- `real_stub`: deterministic placeholder arrays when real ingestion is unavailable.

Primary input:
- Metadata CSV from step 1.

Primary outputs:
- Feature table CSV (`data/processed/*.csv`).
- Sequence arrays NPZ (`data/processed/*.npz`).

```bash
python -m src.data.dataset_builder \
  --metadata data/raw/kepler_metadata.csv \
  --mode synthetic \
  --phase-fold \
  --features-output data/processed/features.csv \
  --sequences-output data/processed/sequences.npz
```

To ingest real Kepler light curves from MAST:

```bash
python -m src.data.dataset_builder \
  --mode real \
  --phase-fold \
  --max-real-targets 200
```

To use the deterministic real-ingestion stub:

```bash
python -m src.data.dataset_builder --mode real_stub
```

### 3. Train baseline models

Purpose:
- Train classical ML baselines (Logistic Regression and Random Forest) on
  engineered features.

Primary input:
- Feature table CSV.

Primary outputs:
- Model files (`*.joblib`), metrics JSON, and evaluation plots.

```bash
python -m src.models.train_baselines \
  --features data/processed/features.csv \
  --output-dir data/processed/models \
  --figures-dir reports/figures \
  --tune \
  --experiment-dir reports/experiments
```

### 4. Train CNN model

Purpose:
- Train a 1D CNN on fixed-length sequence arrays.

Primary input:
- Sequence NPZ file (`sequences`, `labels`).

Primary outputs:
- Best checkpoint (`cnn_best.pt`), metrics/history JSON, and plots.

```bash
python -m src.models.train_cnn \
  --sequences data/processed/sequences.npz \
  --output-dir data/processed/models \
  --figures-dir reports/figures \
  --epochs 20 \
  --tune \
  --tune-trials 6 \
  --experiment-dir reports/experiments
```

### 5. Run the full pipeline

```bash
python -m src.run_pipeline --config configs/default.yaml
```

## Config-Driven Pipeline

`configs/default.yaml` controls:
- archive table/query options
- dataset build mode, MAST cache, and sequence shape
- phase folding using KOI period and epoch columns
- baseline split/random seed
- CNN training hyperparameters
- hyperparameter search options
- experiment tracking output
- artifact output locations

Suggested workflow:
1. Start with a small synthetic run to verify environment setup.
2. Enable tuning/tracking once baseline runtime is acceptable.
3. Use `real` mode with `max_real_targets` for bounded-cost MAST validation.

## Artifacts

Expected outputs include:
- `data/raw/kepler_metadata.csv`
- `data/processed/features.csv`
- `data/processed/sequences.npz`
- `data/processed/models/*.joblib`
- `data/processed/models/*_metrics.json`
- `data/processed/models/cnn_best.pt`
- `reports/figures/*.png`
- `reports/experiments/experiments.jsonl`
- `reports/experiments/runs/*.json`

## Testing

```bash
pytest -q
```

The tests cover:
- preprocessing behavior
- feature extraction keys/basic invariants
- evaluation metric output shape and keys

## Notes for Future Extension

- Expand MAST ingestion with quality-bit filtering and quarter-aware stitching policies.
- Add domain-aware phase-folded feature sets (e.g., transit-shape statistics around phase 0).
- Integrate richer experiment tracking backends and automated report generation.

## Scientific Scope Notes

- The current preprocessing and synthetic simulator are intentionally simple and
  designed for pipeline validation, not publication-grade astrophysical modeling.
- `depth_proxy` and polynomial detrending are approximations that should be
  replaced or validated against stronger domain-specific alternatives for final studies.
