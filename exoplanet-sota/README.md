# exoplanet-sota

State-of-the-art oriented exoplanet transit classification pipeline with:
- DR25 metadata loading from NASA Exoplanet Archive
- realistic synthetic generation (GP stellar variability + instrumental noise)
- 2D phase-folded CNN baseline
- tsfresh + LightGBM baseline

## Structure

```
exoplanet-sota/
├── data/
│   ├── kepler_tce_loader.py
│   └── synthetic_realistic.py
├── preprocess/
│   ├── phase_fold.py
│   └── tsfresh_features.py
├── models/
│   ├── cnn_2d_folded.py
│   └── lightgbm_features.py
├── train.py
└── requirements.txt
```

## Install

```bash
cd exoplanet-sota
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python train.py \
  --workdir artifacts \
  --metadata-limit 2000 \
  --max-real-targets 1000 \
  --synthetic-samples 5000 \
  --real-synthetic-ratio 0.3 \
  --cnn-epochs 25
```

If `tsfresh` is unstable on your platform, run with fallback baseline features:

```bash
python train.py --workdir artifacts --disable-tsfresh
```

If sklearn Gaussian-process noise generation is unstable, use:

```bash
python train.py --workdir artifacts --disable-tsfresh --disable-gp-noise
```

If LightGBM or PyTorch also crash on your platform, use full fallback mode:

```bash
python train.py --workdir artifacts --disable-tsfresh --disable-gp-noise --disable-lightgbm --disable-cnn
```

Or use the one-flag stable smoke preset:

```bash
python train.py --workdir artifacts_smoke --smoke-stable
```

## Outputs

Metrics are written to:
- `artifacts/metrics/metrics_summary.json`
- `artifacts/metrics/lightgbm_metrics.json`
- `artifacts/metrics/cnn_metrics.json`

## Notes

- This implementation enables literature-aligned components but **does not guarantee** target metrics on every run.
- Real performance depends strongly on ingestion scale, class balance, label policy, and Kepler target availability.
