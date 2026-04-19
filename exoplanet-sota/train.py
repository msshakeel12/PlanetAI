"""End-to-end training pipeline for SOTA-style exoplanet transit classification.

Pipeline stages:
1. Download DR25 metadata and map labels.
2. Load real Kepler light curves from MAST (optional bounded subset).
3. Generate realistic synthetic curves with GP stellar noise.
4. Build 2D folded inputs and tsfresh features.
5. Train LightGBM and 2D folded CNN.
6. Save metrics in table-ready JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data.kepler_tce_loader import (
    download_dr25_metadata,
    load_kepler_lightcurves_via_lightkurve,
    make_binary_labels,
)
from data.synthetic_realistic import SyntheticConfig, generate_realistic_synthetic_curve
from models.cnn_2d_folded import CNNTrainConfig, train_folded_cnn
from models.lightgbm_features import train_lightgbm_features
from preprocess.phase_fold import phase_fold_2d
from preprocess.tsfresh_features import extract_tsfresh_features


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float | list[list[int]]]:
    """Compute binary classification metrics with optional probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    roc_auc = float("nan")
    if y_prob is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, np.asarray(y_prob, dtype=float)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _to_arrays(series: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    times: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    labels: list[int] = []
    periods: list[float] = []
    epochs: list[float] = []

    for row in series:
        times.append(np.asarray(row["time"], dtype=float))
        fluxes.append(np.asarray(row["flux"], dtype=float))
        labels.append(int(row["label"]))
        periods.append(float(row.get("period", 0.0)))
        epochs.append(float(row.get("epoch", 0.0)))

    return times, fluxes, np.asarray(labels, dtype=int), np.asarray(periods, dtype=float), np.asarray(epochs, dtype=float)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fallback_stat_features(times: list[np.ndarray], fluxes: list[np.ndarray]) -> np.ndarray:
    """Build compact fallback features if tsfresh extraction is unavailable.

    Args:
        times: List of time arrays.
        fluxes: List of flux arrays.

    Returns:
        Feature matrix with one row per light curve.
    """
    rows: list[list[float]] = []
    for t, f in zip(times, fluxes):
        f = np.asarray(f, dtype=float)
        t = np.asarray(t, dtype=float)
        dt = np.diff(t)
        cadence = float(np.median(dt)) if dt.size > 0 else 0.0
        rows.append(
            [
                float(np.mean(f)),
                float(np.std(f)),
                float(np.min(f)),
                float(np.max(f)),
                float(np.median(f)),
                float(np.quantile(f, 0.1)),
                float(np.quantile(f, 0.9)),
                float(np.max(f) - np.min(f)),
                float(np.median(f) - np.min(f)),
                cadence,
            ]
        )
    return np.asarray(rows, dtype=float)


def _mix_real_synth(
    real_rows: list[dict],
    synth_rows: list[dict],
    synth_ratio: float,
    seed: int,
) -> list[dict]:
    if not real_rows:
        return synth_rows

    rng = np.random.default_rng(seed)
    target_total = int(round(len(real_rows) / max(1e-6, (1.0 - synth_ratio))))
    synth_needed = max(0, target_total - len(real_rows))

    if synth_needed <= len(synth_rows):
        idx = rng.choice(len(synth_rows), size=synth_needed, replace=False)
        selected = [synth_rows[int(i)] for i in idx]
    else:
        idx = rng.choice(len(synth_rows), size=synth_needed, replace=True)
        selected = [synth_rows[int(i)] for i in idx]

    mixed = list(real_rows) + selected
    rng.shuffle(mixed)
    return mixed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("exoplanet-sota/artifacts"))
    parser.add_argument("--metadata-limit", type=int, default=2000)
    parser.add_argument("--max-real-targets", type=int, default=1000)
    parser.add_argument("--synthetic-samples", type=int, default=5000)
    parser.add_argument("--real-synthetic-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnn-epochs", type=int, default=25)
    parser.add_argument("--cnn-batch-size", type=int, default=64)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--disable-tsfresh", action="store_true")
    parser.add_argument("--disable-gp-noise", action="store_true")
    parser.add_argument("--disable-lightgbm", action="store_true")
    parser.add_argument("--disable-cnn", action="store_true")
    parser.add_argument("--smoke-stable", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.smoke_stable:
        # One-flag reproducible smoke profile for unstable native stacks.
        args.metadata_limit = 40
        args.max_real_targets = 0
        args.synthetic_samples = 80
        args.real_synthetic_ratio = 0.5
        args.cnn_epochs = 2
        args.cnn_batch_size = 16
        args.disable_tsfresh = True
        args.disable_gp_noise = True
        args.disable_lightgbm = True
        args.disable_cnn = True

    rng = np.random.default_rng(args.seed)

    workdir = args.workdir
    data_dir = workdir / "data"
    metrics_dir = workdir / "metrics"
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Metadata download and labeling.
    metadata_path = data_dir / "dr25_metadata.csv"
    meta = download_dr25_metadata(metadata_path, limit=args.metadata_limit)
    meta = make_binary_labels(meta)

    # Stage 2: Real Kepler light-curve ingestion.
    real_series_obj = load_kepler_lightcurves_via_lightkurve(
        metadata=meta,
        cache_dir=data_dir / "mast_cache",
        max_targets=args.max_real_targets,
    )
    real_rows = [
        {
            "time": s.time,
            "flux": s.flux,
            "label": s.label,
            "period": s.period,
            "epoch": s.epoch,
            "source": "real",
        }
        for s in real_series_obj
    ]

    # Stage 3: Realistic synthetic generation with GP stellar variability.
    synth_cfg = SyntheticConfig(use_gp_noise=not args.disable_gp_noise)
    synth_rows: list[dict] = []
    for _ in range(args.synthetic_samples):
        y = int(rng.uniform() < 0.5)
        t, f, p, e = generate_realistic_synthetic_curve(label=y, cfg=synth_cfg, rng=rng)
        synth_rows.append({"time": t.tolist(), "flux": f.tolist(), "label": y, "period": p, "epoch": e, "source": "synthetic"})

    # Stage 4: Controlled real/synthetic mixture.
    all_rows = _mix_real_synth(real_rows, synth_rows, synth_ratio=args.real_synthetic_ratio, seed=args.seed)
    if len(all_rows) < 50:
        # Fallback if real ingestion is sparse/unavailable.
        all_rows = synth_rows

    times, fluxes, labels, periods, _epochs = _to_arrays(all_rows)

    # Stage 5: 2D folded image construction.
    folded = np.asarray([phase_fold_2d(t, f, period=p, n_folds=10, n_bins=1000) for t, f, p in zip(times, fluxes, periods)])

    # Stage 6: Stratified 80/10/10 split.
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=args.seed, stratify=labels[train_idx])

    x_train_img, y_train = folded[train_idx], labels[train_idx]
    x_val_img, y_val = folded[val_idx], labels[val_idx]
    x_test_img, y_test = folded[test_idx], labels[test_idx]

    # Stage 7: tsfresh + LightGBM baseline.
    train_times = [times[i] for i in train_idx]
    train_fluxes = [fluxes[i] for i in train_idx]
    test_times = [times[i] for i in test_idx]
    test_fluxes = [fluxes[i] for i in test_idx]

    try:
        if args.disable_tsfresh:
            raise RuntimeError("tsfresh disabled by CLI flag")

        x_train_feat_df = extract_tsfresh_features(train_times, train_fluxes)
        x_test_feat_df = extract_tsfresh_features(test_times, test_fluxes)
        x_train_feat = x_train_feat_df.sort_index().to_numpy()
        x_test_feat = x_test_feat_df.sort_index().to_numpy()
        feature_source = "tsfresh"
    except Exception as exc:
        # Fallback preserves end-to-end execution if tsfresh/numba has runtime issues.
        print(f"WARNING: tsfresh path unavailable ({exc}); using fallback statistical features.")
        x_train_feat = _fallback_stat_features(train_times, train_fluxes)
        x_test_feat = _fallback_stat_features(test_times, test_fluxes)
        feature_source = "fallback_stats"

    if args.disable_lightgbm:
        rf = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)
        rf.fit(x_train_feat, y_train)
        y_prob_lgb = rf.predict_proba(x_test_feat)[:, 1] if len(np.unique(y_train)) == 2 else None
        y_pred_lgb = (y_prob_lgb >= 0.5).astype(int) if y_prob_lgb is not None else rf.predict(x_test_feat)
        lgb_metrics = _compute_metrics(y_test, y_pred_lgb, y_prob_lgb)
        lgb_model_name = "random_forest_fallback"
    else:
        _, lgb_metrics = train_lightgbm_features(x_train_feat, y_train, x_test_feat, y_test, random_state=args.seed)
        lgb_model_name = "lightgbm"

    # Stage 8: 2D folded CNN.
    if args.disable_cnn:
        # Use compact summary features from folded images as a stable fallback classifier.
        tr_flat = x_train_img.reshape(x_train_img.shape[0], -1)
        te_flat = x_test_img.reshape(x_test_img.shape[0], -1)
        x_train_img_feat = np.column_stack(
            [
                np.mean(tr_flat, axis=1),
                np.std(tr_flat, axis=1),
                np.min(tr_flat, axis=1),
                np.max(tr_flat, axis=1),
            ]
        )
        x_test_img_feat = np.column_stack(
            [
                np.mean(te_flat, axis=1),
                np.std(te_flat, axis=1),
                np.min(te_flat, axis=1),
                np.max(te_flat, axis=1),
            ]
        )
        rf_img = RandomForestClassifier(n_estimators=200, random_state=args.seed + 1, n_jobs=-1)
        rf_img.fit(x_train_img_feat, y_train)
        y_prob_cnn = rf_img.predict_proba(x_test_img_feat)[:, 1] if len(np.unique(y_train)) == 2 else None
        y_pred_cnn = (y_prob_cnn >= 0.5).astype(int) if y_prob_cnn is not None else rf_img.predict(x_test_img_feat)
        cnn_metrics = _compute_metrics(y_test, y_pred_cnn, y_prob_cnn)
        cnn_model_name = "folded_image_random_forest_fallback"
    else:
        cnn_cfg = CNNTrainConfig(epochs=args.cnn_epochs, batch_size=args.cnn_batch_size, lr=args.cnn_lr, seed=args.seed)
        _, cnn_metrics = train_folded_cnn(x_train_img, y_train, x_val_img, y_val, x_test_img, y_test, cfg=cnn_cfg)
        cnn_model_name = "cnn_2d_folded"

    summary = {
        "dataset": {
            "n_total": int(len(labels)),
            "n_real": int(len(real_rows)),
            "n_synthetic_pool": int(len(synth_rows)),
            "mix_ratio_synthetic_target": float(args.real_synthetic_ratio),
            "split": "80/10/10 stratified",
            "feature_source": feature_source,
            "lgb_model": lgb_model_name,
            "cnn_model": cnn_model_name,
        },
        "lightgbm_tsfresh": lgb_metrics,
        "cnn_2d_folded": cnn_metrics,
    }

    _save_json(metrics_dir / "metrics_summary.json", summary)
    _save_json(metrics_dir / "lightgbm_metrics.json", lgb_metrics)
    _save_json(metrics_dir / "cnn_metrics.json", cnn_metrics)

    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
