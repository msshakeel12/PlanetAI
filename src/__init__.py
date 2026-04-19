"""PlanetAI package root.

This package contains a reproducible machine-learning pipeline for exoplanet
transit classification from Kepler-style light curves. The top-level
subpackages map to pipeline stages:

- ``src.data``: metadata ingestion and supervised dataset construction.
- ``src.features``: preprocessing and feature engineering.
- ``src.models``: classical and deep learning training + evaluation.
- ``src.utils``: shared helpers for IO, paths, configuration, and tracking.
"""
