# 🎵 Music Genre Classification — Professional ML Pipeline

## Overview

End-to-end music genre classifier using **32,000+ Spotify tracks**.  
Classifies songs into 6 genres: **EDM · Pop · Rap · R&B · Latin · Rock**.

## Architecture

```
archive/spotify_songs.csv
       │
       ▼
src/data_pipeline.py   ← Load → Clean → Feature Engineering (31 features)
       │
       ▼
src/trainer.py         ← XGBoost · LightGBM · CatBoost · RandomForest
                          + Stacking Ensemble · Optuna HPO
       │
       ▼
models/                ← best_model.joblib · scaler · encoder · SHAP cache
       │
       ▼
dashboard.py           ← Streamlit interactive dashboard
```

## Accuracy-boosting Techniques

| Technique | Description |
|-----------|-------------|
| Rich feature engineering | 18 derived features (cyclical key, log-transforms, interaction products) |
| SMOTE (per-fold) | Class balancing — applied only on train fold to prevent leakage |
| Stacking Ensemble | Top-3 base models → LightGBM meta-learner |
| Optuna HPO | 50-trial TPE search with median pruner |
| CatBoost | Gradient boosting with built-in categorical handling |

## Quick Start

```bash
pip install -r requirements.txt

# Fast run (no Optuna, with stacking)
python train.py

# Full professional run (Optuna + stacking)
python train.py --tune --trials=50

# Launch dashboard
streamlit run dashboard.py
```

## Dashboard Pages

| Page | Content |
|------|---------|
| 🏠 Overview | KPIs, genre distribution, radar chart |
| 📊 EDA | Distributions, correlations, boxplots, violins |
| 🤖 Benchmark | Model comparison, confusion matrix, per-class metrics |
| 🎧 Predict | Real-time slider-based prediction with confidence bars |
| 💡 Explainability | Global SHAP, per-class beeswarm, feature importance |

## Feature Set (31 total)

**Base (13):** danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, duration_ms, key, mode, track_popularity

**Engineered (18):** energy_dance_ratio, acoustic_energy_diff, valence_energy,
valence_dance, energy_loudness, acoustic_instrument, energy_valence_sum,
log_instrumentalness, log_speechiness, log_liveness, duration_min, log_duration_ms,
key_sin, key_cos, popularity_bin, tempo_energy, tempo_dance, speech_instrument_ratio
