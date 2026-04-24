# 🎵 Music Genre Classification — Professional ML Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/XGBoost-Latest-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LightGBM-Latest-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit"/>
</p>

> End-to-end production-grade music genre classification system trained on **32,000+ Spotify tracks** with multi-model benchmarking, Optuna hyperparameter tuning, SHAP explainability, and an interactive Streamlit dashboard.

---

## 🚀 Features

| Category | Details |
|---|---|
| **Dataset** | 32,833 Spotify tracks · 6 genres · 13 audio features |
| **Models** | Random Forest · XGBoost · LightGBM · Logistic Regression |
| **Optimisation** | Optuna TPE · 30 trials per model |
| **Balancing** | SMOTE oversampling |
| **Explainability** | SHAP TreeExplainer · Global + instance-level |
| **Dashboard** | Streamlit · Plotly · Real-time prediction |
| **EDA** | Distributions · Correlations · Radar charts · Boxplots |

---

## 📁 Project Structure

```
Music_Genre_Classification/
├── src/
│   ├── config.py          # Paths, features, constants
│   ├── data_pipeline.py   # Load → clean → engineer → SMOTE
│   ├── trainer.py         # Multi-model benchmark + Optuna tuning
│   ├── visualiser.py      # All static figures (EDA, metrics)
│   └── explainer.py       # SHAP explainability
├── archive/
│   └── spotify_songs.csv  # Raw dataset
├── models/                # Saved model artifacts (auto-generated)
├── reports/figures/       # All generated plots (auto-generated)
├── train.py               # 🏋️ Full training pipeline entry-point
├── dashboard.py           # 🎛 Streamlit dashboard
└── requirements.txt
```

---

## ⚙️ Quick Start

```bash
# 1 – Install dependencies
pip install -r requirements.txt

# 2 – Train (fast, no tuning)
python train.py

# 3 – Train with Optuna hyperparameter tuning
python train.py --tune

# 4 – Launch dashboard
streamlit run dashboard.py
```

---

## 📊 Results (baseline)

| Model | Accuracy | F1-Weighted | CV F1 |
|---|---|---|---|
| LightGBM | ~0.87 | ~0.87 | ~0.86 |
| XGBoost | ~0.86 | ~0.86 | ~0.85 |
| Random Forest | ~0.85 | ~0.85 | ~0.84 |
| Logistic Regression | ~0.62 | ~0.62 | ~0.61 |

> Exact numbers depend on SMOTE randomness; run `train.py` to get your results.

---

## 🧠 Genres Classified

| Genre | Tracks | Emoji |
|---|---|---|
| EDM | 6,043 | ⚡ |
| Rap | 5,746 | 🎤 |
| Pop | 5,507 | 🌟 |
| R&B | 5,431 | 🎵 |
| Latin | 5,155 | 💃 |
| Rock | 4,951 | 🎸 |

---

## 👨‍💻 Author

**Selim Bedirhan Öztürk**  
Computer Engineering · Ankara University  
📬 selimbedirhan42@gmail.com
