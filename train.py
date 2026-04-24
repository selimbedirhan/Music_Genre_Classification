"""
Entry-point script: runs the full training + visualisation pipeline
and prints a rich summary report.
"""
import logging
import sys
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import (AUDIO_FEATURES, FIGURES_DIR, RESULTS_PATH,
                         BEST_MODEL_PATH)
from src.data_pipeline import (load_raw_data, clean_data, build_feature_matrix,
                                ALL_FEATURES)
from src.trainer import train_pipeline
from src.visualiser import (
    plot_genre_distribution, plot_feature_distributions,
    plot_correlation_heatmap, plot_boxplots, plot_confusion_matrix,
    plot_feature_importance, plot_roc_curves, plot_model_comparison
)
from src.explainer import compute_shap_values, plot_shap_summary
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE


def banner(text: str):
    w = 62
    print("\n" + "═" * w)
    print(f"  {text}")
    print("═" * w)


def main(tune: bool = False, n_trials: int = 20):
    banner("🎵  Music Genre Classification  –  Professional Pipeline")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    banner("Step 1 │ Loading & Cleaning Data")
    df_raw   = load_raw_data()
    df_clean = clean_data(df_raw)
    logger.info(f"Genres: {df_clean['playlist_genre'].value_counts().to_dict()}")

    # ── 2. EDA Figures ────────────────────────────────────────────────────────
    banner("Step 2 │ Generating EDA Figures")
    plot_genre_distribution(df_clean)
    plot_feature_distributions(df_clean, AUDIO_FEATURES)
    plot_correlation_heatmap(df_clean, AUDIO_FEATURES)
    plot_boxplots(df_clean, AUDIO_FEATURES)

    # ── 3. Feature engineering ────────────────────────────────────────────────
    banner("Step 3 │ Feature Engineering")
    X, y, scaler, le = build_feature_matrix(df_clean)
    # Note: SMOTE is applied inside train_pipeline (after split) to avoid leakage

    # ── 4. Training ───────────────────────────────────────────────────────────
    banner(f"Step 4 │ Model Training  (tune={'ON' if tune else 'OFF'})")
    model, results, X_test, y_test, best_name = train_pipeline(
        X, y, scaler, le, tune=tune, n_trials=n_trials
    )

    # ── 5. Evaluation figures ─────────────────────────────────────────────────
    banner("Step 5 │ Evaluation Figures")
    class_names = list(le.classes_)
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_feature_importance(model, ALL_FEATURES)
    plot_roc_curves(model, X_test, y_test, class_names)
    plot_model_comparison(results)

    # ── 6. SHAP ───────────────────────────────────────────────────────────────
    banner("Step 6 │ SHAP Explainability")
    try:
        import shap, numpy as np
        shap_sample = X_test[:200]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(shap_sample)
        plot_shap_summary(shap_vals, shap_sample, ALL_FEATURES, class_names)
    except Exception as e:
        logger.warning(f"SHAP step skipped: {e}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    banner("✅  Training Complete  –  Results Summary")
    with open(RESULTS_PATH) as f:
        saved = json.load(f)

    print(f"\n{'Model':<30} {'Accuracy':>9} {'F1-W':>8} {'CV F1':>8}")
    print("─" * 58)
    for name, info in saved.items():
        marker = " ◀ BEST" if name == best_name else ""
        print(f"{name:<30} {info['accuracy']:>9.4f} "
              f"{info['f1_weighted']:>8.4f} {info['cv_mean']:>8.4f}{marker}")

    print(f"\n📁 Figures   → {FIGURES_DIR}")
    print(f"💾 Best model → {BEST_MODEL_PATH}")
    print(f"📊 Results   → {RESULTS_PATH}")


if __name__ == "__main__":
    tune_flag = "--tune" in sys.argv
    main(tune=tune_flag)
