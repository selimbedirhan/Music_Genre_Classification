"""
Entry-point: full training + visualisation + SHAP pipeline.

Usage:
  python train.py           # fast run, no Optuna
  python train.py --tune    # Optuna HPO (recommended for best accuracy)
  python train.py --no-stack  # skip stacking (faster)
"""
import logging
import sys
import json
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import (
    AUDIO_FEATURES, FIGURES_DIR, RESULTS_PATH, BEST_MODEL_PATH
)
from src.data_pipeline import (
    load_raw_data, clean_data, build_feature_matrix, ALL_FEATURES
)
from src.trainer import train_pipeline
from src.visualiser import (
    plot_genre_distribution, plot_feature_distributions,
    plot_correlation_heatmap, plot_boxplots, plot_confusion_matrix,
    plot_feature_importance, plot_roc_curves, plot_model_comparison,
)
from src.explainer import compute_shap_values, plot_shap_summary, plot_shap_beeswarm


def banner(text: str):
    w = 66
    print("\n" + "═" * w)
    print(f"  {text}")
    print("═" * w)


def main(tune: bool = False, n_trials: int = 50, build_stack: bool = True):
    banner("🎵  Music Genre Classification  –  Professional Pipeline v2")

    # ── 1. Data ───────────────────────────────────────────────────────────────
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
    banner("Step 3 │ Feature Engineering  "
           f"({len(ALL_FEATURES)} total features)")
    X, y, scaler, le = build_feature_matrix(df_clean)
    n_classes = len(le.classes_)
    logger.info(f"Classes: {list(le.classes_)}")
    logger.info(f"Feature matrix: {X.shape}")

    # ── 4. Training ───────────────────────────────────────────────────────────
    banner(f"Step 4 │ Model Training  "
           f"(tune={'ON' if tune else 'OFF'}, "
           f"stack={'ON' if build_stack else 'OFF'})")
    model, results, X_test, y_test, best_name = train_pipeline(
        X, y, scaler, le,
        feature_names=ALL_FEATURES,
        tune=tune,
        n_trials=n_trials,
        build_stack=build_stack,
    )

    # ── 5. Evaluation figures ─────────────────────────────────────────────────
    banner("Step 5 │ Evaluation Figures")
    class_names = list(le.classes_)
    y_pred      = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_feature_importance(model, ALL_FEATURES)
    plot_roc_curves(model, X_test, y_test, class_names)
    plot_model_comparison(results)

    # ── 6. SHAP ───────────────────────────────────────────────────────────────
    banner("Step 6 │ SHAP Explainability")
    # For Stacking, use the best single tree-based base model (TreeExplainer is fast)
    shap_model = model
    if hasattr(model, "estimators_"):
        for name_k in ["LightGBM", "XGBoost", "RandomForest", "CatBoost"]:
            if name_k in results and hasattr(results[name_k].get("model"), "feature_importances_"):
                shap_model = results[name_k]["model"]
                logger.info(f"SHAP: using {name_k} (TreeExplainer compatible).")
                break

    try:
        shap_values, X_shap = compute_shap_values(
            shap_model, X_test, feature_names=ALL_FEATURES, max_samples=150
        )
        plot_shap_summary(shap_values, X_shap, ALL_FEATURES, class_names)
        # Generate beeswarm only for class 0 (fast; rest can be done in dashboard)
        plot_shap_beeswarm(shap_values, X_shap, ALL_FEATURES,
                           class_idx=0, class_name=class_names[0])
    except Exception as exc:
        logger.warning(f"SHAP step skipped: {exc}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    banner("✅  Training Complete  –  Results Summary")
    with open(RESULTS_PATH) as f:
        saved = json.load(f)

    print(f"\n{'Model':<30} {'Accuracy':>9} {'F1-W':>8} {'CV F1':>8}")
    print("─" * 60)
    for name, info in saved.items():
        marker = "  ◀ BEST" if name == best_name else ""
        print(
            f"{name:<30} {info['accuracy']:>9.4f} "
            f"{info['f1_weighted']:>8.4f} {info['cv_mean']:>8.4f}{marker}"
        )

    print(f"\n📁 Figures    → {FIGURES_DIR}")
    print(f"💾 Best model → {BEST_MODEL_PATH}")
    print(f"📊 Results    → {RESULTS_PATH}")


if __name__ == "__main__":
    args       = sys.argv[1:]
    tune_flag  = "--tune"    in args
    no_stack   = "--no-stack" in args

    # Parse optional --trials=N
    n_trials = 50
    for arg in args:
        if arg.startswith("--trials="):
            try:
                n_trials = int(arg.split("=")[1])
            except ValueError:
                pass

    main(tune=tune_flag, n_trials=n_trials, build_stack=not no_stack)
