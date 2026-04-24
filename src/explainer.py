"""
SHAP-based model explainability module.
Generates global importance plots, beeswarm charts, and per-instance waterfalls.
Handles both TreeExplainer and KernelExplainer (fallback).
"""
import os
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

from src.config import FIGURES_DIR, SHAP_VALUES_PATH

logger = logging.getLogger(__name__)

# ── Style constants ───────────────────────────────────────────────────────────
DARK_BG = "#0f0f1a"
CARD_BG = "#1a1a2e"
TEXT    = "#e0e0f0"


def _dark_fig(fig):
    fig.patch.set_facecolor(DARK_BG)
    for ax in fig.get_axes():
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            item.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")


# ── Core SHAP computation ─────────────────────────────────────────────────────
def compute_shap_values(model, X_explain, feature_names: list,
                        max_samples: int = 300):
    """
    Compute SHAP values and cache them to disk.
    Returns (shap_values, explainer_expected_value).
    """
    X_exp = X_explain[:max_samples]
    logger.info(f"Computing SHAP values on {len(X_exp)} samples …")

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_exp)
        logger.info("SHAP TreeExplainer succeeded.")
    except Exception as exc:
        logger.warning(f"TreeExplainer failed ({exc}); using KernelExplainer.")
        bg = shap.sample(X_exp, min(100, len(X_exp)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        sv = explainer.shap_values(X_exp)

    # Cache for dashboard reuse
    joblib.dump({"shap_values": sv, "X": X_exp, "feature_names": feature_names},
                SHAP_VALUES_PATH)
    logger.info(f"SHAP values cached → {SHAP_VALUES_PATH}")
    return sv, X_exp


# ── Global importance ─────────────────────────────────────────────────────────
def plot_shap_summary(shap_values, X, feature_names: list,
                      class_names: list) -> str:
    """Bar chart of mean |SHAP| averaged across all classes."""
    if isinstance(shap_values, list):
        # TreeExplainer: list of (n_samples, n_features) per class
        abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        abs_mean = np.abs(shap_values)

    # abs_mean shape: (n_samples, n_features) or (n_samples, n_features, n_classes)
    if abs_mean.ndim == 3:
        # KernelExplainer multi-class: average over samples and classes
        feat_importance = abs_mean.mean(axis=0).mean(axis=1)  # → (n_features,)
    else:
        feat_importance = abs_mean.mean(axis=0)               # → (n_features,)

    importance = (
        pd.Series(feat_importance, index=feature_names)
        .sort_values(ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.plasma(np.linspace(0.2, 0.95, len(importance)))
    ax.barh(importance.index, importance.values, color=colors, edgecolor="none")
    ax.set_title("Global SHAP Feature Importance (mean |SHAP|)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    _dark_fig(fig)

    path = os.path.join(FIGURES_DIR, "shap_summary.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_shap_beeswarm(shap_values, X, feature_names: list,
                       class_idx: int = 0, class_name: str = "") -> str | None:
    """SHAP beeswarm for one class. Only works when shap_values is a list (multiclass)."""
    if not isinstance(shap_values, list):
        logger.info("Beeswarm skipped: SHAP values are not per-class list.")
        return None

    sv = shap_values[class_idx]
    expl = shap.Explanation(
        values=sv,
        base_values=np.zeros(len(sv)),
        data=X,
        feature_names=feature_names,
    )
    plt.style.use("dark_background")
    shap.plots.beeswarm(expl, max_display=18, show=False)
    title = f"SHAP Beeswarm – {class_name or f'class {class_idx}'}"
    plt.title(title, color=TEXT, fontsize=12, fontweight="bold")
    fig = plt.gcf()
    fig.patch.set_facecolor(DARK_BG)

    path = os.path.join(FIGURES_DIR, f"shap_beeswarm_class{class_idx}.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved: {path}")
    return path


# ── Per-instance SHAP DataFrame ───────────────────────────────────────────────
def get_instance_shap(shap_values, index: int, feature_names: list,
                      class_names: list, class_idx: int = 0) -> pd.DataFrame:
    """Return a tidy DataFrame of SHAP contributions for one instance."""
    sv = shap_values[class_idx][index] if isinstance(shap_values, list) \
        else shap_values[index]
    return pd.DataFrame({
        "feature":   feature_names,
        "shap_value": sv,
        "abs_shap":  np.abs(sv),
    }).sort_values("abs_shap", ascending=False)
