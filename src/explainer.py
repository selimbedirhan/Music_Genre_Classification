"""
SHAP-based model explainability module.
Generates summary plots, beeswarm charts, and returns shap values for
individual predictions.
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
import shap

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def compute_shap_values(model, X_background, X_explain=None,
                        feature_names=None, max_background: int = 200):
    """
    Compute SHAP values using TreeExplainer (for tree-based models)
    or KernelExplainer as fallback.
    Returns (explainer, shap_values).
    """
    if X_explain is None:
        X_explain = X_background

    # Sample background for speed
    bg_idx = np.random.choice(len(X_background),
                               min(max_background, len(X_background)),
                               replace=False)
    X_bg = X_background[bg_idx]

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain)
        logger.info("SHAP TreeExplainer used.")
    except Exception:
        logger.warning("TreeExplainer failed; falling back to KernelExplainer.")
        explainer = shap.KernelExplainer(model.predict_proba, X_bg)
        sv = explainer.shap_values(X_explain)

    return explainer, sv


def plot_shap_summary(shap_values, X, feature_names: list,
                      class_names: list):
    """Global feature importance via SHAP (multi-class mean |SHAP|)."""
    # Average absolute SHAP across classes if list
    if isinstance(shap_values, list):
        abs_mean = np.mean(
            [np.abs(sv) for sv in shap_values], axis=0
        )
    else:
        abs_mean = np.abs(shap_values)

    importance = pd.Series(abs_mean.mean(axis=0),
                            index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(importance)))
    ax.barh(importance.index, importance.values, color=colors)
    ax.set_title("Global SHAP Feature Importance",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#e0e0f0")
    ax.xaxis.label.set_color("#e0e0f0")
    ax.yaxis.label.set_color("#e0e0f0")
    ax.title.set_color("#e0e0f0")

    path = os.path.join(FIGURES_DIR, "shap_summary.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def get_instance_shap(shap_values, index: int, feature_names: list,
                      class_names: list, class_idx: int = 0):
    """
    Return a DataFrame of SHAP values for a single prediction instance.
    """
    if isinstance(shap_values, list):
        sv = shap_values[class_idx][index]
    else:
        sv = shap_values[index]

    return pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "abs_shap": np.abs(sv),
    }).sort_values("abs_shap", ascending=False)
