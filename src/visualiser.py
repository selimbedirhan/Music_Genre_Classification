"""
All static visualisations: EDA, confusion matrix, feature importance,
ROC curves.  Figures are saved to reports/figures/.
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
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.config import FIGURES_DIR, GENRE_COLORS

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
DARK_BG   = "#0f0f1a"
CARD_BG   = "#1a1a2e"
ACCENT    = "#00D4FF"
TEXT      = "#e0e0f0"
GRID      = "#2a2a4a"

def _apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    axes = ax_list or fig.get_axes()
    for ax in axes:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)


# ── EDA ───────────────────────────────────────────────────────────────────────
def plot_genre_distribution(df: pd.DataFrame):
    counts = df["playlist_genre"].value_counts()
    colors = [GENRE_COLORS.get(g, ACCENT) for g in counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color=colors,
                  edgecolor="#ffffff22", linewidth=0.5)
    ax.set_title("Genre Distribution", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Genre", fontsize=11)
    ax.set_ylabel("Track Count", fontsize=11)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                f"{val:,}", ha="center", va="bottom", color=TEXT, fontsize=9)
    _apply_dark_style(fig)
    path = os.path.join(FIGURES_DIR, "genre_distribution.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_feature_distributions(df: pd.DataFrame, features: list):
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes_flat = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        for genre, grp in df.groupby("playlist_genre"):
            color = GENRE_COLORS.get(genre, ACCENT)
            grp[feat].plot.kde(ax=ax, label=genre, color=color,
                                linewidth=1.5, alpha=0.85)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=7, framealpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Audio Feature Distributions by Genre",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.01)
    _apply_dark_style(fig)
    path = os.path.join(FIGURES_DIR, "feature_distributions.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_correlation_heatmap(df: pd.DataFrame, features: list):
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, linecolor=GRID,
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14,
                  fontweight="bold", pad=12)
    _apply_dark_style(fig)
    path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_boxplots(df: pd.DataFrame, features: list):
    top_features = ["danceability", "energy", "valence",
                    "tempo", "acousticness", "speechiness"]
    top_features = [f for f in top_features if f in features]
    n = len(top_features)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    palette = {g: GENRE_COLORS.get(g, ACCENT)
               for g in df["playlist_genre"].unique()}

    for i, feat in enumerate(top_features):
        ax = axes_flat[i]
        sns.boxplot(data=df, x="playlist_genre", y=feat,
                    palette=palette, ax=ax, linewidth=0.8,
                    flierprops=dict(marker=".", alpha=0.3,
                                    markersize=2, color=ACCENT))
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Feature Boxplots by Genre",
                 fontsize=14, fontweight="bold", color=TEXT)
    _apply_dark_style(fig)
    path = os.path.join(FIGURES_DIR, "boxplots.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ── Model evaluation ─────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names: list):
    from sklearn.metrics import confusion_matrix as cm
    matrix = cm(y_true, y_pred)
    matrix_norm = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, fmt in zip(
        axes,
        [matrix, matrix_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt,
                    xticklabels=class_names, yticklabels=class_names,
                    cmap="Blues", linewidths=0.4, linecolor=GRID,
                    ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    _apply_dark_style(fig)
    path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature_importances_; skipping plot.")
        return None

    importances = pd.Series(model.feature_importances_,
                             index=feature_names).nlargest(top_n)
    colors = [ACCENT if i < 5 else "#6060aa" for i in range(len(importances))]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(importances.index[::-1], importances.values[::-1],
                   color=colors[::-1], edgecolor="#ffffff22")
    ax.set_title(f"Top {top_n} Feature Importances",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=11)
    _apply_dark_style(fig)
    path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_roc_curves(model, X_test, y_test, class_names: list):
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    y_score = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        color = list(GENRE_COLORS.values())[i % len(GENRE_COLORS)]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cname} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    _apply_dark_style(fig)
    path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_model_comparison(results: dict):
    names = list(results.keys())
    accs  = [results[n]["accuracy"]    for n in names]
    f1s   = [results[n]["f1_weighted"] for n in names]
    cvs   = [results[n]["cv_mean"]     for n in names]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, accs, w, label="Accuracy",    color="#00D4FF", alpha=0.85)
    ax.bar(x,     f1s,  w, label="F1-Weighted", color="#FF6B9D", alpha=0.85)
    ax.bar(x + w, cvs,  w, label="CV F1",       color="#FFD700", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.01)
    ax.set_title("Model Benchmark Comparison",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    _apply_dark_style(fig)
    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path
