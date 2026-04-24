"""
Multi-model benchmark training with cross-validation and hyperparameter
optimisation via Optuna.  Saves best model + artefacts to disk.
"""
import json
import logging
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (
    BEST_MODEL_PATH, SCALER_PATH, ENCODER_PATH, RESULTS_PATH,
    RANDOM_STATE, TEST_SIZE
)

logger = logging.getLogger(__name__)


# ── Candidate models (fast baseline) ─────────────────────────────────────────
def _get_candidates(n_classes: int):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=25,
            min_samples_split=4, n_jobs=-1,
            random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=RANDOM_STATE,
            num_class=n_classes, objective="multi:softprob"),
        "LightGBM": LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.08,
            subsample=0.8, n_jobs=-1,
            random_state=RANDOM_STATE, verbose=-1),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, n_jobs=-1,
            random_state=RANDOM_STATE),
    }


def benchmark_models(X_train, X_test, y_train, y_test, n_classes: int,
                     cv: int = 5):
    """Train all candidates, collect metrics, return results dict."""
    candidates = _get_candidates(n_classes)
    results = {}
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                                  random_state=RANDOM_STATE)

    for name, clf in candidates.items():
        logger.info(f"  ▶ Training {name} …")
        t0 = time.time()

        cv_scores = cross_val_score(clf, X_train, y_train,
                                    cv=cv_splitter, scoring="f1_weighted",
                                    n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        elapsed = time.time() - t0

        results[name] = {
            "model":        clf,
            "accuracy":     float(accuracy_score(y_test, y_pred)),
            "f1_weighted":  float(f1_score(y_test, y_pred, average="weighted")),
            "cv_mean":      float(cv_scores.mean()),
            "cv_std":       float(cv_scores.std()),
            "elapsed_s":    round(elapsed, 2),
            "report":       classification_report(y_test, y_pred,
                                                   output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        logger.info(
            f"    {name}: acc={results[name]['accuracy']:.4f}  "
            f"f1={results[name]['f1_weighted']:.4f}  "
            f"cv={results[name]['cv_mean']:.4f}±{results[name]['cv_std']:.4f}  "
            f"({elapsed:.1f}s)"
        )

    return results


def tune_best_model(X_train, y_train, best_name: str,
                    n_classes: int, n_trials: int = 30):
    """Run Optuna hyper-parameter search on the winning model."""
    logger.info(f"Tuning {best_name} with {n_trials} Optuna trials …")

    def objective(trial):
        if best_name == "LightGBM":
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 200, 600),
                "num_leaves":    trial.suggest_int("num_leaves", 31, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            }
            clf = LGBMClassifier(**params, n_jobs=-1,
                                  random_state=RANDOM_STATE, verbose=-1)
        elif best_name == "XGBoost":
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 200, 600),
                "max_depth":       trial.suggest_int("max_depth", 4, 12),
                "learning_rate":   trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
            clf = XGBClassifier(**params, use_label_encoder=False,
                                 eval_metric="mlogloss", n_jobs=-1,
                                 random_state=RANDOM_STATE,
                                 num_class=n_classes,
                                 objective="multi:softprob")
        else:  # RandomForest fallback
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                "max_depth":    trial.suggest_int("max_depth", 15, 35),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            clf = RandomForestClassifier(**params, n_jobs=-1,
                                          random_state=RANDOM_STATE)

        cv = StratifiedKFold(n_splits=3, shuffle=True,
                              random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X_train, y_train,
                                  cv=cv, scoring="f1_weighted", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(
                                     seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best trial: {study.best_params}  →  f1={study.best_value:.4f}")
    return study.best_params, study.best_value


def save_artifacts(model, scaler, label_encoder, results: dict):
    """Persist model, scaler, encoder, and benchmark metrics."""
    joblib.dump(model,         BEST_MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    serialisable = {
        name: {k: v for k, v in info.items() if k != "model"}
        for name, info in results.items()
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)

    logger.info(f"Artifacts saved → {BEST_MODEL_PATH}")


def train_pipeline(X, y, scaler, le, tune: bool = True, n_trials: int = 30):
    """
    Full training pipeline:
      split → benchmark → (optional) tune → save → return best model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )

    # Apply SMOTE only on training data to prevent leakage
    from src.data_pipeline import apply_smote
    X_train, y_train = apply_smote(X_train, y_train)

    n_classes = len(np.unique(y))

    logger.info("── Baseline benchmark ───────────────────────────────────")
    results = benchmark_models(X_train, X_test, y_train, y_test, n_classes)

    # Pick winner by weighted-F1
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    logger.info(f"\n🏆 Best baseline model: {best_name}")

    if tune:
        best_params, _ = tune_best_model(X_train, y_train,
                                          best_name, n_classes, n_trials)
        # Retrain with best params on FULL train set
        candidates = _get_candidates(n_classes)
        tuned_clf = candidates[best_name].__class__(
            **best_params,
            n_jobs=-1 if best_name != "LogisticRegression" else None,
            random_state=RANDOM_STATE,
            **({"verbose": -1} if best_name == "LightGBM" else {}),
            **({"use_label_encoder": False, "eval_metric": "mlogloss",
                "num_class": n_classes, "objective": "multi:softprob"}
               if best_name == "XGBoost" else {}),
        )
        tuned_clf.fit(X_train, y_train)
        y_pred_tuned = tuned_clf.predict(X_test)
        results[f"{best_name}_Tuned"] = {
            "model":        tuned_clf,
            "accuracy":     float(accuracy_score(y_test, y_pred_tuned)),
            "f1_weighted":  float(f1_score(y_test, y_pred_tuned,
                                           average="weighted")),
            "cv_mean":      results[best_name]["cv_mean"],
            "cv_std":       results[best_name]["cv_std"],
            "elapsed_s":    0,
            "report":       classification_report(y_test, y_pred_tuned,
                                                   output_dict=True),
            "confusion_matrix": confusion_matrix(
                y_test, y_pred_tuned).tolist(),
        }
        best_name = f"{best_name}_Tuned"

    final_model = results[best_name]["model"]
    save_artifacts(final_model, scaler, le, results)
    return final_model, results, X_test, y_test, best_name
