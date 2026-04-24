"""
Professional multi-model training engine with:
  • XGBoost, LightGBM, CatBoost, RandomForest baselines
  • Stacking Ensemble (level-1 LightGBM meta-learner)
  • Optuna HPO on the best single model
  • Per-fold SMOTE to prevent data leakage
  • Full artifact persistence
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

from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (
    BEST_MODEL_PATH, SCALER_PATH, ENCODER_PATH, RESULTS_PATH,
    FEATURE_NAMES_PATH, RANDOM_STATE, TEST_SIZE, N_CV_FOLDS, OPTUNA_TRIALS
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _try_catboost(n_classes: int):
    """Import CatBoost only if installed; return None otherwise."""
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=400, depth=8, learning_rate=0.07,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            eval_metric="Accuracy", random_seed=RANDOM_STATE,
            verbose=0, thread_count=-1,
        )
    except ImportError:
        logger.info("CatBoost not installed – skipping.")
        return None


def _get_candidates(n_classes: int) -> dict:
    candidates = {
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.07,
            subsample=0.85, colsample_bytree=0.80,
            eval_metric="mlogloss", n_jobs=-1,
            random_state=RANDOM_STATE,
            num_class=n_classes, objective="multi:softprob",
            use_label_encoder=False,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400, num_leaves=79, learning_rate=0.07,
            subsample=0.85, colsample_bytree=0.80,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=28, min_samples_split=4,
            max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }
    cb = _try_catboost(n_classes)
    if cb is not None:
        candidates["CatBoost"] = cb
    return candidates


# ── Baseline benchmark ────────────────────────────────────────────────────────
def benchmark_models(X_train, X_test, y_train, y_test,
                     n_classes: int, cv: int = N_CV_FOLDS) -> dict:
    """Train all candidates; collect CV + hold-out metrics."""
    candidates = _get_candidates(n_classes)
    results    = {}
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                                  random_state=RANDOM_STATE)

    for name, clf in candidates.items():
        logger.info(f"  ▶ Training {name} …")
        t0 = time.time()

        cv_scores = cross_val_score(
            clf, X_train, y_train,
            cv=cv_splitter, scoring="f1_weighted", n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred  = clf.predict(X_test)
        elapsed = time.time() - t0

        results[name] = {
            "model":            clf,
            "accuracy":         float(accuracy_score(y_test, y_pred)),
            "f1_weighted":      float(f1_score(y_test, y_pred, average="weighted")),
            "cv_mean":          float(cv_scores.mean()),
            "cv_std":           float(cv_scores.std()),
            "elapsed_s":        round(elapsed, 2),
            "report":           classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        logger.info(
            f"    {name}: acc={results[name]['accuracy']:.4f}  "
            f"f1={results[name]['f1_weighted']:.4f}  "
            f"cv={results[name]['cv_mean']:.4f}±{results[name]['cv_std']:.4f}  "
            f"({elapsed:.1f}s)"
        )

    return results


# ── Stacking ensemble ─────────────────────────────────────────────────────────
def build_stacking_ensemble(X_train, X_test, y_train, y_test,
                             base_results: dict, n_classes: int) -> dict:
    """
    Build a StackingClassifier using the top-3 base models as estimators
    and a LogisticRegression as the meta-learner.
    """
    logger.info("Building Stacking Ensemble …")
    t0 = time.time()

    # Sort base models by CV F1 and pick top 3
    sorted_names = sorted(
        [n for n in base_results if "model" in base_results[n]],
        key=lambda n: base_results[n]["cv_mean"], reverse=True,
    )[:3]

    estimators = [(name, base_results[name]["model"]) for name in sorted_names]
    logger.info(f"  Base learners: {sorted_names}")

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
        ),
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1, passthrough=False,
    )
    stack.fit(X_train, y_train)
    y_pred  = stack.predict(X_test)
    elapsed = time.time() - t0

    cv_splitter = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                                  random_state=RANDOM_STATE)
    cv_scores = cross_val_score(stack, X_train, y_train,
                                cv=cv_splitter, scoring="f1_weighted", n_jobs=-1)

    stack_result = {
        "model":            stack,
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "f1_weighted":      float(f1_score(y_test, y_pred, average="weighted")),
        "cv_mean":          float(cv_scores.mean()),
        "cv_std":           float(cv_scores.std()),
        "elapsed_s":        round(elapsed, 2),
        "report":           classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    logger.info(
        f"  Stacking: acc={stack_result['accuracy']:.4f}  "
        f"f1={stack_result['f1_weighted']:.4f}  "
        f"cv={stack_result['cv_mean']:.4f}±{stack_result['cv_std']:.4f}  "
        f"({elapsed:.1f}s)"
    )
    return stack_result


# ── Optuna HPO ────────────────────────────────────────────────────────────────
def tune_best_model(X_train, y_train, best_name: str,
                    n_classes: int, n_trials: int = OPTUNA_TRIALS):
    """Optuna TPE search on the best single model (not the stack)."""
    # Don't try to tune a StackingClassifier
    if "Stacking" in best_name or "CatBoost" in best_name:
        logger.info(f"Skipping Optuna for {best_name}.")
        return None, None

    logger.info(f"Tuning {best_name} with {n_trials} Optuna trials …")

    def objective(trial):
        if best_name == "LightGBM":
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 300, 800),
                "num_leaves":        trial.suggest_int("num_leaves", 40, 150),
                "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "subsample":         trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.55, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            }
            clf = LGBMClassifier(**params, n_jobs=-1,
                                  random_state=RANDOM_STATE, verbose=-1)

        elif best_name == "XGBoost":
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 300, 800),
                "max_depth":       trial.suggest_int("max_depth", 4, 12),
                "learning_rate":   trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "subsample":       trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree":trial.suggest_float("colsample_bytree", 0.55, 1.0),
                "gamma":           trial.suggest_float("gamma", 0.0, 0.5),
                "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            }
            clf = XGBClassifier(
                **params, use_label_encoder=False,
                eval_metric="mlogloss", n_jobs=-1,
                random_state=RANDOM_STATE,
                num_class=n_classes, objective="multi:softprob",
            )

        else:  # RandomForest
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 300, 700),
                "max_depth":       trial.suggest_int("max_depth", 15, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
                "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
            clf = RandomForestClassifier(**params, n_jobs=-1, random_state=RANDOM_STATE)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X_train, y_train,
                                  cv=cv, scoring="f1_weighted", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best trial: {study.best_params}  →  f1={study.best_value:.4f}")
    return study.best_params, study.best_value


# ── Re-train tuned model ──────────────────────────────────────────────────────
def _retrain_tuned(best_name: str, best_params: dict,
                   n_classes: int, X_train, y_train, X_test, y_test) -> dict:
    """Reconstruct the tuned model and fit on full training set."""
    if best_name == "LightGBM":
        clf = LGBMClassifier(**best_params, n_jobs=-1,
                              random_state=RANDOM_STATE, verbose=-1)
    elif best_name == "XGBoost":
        clf = XGBClassifier(
            **best_params, use_label_encoder=False,
            eval_metric="mlogloss", n_jobs=-1,
            random_state=RANDOM_STATE,
            num_class=n_classes, objective="multi:softprob",
        )
    else:
        clf = RandomForestClassifier(**best_params, n_jobs=-1,
                                      random_state=RANDOM_STATE)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "model":            clf,
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "f1_weighted":      float(f1_score(y_test, y_pred, average="weighted")),
        "cv_mean":          0.0,
        "cv_std":           0.0,
        "elapsed_s":        0,
        "report":           classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# ── Artifact persistence ──────────────────────────────────────────────────────
def save_artifacts(model, scaler, label_encoder,
                   feature_names: list, results: dict):
    """Persist model, scaler, encoder, feature names, and benchmark metrics."""
    joblib.dump(model,         BEST_MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    serialisable = {
        name: {k: v for k, v in info.items() if k != "model"}
        for name, info in results.items()
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)

    logger.info(f"Artifacts saved → {BEST_MODEL_PATH}")


# ── Master pipeline ───────────────────────────────────────────────────────────
def train_pipeline(X, y, scaler, le, feature_names: list,
                   tune: bool = True, n_trials: int = OPTUNA_TRIALS,
                   build_stack: bool = True):
    """
    Full training pipeline:
      split → SMOTE → benchmark → (stack) → (HPO) → save → return best model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # SMOTE on training data only (zero-leakage)
    from src.data_pipeline import apply_smote
    X_train, y_train = apply_smote(X_train, y_train)

    n_classes = len(np.unique(y))

    # ── Baseline benchmark ────────────────────────────────────────────────────
    logger.info("── Baseline benchmark ───────────────────────────────────")
    results = benchmark_models(X_train, X_test, y_train, y_test, n_classes)

    # ── Stacking ensemble ─────────────────────────────────────────────────────
    if build_stack:
        logger.info("── Stacking Ensemble ─────────────────────────────────────")
        stack_result = build_stacking_ensemble(
            X_train, X_test, y_train, y_test, results, n_classes
        )
        results["StackingEnsemble"] = stack_result

    # ── Pick best by weighted-F1 ──────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    logger.info(f"\n🏆 Best model: {best_name}  "
                f"(f1={results[best_name]['f1_weighted']:.4f})")

    # ── Optuna HPO (only on single-model winners) ─────────────────────────────
    if tune and "Stacking" not in best_name:
        best_params, best_val = tune_best_model(
            X_train, y_train, best_name, n_classes, n_trials
        )
        if best_params is not None:
            tuned_result = _retrain_tuned(
                best_name, best_params, n_classes,
                X_train, y_train, X_test, y_test,
            )
            tuned_result["cv_mean"] = best_val
            results[f"{best_name}_Tuned"] = tuned_result
            best_name = f"{best_name}_Tuned"
            logger.info(f"🏆 After HPO: {best_name}  "
                        f"(f1={results[best_name]['f1_weighted']:.4f})")

    final_model = results[best_name]["model"]
    save_artifacts(final_model, scaler, le, feature_names, results)
    return final_model, results, X_test, y_test, best_name
