"""
train_model.py
Trains and compares multiple ML algorithms for a two-stage intrusion
detection pipeline:

    Stage 1 (binary):     Normal vs Attack
    Stage 2 (multiclass):  DoS / DDoS / Port Scan / Brute Force / Suspicious Traffic
                            (only applied to records already flagged ATTACK)

The best-performing algorithm for each stage is selected automatically
based on F1-score and saved to models/. All metrics are computed from
actual evaluation on a held-out test set -- nothing here is hard-coded.

Run directly:
    python src/train_model.py
"""

import json
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    METRICS_PATH,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAINING_DATA_PATH,
    MODELS_DIR,
)
from src.preprocessing import (  # noqa: E402
    clean_dataframe,
    fit_preprocessing_objects,
    save_preprocessing_objects,
    transform,
)
from src.utils import get_logger, safe_makedirs  # noqa: E402

logger = get_logger(__name__)

CANDIDATE_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    "SVM": SVC(probability=True, random_state=RANDOM_SEED),
}


def evaluate_candidates(X_train, y_train, X_test, y_test, stage_name: str):
    """Train each candidate model, evaluate it, and return comparison results."""
    results = {}
    fitted_models = {}

    for name, model in CANDIDATE_MODELS.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - start, 3)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "training_time_sec": train_time,
        }
        fitted_models[name] = model

        logger.info(
            "[%s] %s -> acc=%.4f prec=%.4f rec=%.4f f1=%.4f (%.3fs)",
            stage_name, name, acc, prec, rec, f1, train_time,
        )

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    logger.info("[%s] Best model: %s", stage_name, best_name)
    return best_name, fitted_models[best_name], results


def main():
    safe_makedirs(MODELS_DIR)

    logger.info("Loading training and test data...")
    train_df = pd.read_csv(TRAINING_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    train_df = clean_dataframe(train_df)
    test_df = clean_dataframe(test_df)

    # Fit scaler + categorical/target encoders on TRAIN data only (no leakage)
    scaler, encoders = fit_preprocessing_objects(train_df)
    save_preprocessing_objects(scaler, encoders)

    # ---- Stage 1: Normal vs Attack -----------------------------------
    logger.info("=== Stage 1: Intrusion Detection (Normal vs Attack) ===")

    train_feat = transform(train_df, scaler, encoders, include_target=False)
    test_feat = transform(test_df, scaler, encoders, include_target=False)

    y_train_stage1 = (train_df[TARGET_COLUMN] != "Normal").astype(int)
    y_test_stage1 = (test_df[TARGET_COLUMN] != "Normal").astype(int)

    X_train = train_feat[FEATURE_COLUMNS]
    X_test = test_feat[FEATURE_COLUMNS]

    best_name_1, best_model_1, results_1 = evaluate_candidates(
        X_train, y_train_stage1, X_test, y_test_stage1, "Stage1"
    )

    preds_1 = best_model_1.predict(X_test)
    cm_1 = confusion_matrix(y_test_stage1, preds_1).tolist()
    report_1 = classification_report(
        y_test_stage1, preds_1, target_names=["Normal", "Attack"], output_dict=True
    )

    joblib.dump(best_model_1, STAGE1_MODEL_PATH)
    logger.info("Saved Stage 1 model (%s) to %s", best_name_1, STAGE1_MODEL_PATH)

    # ---- Stage 2: Attack type classification (attack rows only) ------
    logger.info("=== Stage 2: Attack Type Classification ===")

    train_attack_mask = train_df[TARGET_COLUMN] != "Normal"
    test_attack_mask = test_df[TARGET_COLUMN] != "Normal"

    X_train_2 = X_train[train_attack_mask.values]
    X_test_2 = X_test[test_attack_mask.values]
    y_train_2 = train_df.loc[train_attack_mask, TARGET_COLUMN]
    y_test_2 = test_df.loc[test_attack_mask, TARGET_COLUMN]

    best_name_2, best_model_2, results_2 = evaluate_candidates(
        X_train_2, y_train_2, X_test_2, y_test_2, "Stage2"
    )

    preds_2 = best_model_2.predict(X_test_2)
    attack_classes = sorted(y_test_2.unique().tolist())
    cm_2 = confusion_matrix(y_test_2, preds_2, labels=attack_classes).tolist()
    report_2 = classification_report(
        y_test_2, preds_2, labels=attack_classes, output_dict=True, zero_division=0
    )

    joblib.dump(best_model_2, STAGE2_MODEL_PATH)
    logger.info("Saved Stage 2 model (%s) to %s", best_name_2, STAGE2_MODEL_PATH)

    # ---- Feature importance (if Random Forest was selected) ----------
    feature_importance = {}
    for stage_label, model, name in [
        ("stage1", best_model_1, best_name_1),
        ("stage2", best_model_2, best_name_2),
    ]:
        if hasattr(model, "feature_importances_"):
            feature_importance[stage_label] = dict(
                zip(FEATURE_COLUMNS, [round(float(v), 4) for v in model.feature_importances_])
            )

    # ---- Persist all metrics for the dashboard ------------------------
    metrics = {
        "stage1": {
            "best_model": best_name_1,
            "comparison": results_1,
            "confusion_matrix": cm_1,
            "classification_report": report_1,
            "labels": ["Normal", "Attack"],
        },
        "stage2": {
            "best_model": best_name_2,
            "comparison": results_2,
            "confusion_matrix": cm_2,
            "classification_report": report_2,
            "labels": attack_classes,
        },
        "feature_importance": feature_importance,
        "feature_columns": FEATURE_COLUMNS,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", METRICS_PATH)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
