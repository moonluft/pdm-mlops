"""Train candidate models, track them in MLflow, register the best one as @production."""
import json
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data import FEATURES, TARGET, make_data

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("machine-failure")

# TODO 2a: set a THRESHOLD to turn failure probability into a hard 0/1 prediction.
# HINT: with rare classes, the default 0.5 threshold gets low recall.
#       Set a lower threshold to catch more failures (higher recall) while tolerating a few false alarms.
THRESHOLD = 0.3

# TODO 2b: set a minimum PR-AUC quality gate threshold.
# HINT: must be a float between 0.5 and 0.8. If the best model has a lower score, training fails.
MIN_PR_AUC = 0.6

# TODO 2e: which metric should select the winner?
# Options: "pr_auc", "recall", "precision", "roc_auc"
# HINT: on rare classes, we care about the overall ranking quality on the positive class.
SELECT_BY = "pr_auc"


def main():
    print("Loading data...")
    df = make_data()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # TODO 2c: choose three candidate models of different classes.
    # One linear and one forest are provided. Add/enable a third one.
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "xgboost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss"),
    }

    best_score = -1.0
    best_model_name = None
    best_pipeline = None
    best_run_id = None
    best_metrics = {}

    print("Training models...")
    for name, clf in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            clf.fit(X_train, y_train)

            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= THRESHOLD).astype(int)

            # Calculate metrics
            roc_auc = float(roc_auc_score(y_test, y_prob))
            pr_auc = float(average_precision_score(y_test, y_prob))

            # TODO 2d: fill in the formula for recall and precision
            recall = float(recall_score(y_test, y_pred))
            precision = float(precision_score(y_test, y_pred, zero_division=0))

            # Log parameters
            mlflow.log_param("model_type", name)
            mlflow.log_param("threshold", THRESHOLD)
            if hasattr(clf, "steps"):
                mlflow.log_param("steps", [s[0] for s in clf.steps])
                actual_clf = clf.steps[-1][1]
            else:
                actual_clf = clf
            for p in ["C", "max_depth", "n_estimators", "n_neighbors", "kernel", "learning_rate"]:
                if hasattr(actual_clf, p):
                    mlflow.log_param(p, getattr(actual_clf, p))

            # Log metrics
            metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc, "recall": recall, "precision": precision}
            mlflow.log_metrics(metrics)

            # Log model with signature
            signature = mlflow.models.infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(clf, "model", signature=signature, serialization_format="pickle")

            print(f"  {name:<20} | PR-AUC: {pr_auc:.3f} | Recall: {recall:.3f} | Precision: {precision:.3f}")

            score = metrics[SELECT_BY]
            if score > best_score:
                best_score = score
                best_model_name = name
                best_pipeline = clf
                best_run_id = run.info.run_id
                best_metrics = metrics

    print(f"Best model: {best_model_name} with {SELECT_BY} = {best_score:.3f}")

    # Quality Gate Check
    if best_score < MIN_PR_AUC:
        raise RuntimeError(f"Quality Gate Failed: best score {best_score:.3f} < threshold {MIN_PR_AUC:.3f}")
    print("Quality Gate Passed!")

    # Save winning model to disk
    joblib.dump(best_pipeline, "model.joblib")
    print("Saved model.joblib")

    # Write metrics summary json
    summary = {
        "best_model": best_model_name,
        "selected_by": SELECT_BY,
        "threshold": THRESHOLD,
        "min_pr_auc": MIN_PR_AUC,
        **best_metrics,
    }
    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # TODO 2f: model registration
    # HINT: 1. Register the model under name "machine-failure-model"
    #       2. Point the production alias to the newly created version
    model_name = "machine-failure-model"
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)

    client = MlflowClient()
    client.set_registered_model_alias(model_name, "production", mv.version)
    print(f"Registered model version {mv.version} and set alias @production")


if __name__ == "__main__":
    main()
