"""
train.py
Credit Risk Project — Model training pipeline.
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
"""

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")


def train_models(X_train, y_train):
    """
    Train three classifiers:
    1. Logistic Regression (baseline)
    2. Random Forest
    3. XGBoost
    Returns dict of fitted models.
    """
    print("[3/5] Training models...")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="logloss", random_state=42, verbosity=0
        ),
    }

    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"    Trained: {name}")

    return fitted


def add_cluster_features(X_train, X_test, n_clusters=4):
    """
    Hybrid pipeline: unsupervised K-Means borrower segmentation.
    Appends cluster label as an additional feature.
    """
    print("    Adding K-Means cluster features (hybrid pipeline)...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_train)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["cluster"] = km.predict(X_train)
    X_test["cluster"] = km.predict(X_test)
    return X_train, X_test, km


def evaluate(models, X_test, y_test):
    """
    Evaluate all models, return results dict.
    """
    print("[4/5] Evaluating models...")
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "report": classification_report(y_test, y_pred),
            "cm": confusion_matrix(y_test, y_pred),
        }
        print(f"    {name}: AUC={results[name]['auc_roc']:.3f}, F1={results[name]['f1']:.3f}, Acc={results[name]['accuracy']:.3f}")
    return results


def save_models(models, path="models/"):
    os.makedirs(path, exist_ok=True)
    for name, model in models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(path, fname))
    print(f"    Models saved to {path}")


if __name__ == "__main__":
    from preprocess import load_data, preprocess
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)
    X_train_c, X_test_c, km = add_cluster_features(X_train, X_test)
    models = train_models(X_train_c, y_train)
    results = evaluate(models, X_test_c, y_test)
    save_models(models, path="/home/claude/credit-risk-project/models/")
    print("Training complete.")
