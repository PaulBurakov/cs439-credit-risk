"""
preprocess.py
Credit Risk Project — Data loading and preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings("ignore")


def load_data(path="data/credit_data.csv"):
    """
    Load the German Credit-style dataset.
    Returns raw DataFrame.
    """
    print("[1/5] Loading dataset...")
    df = pd.read_csv(path)
    return df


def preprocess(df):
    """
    Full preprocessing pipeline:
    - Binary encode target
    - One-hot encode categoricals
    - Standard scale numerics
    - Train/test split (no leakage)
    Returns X_train, X_test, y_train, y_test, feature_names
    """
    print("[2/5] Preprocessing data...")

    # Target already binary encoded (0=no default, 1=default)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "target"]

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df_encoded.drop(columns=["target"])
    y = df_encoded["target"]

    feature_names = X.columns.tolist()

    # Train/test split BEFORE scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale only numeric features; fit on train only
    scaler = StandardScaler()
    num_cols_encoded = [c for c in num_cols if c in X_train.columns]
    X_train[num_cols_encoded] = scaler.fit_transform(X_train[num_cols_encoded])
    X_test[num_cols_encoded] = scaler.transform(X_test[num_cols_encoded])

    print(f"    Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"    Default rate (train): {y_train.mean():.2%}")
    print(f"    Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)
    print("Preprocessing complete.")
