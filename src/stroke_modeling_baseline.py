"""
stroke_modeling.py

Models + Evaluation for our stroke prediction project

Uses cleaned dataset:
    Cleaned-healthcare-dataset-stroke-data_.csv

Models:
    - Logistic Regression
    - Random Forest
    - Naive Bayes (GaussianNB)

Key steps:
    - Load dataset
    - Encode Yes/No -> 1/0 for binary risk factors
    - One-hot encode multi-category categorical variables
    - Train/test split with stratification on stroke
    - Compute class weights for minority class
    - Fit models with sample_weight (to weight minority class more)
    - Evaluate with:
        * Accuracy
        * Precision
        * Recall
        * F1-score
        * ROC-AUC
        * Confusion matrix
    - Plot ROC curve & confusion matrix for each model
    - Print comparison table of metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# ---------------------------- CONFIG ---------------------------------

DATA_PATH = "data/Cleaned-healthcare-dataset-stroke-data_.csv"
TARGET_COL = "stroke"
FIG_DIR = "figures/models/baseline"

# Binary Yes/No columns
BINARY_YES_NO_COLS = ["hypertension", "heart_disease", "ever_married"]

# Multi-category categorical columns
CATEGORICAL_COLS = [
    "bmi",
    "gender",
    "work_type",
    "Residence_type",
    "smoking_status",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------- UTILS ----------------------------------------


def ensure_fig_dir():
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)


def savefig(name):
    ensure_fig_dir()
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"[FIG] Saved {path}")
    plt.close()


# ---------------------- DATA PREP ------------------------------------


def load_and_encode_data():
    print(f"[INFO] Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print("[INFO] Data shape:", df.shape)

    # Encode Yes/No binary columns to 1/0
    for col in BINARY_YES_NO_COLS:
        if col in df.columns:
            if df[col].dtype == "O":
                df[col] = df[col].map(
                    {"Yes": 1, "No": 0,
                     "YES": 1, "NO": 0,
                     "yes": 1, "no": 0}
                )
                print(f"[ENCODE] {col}: Yes/No -> 1/0")
            else:
                print(f"[ENCODE] {col}: already numeric")
        else:
            print(f"[WARN] {col} not found in dataset")

    # Ensure stroke is numeric 0/1
    if df[TARGET_COL].dtype == "O":
        df[TARGET_COL] = df[TARGET_COL].map(
            {"Yes": 1, "No": 0,
             "YES": 1, "NO": 0,
             "yes": 1, "no": 0}
        )
        print(f"[ENCODE] {TARGET_COL}: Yes/No -> 0/1")
    else:
        print(f"[ENCODE] {TARGET_COL}: already numeric")

    # Separate features and target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)
    print("[INFO] After one-hot encoding, feature shape:", X.shape)

    return X, y


def train_test_split_stratified(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("[INFO] Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("[INFO] Stroke distribution in train:\n", y_train.value_counts(normalize=True))
    print("[INFO] Stroke distribution in test:\n", y_test.value_counts(normalize=True))
    return X_train, X_test, y_train, y_test


def compute_sample_weights(y_train):
    """
    Compute class weights for imbalance and expand to sample_weight array.
    """
    classes = np.array(sorted(y_train.unique()))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.values
    )
    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}
    print("[INFO] Class weights:", class_weight_dict)

    # Build sample_weight array
    sample_weights = y_train.map(class_weight_dict).values
    return sample_weights, class_weight_dict


# ---------------------- MODELING -------------------------------------


def build_models():
    """
    Return a dict of models we want to train.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "NaiveBayes": GaussianNB(),
    }
    return models


def evaluate_model(name, model, X_test, y_test):
    """
    Compute metrics & confusion matrix for a trained model.
    Returns metrics dict and also plots ROC & confusion matrix.
    """
    y_pred = model.predict(X_test)

    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
       
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # Min-max transform to [0,1] for ROC-AUC
            y_proba = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = np.nan

    print(f"\n=== {name} ===")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)
    print("ROC-AUC  :", auc)
    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    savefig(f"{name}_confusion_matrix.png")

    # ROC curve
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} - ROC Curve")
        plt.legend()
        savefig(f"{name}_roc_curve.png")

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def main():
    sns.set(style="whitegrid")
    ensure_fig_dir()

    # 1) Load and encode
    X, y = load_and_encode_data()

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    # 3) Compute sample weights for minority class
    sample_weights, class_weight_dict = compute_sample_weights(y_train)

    # 4) Build models
    models = build_models()

    # 5) Train & evaluate each model
    results = []
    for name, model in models.items():
        print(f"\n[TRAIN] Fitting {name} ...")
        # Fit with sample_weight to emphasize minority class
        model.fit(X_train, y_train, sample_weight=sample_weights)
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

    # 6) Summary comparison table
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]])


if __name__ == "__main__":
    main()
