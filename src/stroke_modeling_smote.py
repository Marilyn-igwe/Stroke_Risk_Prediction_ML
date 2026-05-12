"""
stroke_modeling.py

Models + Evaluation for stroke prediction project.

Dataset:
    Cleaned-healthcare-dataset-stroke-data_.csv

Models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Naive Bayes (GaussianNB)

Pipeline:
    - Load cleaned dataset
    - Encode Yes/No -> 1/0 for binary columns
    - One-hot encode categorical variables
    - Train/test split with stratification on stroke
    - Apply SMOTE on training set to balance classes
    - Train all models on SMOTE-balanced training data
    - Evaluate on original (imbalanced) test data:
        * Accuracy
        * Precision
        * Recall
        * F1-score
        * ROC-AUC
        * Confusion matrix
    - For Logistic Regression, Random Forest & XGBoost:
      use threshold = 0.30 for final evaluation
    - For Logistic Regression: also print threshold exploration
    - Save ROC curves & confusion matrices into ./figures_models
    - Print comparison table of all models
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE  # for class balancing
from xgboost import XGBClassifier          

# ---------------------------- CONFIG ---------------------------------

DATA_PATH = "data/Cleaned-healthcare-dataset-stroke-data_.csv"
TARGET_COL = "stroke"
FIG_DIR = "figures/models/smote"

# Binary Yes/No columns
BINARY_YES_NO_COLS = ["hypertension", "heart_disease", "ever_married"]

# Categorical columns – will be one-hot encoded 
CATEGORICAL_COLS = [
    "bmi",
    "gender",
    "work_type",
    "Residence_type",
    "smoking_status",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Final chosen threshold for LR, RF, XGB
FINAL_THRESHOLD = 0.30


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
    print("\n[INFO] First 5 rows:")
    print(df.head())

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

    # One-hot encode all object-type columns 
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


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to the training data to balance classes.
    """
    print("\n[SMOTE] Before SMOTE, class distribution (train):")
    print(y_train.value_counts())

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("\n[SMOTE] After SMOTE, class distribution (train):")
    print(y_res.value_counts())

    return X_res, y_res


# ---------------------- MODELING -------------------------------------


def build_models():
    """
    Return a dict of models to train.
    RandomForest & XGBoost trained on SMOTE-balanced data.
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
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "NaiveBayes": GaussianNB(),
    }
    return models


def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    """
    Compute metrics & confusion matrix for a trained model.
    Uses the given probability threshold if predict_proba is available.
    Returns metrics dict and also plots ROC & confusion matrix.
    """
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_proba = None
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = np.nan

    print(f"\n=== {name} (threshold={threshold:.2f}) ===")
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
    plt.title(f"{name} - Confusion Matrix (thr={threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    fname = f"{name}_confusion_matrix_thr_{threshold:.2f}.png".replace(".", "_")
    savefig(fname)

    # ROC curve (independent of threshold used)
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
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def threshold_explore(model, X_test, y_test, name="Model"):
    """
    Explore different probability thresholds for a model
    with predict_proba, and print precision/recall/F1.
    """
    if not hasattr(model, "predict_proba"):
        print(f"[{name}] No predict_proba, skipping threshold analysis.")
        return

    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n[{name}] Threshold analysis (precision, recall, f1):")
    for thr in [0.20, 0.30, 0.40, 0.50]:
        y_pred_thr = (y_proba >= thr).astype(int)
        prec = precision_score(y_test, y_pred_thr, zero_division=0)
        rec = recall_score(y_test, y_pred_thr, zero_division=0)
        f1 = f1_score(y_test, y_pred_thr, zero_division=0)
        print(f"  thr={thr:.2f} -> precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")


# ---------------------- MAIN ----------------------------------------


def main():
    sns.set(style="whitegrid")
    ensure_fig_dir()

    # 1) Load & encode
    X, y = load_and_encode_data()

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    # 3) Apply SMOTE on training set (for class balancing)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # 4) Build models
    models = build_models()

    # 5) Train & evaluate with chosen thresholds
    results = []

    for name, model in models.items():
        print(f"\n[TRAIN] Fitting {name} on SMOTE-balanced data ...")
        model.fit(X_train_sm, y_train_sm)

        # Use threshold 0.30 for Logistic Regression, RandomForest, XGBoost
        if name in ["LogisticRegression", "RandomForest", "XGBoost"]:
            metrics = evaluate_model(name, model, X_test, y_test, threshold=FINAL_THRESHOLD)

            # For Logistic Regression, also show threshold exploration table
            if name == "LogisticRegression":
                threshold_explore(model, X_test, y_test, name="LogisticRegression")
        else:
            # Naive Bayes → default 0.5 threshold
            metrics = evaluate_model(name, model, X_test, y_test, threshold=0.50)

        results.append(metrics)

    # 6) Summary comparison table
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison (with configured thresholds) ===")
    print(results_df[["model", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc"]])


if __name__ == "__main__":
    main()
