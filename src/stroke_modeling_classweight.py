"""
stroke_modeling_classweight.py

Same pipeline as before but:
- NO SMOTE
- Class weights used instead:
    LogisticRegression(class_weight="balanced")
    RandomForest(class_weight="balanced")
    XGBoost(scale_pos_weight = weight ratio)
    NaiveBayes -> no class weight
Purpose: Train and evaluate stroke prediction models using class weights to handle class imbalance.
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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier


# ------------------------ CONFIG --------------------------

DATA_PATH = "data/Cleaned-healthcare-dataset-stroke-data_.csv"
TARGET_COL = "stroke"
FIG_DIR = "figures/models/classweight"
RANDOM_STATE = 42
TEST_SIZE = 0.2
FINAL_THRESHOLD = 0.30

# Encode Yes/No columns
BINARY_COLS = ["hypertension", "heart_disease", "ever_married"]


def ensure_fig_dir():
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)


def savefig(name):
    ensure_fig_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300)
    print(f"[FIG] Saved {os.path.join(FIG_DIR, name)}")
    plt.close()


# ------------------------- DATA ---------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    print("[INFO] Loaded:", df.shape)

    # Encode binary cols
    for col in BINARY_COLS:
        if df[col].dtype == "O":
            df[col] = df[col].map({"Yes": 1, "No": 0})

    if df[TARGET_COL].dtype == "O":
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)

    print("[INFO] After one-hot:", X.shape)
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("[INFO] Train distribution:")
    print(y_train.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


# -------------------------- MODELS -------------------------

def build_models(y_train):

    # compute ratio for XGBoost
    count_0 = (y_train == 0).sum()
    count_1 = (y_train == 1).sum()
    scale_pos_weight = count_0 / count_1

    print(f"[INFO] XGBoost scale_pos_weight = {scale_pos_weight:.2f}")

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        ),

        "NaiveBayes": GaussianNB()
    }

    return models


# ----------------------- EVALUATION -------------------------

def evaluate(name, model, X_test, y_test, threshold):

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    print(f"\n=== {name} (Class Weight, thr={threshold}) ===")
    print("Acc :", acc)
    print("Prec:", prec)
    print("Rec :", rec)
    print("F1  :", f1)
    print("AUC :", auc)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} (Class Weight)\nConfusion Matrix (thr={threshold:.2f})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    savefig(f"{name}_cm_classweight_thr_{threshold}.png")

    # ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="grey", label="Random")
        plt.title(f"{name} (Class Weight)\nROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        savefig(f"{name}_roc_classweight.png")

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "threshold": threshold,
    }


# --------------------------- MAIN ---------------------------

def main():
    sns.set(style="whitegrid")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = build_models(y_train)

    results = []
    for name, model in models.items():
        print(f"\n[TRAIN] {name} with CLASS WEIGHTS ...")
        model.fit(X_train, y_train)

        thr = FINAL_THRESHOLD if name != "NaiveBayes" else 0.5

        res = evaluate(name, model, X_test, y_test, thr)
        results.append(res)

    print("\n=== FINAL CLASS WEIGHT RESULTS ===")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
