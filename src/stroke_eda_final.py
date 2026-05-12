"""
stroke_eda_final.py

Exploratory Data Analysis for the stroke prediction project.

Works with dataset:
    Cleaned-healthcare-dataset-stroke-data_.csv

Assumes columns (typical cleaned stroke dataset):
    bmi (categorical: underweight/normal/overweight/obese)
    gender
    age (numeric)
    hypertension (Yes/No)
    heart_disease (Yes/No)
    ever_married (Yes/No)
    work_type
    Residence_type
    avg_glucose_level (numeric)
    smoking_status
    stroke (0/1 integer)

What this script does:
    - Loads the dataset
    - Encodes binary Yes/No columns to 1/0 (for maths)
    - Prints descriptive statistics and missing values
    - Shows value counts for categorical variables
    - Computes Cramér's V (categorical vs stroke)
    - Plots numeric correlation heatmap
    - Plots distributions, bar charts, boxplots
    - Runs DBSCAN on numeric features to detect noise/outliers
    - Saves all plots into ./figures
    - Saves a copy of the dataframe with DBSCAN cluster labels
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# ---------------------------- CONFIG ---------------------------------

DATA_PATH = "data/Cleaned-healthcare-dataset-stroke-data_.csv"
TARGET_COL = "stroke"
FIG_DIR = "figures/eda"


# ---------------------- SMALL UTILITY HELPERS ------------------------


def ensure_fig_dir():
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)


def savefig(name, tight=True):
    """Helper to save figures into the figures/ folder."""
    if tight:
        plt.tight_layout()
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=300)
    print(f"[FIG] Saved {path}")
    plt.close()


# ---------------------------- LOADING --------------------------------


def load_data():
    print(f"[INFO] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print("\n[INFO] First 5 rows:")
    print(df.head())
    print("\n[INFO] Shape (rows, cols):", df.shape)
    return df


# ---------------------- BASIC ENCODING / CLEANING --------------------


def encode_binary_yes_no(df):
    """
    Convert Yes/No style binary columns to 1/0.
    We assume:
        hypertension      -> Yes/No
        heart_disease     -> Yes/No
        ever_married      -> Yes/No
    stroke is already numeric 0/1 and will be kept as is.
    """
    binary_yes_no_cols = ["hypertension", "heart_disease", "ever_married"]

    for col in binary_yes_no_cols:
        if col not in df.columns:
            print(f"[WARN] Column not found (skipped): {col}")
            continue

        if df[col].dtype == "object" or df[col].dtype=="string":  # object, likely Yes/No
            df[col] = df[col].map(
                {"Yes": 1, "No": 0, "YES": 1, "NO": 0, "yes": 1, "no": 0}
            )
            print(f"[ENCODE] Converted {col} from Yes/No → 1/0")
        else:
            print(f"[ENCODE] {col} already numeric, left as is")

    # Ensure stroke is numeric (0/1)
    if TARGET_COL in df.columns:
        if df[TARGET_COL].dtype == "O":
            df[TARGET_COL] = df[TARGET_COL].map(
                {"Yes": 1, "No": 0, "YES": 1, "NO": 0, "yes": 1, "no": 0}
            )
            print(f"[ENCODE] Converted {TARGET_COL} to numeric 0/1 from Yes/No")
        else:
            print(f"[ENCODE] {TARGET_COL} already numeric (0/1), OK")
    else:
        print(f"[ERROR] Target column '{TARGET_COL}' not found!")

    return df


# ---------------------- DESCRIPTIVE STATISTICS ------------------------


def basic_descriptives(df):
    print("\n[INFO] Descriptive statistics (numeric):")
    print(df.describe())

    print("\n[INFO] Descriptive statistics (including categorical):")
    print(df.describe(include="all"))

    print("\n[INFO] Missing values per column:")
    print(df.isna().sum())


def value_counts_categoricals(df):
    cat_cols = df.select_dtypes(include="object").columns
    print("\n[INFO] Value counts for categorical variables:")
    for col in cat_cols:
        print("\n---", col, "---")
        print(df[col].value_counts(dropna=False))


# -------------------------- CRAMER'S V --------------------------------


def cramers_v(x, y):
    """Cramér's V for two categorical vectors."""
    confusion = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def categorical_vs_stroke_cramers(df):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    cat_cols = df.select_dtypes(include="object").columns
    print("\n[INFO] Cramér's V (categorical vs stroke):")
    results = []
    for col in cat_cols:
        try:
            v = cramers_v(df[col], df[TARGET_COL])
            results.append((col, v))
            print(f"{col:20s}  Cramér's V = {v:.3f}")
        except Exception as e:
            print(f"[WARN] Could not compute Cramér's V for {col}: {e}")

    results_sorted = sorted(results, key=lambda t: abs(t[1]), reverse=True)
    print("\n[INFO] Cramér's V sorted (strongest association first):")
    for col, v in results_sorted:
        print(f"{col:20s}  {v:.3f}")


# -------------------------- CORRELATION -------------------------------


def numeric_correlations(df):
    """
    Compute and plot correlation matrix for numeric features:
        age, avg_glucose_level, hypertension, heart_disease,
        ever_married (0/1), stroke (0/1)
    """
    candidate_cols = [
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
        "ever_married",
        TARGET_COL,
    ]
    num_cols = [c for c in candidate_cols if c in df.columns]

    subset = df[num_cols].copy()
    print("\n[INFO] Numeric correlation matrix (subset):")
    print(subset.corr())

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        subset.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f"
    )
    plt.title("Numeric Correlations")
    savefig("corr_numeric_heatmap.png")


# -------------------------- DISTRIBUTIONS -----------------------------


def plot_distributions(df):
    ensure_fig_dir()

    # Numeric distributions
    numeric_cols = ["age", "avg_glucose_level"]
    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(5, 3))
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            savefig(f"dist_{col}.png")

    # Stroke imbalance
    if TARGET_COL in df.columns:
        plt.figure(figsize=(4, 3))
        sns.countplot(x=df[TARGET_COL])
        plt.title("Stroke Class Distribution (0 = No, 1 = Yes)")
        savefig("dist_stroke.png")

    # Important categorical distributions
    cat_to_plot = [
        "bmi",
        "smoking_status",
        "hypertension",
        "heart_disease",
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
    ]

    for col in cat_to_plot:
        if col in df.columns:
            plt.figure(figsize=(6, 3))
            sns.countplot(x=df[col])
            plt.xticks(rotation=30, ha="right")
            plt.title(f"Distribution of {col}")
            savefig(f"dist_{col}.png")


# ------------------- RELATIONSHIPS WITH STROKE ------------------------


def plot_relationships(df):
    ensure_fig_dir()

    # Categorical vs Stroke – grouped bars
    cat_cols = [
        "bmi",
        "smoking_status",
        "hypertension",
        "heart_disease",
        "ever_married",
        "gender",
        "work_type",
        "Residence_type",
    ]

    for col in cat_cols:
        if col not in df.columns:
            continue

        plt.figure(figsize=(6, 3))
        sns.countplot(x=df[col], hue=df[TARGET_COL])
        plt.xticks(rotation=30, ha="right")
        plt.title(f"{col} vs Stroke")
        savefig(f"{col}_vs_stroke.png")

    # Numeric vs Stroke – boxplots
    for col in ["age", "avg_glucose_level"]:
        if col in df.columns:
            plt.figure(figsize=(5, 3))
            sns.boxplot(x=df[TARGET_COL], y=df[col])
            plt.title(f"{col} by Stroke (0/1)")
            savefig(f"box_{col}_by_stroke.png")


# ------------------------- OUTLIERS / DBSCAN --------------------------


def run_dbscan(df):
    """
    DBSCAN on numeric features for outlier exploration.
    We use numeric subset:
        age, avg_glucose_level, hypertension, heart_disease, ever_married (if present)
    Noise points (label -1) are NOT removed; they are just analysed.
    """
    ensure_fig_dir()

    numeric_cols = [
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
        "ever_married",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    if len(numeric_cols) < 2:
        print("[DBSCAN] Not enough numeric columns for DBSCAN, skipping.")
        return df

    X = df[numeric_cols].copy()

    # Standardize for DBSCAN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=0.8, min_samples=10)
    labels = db.fit_predict(X_scaled)
    df["dbscan_cluster"] = labels

    print("\n[DBSCAN] Cluster label counts (including noise = -1):")
    print(pd.Series(labels).value_counts())

    # 2D scatter: Age vs Glucose colored by cluster (if both exist)
    if "age" in df.columns and "avg_glucose_level" in df.columns:
        plt.figure(figsize=(6, 4))
        scatter = plt.scatter(
            df["age"],
            df["avg_glucose_level"],
            c=labels,
            cmap="tab10",
            s=10,
            alpha=0.8,
        )
        plt.xlabel("Age")
        plt.ylabel("Average Glucose Level")
        plt.title("DBSCAN Clusters (Age vs Glucose)\nCluster ID (-1 = Noise)")
        plt.colorbar(scatter, label="Cluster ID")
        savefig("dbscan_age_glucose.png")

    return df


# ------------------------------ MAIN ----------------------------------


def main():
    ensure_fig_dir()
    df = load_data()

    # 1) Convert Yes/No to 1/0 for relevant columns
    df = encode_binary_yes_no(df)

    # 2) Basic stats
    basic_descriptives(df)
    value_counts_categoricals(df)

    # 3) Categorical association with stroke via Cramér's V
    categorical_vs_stroke_cramers(df)

    # 4) Numeric correlation matrix
    numeric_correlations(df)

    # 5) Distributions and relationships
    plot_distributions(df)
    plot_relationships(df)

    # 6) DBSCAN for outlier/noise exploration (kept for analysis, not removed)
    df = run_dbscan(df)

    # 7) Save dataframe with DBSCAN labels
    out_path = "data/stroke_with_dbscan_labels.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved dataframe with DBSCAN labels to: {out_path}")


if __name__ == "__main__":
    main()
