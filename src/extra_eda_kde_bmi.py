import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- CONFIG ----------
DATA_PATH = "data/Cleaned-healthcare-dataset-stroke-data_.csv"
FIG_DIR = "figures/extra_eda"
# ----------------------------

# Create folder for figures
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    """Save current figure into the extra EDA figures folder."""
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"[FIG] Saved {path}")
    plt.close()

# ---------- LOAD DATA ----------
df = pd.read_csv(DATA_PATH)

# Make sure stroke is numeric 0/1 
if df["stroke"].dtype == "O":
    df["stroke"] = df["stroke"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})

# ================== 1) KDE PLOTS (AGE & GLUCOSE) ==================

def kde_plot_by_stroke(df, column):
    """Draw KDE for a numeric column, split by stroke = 0/1."""
    plt.figure(figsize=(6, 4))

    sns.kdeplot(
        df[df["stroke"] == 0][column],
        label="No Stroke",
        fill=True
    )
    sns.kdeplot(
        df[df["stroke"] == 1][column],
        label="Stroke",
        fill=True
    )

    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(f"KDE: {column} by Stroke Outcome")
    plt.legend()

    savefig(f"kde_{column}.png")

# Run KDE for age and avg_glucose_level
kde_plot_by_stroke(df, "avg_glucose_level")
kde_plot_by_stroke(df, "age")

# ================== 2) BMI EDA (CATEGORICAL) ==================

# 2a) BMI distribution (how many in each category)
plt.figure(figsize=(6, 3))
sns.countplot(x=df["bmi"], order=df["bmi"].value_counts().index)
plt.xticks(rotation=25, ha="right")
plt.title("BMI Category Distribution")
plt.xlabel("BMI Category")
plt.ylabel("Count")
savefig("bmi_distribution.png")

# 2b) Stroke probability by BMI category (mean of stroke within each BMI)
plt.figure(figsize=(6, 4))
bmi_stroke_rate = df.groupby("bmi")["stroke"].mean().sort_values()
bmi_stroke_rate.plot(kind="bar", color="teal")
plt.ylabel("Stroke Rate (Proportion)")
plt.xlabel("BMI Category")
plt.title("Stroke Probability by BMI Category")
savefig("bmi_stroke_rate.png")

# 2c) Stacked bar: distribution of stroke vs no-stroke within each BMI category
crosstab = pd.crosstab(df["bmi"], df["stroke"], normalize="index")
crosstab.plot(kind="bar", stacked=True, figsize=(6, 4), colormap="viridis")
plt.ylabel("Proportion within BMI Category")
plt.xlabel("BMI Category")
plt.title("Stroke vs No Stroke Distribution within BMI Categories")
plt.legend(title="Stroke", labels=["No (0)", "Yes (1)"])
savefig("bmi_stroke_stacked.png")

# 2d) Age distribution by BMI category (boxplot)
plt.figure(figsize=(7, 4))
sns.boxplot(x="bmi", y="age", data=df, order=df["bmi"].value_counts().index)
plt.xticks(rotation=25, ha="right")
plt.xlabel("BMI Category")
plt.ylabel("Age")
plt.title("Age Distribution by BMI Category")
savefig("bmi_age_boxplot.png")

print("\n[INFO] Extra EDA (KDE + BMI plots) completed.")
print(f"Figures saved in folder: {FIG_DIR}")
