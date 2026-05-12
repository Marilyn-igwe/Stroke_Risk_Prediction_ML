# Stroke Prediction

Stroke is one of the leading causes of death and disability worldwide. This project uses machine learning to predict stroke risk based on patient health and demographic data. We conduct exploratory data analysis to identify key risk factors, then compare three modelling strategies to address the dataset's class imbalance.

---

## Dataset

A cleaned version of the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), containing 5,110 patient records with features including age, BMI, average glucose level, hypertension, heart disease, and smoking status.

---

## Modelling Approaches

| Script | Strategy | Models |
|---|---|---|
| `stroke_modeling_baseline.py` | Sample weights | Logistic Regression, Random Forest, Naive Bayes |
| `stroke_modeling_smote.py` | SMOTE oversampling | Logistic Regression, Random Forest, Naive Bayes, XGBoost |
| `stroke_modeling_classweight.py` | Class weights | Logistic Regression, Random Forest, Naive Bayes, XGBoost |

---
## Results

---

## Authors

Blen Nima
Marilyn Igwe 
Swati Poojary
