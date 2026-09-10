"""
Week 4 Task: Supervised Learning Model Implementation
Problem Type : Binary Classification
Dataset      : Breast Cancer Wisconsin (Diagnostic) Data Set
               (publicly available; bundled with scikit-learn,
               originally from the UCI Machine Learning Repository)
Goal         : Predict whether a breast tumour is Malignant (M) or
               Benign (B) from digitized fine needle aspirate (FNA)
               features of a breast mass.
Author       : Mayank
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42
FIG_DIR = "figures"
DATA_DIR = "data"
OUT_DIR = "outputs"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# --------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.rename(columns={"target": "diagnosis"}, inplace=True)
# In this dataset target=0 -> malignant, target=1 -> benign (sklearn convention)
df["diagnosis_label"] = df["diagnosis"].map({0: "malignant", 1: "benign"})

df.to_csv(f"{DATA_DIR}/breast_cancer_raw.csv", index=False)

print("Dataset shape:", df.shape)
print(df["diagnosis_label"].value_counts())

# --------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------------------------
summary_stats = df.drop(columns=["diagnosis", "diagnosis_label"]).describe().T
summary_stats.to_csv(f"{OUT_DIR}/summary_statistics.csv")

missing = df.isnull().sum().sum()
print("Total missing values:", missing)

# Class balance plot
plt.figure(figsize=(5, 4))
sns.countplot(x="diagnosis_label", data=df, palette=["#e74c3c", "#2ecc71"])
plt.title("Class Distribution: Malignant vs Benign")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/class_distribution.png")
plt.close()

# Correlation heatmap (top features)
plt.figure(figsize=(10, 8))
corr = df.drop(columns=["diagnosis", "diagnosis_label"]).corr()
top_corr_features = corr["mean radius"].abs().sort_values(ascending=False).index[:12]
sns.heatmap(df[top_corr_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Top Features Correlated with 'mean radius')")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap.png")
plt.close()

# Distribution of a couple of key features by class
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x="diagnosis_label", y="mean radius", data=df, ax=axes[0], palette=["#e74c3c", "#2ecc71"])
axes[0].set_title("Mean Radius by Diagnosis")
sns.boxplot(x="diagnosis_label", y="mean texture", data=df, ax=axes[1], palette=["#e74c3c", "#2ecc71"])
axes[1].set_title("Mean Texture by Diagnosis")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_boxplots.png")
plt.close()

# --------------------------------------------------------------------
# 3. FEATURE ENGINEERING & PREPROCESSING
# --------------------------------------------------------------------
X = df.drop(columns=["diagnosis", "diagnosis_label"])
y = df["diagnosis"]  # 0 = malignant, 1 = benign

# Engineered ratio features (domain-informed feature engineering)
X["radius_texture_ratio"] = X["mean radius"] / (X["mean texture"] + 1e-6)
X["concavity_area_ratio"] = X["mean concavity"] / (X["mean area"] + 1e-6)
X["perimeter_radius_ratio"] = X["mean perimeter"] / (X["mean radius"] + 1e-6)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# --------------------------------------------------------------------
# 4. MODEL PIPELINES
# --------------------------------------------------------------------
# Pipeline = scaling + feature selection (top-k ANOVA F-value) + classifier
# This keeps preprocessing inside cross-validation folds, avoiding leakage.

pipelines = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=15)),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=15)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=15)),
        ("clf", SVC(probability=True, random_state=RANDOM_STATE))
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = {}
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    cv_results[name] = {"mean_f1": scores.mean(), "std_f1": scores.std(), "folds": scores.tolist()}
    print(f"{name}: CV F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

# --------------------------------------------------------------------
# 5. HYPERPARAMETER TUNING (best candidate: Logistic Regression)
# --------------------------------------------------------------------
best_name = max(cv_results, key=lambda n: cv_results[n]["mean_f1"])
print("Best model by CV F1:", best_name)

param_grids = {
    "Logistic Regression": {
        "select__k": [10, 15, 20],
        "clf__C": [0.01, 0.1, 1, 10],
    },
    "Random Forest": {
        "select__k": [10, 15, 20],
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 5, 10],
    },
    "SVM (RBF)": {
        "select__k": [10, 15, 20],
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", "auto"],
    },
}

grid = GridSearchCV(
    pipelines[best_name], param_grids[best_name], cv=cv, scoring="f1", n_jobs=-1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

# --------------------------------------------------------------------
# 6. FINAL EVALUATION ON HELD-OUT TEST SET
# --------------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
print("\nTest-set metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

report_text = classification_report(y_test, y_pred, target_names=["malignant", "benign"])
print(report_text)

with open(f"{OUT_DIR}/classification_report.txt", "w") as f:
    f.write(report_text)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["malignant", "benign"], yticklabels=["malignant", "benign"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix.png")
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}", color="#2980b9")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/roc_curve.png")
plt.close()

# Cross-validation comparison plot across all models
plt.figure(figsize=(6, 4))
names = list(cv_results.keys())
means = [cv_results[n]["mean_f1"] for n in names]
stds = [cv_results[n]["std_f1"] for n in names]
plt.bar(names, means, yerr=stds, capsize=6, color=["#3498db", "#e67e22", "#9b59b6"])
plt.ylabel("Cross-Validated F1 Score")
plt.title("Model Comparison (5-Fold Stratified CV)")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/model_comparison.png")
plt.close()

# Feature importance (if Random Forest was selected) or coefficients (Logistic Regression)
selected_mask = best_model.named_steps["select"].get_support()
selected_features = X_train.columns[selected_mask]

plt.figure(figsize=(7, 5))
clf_step = best_model.named_steps["clf"]
if hasattr(clf_step, "feature_importances_"):
    importances = clf_step.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.barh(np.array(selected_features)[order][:15][::-1], importances[order][:15][::-1], color="#16a085")
    plt.title("Feature Importances")
elif hasattr(clf_step, "coef_"):
    coefs = clf_step.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    plt.barh(np.array(selected_features)[order][:15][::-1], coefs[order][:15][::-1], color="#16a085")
    plt.title("Logistic Regression Coefficients (selected features)")
plt.xlabel("Importance / Coefficient magnitude")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_importance.png")
plt.close()

# --------------------------------------------------------------------
# 7. SAVE ALL RESULTS FOR REPORT GENERATION
# --------------------------------------------------------------------
results_summary = {
    "dataset_shape": df.shape,
    "class_counts": df["diagnosis_label"].value_counts().to_dict(),
    "missing_values": int(missing),
    "cv_results": cv_results,
    "best_model_name": best_name,
    "best_params": grid.best_params_,
    "best_cv_f1": grid.best_score_,
    "test_metrics": metrics,
    "selected_features": list(selected_features),
}

with open(f"{OUT_DIR}/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

print("\nAll outputs saved to 'outputs/' and 'figures/' directories.")
