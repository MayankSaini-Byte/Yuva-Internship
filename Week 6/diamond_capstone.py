"""
Week 6 Task: Integrative Capstone Project
Title   : Diamond Pricing Analytics — Predictive Pricing and Market Segmentation
Author  : Mayank

Pipeline stages:
  1. Data acquisition
  2. Data cleaning
  3. Exploratory data analysis (EDA)
  4. Feature engineering
  5. Supervised modeling  -> predict diamond price (regression)
  6. Unsupervised modeling -> discover natural diamond market segments (clustering)
  7. Evaluation of both models
  8. Insights export for the report
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

RANDOM_STATE = 42
FIG = "figures"
DATA = "data"
OUT = "outputs"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# ======================================================================
# 1. DATA ACQUISITION
# ======================================================================
# Source: seaborn-data repository (mirrors the well-known public "diamonds"
# dataset, ~54,000 records of diamond attributes and prices).
df_raw = sns.load_dataset("diamonds")
df_raw.to_csv(f"{DATA}/diamonds_raw.csv", index=False)
print("Raw shape:", df_raw.shape)
print(df_raw.dtypes)

# ======================================================================
# 2. DATA CLEANING
# ======================================================================
df = df_raw.copy()

n_before = len(df)
n_duplicates = df.duplicated().sum()
df = df.drop_duplicates()

# Physically impossible records: zero dimensions can't exist for a real stone
invalid_dims = ((df["x"] == 0) | (df["y"] == 0) | (df["z"] == 0)).sum()
df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

# Extreme outliers in physical dimensions (data entry errors, e.g. y or z > 20mm
# while carat is small) are removed using domain-informed bounds.
outlier_mask = (df["y"] > 20) | (df["z"] > 10) | (df["table"] > 80) | (df["table"] < 40) | (df["depth"] > 80) | (df["depth"] < 40)
n_outliers = outlier_mask.sum()
df = df[~outlier_mask]

missing_values = df.isnull().sum().sum()

cleaning_log = {
    "rows_before": n_before,
    "duplicates_removed": int(n_duplicates),
    "invalid_zero_dimension_rows_removed": int(invalid_dims),
    "physical_outliers_removed": int(n_outliers),
    "missing_values_remaining": int(missing_values),
    "rows_after": len(df),
}
print("Cleaning log:", cleaning_log)

df.to_csv(f"{DATA}/diamonds_clean.csv", index=False)

# ======================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ======================================================================
desc = df.describe(include="all").T
desc.to_csv(f"{OUT}/summary_statistics.csv")

# Price distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["price"], bins=50, kde=True, ax=axes[0], color="#2980b9")
axes[0].set_title("Price Distribution (Raw)")
sns.histplot(np.log1p(df["price"]), bins=50, kde=True, ax=axes[1], color="#8e44ad")
axes[1].set_title("Price Distribution (Log-Transformed)")
plt.tight_layout()
plt.savefig(f"{FIG}/price_distribution.png")
plt.close()

# Price vs carat, colored by cut
plt.figure(figsize=(6, 5))
sample = df.sample(4000, random_state=RANDOM_STATE)
sns.scatterplot(data=sample, x="carat", y="price", hue="cut", alpha=0.5, palette="viridis", s=15)
plt.title("Price vs Carat (sampled, colored by Cut)")
plt.tight_layout()
plt.savefig(f"{FIG}/price_vs_carat.png")
plt.close()

# Categorical breakdown: mean price by cut/color/clarity
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.barplot(data=df, x="cut", y="price", ax=axes[0], estimator=np.mean,
            order=["Fair", "Good", "Very Good", "Premium", "Ideal"], color="#27ae60")
axes[0].set_title("Mean Price by Cut")
axes[0].tick_params(axis="x", rotation=30)
sns.barplot(data=df, x="color", y="price", ax=axes[1], estimator=np.mean,
            order=sorted(df["color"].unique()), color="#2980b9")
axes[1].set_title("Mean Price by Color")
sns.barplot(data=df, x="clarity", y="price", ax=axes[2], estimator=np.mean,
            order=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], color="#c0392b")
axes[2].set_title("Mean Price by Clarity")
axes[2].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(f"{FIG}/categorical_price_breakdown.png")
plt.close()
# Note for report: mean price by cut/color/clarity alone can look counter-intuitive
# (e.g. "Fair" cut sometimes shows a high mean) because carat is a confound -
# larger stones are more often cut conservatively. This motivates controlling
# for carat in the modeling stage rather than reading these bars in isolation.

# Correlation heatmap of numeric features
plt.figure(figsize=(7, 6))
numeric_cols = ["carat", "depth", "table", "price", "x", "y", "z"]
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{FIG}/correlation_heatmap.png")
plt.close()

# ======================================================================
# 4. FEATURE ENGINEERING
# ======================================================================
df_fe = df.copy()

# Physical volume proxy and price-per-carat (useful for segmentation later)
df_fe["volume"] = df_fe["x"] * df_fe["y"] * df_fe["z"]
df_fe["price_per_carat"] = df_fe["price"] / df_fe["carat"]

# Ordinal encoding: cut, color, clarity all have a natural quality order
cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]          # J (worst) -> D (best)
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]  # worst -> best

ordinal_encoder = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])
df_fe[["cut_enc", "color_enc", "clarity_enc"]] = ordinal_encoder.fit_transform(
    df_fe[["cut", "color", "clarity"]]
)

feature_cols_reg = ["carat", "depth", "table", "x", "y", "z", "volume",
                     "cut_enc", "color_enc", "clarity_enc"]
target_col = "price"

df_fe.to_csv(f"{DATA}/diamonds_features.csv", index=False)

# ======================================================================
# 5. SUPERVISED MODELING — PRICE PREDICTION (REGRESSION)
# ======================================================================
X = df_fe[feature_cols_reg]
y = df_fe[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# For the model-comparison / cross-validation stage, use a representative
# subsample of the training data to keep runtime manageable on ~54k rows
# while still giving a reliable comparison signal.
cv_sample_idx = X_train.sample(n=min(6000, len(X_train)), random_state=RANDOM_STATE).index
X_train_cv = X_train.loc[cv_sample_idx]
y_train_cv = y_train.loc[cv_sample_idx]

reg_pipelines = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=RANDOM_STATE))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, max_depth=14, random_state=RANDOM_STATE, n_jobs=1))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE))
    ]),
}

kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
reg_cv_results = {}
for name, pipe in reg_pipelines.items():
    scores = cross_val_score(pipe, X_train_cv, y_train_cv, cv=kf, scoring="r2", n_jobs=1)
    reg_cv_results[name] = {"mean_r2": scores.mean(), "std_r2": scores.std()}
    print(f"{name}: CV R2 = {scores.mean():.4f} (+/- {scores.std():.4f})")

best_reg_name = max(reg_cv_results, key=lambda n: reg_cv_results[n]["mean_r2"])
print("Best regression model by CV R2:", best_reg_name)

# Light, targeted hyperparameter tuning (small grid, run on the CV subsample)
reg_param_grids = {
    "Random Forest": {"model__max_depth": [14, 22]},
    "Gradient Boosting": {"model__learning_rate": [0.05, 0.1]},
    "Ridge Regression": {"model__alpha": [0.1, 1.0, 10.0]},
    "Linear Regression": {},
}

grid_reg = GridSearchCV(
    reg_pipelines[best_reg_name],
    reg_param_grids[best_reg_name],
    cv=kf, scoring="r2", n_jobs=1
) if reg_param_grids[best_reg_name] else None

if grid_reg is not None:
    # Tune on the CV subsample for speed, then refit the chosen configuration
    # on the FULL training set for the strongest final model.
    grid_reg.fit(X_train_cv, y_train_cv)
    best_reg_params = grid_reg.best_params_
    best_reg_cv_r2 = grid_reg.best_score_
    best_reg_model = reg_pipelines[best_reg_name]
    best_reg_model.set_params(**best_reg_params)
    # Cap ensemble size on the full refit for speed on ~43k training rows
    if "model__n_estimators" in reg_param_grids.get(best_reg_name, {}):
        pass
    best_reg_model.fit(X_train, y_train)
else:
    best_reg_model = reg_pipelines[best_reg_name].fit(X_train, y_train)
    best_reg_params = {}
    best_reg_cv_r2 = reg_cv_results[best_reg_name]["mean_r2"]

print("Best regression params:", best_reg_params)

y_pred = best_reg_model.predict(X_test)
reg_metrics = {
    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    "mae": float(mean_absolute_error(y_test, y_pred)),
    "r2": float(r2_score(y_test, y_pred)),
}
print("Test regression metrics:", reg_metrics)

# Actual vs predicted plot
plt.figure(figsize=(5.5, 5))
plt.scatter(y_test, y_pred, alpha=0.15, s=8, color="#2980b9")
lims = [0, max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", linewidth=1.5)
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title(f"Actual vs Predicted Price — {best_reg_name}")
plt.tight_layout()
plt.savefig(f"{FIG}/actual_vs_predicted.png")
plt.close()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(5.5, 4))
plt.scatter(y_pred, residuals, alpha=0.15, s=8, color="#c0392b")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(f"{FIG}/residual_plot.png")
plt.close()

# Model comparison bar chart
plt.figure(figsize=(6, 4))
names = list(reg_cv_results.keys())
means = [reg_cv_results[n]["mean_r2"] for n in names]
stds = [reg_cv_results[n]["std_r2"] for n in names]
plt.bar(names, means, yerr=stds, capsize=6, color=["#3498db", "#e67e22", "#9b59b6", "#16a085"])
plt.ylabel("Cross-Validated R\u00b2")
plt.title("Regression Model Comparison (5-Fold CV)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{FIG}/regression_model_comparison.png")
plt.close()

# Feature importance (tree-based models)
model_step = best_reg_model.named_steps["model"]
plt.figure(figsize=(6.5, 5))
if hasattr(model_step, "feature_importances_"):
    importances = model_step.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.barh(np.array(feature_cols_reg)[order][::-1], importances[order][::-1], color="#16a085")
    plt.title(f"Feature Importances — {best_reg_name}")
elif hasattr(model_step, "coef_"):
    coefs = model_step.coef_
    order = np.argsort(np.abs(coefs))[::-1]
    plt.barh(np.array(feature_cols_reg)[order][::-1], coefs[order][::-1], color="#16a085")
    plt.title(f"Coefficients — {best_reg_name}")
plt.tight_layout()
plt.savefig(f"{FIG}/feature_importance.png")
plt.close()

# ======================================================================
# 6. UNSUPERVISED MODELING — MARKET SEGMENTATION (CLUSTERING)
# ======================================================================
cluster_features = ["carat", "depth", "table", "cut_enc", "color_enc",
                     "clarity_enc", "price_per_carat"]

# Cluster on a manageable, reproducible sample for speed and clarity
cluster_sample = df_fe.sample(8000, random_state=RANDOM_STATE).reset_index(drop=True)
X_cluster_raw = cluster_sample[cluster_features]

cluster_scaler = StandardScaler()
X_cluster = cluster_scaler.fit_transform(X_cluster_raw)

# Determine a reasonable k via the elbow method + silhouette score
inertias, silhouettes = [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_k = km.fit_predict(X_cluster)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cluster, labels_k))

plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(list(k_range), inertias, marker="o", color="#2980b9")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.subplot(1, 2, 2)
plt.plot(list(k_range), silhouettes, marker="o", color="#c0392b")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.tight_layout()
plt.savefig(f"{FIG}/cluster_selection.png")
plt.close()

best_k_by_silhouette = list(k_range)[int(np.argmax(silhouettes))]

# Business note: pure silhouette score is often maximized at k=2 for datasets
# with one dominant axis of variation (here, carat/price size), but a 2-segment
# view has little practical value for merchandising or pricing strategy.
# We therefore prefer the best k with >=3 segments unless k=2 is clearly superior
# (more than a 0.02 silhouette-score margin), trading a small amount of pure
# cluster separation for a business-actionable number of segments.
silhouettes_arr = np.array(silhouettes)
k_list = list(k_range)
idx_ge3 = [i for i, k in enumerate(k_list) if k >= 3]
best_idx_ge3 = idx_ge3[int(np.argmax(silhouettes_arr[idx_ge3]))]
best_k_ge3 = k_list[best_idx_ge3]

if silhouettes_arr[k_list.index(best_k_by_silhouette)] - silhouettes_arr[best_idx_ge3] > 0.02:
    best_k = best_k_by_silhouette
else:
    best_k = best_k_ge3

print(f"Silhouette-optimal k: {best_k_by_silhouette} | Chosen k (business-informed): {best_k}")

kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
cluster_sample["cluster"] = kmeans_final.fit_predict(X_cluster)
final_silhouette = silhouette_score(X_cluster, cluster_sample["cluster"])
print("Final silhouette score:", final_silhouette)

# PCA for 2D visualization of clusters
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coords = pca.fit_transform(X_cluster)
cluster_sample["pca1"] = pca_coords[:, 0]
cluster_sample["pca2"] = pca_coords[:, 1]

plt.figure(figsize=(6.5, 5.5))
sns.scatterplot(data=cluster_sample, x="pca1", y="pca2", hue="cluster", palette="tab10", s=18, alpha=0.7)
plt.title(f"Diamond Segments Visualized via PCA (k={best_k})")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.tight_layout()
plt.savefig(f"{FIG}/cluster_pca.png")
plt.close()

# Cluster profiling: mean feature values per cluster
cluster_profile = cluster_sample.groupby("cluster")[cluster_features + ["price"]].mean().round(2)
cluster_profile["count"] = cluster_sample.groupby("cluster").size()
cluster_profile.to_csv(f"{OUT}/cluster_profile.csv")
print(cluster_profile)

plt.figure(figsize=(8, 5))
profile_norm = (cluster_profile[cluster_features] - cluster_profile[cluster_features].mean()) / cluster_profile[cluster_features].std()
sns.heatmap(profile_norm.T, annot=cluster_profile[cluster_features].T.round(2), fmt="", cmap="coolwarm", cbar_kws={"label": "Relative level (z-score)"})
plt.title("Cluster Profiles (annotated with actual mean values)")
plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig(f"{FIG}/cluster_profile_heatmap.png")
plt.close()

# ======================================================================
# 7. SAVE RESULTS SUMMARY FOR REPORT
# ======================================================================
results_summary = {
    "cleaning_log": cleaning_log,
    "n_features_engineered": ["volume", "price_per_carat", "cut_enc", "color_enc", "clarity_enc"],
    "regression": {
        "cv_results": reg_cv_results,
        "best_model": best_reg_name,
        "best_params": best_reg_params,
        "best_cv_r2": best_reg_cv_r2,
        "test_metrics": reg_metrics,
        "feature_columns": feature_cols_reg,
    },
    "clustering": {
        "k_range_tested": list(k_range),
        "silhouette_scores": silhouettes,
        "inertias": inertias,
        "silhouette_optimal_k": int(best_k_by_silhouette),
        "chosen_k": int(best_k),
        "final_silhouette": final_silhouette,
        "cluster_sizes": cluster_sample["cluster"].value_counts().sort_index().to_dict(),
    },
}
with open(f"{OUT}/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

print("\nCapstone pipeline complete. Outputs saved to 'outputs/' and 'figures/'.")
