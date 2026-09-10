"""
Week 3: Unsupervised Learning & Clustering Analysis Pipeline
Author: Mayank Saini
Dataset: Customer Segmentation (Mall Customers Benchmark)
Algorithms: K-Means Clustering & Agglomerative Hierarchical Clustering
"""

import os
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Directory Initialization
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# 2. Data Acquisition
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
MALL_URL = "https://raw.githubusercontent.com/SteffiPeT/Customer-Segmentation-using-RFM-and-K-Means/master/Mall_Customers.csv"
csv_path = os.path.join("data", "Mall_Customers.csv")

if not os.path.exists(csv_path):
    print("Fetching dataset...")
    urllib.request.urlretrieve(MALL_URL, csv_path)
    print("Dataset saved to:", csv_path)

df = pd.read_csv(csv_path)

# Rename columns for programmatic consistency
df.columns = ["CustomerID", "Gender", "Age", "Annual_Income_k", "Spending_Score"]
print("\nDataset Snapshot:")
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())

# 3. Preprocessing & Feature Transformation
# Primary numerical features selected for customer behavioral segmentation
features = ["Age", "Annual_Income_k", "Spending_Score"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Hyperparameter Optimization: Elbow Method & Silhouette Analysis
wcss = []
silhouette_scores = []
db_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=25, max_iter=400, random_state=42)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# Plot 1: Elbow and Silhouette Analysis
fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=300)

color = "tab:blue"
ax1.set_xlabel("Number of Clusters (k)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Inertia / WCSS", color=color, fontsize=11, fontweight="bold")
ax1.plot(k_range, wcss, marker="o", color=color, linewidth=2, label="WCSS")
ax1.tick_params(axis="y", labelcolor=color)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Silhouette Score", color=color, fontsize=11, fontweight="bold")
ax2.plot(k_range, silhouette_scores, marker="s", color=color, linewidth=2, linestyle="--", label="Silhouette")
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Elbow Method & Silhouette Coefficients Across Cluster Cardinalities", fontsize=12, pad=12)
fig.tight_layout()
plt.savefig("figures/01_elbow_silhouette.png")
plt.close()

# 5. Hierarchical Clustering (Dendrogram)
plt.figure(figsize=(10, 5), dpi=300)
linked = linkage(X_scaled, method="ward")
dendrogram(linked, orientation="top", distance_sort="descending", show_leaf_counts=False, no_labels=True)
plt.axhline(y=7.0, color="crimson", linestyle="--", label="Optimal Cutoff (k=5)")
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=12, pad=10)
plt.xlabel("Sample Index", fontsize=10)
plt.ylabel("Euclidean Cophenetic Distance", fontsize=10)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("figures/02_dendrogram.png")
plt.close()

# 6. Final Model Fitting (Optimal k = 5)
OPTIMAL_K = 5
kmeans_final = KMeans(n_clusters=OPTIMAL_K, init="k-means++", n_init=30, random_state=42)
df["Cluster"] = kmeans_final.fit_predict(X_scaled)

# Validation Metrics
final_sil = silhouette_score(X_scaled, df["Cluster"])
final_db = davies_bouldin_score(X_scaled, df["Cluster"])
final_ch = calinski_harabasz_score(X_scaled, df["Cluster"])

print(f"\nFinal Validation Metrics (k={OPTIMAL_K}):")
print(f"Silhouette Score:         {final_sil:.4f}")
print(f"Davies-Bouldin Index:     {final_db:.4f}")
print(f"Calinski-Harabasz Score:  {final_ch:.4f}")

# 7. Cluster Visualization (PCA Projection & Direct Domain Features)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Scatter 1: Domain Dimensions (Income vs. Spending)
sns.scatterplot(
    data=df,
    x="Annual_Income_k",
    y="Spending_Score",
    hue="Cluster",
    palette=palette,
    style="Cluster",
    s=80,
    ax=axes[0]
)
axes[0].set_title("Customer Segments: Income vs. Spending Score", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Annual Income ($k)", fontsize=10)
axes[0].set_ylabel("Spending Score (1-100)", fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.5)

# Scatter 2: PCA Reduced Space
sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette=palette,
    style="Cluster",
    s=80,
    ax=axes[1]
)
axes[1].set_title(
    f"PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
    fontsize=11,
    fontweight="bold"
)
axes[1].set_xlabel("Principal Component 1", fontsize=10)
axes[1].set_ylabel("Principal Component 2", fontsize=10)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("figures/03_kmeans_clusters.png")
plt.close()

# 8. Cluster Profiling Aggregations
profile = df.groupby("Cluster")[features].mean().round(2)
profile["Customer_Count"] = df.groupby("Cluster")["CustomerID"].count()
profile["Percentage"] = (profile["Customer_Count"] / len(df) * 100).round(1)
print("\nCluster Architectural Profiles:")
print(profile)

# Plot 4: Feature Distribution across Clusters
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
features_to_plot = ["Age", "Annual_Income_k", "Spending_Score"]
titles = ["Mean Age per Cluster", "Annual Income ($k)", "Spending Score (1-100)"]

for idx, col in enumerate(features_to_plot):
    sns.barplot(x=profile.index, y=profile[col], ax=axes[idx], palette=palette)
    axes[idx].set_title(titles[idx], fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("Cluster ID", fontsize=10)
    axes[idx].set_ylabel("")
    axes[idx].grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("figures/04_cluster_profiles.png")
plt.close()

print("\nPipeline execution complete. All figures rendered to ./figures/")