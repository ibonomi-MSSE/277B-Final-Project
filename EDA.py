from encoding import full_data_pipeline  # change to your actual filename
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import umap
import os

OUTPUT_DIR = "EDA_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load processed data
data_clean, data_genomic_positions, drug_lookup = full_data_pipeline()

# Choosing the dataset with genomic positions for EDA
df = data_genomic_positions.copy()
df.head()

# 2. Separate features and target
y = df["resistant"]
X = df.drop(columns=["resistant"])

# 3. Identify and handle different column types
# Separate numeric and non-numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# One-hot encode categorical columns (like drug names)
if categorical_cols:
    X_encoded = pd.get_dummies(X[categorical_cols], drop_first=True)
    # Combine with numeric features
    X_processed = pd.concat([X[numeric_cols], X_encoded], axis=1)
else:
    X_processed = X[numeric_cols]

# Check for any remaining non-numeric columns
non_numeric_remaining = X_processed.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_remaining:
    # print number of non-numeric columns remaining
    print(f"Warning: Still have non-numeric columns: {len(non_numeric_remaining)}")
    # Drop them if they can't be encoded
    X_processed = X_processed.drop(columns=non_numeric_remaining)

# Remove any columns with missing values
X_processed = X_processed.dropna(axis=1)

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# ----------------------------
# PCA: Explained Variance Plot
# ----------------------------
pca = PCA()
pca.fit(X_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'b-')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.xlim(1, 15)  # Focus on the first 15 components
plt.grid(True, alpha=0.3)

plt.savefig(os.path.join(OUTPUT_DIR, "PCA Explained Variance.png"), dpi=300)
plt.close()

# print the number of components needed to explain 95% of the variance
n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"Number of PCA components needed to explain 95% variance: {n_components_95}")

# ----------------------------
# UMAP
# ----------------------------

print(f"Reducing from {X_scaled.shape[1]} to 10 dimensions with PCA first to preserve 95% variance...")
pca_pre = PCA(n_components=min(10, X_scaled.shape[1]))
X_for_umap = pca_pre.fit_transform(X_scaled)

umap_model = umap.UMAP(
    n_neighbors=min(15, len(X_for_umap) - 1),  # Handle small datasets
    min_dist=0.1,
    n_components=2,
    random_state=42
)

X_umap = umap_model.fit_transform(X_for_umap)

umap_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
umap_df["resistant"] = y.values

# First, check what's in your resistant column
print("Unique values in 'resistant' column:")
print(y.value_counts())
print(f"Number of unique classes: {y.nunique()}")

# Create dynamic colors based on number of unique classes
unique_classes = sorted(umap_df["resistant"].unique())
n_classes = len(unique_classes)

# Generate a color for each class
if n_classes <= 10:
    # Use tab10 colormap for up to 10 classes
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
elif n_classes <= 20:
    # Use tab20 for up to 20 classes
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
else:
    # Use viridis for many classes
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))


plt.figure(figsize=(10, 8))
for i, label in enumerate(sorted(umap_df["resistant"].unique())):
    subset = umap_df[umap_df["resistant"] == label]
    plt.scatter(subset["UMAP1"], subset["UMAP2"], 
               label=f"Resistant={label}", 
               alpha=0.6, 
               c=[colors[i]],
               edgecolors='white',
               linewidth=0.5,
               )

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP Projection - Drug Resistance Patterns")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "UMAP Projection - Drug Resistance Patterns.png"), dpi =300)


"""
# -------------------------
# Correlation Heatmap
# -------------------------

# Calculate correlation matrix
# Only use numeric columns (skip the target if it's in the dataframe)
corr_features = X_processed.copy()

# Option 1: Full correlation matrix (might be large)
plt.figure(figsize=(16, 14))
correlation_matrix = corr_features.corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# Plot heatmap
sns.heatmap(correlation_matrix, 
            mask=mask,
            cmap='RdBu_r',  # Red-Blue diverging colormap
            center=0,        # Center colormap at 0
            annot=False,     # Set to True if you have few features
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            vmin=-1, 
            vmax=1)

plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.show()
"""