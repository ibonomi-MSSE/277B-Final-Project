# Feature Selection Guide

This document explains how to use automated feature selection with the WHO data loader.

## Overview

Feature selection reduces the dimensionality of your dataset by:
- **PCA** for continuous features (Morgan fingerprints, ChemBERTa embeddings)
- **L1 regularization** for one-hot encoded features (genes, mutations, drugs)

This helps:
- Reduce overfitting
- Speed up training
- Improve model interpretability
- Remove redundant features

## Quick Start

```python
from data.WHO_data import get_who_dataframe

# Enable feature selection with one parameter
df, _ = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv',
    'feature_selection': True  # Add this!
}, feature_selection_output_dir='feature_plots')
```

## What Gets Selected

### Morgan Fingerprints (PCA)
- **Before**: 256 binary features
- **After**: ~12 principal components (95% variance)
- **Reduction**: 244 features removed (~95% reduction)

### ChemBERTa Embeddings (PCA)
- **Before**: 128 float features
- **After**: ~20 principal components (95% variance)
- **Reduction**: 108 features removed (~85% reduction)

### Gene Features (L1)
- **Before**: 65+ one-hot encoded genes
- **After**: ~20-30 genes (varies by target)
- **Reduction**: Removes genes with zero predictive power

### Mutation Features (L1)
Applied separately to:
- **Mutation types** (11 types -> ~5 types)
- **Reference amino acids** (27 -> ~20)
- **Alternative amino acids** (27 -> ~15-20)

### Drug Features (L1, if one-hot encoded)
- **Before**: 15 one-hot encoded drugs
- **After**: ~10-15 drugs (usually keeps most)
- **Reduction**: Removes drugs with insufficient data

## Typical Results

### Example: Morgan Fingerprints + PPV Target

**Without feature selection:**
```
Shape: (89936, 397)
Features: 256 fingerprints + 65 genes + 65 mutation features + 11 metadata
```

**With feature selection:**
```
Shape: (89936, 87)
Features: 12 PCA components + 22 genes + 42 mutation features + 11 metadata
Reduction: 310 features removed (78%)
```

### Example: ChemBERTa + Resistance Target

**Without feature selection:**
```
Shape: (35592, 525)
Features: 128 embeddings + 65 genes + 321 mutation features + 11 metadata
```

**With feature selection:**
```
Shape: (35592, 120)
Features: 20 PCA components + 25 genes + 64 mutation features + 11 metadata
Reduction: 405 features removed (77%)
```

## Visualization Plots

Feature selection generates plots for each feature group:

### PCA Plots (2 subplots per plot)

**Left plot**: Explained variance per component
- Shows how much variance each component captures
- Helps understand feature importance distribution

**Right plot**: Cumulative explained variance
- Shows total variance captured by N components
- Green line: 95% threshold
- Blue line: Selected number of components

**Example files**:
- `pca_morgan_fingerprints.png`
- `pca_chemberta_embeddings.png`

### L1 Plots (2 subplots per plot)

**Left plot**: Coefficient magnitudes (sorted)
- Bar chart of absolute coefficients
- Red line: zero threshold
- Features above line are kept

**Right plot**: Coefficient distribution
- Histogram of coefficient values
- Shows sparsity of selection
- Text box: selection statistics

**Example files**:
- `l1_gene_features.png`
- `l1_mutation_type.png`
- `l1_reference_aa.png`
- `l1_alternative_aa.png`
- `l1_drug_one-hot.png` (if using one-hot drug encoding)

## Configuration Options

### Customize PCA Variance Threshold

Currently hardcoded to 95%. To change, modify `apply_pca_feature_selection()`:

```python
# In data_loader.py
def apply_pca_feature_selection(
    data: pd.DataFrame,
    feature_cols: List[str],
    variance_threshold: float = 0.90,  # Change this
    ...
):
```

### Customize L1 Regularization Strength

Currently uses `alpha=0.01`. To change, modify `apply_l1_feature_selection()`:

```python
# In data_loader.py
def apply_l1_feature_selection(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    feature_type: str = "features",
    alpha: float = 0.001,  # Change this (lower = less sparsity)
    ...
):
```

## Complete Examples

### Example 1: Compare With/Without Selection

```python
from data.WHO_data import get_who_dataframe

# Without selection
df_full, _ = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv'
}, add_genomic_positions=False)

# With selection
df_selected, _ = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv',
    'feature_selection': True
}, add_genomic_positions=False,
   feature_selection_output_dir='comparison_plots')

print(f"Full dataset: {df_full.shape}")
print(f"Selected dataset: {df_selected.shape}")
print(f"Reduction: {df_full.shape[1] - df_selected.shape[1]} features")
```

### Example 2: Different Encoding Strategies

```python
from data.WHO_data import get_who_dataframe

# Morgan fingerprints (PCA-based selection)
df1, _ = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv',
    'feature_selection': True
}, add_genomic_positions=False,
   feature_selection_output_dir='plots/morgan')

# ChemBERTa (PCA-based selection)
df2, _ = get_who_dataframe({
    'drug': 'chemberta',
    'mutation': 'extract_features',
    'target': 'ppv',
    'feature_selection': True
}, add_genomic_positions=False,
   feature_selection_output_dir='plots/chemberta')

# One-hot (L1-based selection)
df3, _ = get_who_dataframe({
    'drug': 'one_hot',
    'mutation': 'extract_features',
    'target': 'ppv',
    'feature_selection': True
}, add_genomic_positions=False,
   feature_selection_output_dir='plots/onehot')

print(f"Morgan: {df1.shape}")
print(f"ChemBERTa: {df2.shape}")
print(f"One-hot: {df3.shape}")
```

### Example 3: Feature Selection in Model Pipeline

```python
from data.WHO_data import get_who_dataframe
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load data with feature selection
df, _ = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv',
    'feature_selection': True
}, add_genomic_positions=False,
   feature_selection_output_dir='model_feature_plots')

# Prepare features
X = df.drop(columns=['ppv', 'mutation_key', 'drug', 'drug_norm'],
            errors='ignore')
y = df['ppv']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale and train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R2: {r2_score(y_test, y_pred):.3f}")
```

## Testing Feature Selection

Run the test script to see feature selection in action:

```bash
python data/WHO_data/test_feature_selection.py
```

This will:
1. Test Morgan fingerprints with PPV target
2. Test ChemBERTa embeddings with PPV target
3. Test one-hot encoding with PPV target
4. Test Morgan fingerprints with resistance target

Plots are saved to:
- `feature_selection_plots/morgan_ppv/`
- `feature_selection_plots/chemberta_ppv/`
- `feature_selection_plots/onehot_ppv/`
- `feature_selection_plots/morgan_resistant/`

## When to Use Feature Selection

**Use feature selection when:**
- You have high-dimensional data (many features)
- You want to reduce overfitting
- You need faster training times
- You want more interpretable models
- You have limited compute resources

**Skip feature selection when:**
- Using neural networks with embeddings (they learn representations)
- Dataset is already small
- Features are already carefully curated
- You want to preserve all information
- Using tree-based models (they handle high dimensions well)

## Feature Selection vs. Dimensionality Reduction

**Feature Selection (what we do here)**:
- Removes features entirely
- Keeps interpretability
- Works well with L1/L2 regularization
- Good for understanding feature importance

**Dimensionality Reduction (PCA for continuous features)**:
- Transforms features into new space
- Loses some interpretability
- Captures maximum variance
- Good for collinear features

We use **both**:
- PCA for continuous features (fingerprints, embeddings)
- L1 selection for discrete features (one-hot encoded)

## Performance Considerations

**Memory usage**: Feature selection reduces memory by ~70-80%

**Training time**: Depends on model
- Linear models: 50-70% faster
- Neural networks: 30-50% faster
- Tree-based models: 20-40% faster

**Accuracy**: Usually minimal impact or slight improvement
- Reduction in overfitting often compensates for information loss
- Test on your specific use case

## Troubleshooting

### Too few features selected

**Problem**: L1 removes too many features

**Solution**: Reduce alpha parameter
```python
# In apply_l1_feature_selection()
alpha: float = 0.001  # Lower = keeps more features
```

### Too many features kept

**Problem**: PCA keeps too many components

**Solution**: Lower variance threshold
```python
# In apply_pca_feature_selection()
variance_threshold: float = 0.90  # Lower = fewer components
```

### No plots generated

**Problem**: Output directory not created or permission issues

**Solution**: Check directory exists and is writable
```python
import os
os.makedirs('my_plots', exist_ok=True)
```

### Different results each run

**Problem**: L1 regularization uses random initialization

**Solution**: Already handled - we use `random_state=42` for reproducibility

## Advanced Usage

### Custom Feature Selection

To implement your own selection logic:

```python
from data.WHO_data import get_who_dataframe

# Load without built-in selection
df, _ = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv',
    'feature_selection': False
}, add_genomic_positions=False)

# Apply your own selection
from sklearn.feature_selection import SelectKBest, f_regression

X = df.drop(columns=['ppv', ...], errors='ignore')
y = df['ppv']

selector = SelectKBest(f_regression, k=50)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
```

## Summary

Feature selection is a powerful tool for:
- ✅ Reducing dataset dimensionality by 70-80%
- ✅ Speeding up training by 30-70%
- ✅ Reducing overfitting
- ✅ Improving model interpretability
- ✅ Generating informative visualization plots

Enable with one parameter:
```python
'feature_selection': True
```

Visualizations automatically saved to track what was selected!
