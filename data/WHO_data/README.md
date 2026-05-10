# WHO Data Loader

Flexible API for loading and encoding WHO tuberculosis resistance data with various encoding options.

## Quick Start

```python
from data.WHO_data.data_loader import get_who_dataframe

# Load data with specific encodings
df, drug_lookup = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv'
})
```

## Configuration Options

### Drug Encoding Options

- **`'one_hot'`**: One-hot encode drug names
- **`'morgan_fingerprint'`**: Use Morgan molecular fingerprints (256-bit)
- **`'chemberta'`**: Use ChemBERTa embeddings
- **`'trained_embedding'`**: Keep as categorical codes for embedding layer in neural networks

### Mutation Encoding Options

- **`'one_hot'` or `'extract_features'`**: Extract features from mutation strings (position, type, ref/alt amino acids) and one-hot encode
- **`'trained_embedding'`**: Keep as categorical codes for embedding layer in neural networks

### Target Variable Options

- **`'ppv'`**: Positive predictive value (continuous, 0-1)
- **`'resistant'`**: Resistance classification (0-3):
  - 0: Not associated with resistance
  - 1: Not associated with resistance (interim)
  - 2: Associated with resistance (interim)
  - 3: Associated with resistance

### Feature Selection Option

- **`'feature_selection': True`**: (optional) Apply automated feature selection
  - **PCA** for continuous features (Morgan fingerprints, ChemBERTa) - keeps components explaining 95% variance
  - **L1 regularization** for one-hot features (genes, mutations, drugs) - keeps non-zero coefficients
  - Generates visualization plots showing which features are selected/removed

## Additional Parameters

```python
df, drug_lookup = get_who_dataframe(
    config={
        'drug': '...',
        'mutation': '...',
        'target': '...',
        'feature_selection': True        # (optional) Apply feature selection
    },
    add_genomic_positions=True,          # Add genomic position features
    drop_rare_gene_threshold=10,         # Drop genes appearing < N times
    feature_selection_output_dir='plots' # Directory for feature selection plots
)
```

## Examples

### Example 1: PPV Prediction with Morgan Fingerprints

```python
from data.WHO_data.data_loader import get_who_dataframe

# Load dataset
df, _ = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv'
})

# Prepare for modeling
from data.WHO_data.data_loader import get_standard_splits

X_train, X_test, y_train, y_test = get_standard_splits(
    df,
    target_col='ppv',
    test_size=0.2,
    random_state=42
)
```

### Example 2: Resistance Classification with ChemBERTa

```python
from data.WHO_data.data_loader import get_resistance_dataset_with_chemberta

# Use convenience function
df, _ = get_resistance_dataset_with_chemberta()

# The dataset is ready for classification models
print(df['resistant'].value_counts())
```

### Example 3: Neural Network with Learned Embeddings

```python
from data.WHO_data.data_loader import get_embedding_ready_dataset

# Load dataset with categorical codes
df, drug_lookup = get_embedding_ready_dataset(target='ppv')

# Now df has 'drug_code' and 'mutation_code' columns
# ready for embedding layers in PyTorch or TensorFlow
print(f"Number of unique drugs: {len(drug_lookup)}")
print(f"Drug codes range: {df['drug_code'].min()} to {df['drug_code'].max()}")
```

### Example 4: Feature Selection

```python
from data.WHO_data.data_loader import get_who_dataframe

# Load with automated feature selection
df, _ = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv',
    'feature_selection': True  # Enable feature selection
}, add_genomic_positions=False,
   feature_selection_output_dir='my_feature_plots')

# Feature selection applied:
# - PCA on Morgan fingerprints (256 -> ~12 components for 95% variance)
# - L1 on mutation types, ref/alt amino acids, and genes
# - Plots saved to 'my_feature_plots/'

print(f"Dataset shape: {df.shape}")
# Typical reduction: 397 features -> ~87 features
```

**Feature Selection Details:**

The feature selection process automatically:
1. **For continuous features** (Morgan fingerprints, ChemBERTa):
   - Applies PCA and selects components explaining 95% of variance
   - Typically reduces 256 fingerprint features to ~12 components
   - Typically reduces 128 ChemBERTa features to ~20 components

2. **For one-hot encoded features** (genes, mutations, drugs):
   - Trains L1-regularized model (Lasso or LogisticRegression)
   - Keeps only features with non-zero coefficients
   - Removes features that don't contribute to predictions

3. **Generates visualization plots** for each feature group:
   - PCA: explained variance and cumulative variance plots
   - L1: coefficient magnitudes and distribution plots

## Convenience Functions

Pre-configured dataset loaders:

```python
from data.WHO_data.data_loader import (
    get_ppv_dataset_with_fingerprints,
    get_resistance_dataset_with_chemberta,
    get_embedding_ready_dataset
)

# PPV prediction with Morgan fingerprints
df1, _ = get_ppv_dataset_with_fingerprints()

# Resistance classification with ChemBERTa
df2, _ = get_resistance_dataset_with_chemberta()

# Dataset ready for embedding layers
df3, drug_lookup = get_embedding_ready_dataset('ppv')
```

## Migrating Existing Models

If you have existing model files using the old `full_data_pipeline()` approach:

### Old Code (from encoding.py)
```python
from encoding import full_data_pipeline

data, data_genomic_positions, drug_lookup = full_data_pipeline()
# Use data_genomic_positions for modeling
```

### New Code (with data_loader)
```python
from data.WHO_data.data_loader import get_who_dataframe

# Equivalent to full_data_pipeline with genomic positions
df, drug_lookup = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv'  # or 'resistant'
}, add_genomic_positions=True)
```

## Dataset Features

After loading with genomic positions enabled, the dataset includes:

- **Drug features**: Fingerprints, embeddings, or one-hot encoded
- **Mutation features**: Extracted features (position, type, ref/alt) or embeddings
- **Genomic position features**:
  - `norm_position`: Normalized position within gene (0-1)
  - `distance_to_end`: Distance to gene end
  - `codon_position`: Position within codon (0-2)
- **MIC features**:
  - `mean_log2mic_final`: Log2 MIC values from CRyPTIC consortium
  - `mic_count_final`: Number of MIC observations
  - `has_exact_mic`: Binary indicator for exact match
  - `has_variant_mic`: Binary indicator for variant-level match
- **Gene features**: One-hot encoded gene names (rare genes dropped)

## Notes

- The API automatically handles data cleaning, rare gene filtering, and feature engineering
- Drug lookups are returned when using `trained_embedding` encoding
- All relative paths are handled automatically - just import and use
- Progress messages are printed during data loading to track progress
