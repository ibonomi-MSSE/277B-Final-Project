"""
Flexible data loading API for WHO tuberculosis resistance data.

This module provides a standardized interface for loading and encoding the WHO dataset
with various encoding options for drugs, mutations, and target variables.

Example usage:
    df = get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'one_hot',
        'target': 'ppv'
    })
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Literal, Optional, Tuple, List
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add parent directories to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from load_data import main as load_raw_data

# Type aliases for valid options
DrugEncoding = Literal['one_hot', 'morgan_fingerprint', 'chemberta', 'trained_embedding']
MutationEncoding = Literal['one_hot', 'extract_features', 'trained_embedding']
TargetVariable = Literal['ppv', 'resistant']


def _ensure_in_project_root():
    """Helper to ensure we're running from project root for relative paths."""
    import inspect
    current_file = inspect.getfile(inspect.currentframe())
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '../..'))
    if os.getcwd() != project_root:
        os.chdir(project_root)


def encode_target(data: pd.DataFrame, target: TargetVariable) -> pd.DataFrame:
    """
    Encode the target variable.

    Args:
        data: Input dataframe
        target: Target variable to use ('ppv' or 'resistant')

    Returns:
        DataFrame with encoded target
    """
    if target == 'resistant':
        # Drop uncertain resistance
        df = data[data['FINAL CONFIDENCE GRADING'] != '3) Uncertain significance'].copy()

        grading_scheme = {
            "5) Not assoc w R": 0,
            "4) Not assoc w R - Interim": 1,
            "2) Assoc w R - Interim": 2,
            "1) Assoc w R": 3
        }

        df["resistant"] = df["FINAL CONFIDENCE GRADING"].map(grading_scheme)
        df = df.dropna(subset=["resistant"]).copy()
        df = df.drop(columns=["FINAL CONFIDENCE GRADING"])

        return df

    elif target == 'ppv':
        # PPV is already in the data, just need to ensure it exists
        if 'ppv' not in data.columns:
            raise ValueError("PPV column not found in data")
        return data

    else:
        raise ValueError(f"Unknown target: {target}. Must be 'ppv' or 'resistant'")


def encode_mutations_one_hot(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode mutations with feature extraction."""
    from encoding import encode_mutations
    return encode_mutations(data)


def encode_drug_one_hot(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """One-hot encode drug names."""
    lookup_drugs = dict(enumerate(data["drug"].astype("category").cat.categories))
    data = pd.get_dummies(data, columns=['drug'], prefix='drug')
    return data, lookup_drugs


def encode_drug_morgan_fingerprint(data: pd.DataFrame) -> pd.DataFrame:
    """Encode drugs using Morgan fingerprints."""
    from encoding import get_drug_smiles

    data = get_drug_smiles(data)

    # Expand fingerprint into separate columns
    encoded_df = pd.DataFrame(data["fingerprint"].tolist(), index=data.index)
    encoded_df.columns = [f"fp_{i}" for i in range(encoded_df.shape[1])]

    data = pd.concat([data.drop(columns=["fingerprint", "smiles"]), encoded_df], axis=1)
    data = data.loc[:, ~data.columns.duplicated()].copy()

    return data


def encode_drug_chemberta(data: pd.DataFrame) -> pd.DataFrame:
    """Encode drugs using ChemBERTa embeddings."""
    from encoding import ChemBerta_embedding
    return ChemBerta_embedding(data)


def encode_drug_trained_embedding(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Keep drug as categorical code for embedding layer."""
    lookup_drugs = dict(enumerate(data["drug"].astype("category").cat.categories))
    data["drug_code"] = data["drug"].astype("category").cat.codes

    # Save lookup for reference
    with open("Drug_lookup.txt", "w") as f:
        f.write(f"Drug encoding map: {lookup_drugs}")

    # Drop original drug column to avoid confusion
    data = data.drop(columns=["drug"])

    return data, lookup_drugs


def encode_mutation_trained_embedding(data: pd.DataFrame) -> pd.DataFrame:
    """Keep mutation as categorical for embedding layer."""
    data["mutation_code"] = data["mutation"].astype("category").cat.codes
    return data


def add_genomic_position_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add genomic position features."""
    from encoding import genomic_positions
    return genomic_positions(data)


def drop_rare_genes_func(data: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """Drop genes that appear less than threshold times."""
    # Replicate drop_rare_genes logic but handle both ppv and resistant targets
    gene_cols = [c for c in data.columns if c.startswith("gene_")]
    rare = [c for c in gene_cols if data[c].sum() < threshold]

    data = data.drop(columns=rare)

    # Handle reference/alternative nucleotides if they exist
    if "reference_nucleotide" in data.columns and "alternative_nucleotide" in data.columns:
        data["ref_len"] = data["reference_nucleotide"].astype(str).str.len()
        data["alt_len"] = data["alternative_nucleotide"].astype(str).str.len()

        data["ref_base"] = data["reference_nucleotide"].where(
            data["ref_len"] == 1, "LONG"
        )
        data["alt_base"] = data["alternative_nucleotide"].where(
            data["alt_len"] == 1, "LONG"
        )
        data = data.drop(
            columns=["reference_nucleotide", "alternative_nucleotide"],
            errors="ignore"
        )

        data = pd.get_dummies(data, columns=["ref_base", "alt_base"])

    # Drop rows with NaN in target columns (only if they exist)
    target_cols = ['resistant', 'ppv']
    existing_targets = [col for col in target_cols if col in data.columns]
    if existing_targets:
        data = data.dropna(subset=existing_targets).copy()

    return data


def apply_pca_feature_selection(
    data: pd.DataFrame,
    feature_cols: List[str],
    variance_threshold: float = 0.95,
    feature_type: str = "features",
    output_dir: str = "feature_selection_plots"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply PCA to continuous features and select components explaining variance_threshold.

    Args:
        data: Input dataframe
        feature_cols: List of column names to apply PCA to
        variance_threshold: Cumulative variance to retain (default 0.95)
        feature_type: Name for plotting (e.g., 'Morgan Fingerprints', 'ChemBERTa')
        output_dir: Directory to save plots

    Returns:
        Tuple of (transformed dataframe, list of selected component names)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract features
    X = data[feature_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Calculate cumulative variance
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1

    print(f"  PCA on {feature_type}:")
    print(f"    Original features: {len(feature_cols)}")
    print(f"    Components for {variance_threshold*100:.0f}% variance: {n_components}")
    print(f"    Reduction: {len(feature_cols) - n_components} features removed")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Explained variance ratio
    ax1.plot(range(1, min(51, len(cumsum_variance)+1)),
             pca.explained_variance_ratio_[:50], 'bo-', markersize=4)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'{feature_type}: Explained Variance per Component')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(51, len(cumsum_variance)+1))

    # Plot 2: Cumulative variance
    ax2.plot(range(1, len(cumsum_variance)+1), cumsum_variance, 'ro-', markersize=3)
    ax2.axhline(y=variance_threshold, color='g', linestyle='--',
                label=f'{variance_threshold*100:.0f}% threshold')
    ax2.axvline(x=n_components, color='b', linestyle='--',
                label=f'{n_components} components')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title(f'{feature_type}: Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(100, len(cumsum_variance)))

    plt.tight_layout()
    safe_name = feature_type.lower().replace(' ', '_').replace('/', '_')
    plt.savefig(f"{output_dir}/pca_{safe_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Keep only selected components
    component_names = [f"pca_{safe_name}_{i}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca[:, :n_components],
                          index=data.index,
                          columns=component_names)

    # Drop original features and add PCA components
    data = data.drop(columns=feature_cols)
    data = pd.concat([data, pca_df], axis=1)

    return data, component_names


def apply_l1_feature_selection(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    feature_type: str = "features",
    alpha: float = 0.01,
    output_dir: str = "feature_selection_plots"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply L1 regularization to select one-hot encoded features.

    Args:
        data: Input dataframe
        feature_cols: List of column names to apply L1 selection to
        target_col: Target column name for training
        feature_type: Name for plotting (e.g., 'Gene Features', 'Mutation Features')
        alpha: L1 regularization strength (higher = more sparsity)
        output_dir: Directory to save plots

    Returns:
        Tuple of (dataframe with selected features, list of selected feature names)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract features and target
    X = data[feature_cols].values
    y = data[target_col].values

    # Determine if classification or regression
    is_classification = target_col == 'resistant'

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply L1 regularization
    if is_classification:
        model = LogisticRegression(penalty='l1', solver='liblinear', C=1/alpha,
                                   max_iter=1000, random_state=42)
    else:
        model = Lasso(alpha=alpha, max_iter=1000, random_state=42)

    model.fit(X_scaled, y)

    # Get coefficients
    if is_classification:
        if len(model.coef_.shape) > 1:
            # Multi-class: take max absolute coefficient across classes
            coefficients = np.max(np.abs(model.coef_), axis=0)
        else:
            coefficients = np.abs(model.coef_.ravel())
    else:
        coefficients = np.abs(model.coef_)

    # Select non-zero coefficients
    selected_mask = coefficients > 0
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]

    print(f"  L1 selection on {feature_type}:")
    print(f"    Original features: {len(feature_cols)}")
    print(f"    Selected features: {len(selected_features)}")
    print(f"    Removed: {len(feature_cols) - len(selected_features)} features")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Coefficient magnitudes (sorted)
    sorted_indices = np.argsort(coefficients)[::-1]
    sorted_coeffs = coefficients[sorted_indices]

    ax1.bar(range(len(sorted_coeffs)), sorted_coeffs, color='steelblue', alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero threshold')
    ax1.set_xlabel('Feature Index (sorted by coefficient)')
    ax1.set_ylabel('|Coefficient|')
    ax1.set_title(f'{feature_type}: L1 Coefficient Magnitudes')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Histogram of coefficients
    ax2.hist(coefficients, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero threshold')
    ax2.set_xlabel('|Coefficient|')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{feature_type}: Coefficient Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add text box with stats
    stats_text = f'Selected: {len(selected_features)}/{len(feature_cols)}\n'
    stats_text += f'Removed: {len(feature_cols) - len(selected_features)}'
    ax2.text(0.98, 0.98, stats_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    safe_name = feature_type.lower().replace(' ', '_').replace('/', '_')
    plt.savefig(f"{output_dir}/l1_{safe_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Keep only selected features
    data = data.drop(columns=[f for f in feature_cols if f not in selected_features])

    return data, selected_features


def apply_feature_selection(
    data: pd.DataFrame,
    drug_encoding: str,
    mutation_encoding: str,
    target_col: str,
    output_dir: str = "feature_selection_plots"
) -> pd.DataFrame:
    """
    Apply appropriate feature selection based on encoding types.

    Args:
        data: Input dataframe
        drug_encoding: Drug encoding method used
        mutation_encoding: Mutation encoding method used
        target_col: Target column name
        output_dir: Directory to save plots

    Returns:
        DataFrame with selected features
    """
    print("\nApplying feature selection...")
    os.makedirs(output_dir, exist_ok=True)

    # Feature selection for drugs
    if drug_encoding == 'morgan_fingerprint':
        fp_cols = [c for c in data.columns if c.startswith('fp_')]
        if fp_cols:
            data, _ = apply_pca_feature_selection(
                data, fp_cols,
                variance_threshold=0.95,
                feature_type="Morgan Fingerprints",
                output_dir=output_dir
            )

    elif drug_encoding == 'chemberta':
        emb_cols = [c for c in data.columns if c.startswith('drug_embedding_')]
        if emb_cols:
            data, _ = apply_pca_feature_selection(
                data, emb_cols,
                variance_threshold=0.95,
                feature_type="ChemBERTa Embeddings",
                output_dir=output_dir
            )

    elif drug_encoding == 'one_hot':
        drug_cols = [c for c in data.columns if c.startswith('drug_')]
        if drug_cols:
            data, _ = apply_l1_feature_selection(
                data, drug_cols, target_col,
                feature_type="Drug One-Hot",
                alpha=0.01,
                output_dir=output_dir
            )

    # Feature selection for mutations
    if mutation_encoding in ['one_hot', 'extract_features']:
        # One-hot encoded mutation types
        mut_type_cols = [c for c in data.columns if c.startswith('mut_type_')]
        if mut_type_cols:
            data, _ = apply_l1_feature_selection(
                data, mut_type_cols, target_col,
                feature_type="Mutation Type",
                alpha=0.01,
                output_dir=output_dir
            )

        # One-hot encoded reference amino acids
        ref_cols = [c for c in data.columns if c.startswith('ref_')]
        if ref_cols:
            data, _ = apply_l1_feature_selection(
                data, ref_cols, target_col,
                feature_type="Reference AA",
                alpha=0.01,
                output_dir=output_dir
            )

        # One-hot encoded alternative amino acids
        alt_cols = [c for c in data.columns if c.startswith('alt_')]
        if alt_cols:
            data, _ = apply_l1_feature_selection(
                data, alt_cols, target_col,
                feature_type="Alternative AA",
                alpha=0.01,
                output_dir=output_dir
            )

    # Feature selection for genes (always one-hot)
    gene_cols = [c for c in data.columns if c.startswith('gene_')]
    if gene_cols:
        data, _ = apply_l1_feature_selection(
            data, gene_cols, target_col,
            feature_type="Gene Features",
            alpha=0.01,
            output_dir=output_dir
        )

    print(f"\nFeature selection complete. Plots saved to {output_dir}/")
    return data


def get_who_dataframe(
    config: Dict[str, str],
    add_genomic_positions: bool = True,
    drop_rare_gene_threshold: Optional[int] = 10,
    feature_selection: bool = False,
    feature_selection_output_dir: str = "feature_selection_plots"
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Load and encode WHO tuberculosis resistance data with flexible encoding options.

    Args:
        config: Dictionary specifying encoding options:
            - 'drug': Drug encoding method ('one_hot', 'morgan_fingerprint', 'chemberta', 'trained_embedding')
            - 'mutation': Mutation encoding method ('one_hot', 'extract_features', 'trained_embedding')
            - 'target': Target variable ('ppv', 'resistant')
            - 'feature_selection': (optional) Enable feature selection (default: False)
        add_genomic_positions: Whether to add genomic position features (default True)
        drop_rare_gene_threshold: Minimum count threshold for genes to keep (default 10)
        feature_selection: Apply PCA/L1 feature selection (default False)
        feature_selection_output_dir: Directory for feature selection plots

    Returns:
        Tuple of (processed DataFrame, drug_lookup dict if applicable)

    Example:
        >>> df, drug_lookup = get_who_dataframe({
        ...     'drug': 'morgan_fingerprint',
        ...     'mutation': 'extract_features',
        ...     'target': 'ppv'
        ... })
    """
    # Ensure we're in project root for file paths
    _ensure_in_project_root()

    # Validate config
    required_keys = ['drug', 'mutation', 'target']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    drug_encoding = config['drug']
    mutation_encoding = config['mutation']
    target = config['target']

    # Load raw data
    print(f"Loading raw data...")
    data = load_raw_data()
    print(f"Raw data shape: {data.shape}")

    # Encode target first (may filter rows)
    print(f"Encoding target: {target}...")
    data = encode_target(data, target)
    print(f"After target encoding: {data.shape}")

    # Encode mutations
    print(f"Encoding mutations: {mutation_encoding}...")
    if mutation_encoding == 'one_hot' or mutation_encoding == 'extract_features':
        data = encode_mutations_one_hot(data)
    elif mutation_encoding == 'trained_embedding':
        data = encode_mutation_trained_embedding(data)
    else:
        raise ValueError(f"Unknown mutation encoding: {mutation_encoding}")
    print(f"After mutation encoding: {data.shape}")

    # Encode drugs
    print(f"Encoding drugs: {drug_encoding}...")
    drug_lookup = None
    if drug_encoding == 'one_hot':
        data, drug_lookup = encode_drug_one_hot(data)
    elif drug_encoding == 'morgan_fingerprint':
        data = encode_drug_morgan_fingerprint(data)
    elif drug_encoding == 'chemberta':
        data = encode_drug_chemberta(data)
    elif drug_encoding == 'trained_embedding':
        data, drug_lookup = encode_drug_trained_embedding(data)
    else:
        raise ValueError(f"Unknown drug encoding: {drug_encoding}")
    print(f"After drug encoding: {data.shape}")

    # Drop rare genes if threshold specified
    if drop_rare_gene_threshold is not None:
        print(f"Dropping rare genes (threshold={drop_rare_gene_threshold})...")
        data = drop_rare_genes_func(data, threshold=drop_rare_gene_threshold)
        print(f"After dropping rare genes: {data.shape}")

    # Add genomic positions if requested
    if add_genomic_positions:
        print("Adding genomic position features...")
        data = add_genomic_position_features(data)
        print(f"After adding genomic positions: {data.shape}")

    # Clean up duplicates and remove intermediate columns
    data = data.loc[:, ~data.columns.duplicated()].copy()

    # Drop columns that are typically not used in modeling
    columns_to_drop = [
        "mean_log2mic",
        "median_log2mic",
        "count",
        "mean_log2mic_variant",
        "median_log2mic_variant",
        "count_variant"
    ]
    data = data.drop(columns=columns_to_drop, errors="ignore")

    # Apply feature selection if requested (or if in config)
    if feature_selection or config.get('feature_selection', False):
        print(f"\nBefore feature selection: {data.shape}")
        data = apply_feature_selection(
            data,
            drug_encoding,
            mutation_encoding,
            target,
            output_dir=feature_selection_output_dir
        )
        print(f"After feature selection: {data.shape}")

    print(f"\nFinal dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns[:20])}{'...' if len(data.columns) > 20 else ''}")

    return data, drug_lookup


def get_standard_splits(
    df: pd.DataFrame,
    target_col: str = 'ppv',
    exclude_cols: list = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Get standard train/test splits for the WHO dataset.

    Args:
        df: Processed dataframe from get_who_dataframe
        target_col: Name of target column
        exclude_cols: Additional columns to exclude from features
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    # Separate features and target
    default_exclude = ['ppv', 'resistant', 'mutation_key', 'drug', 'drug_norm']
    if exclude_cols:
        default_exclude.extend(exclude_cols)

    feature_cols = [c for c in df.columns if c not in default_exclude and c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Convenience functions for common configurations
def get_ppv_dataset_with_fingerprints() -> Tuple[pd.DataFrame, None]:
    """Load dataset for PPV prediction with Morgan fingerprints."""
    return get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'extract_features',
        'target': 'ppv'
    })


def get_resistance_dataset_with_chemberta() -> Tuple[pd.DataFrame, None]:
    """Load dataset for resistance classification with ChemBERTa embeddings."""
    return get_who_dataframe({
        'drug': 'chemberta',
        'mutation': 'extract_features',
        'target': 'resistant'
    })


def get_embedding_ready_dataset(target: TargetVariable = 'ppv') -> Tuple[pd.DataFrame, dict]:
    """Load dataset with categorical codes ready for embedding layers."""
    return get_who_dataframe({
        'drug': 'trained_embedding',
        'mutation': 'trained_embedding',
        'target': target
    })


if __name__ == "__main__":
    # Test the API
    print("="*80)
    print("Testing flexible data loading API")
    print("="*80)

    print("\n" + "="*80)
    print("Test 1: PPV dataset with Morgan fingerprints")
    print("="*80)
    df1, _ = get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'extract_features',
        'target': 'ppv'
    })
    print(f"\n✓ Success! Shape: {df1.shape}")
    print(f"  Sample columns: {list(df1.columns[:10])}")
    print(f"  Target range: [{df1['ppv'].min():.3f}, {df1['ppv'].max():.3f}]")

    print("\n" + "="*80)
    print("Test 2: Resistance dataset with ChemBERTa")
    print("="*80)
    df2, _ = get_who_dataframe({
        'drug': 'chemberta',
        'mutation': 'extract_features',
        'target': 'resistant'
    })
    print(f"\n✓ Success! Shape: {df2.shape}")
    print(f"  Target distribution:\n{df2['resistant'].value_counts()}")

    print("\n" + "="*80)
    print("Test 3: Dataset ready for embeddings")
    print("="*80)
    df3, drug_lookup = get_embedding_ready_dataset('ppv')
    print(f"\n✓ Success! Shape: {df3.shape}")
    print(f"  Has drug_code: {'drug_code' in df3.columns}")
    print(f"  Has mutation_code: {'mutation_code' in df3.columns}")
    if drug_lookup:
        print(f"  Number of drugs: {len(drug_lookup)}")

    print("\n" + "="*80)
    print("API tests complete!")
    print("="*80)
