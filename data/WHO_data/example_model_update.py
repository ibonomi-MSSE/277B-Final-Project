"""
Example: How to update existing model files to use the new data_loader API.

This shows a side-by-side comparison of old vs new approaches for model_ANN_PPV.py
"""

# ============================================================================
# BEFORE: Using encoding.py full_data_pipeline()
# ============================================================================

def old_model_ann_ppv_main():
    """
    Original approach from model_ANN_PPV.py
    """
    from encoding import full_data_pipeline
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    # Get data using the old approach
    data, data_genomic_positions, drug_lookup = full_data_pipeline()

    # Prepare features
    X = data_genomic_positions.drop(
        columns=["ppv", "ppv_bin", "resistant", "mutation_key", "drug", "drug_norm"],
        errors="ignore"
    )
    y = data_genomic_positions["ppv"]
    groups = data_genomic_positions["mutation_key"].fillna("unknown_mutation")

    # Rest of model code...
    print(f"Old approach - Data shape: {X.shape}")
    return X, y, groups


# ============================================================================
# AFTER: Using new data_loader API
# ============================================================================

def new_model_ann_ppv_main():
    """
    Updated approach using data_loader.get_who_dataframe()
    """
    from data.WHO_data import get_who_dataframe
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    # Get data using the new flexible API
    data, drug_lookup = get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'extract_features',
        'target': 'ppv'
    }, add_genomic_positions=True)  # Note: requires genome annotation file

    # Prepare features (same as before)
    X = data.drop(
        columns=["ppv", "ppv_bin", "resistant", "mutation_key", "drug", "drug_norm"],
        errors="ignore"
    )
    y = data["ppv"]
    groups = data["mutation_key"].fillna("unknown_mutation")

    # Rest of model code...
    print(f"New approach - Data shape: {X.shape}")
    return X, y, groups


# ============================================================================
# BENEFITS OF NEW APPROACH
# ============================================================================

"""
1. EXPLICIT CONFIGURATION
   - Clear what encodings are used
   - Easy to experiment with different encodings
   - Self-documenting code

2. FLEXIBILITY
   - Try ChemBERTa instead of Morgan fingerprints with one line change:
     'drug': 'chemberta' instead of 'morgan_fingerprint'

   - Try different targets:
     'target': 'resistant' instead of 'ppv'

   - Use embeddings for neural networks:
     'drug': 'trained_embedding', 'mutation': 'trained_embedding'

3. CONSISTENCY
   - All models use the same API
   - Easier to compare model results
   - Reduces code duplication

4. MAINTAINABILITY
   - Changes to data pipeline are centralized
   - No need to modify encoding.py for new encodings
   - Better separation of concerns
"""

# ============================================================================
# EXAMPLE: Multiple model configurations
# ============================================================================

def compare_drug_encodings():
    """
    Example: Compare different drug encodings for the same model
    """
    from data.WHO_data import get_who_dataframe

    configs = [
        {'drug': 'morgan_fingerprint', 'mutation': 'extract_features', 'target': 'ppv'},
        {'drug': 'chemberta', 'mutation': 'extract_features', 'target': 'ppv'},
        {'drug': 'one_hot', 'mutation': 'extract_features', 'target': 'ppv'},
    ]

    for i, config in enumerate(configs, 1):
        df, _ = get_who_dataframe(config, add_genomic_positions=False)
        print(f"\nConfig {i}: {config['drug']}")
        print(f"  Shape: {df.shape}")
        print(f"  Feature count: {df.shape[1] - 5}")  # Subtract metadata columns


# ============================================================================
# MINIMAL CHANGES NEEDED TO UPDATE MODEL FILES
# ============================================================================

"""
To update a model file like model_ANN_PPV.py:

1. Change the import:
   FROM: from encoding import full_data_pipeline
   TO:   from data.WHO_data import get_who_dataframe

2. Change the data loading:
   FROM: data, data_genomic_positions, drug_lookup = full_data_pipeline()
   TO:   data, drug_lookup = get_who_dataframe({
             'drug': 'morgan_fingerprint',
             'mutation': 'extract_features',
             'target': 'ppv'
         }, add_genomic_positions=True)

3. Update variable name:
   FROM: data_genomic_positions
   TO:   data

That's it! The rest of the model code stays the same.
"""

if __name__ == "__main__":
    import sys
    import os
    # Add project root to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    print("=" * 80)
    print("Comparing old vs new data loading approaches")
    print("=" * 80)

    # Uncomment to test old approach (requires encoding.py)
    # print("\nOld approach:")
    # X_old, y_old, groups_old = old_model_ann_ppv_main()

    print("\nNew approach (without genomic positions):")
    # Test without genomic positions since genome file is missing
    from data.WHO_data import get_who_dataframe

    data, _ = get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'extract_features',
        'target': 'ppv'
    }, add_genomic_positions=False)

    X = data.drop(columns=["ppv", "ppv_bin", "resistant", "mutation_key", "drug", "drug_norm"], errors="ignore")
    y = data["ppv"]
    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"PPV range: [{y.min():.3f}, {y.max():.3f}]")

    print("\n" + "=" * 80)
    print("Comparing different drug encodings")
    print("=" * 80)
    compare_drug_encodings()

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
