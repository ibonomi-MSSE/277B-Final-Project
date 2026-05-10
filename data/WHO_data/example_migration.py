"""
Example migration from old encoding.py approach to new data_loader API.

This file demonstrates how to update existing model files to use the new standardized data loading API.
"""

# ============================================================================
# OLD APPROACH (from existing model files)
# ============================================================================

def old_approach():
    """
    Example of the old approach used in model files like model_ANN_PPV.py
    """
    from encoding import full_data_pipeline

    # Old way: hardcoded pipeline that does everything
    data, data_genomic_positions, drug_lookup = full_data_pipeline()

    # Data includes:
    # - One-hot encoded mutations
    # - Morgan fingerprints for drugs
    # - ChemBERTa embeddings for drugs
    # - Both versions with/without genomic positions

    return data_genomic_positions


# ============================================================================
# NEW APPROACH (with data_loader)
# ============================================================================

def new_approach_ppv_morgan():
    """
    New approach: Flexible configuration for PPV prediction with Morgan fingerprints
    """
    from data.WHO_data.data_loader import get_who_dataframe

    df, drug_lookup = get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'extract_features',
        'target': 'ppv'
    }, add_genomic_positions=True)

    return df


def new_approach_resistance_chemberta():
    """
    New approach: Resistance classification with ChemBERTa embeddings
    """
    from data.WHO_data.data_loader import get_who_dataframe

    df, drug_lookup = get_who_dataframe({
        'drug': 'chemberta',
        'mutation': 'extract_features',
        'target': 'resistant'
    }, add_genomic_positions=True)

    return df


def new_approach_embedding_layer():
    """
    New approach: For neural networks with embedding layers
    """
    from data.WHO_data.data_loader import get_who_dataframe

    df, drug_lookup = get_who_dataframe({
        'drug': 'trained_embedding',
        'mutation': 'trained_embedding',
        'target': 'ppv'
    }, add_genomic_positions=True)

    # Now df has 'drug_code' and 'mutation_code' columns
    # ready for embedding layers
    return df, drug_lookup


# ============================================================================
# MIGRATION EXAMPLE: Updating a model file
# ============================================================================

# Before (model_ANN_PPV.py):
"""
from encoding import full_data_pipeline
...
def main():
    data, data_genomic_positions, drug_lookup = full_data_pipeline()
    ann_model_ppv = ANN_model_PPV(data_genomic_positions)
"""

# After (with data_loader):
"""
from data.WHO_data.data_loader import get_who_dataframe
...
def main():
    data, drug_lookup = get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'extract_features',
        'target': 'ppv'
    }, add_genomic_positions=True)

    ann_model_ppv = ANN_model_PPV(data)
"""

# ============================================================================
# BENEFITS OF NEW APPROACH
# ============================================================================

"""
1. **Flexibility**: Choose encoding methods per model without changing global pipeline
2. **Clarity**: Explicit configuration makes it clear what each model uses
3. **Reusability**: Easy to experiment with different encodings
4. **Consistency**: Standard API across all models
5. **Documentation**: Clear parameter options and examples

Example use cases:

# Compare Morgan fingerprints vs ChemBERTa for same model
df1, _ = get_who_dataframe({'drug': 'morgan_fingerprint', 'mutation': 'extract_features', 'target': 'ppv'})
df2, _ = get_who_dataframe({'drug': 'chemberta', 'mutation': 'extract_features', 'target': 'ppv'})

# Test embedding layers
df3, drug_lookup = get_who_dataframe({'drug': 'trained_embedding', 'mutation': 'trained_embedding', 'target': 'ppv'})

# Quick one-liners with convenience functions
from data.WHO_data.data_loader import get_ppv_dataset_with_fingerprints
df, _ = get_ppv_dataset_with_fingerprints()
"""

if __name__ == "__main__":
    print("Running migration examples...")

    print("\n1. Old approach:")
    # Note: This requires all the dependencies from encoding.py
    # old_df = old_approach()
    # print(f"   Shape: {old_df.shape}")

    print("\n2. New approach - PPV with Morgan fingerprints:")
    new_df1 = new_approach_ppv_morgan()
    print(f"   Shape: {new_df1.shape}")
    print(f"   Has fingerprints: {any('fp_' in col for col in new_df1.columns)}")

    print("\n3. New approach - Resistance with ChemBERTa:")
    new_df2 = new_approach_resistance_chemberta()
    print(f"   Shape: {new_df2.shape}")
    print(f"   Has embeddings: {any('drug_embedding_' in col for col in new_df2.columns)}")
    print(f"   Target distribution: {new_df2['resistant'].value_counts().to_dict()}")

    print("\n4. New approach - With embedding layers:")
    new_df3, drug_lookup = new_approach_embedding_layer()
    print(f"   Shape: {new_df3.shape}")
    print(f"   Has drug_code: {'drug_code' in new_df3.columns}")
    print(f"   Has mutation_code: {'mutation_code' in new_df3.columns}")
    print(f"   Num drugs: {len(drug_lookup)}")

    print("\nMigration examples complete!")
