"""
Test script for feature selection functionality.

This demonstrates how to use feature selection with the data loader API.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.WHO_data import get_who_dataframe


def test_feature_selection_morgan():
    """Test feature selection with Morgan fingerprints."""
    print("=" * 80)
    print("Test 1: Feature Selection with Morgan Fingerprints")
    print("=" * 80)

    # Without feature selection
    print("\nWithout feature selection:")
    df1, _ = get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'extract_features',
        'target': 'ppv'
    }, add_genomic_positions=False)
    print(f"Final shape: {df1.shape}")

    # With feature selection
    print("\n" + "=" * 80)
    print("With feature selection:")
    df2, _ = get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'extract_features',
        'target': 'ppv',
        'feature_selection': True
    }, add_genomic_positions=False,
       feature_selection_output_dir="feature_selection_plots/morgan_ppv")
    print(f"Final shape: {df2.shape}")

    print(f"\n✓ Reduced from {df1.shape[1]} to {df2.shape[1]} features")
    print(f"  ({df1.shape[1] - df2.shape[1]} features removed)")


def test_feature_selection_chemberta():
    """Test feature selection with ChemBERTa embeddings."""
    print("\n" + "=" * 80)
    print("Test 2: Feature Selection with ChemBERTa Embeddings")
    print("=" * 80)

    # Without feature selection
    print("\nWithout feature selection:")
    df1, _ = get_who_dataframe({
        'drug': 'chemberta',
        'mutation': 'extract_features',
        'target': 'ppv'
    }, add_genomic_positions=False)
    print(f"Final shape: {df1.shape}")

    # With feature selection
    print("\n" + "=" * 80)
    print("With feature selection:")
    df2, _ = get_who_dataframe({
        'drug': 'chemberta',
        'mutation': 'extract_features',
        'target': 'ppv',
        'feature_selection': True
    }, add_genomic_positions=False,
       feature_selection_output_dir="feature_selection_plots/chemberta_ppv")
    print(f"Final shape: {df2.shape}")

    print(f"\n✓ Reduced from {df1.shape[1]} to {df2.shape[1]} features")
    print(f"  ({df1.shape[1] - df2.shape[1]} features removed)")


def test_feature_selection_onehot():
    """Test feature selection with one-hot encoding."""
    print("\n" + "=" * 80)
    print("Test 3: Feature Selection with One-Hot Encoding")
    print("=" * 80)

    # Without feature selection
    print("\nWithout feature selection:")
    df1, _ = get_who_dataframe({
        'drug': 'one_hot',
        'mutation': 'extract_features',
        'target': 'ppv'
    }, add_genomic_positions=False)
    print(f"Final shape: {df1.shape}")

    # With feature selection
    print("\n" + "=" * 80)
    print("With feature selection:")
    df2, _ = get_who_dataframe({
        'drug': 'one_hot',
        'mutation': 'extract_features',
        'target': 'ppv',
        'feature_selection': True
    }, add_genomic_positions=False,
       feature_selection_output_dir="feature_selection_plots/onehot_ppv")
    print(f"Final shape: {df2.shape}")

    print(f"\n✓ Reduced from {df1.shape[1]} to {df2.shape[1]} features")
    print(f"  ({df1.shape[1] - df2.shape[1]} features removed)")


def test_feature_selection_resistant():
    """Test feature selection with resistance classification target."""
    print("\n" + "=" * 80)
    print("Test 4: Feature Selection with Resistance Target")
    print("=" * 80)

    # Without feature selection
    print("\nWithout feature selection:")
    df1, _ = get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'extract_features',
        'target': 'resistant'
    }, add_genomic_positions=False)
    print(f"Final shape: {df1.shape}")

    # With feature selection
    print("\n" + "=" * 80)
    print("With feature selection:")
    df2, _ = get_who_dataframe({
        'drug': 'morgan_fingerprint',
        'mutation': 'extract_features',
        'target': 'resistant',
        'feature_selection': True
    }, add_genomic_positions=False,
       feature_selection_output_dir="feature_selection_plots/morgan_resistant")
    print(f"Final shape: {df2.shape}")

    print(f"\n✓ Reduced from {df1.shape[1]} to {df2.shape[1]} features")
    print(f"  ({df1.shape[1] - df2.shape[1]} features removed)")


if __name__ == "__main__":
    print("=" * 80)
    print("FEATURE SELECTION TESTING")
    print("=" * 80)
    print("\nThis script tests feature selection with different encoding configurations.")
    print("Plots will be saved to feature_selection_plots/*/")
    print()

    # Run all tests
    test_feature_selection_morgan()
    test_feature_selection_chemberta()
    test_feature_selection_onehot()
    test_feature_selection_resistant()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nCheck the following directories for plots:")
    print("  - feature_selection_plots/morgan_ppv/")
    print("  - feature_selection_plots/chemberta_ppv/")
    print("  - feature_selection_plots/onehot_ppv/")
    print("  - feature_selection_plots/morgan_resistant/")
    print()
    print("Each directory contains:")
    print("  - PCA plots (for Morgan fingerprints and ChemBERTa)")
    print("  - L1 selection plots (for one-hot encoded features)")
    print()
