"""
Calculate Tanimoto coefficients between drug molecules using molecular fingerprints.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pubchempy as pcp


# TB drug names
DRUG_NAMES = [
    'Amikacin', 'Bedaquiline', 'Capreomycin', 'Clofazimine', 'Delamanid',
    'Ethambutol', 'Ethionamide', 'Isoniazid', 'Kanamycin', 'Levofloxacin',
    'Linezolid', 'Moxifloxacin', 'Pyrazinamide', 'Rifampicin', 'Streptomycin'
]


def get_smiles_from_pubchem(drug_name):
    """Get SMILES string for a drug from PubChem."""
    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            return compounds[0].canonical_smiles
    except:
        pass
    return None


def calculate_tanimoto_coefficients(drug_names):
    """
    Calculate Tanimoto coefficients between all pairs of drugs.

    Returns:
        dict: Dictionary mapping drug names to SMILES
        np.ndarray: Matrix of Tanimoto coefficients
        list: Drug names (in order)
    """
    print("Fetching SMILES from PubChem...")
    smiles_dict = {}
    for drug in drug_names:
        smiles = get_smiles_from_pubchem(drug)
        if smiles:
            smiles_dict[drug] = smiles
            print(f"  {drug}: {smiles[:50]}...")
        else:
            print(f"  {drug}: Could not retrieve SMILES")

    print(f"\nSuccessfully retrieved SMILES for {len(smiles_dict)}/{len(drug_names)} drugs")

    # Calculate fingerprints
    print("\nCalculating molecular fingerprints...")
    fps = {}
    valid_drugs = []
    for drug, smiles in smiles_dict.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Morgan fingerprint (equivalent to ECFP)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps[drug] = fp
            valid_drugs.append(drug)
            print(f"  {drug}: fingerprint generated")

    # Calculate Tanimoto coefficients
    print("\nCalculating Tanimoto coefficients...")
    n_drugs = len(valid_drugs)
    tanimoto_matrix = np.zeros((n_drugs, n_drugs))

    for i, drug1 in enumerate(valid_drugs):
        for j, drug2 in enumerate(valid_drugs):
            if i == j:
                tanimoto_matrix[i, j] = 1.0
            else:
                tanimoto = DataStructs.TanimotoSimilarity(fps[drug1], fps[drug2])
                tanimoto_matrix[i, j] = tanimoto

    return smiles_dict, tanimoto_matrix, valid_drugs


def print_top_similarities(tanimoto_matrix, drug_names, top_n=10):
    """Print the top N most similar drug pairs."""
    pairs = []
    n = len(drug_names)

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((drug_names[i], drug_names[j], tanimoto_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP {top_n} MOST SIMILAR DRUG PAIRS (by Tanimoto coefficient)")
    print(f"{'='*70}")
    for idx, (drug1, drug2, coef) in enumerate(pairs[:top_n], 1):
        print(f"{idx:2d}. {drug1:15s} <-> {drug2:15s}  Tanimoto: {coef:.4f}")

    print(f"\n{'='*70}")
    print(f"TOP {top_n} LEAST SIMILAR DRUG PAIRS (by Tanimoto coefficient)")
    print(f"{'='*70}")
    for idx, (drug1, drug2, coef) in enumerate(pairs[-top_n:], 1):
        print(f"{idx:2d}. {drug1:15s} <-> {drug2:15s}  Tanimoto: {coef:.4f}")


def print_aminoglycoside_comparisons(tanimoto_matrix, drug_names):
    """Print comparisons specifically for the aminoglycosides."""
    aminoglycosides = ['Amikacin', 'Kanamycin', 'Streptomycin']

    print(f"\n{'='*70}")
    print("AMINOGLYCOSIDE TANIMOTO COEFFICIENTS")
    print("(These are the drugs held out by scaffold splitting)")
    print(f"{'='*70}")

    for i, drug1 in enumerate(aminoglycosides):
        if drug1 not in drug_names:
            continue
        idx1 = drug_names.index(drug1)
        for j, drug2 in enumerate(aminoglycosides):
            if i >= j or drug2 not in drug_names:
                continue
            idx2 = drug_names.index(drug2)
            coef = tanimoto_matrix[idx1, idx2]
            print(f"  {drug1:15s} <-> {drug2:15s}  Tanimoto: {coef:.4f}")


def create_comparison_table(tanimoto_matrix, cosine_matrix, drug_names):
    """Create a comparison between Tanimoto and cosine similarity."""
    pairs = []
    n = len(drug_names)

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'Drug 1': drug_names[i],
                'Drug 2': drug_names[j],
                'Tanimoto': tanimoto_matrix[i, j],
                'Cosine': cosine_matrix[i, j],
                'Difference': abs(tanimoto_matrix[i, j] - cosine_matrix[i, j])
            })

    df = pd.DataFrame(pairs)
    df = df.sort_values('Tanimoto', ascending=False).reset_index(drop=True)

    print(f"\n{'='*90}")
    print("COMPARISON: TANIMOTO vs COSINE SIMILARITY (Top 10)")
    print(f"{'='*90}")
    print(df.head(10).to_string(index=False))

    return df


def main():
    print("Starting Tanimoto coefficient analysis for TB drugs...")
    print(f"Analyzing {len(DRUG_NAMES)} drugs\n")

    smiles_dict, tanimoto_matrix, valid_drugs = calculate_tanimoto_coefficients(DRUG_NAMES)

    # Print top similarities
    print_top_similarities(tanimoto_matrix, valid_drugs, top_n=10)

    # Print aminoglycoside comparisons
    print_aminoglycoside_comparisons(tanimoto_matrix, valid_drugs)

    # Load cosine similarities from previous analysis
    print("\n" + "="*70)
    print("LOADING COSINE SIMILARITIES FROM EMBEDDINGS...")
    print("="*70)

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from feature_encoding.chemBERTa_mtr_embeddings import get_drug_embeddings
    from sklearn.metrics.pairwise import cosine_similarity

    embeddings = get_drug_embeddings()
    drug_names_embedding = [d for d in valid_drugs if d in embeddings]

    # Reorder to match valid_drugs
    cosine_matrix = np.zeros((len(valid_drugs), len(valid_drugs)))
    embedding_matrix = np.array([embeddings[drug] for drug in valid_drugs if drug in embeddings])
    cosine_full = cosine_similarity(embedding_matrix)

    for i, d1 in enumerate(valid_drugs):
        if d1 not in embeddings:
            continue
        idx1 = drug_names_embedding.index(d1)
        for j, d2 in enumerate(valid_drugs):
            if d2 not in embeddings:
                continue
            idx2 = drug_names_embedding.index(d2)
            cosine_matrix[i, j] = cosine_full[idx1, idx2]

    # Create comparison table
    comparison_df = create_comparison_table(tanimoto_matrix, cosine_matrix, valid_drugs)

    # Save results
    comparison_df.to_csv('feature_selection/tanimoto_cosine_comparison.csv', index=False)
    print("\n" + "="*70)
    print("Results saved to: feature_selection/tanimoto_cosine_comparison.csv")
    print("="*70)


if __name__ == '__main__':
    main()
