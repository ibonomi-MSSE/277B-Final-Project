"""
Analysis of drug embeddings from ChemBERTa MTR model.
Creates cosine similarity heatmap and UMAP visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from feature_encoding.chemBERTa_mtr_embeddings import get_drug_embeddings


def plot_cosine_similarity_heatmap(embeddings_dict, output_path='drug_cosine_similarity.png'):
    """
    Create and save a heatmap of cosine similarities between drug embeddings.

    Parameters:
    -----------
    embeddings_dict : dict
        Dictionary mapping drug names to embedding vectors
    output_path : str
        Path to save the heatmap
    """
    drug_names = list(embeddings_dict.keys())
    embeddings_matrix = np.array([embeddings_dict[drug] for drug in drug_names])

    cosine_sim = cosine_similarity(embeddings_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cosine_sim,
        xticklabels=drug_names,
        yticklabels=drug_names,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title('Cosine Similarity Between Drug Embeddings', fontsize=16, pad=20)
    plt.xlabel('Drug', fontsize=12)
    plt.ylabel('Drug', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cosine similarity heatmap saved to: {output_path}")
    plt.close()

    return cosine_sim


def plot_umap_embeddings(embeddings_dict, output_path='drug_umap.png'):
    """
    Create and save a UMAP plot of drug embeddings with labels.

    Parameters:
    -----------
    embeddings_dict : dict
        Dictionary mapping drug names to embedding vectors
    output_path : str
        Path to save the UMAP plot
    """
    drug_names = list(embeddings_dict.keys())
    embeddings_matrix = np.array([embeddings_dict[drug] for drug in drug_names])

    reducer = UMAP(
        n_neighbors=5,
        min_dist=0.3,
        metric='cosine',
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings_matrix)

    plt.figure(figsize=(12, 10))
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        s=200,
        alpha=0.7,
        c=range(len(drug_names)),
        cmap='tab20'
    )

    for i, drug_name in enumerate(drug_names):
        plt.annotate(
            drug_name,
            (embedding_2d[i, 0], embedding_2d[i, 1]),
            fontsize=10,
            ha='center',
            va='bottom',
            xytext=(0, 5),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray')
        )

    plt.title('UMAP Projection of Drug Embeddings', fontsize=16, pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"UMAP visualization saved to: {output_path}")
    plt.close()

    return embedding_2d


def main():
    print("Loading drug embeddings...")
    embeddings = get_drug_embeddings()
    print(f"Loaded embeddings for {len(embeddings)} drugs")
    print(f"Embedding dimension: {list(embeddings.values())[0].shape[0]}")

    print("\nGenerating cosine similarity heatmap...")
    cosine_sim = plot_cosine_similarity_heatmap(
        embeddings,
        output_path='feature_selection/drug_cosine_similarity.png'
    )

    print("\nTop 5 most similar drug pairs:")
    drug_names = list(embeddings.keys())
    for i in range(len(drug_names)):
        for j in range(i + 1, len(drug_names)):
            if i != j:
                sim = cosine_sim[i, j]
                print(f"  {drug_names[i]} <-> {drug_names[j]}: {sim:.3f}")

    sorted_pairs = []
    for i in range(len(drug_names)):
        for j in range(i + 1, len(drug_names)):
            sorted_pairs.append((drug_names[i], drug_names[j], cosine_sim[i, j]))
    sorted_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 5 most similar drug pairs:")
    for drug1, drug2, sim in sorted_pairs[:5]:
        print(f"  {drug1} <-> {drug2}: {sim:.3f}")

    print("\nTop 5 least similar drug pairs:")
    for drug1, drug2, sim in sorted_pairs[-5:]:
        print(f"  {drug1} <-> {drug2}: {sim:.3f}")

    print("\nGenerating UMAP visualization...")
    embedding_2d = plot_umap_embeddings(
        embeddings,
        output_path='feature_selection/drug_umap.png'
    )

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
