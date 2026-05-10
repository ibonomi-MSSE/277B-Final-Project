"""
Create a summary comparison of Tanimoto coefficients vs Cosine similarities.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the comparison data
df = pd.read_csv('feature_selection/tanimoto_cosine_comparison.csv')

# Create summary figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Scatter plot: Tanimoto vs Cosine
ax1 = axes[0, 0]
ax1.scatter(df['Tanimoto'], df['Cosine'], alpha=0.6, s=50)

# Highlight aminoglycosides
aminoglycoside_pairs = [
    ('Amikacin', 'Kanamycin'),
    ('Amikacin', 'Streptomycin'),
    ('Kanamycin', 'Streptomycin')
]

for drug1, drug2 in aminoglycoside_pairs:
    mask = ((df['Drug 1'] == drug1) & (df['Drug 2'] == drug2)) | \
           ((df['Drug 1'] == drug2) & (df['Drug 2'] == drug1))
    subset = df[mask]
    if not subset.empty:
        ax1.scatter(subset['Tanimoto'], subset['Cosine'],
                   color='red', s=200, alpha=0.8, marker='*',
                   edgecolors='darkred', linewidth=2, zorder=5)
        # Add label
        ax1.annotate(f"{drug1[:4]}-{drug2[:4]}",
                    (subset['Tanimoto'].values[0], subset['Cosine'].values[0]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Add diagonal line
max_val = max(df['Tanimoto'].max(), df['Cosine'].max())
ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x')

ax1.set_xlabel('Tanimoto Coefficient (Structural)', fontsize=12)
ax1.set_ylabel('Cosine Similarity (Embedding)', fontsize=12)
ax1.set_title('Structural vs Embedding Similarity\n(Red stars = Aminoglycosides held out)', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Top 10 comparison bar plot
ax2 = axes[0, 1]
top10 = df.nlargest(10, 'Tanimoto').copy()
top10['pair'] = top10['Drug 1'].str[:5] + '-' + top10['Drug 2'].str[:5]

x = range(len(top10))
width = 0.35

bars1 = ax2.bar([i - width/2 for i in x], top10['Tanimoto'], width,
               label='Tanimoto', alpha=0.8, color='steelblue')
bars2 = ax2.bar([i + width/2 for i in x], top10['Cosine'], width,
               label='Cosine', alpha=0.8, color='coral')

ax2.set_xlabel('Drug Pair', fontsize=11)
ax2.set_ylabel('Similarity Score', fontsize=11)
ax2.set_title('Top 10 Most Similar Pairs\nComparison of Metrics', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(top10['pair'], rotation=45, ha='right', fontsize=9)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Distribution comparison
ax3 = axes[1, 0]
ax3.hist(df['Tanimoto'], bins=30, alpha=0.6, label='Tanimoto', color='steelblue', edgecolor='black')
ax3.hist(df['Cosine'], bins=30, alpha=0.6, label='Cosine', color='coral', edgecolor='black')
ax3.set_xlabel('Similarity Score', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Distribution of Similarity Scores', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Difference analysis
ax4 = axes[1, 1]
top_diff = df.nlargest(10, 'Difference').copy()
top_diff['pair'] = top_diff['Drug 1'].str[:5] + '-' + top_diff['Drug 2'].str[:5]

colors = ['red' if any((d1 in ['Amikacin', 'Kanamycin', 'Streptomycin'] and
                        d2 in ['Amikacin', 'Kanamycin', 'Streptomycin'])
                      for d1, d2 in [(row['Drug 1'], row['Drug 2'])])
         else 'gray' for _, row in top_diff.iterrows()]

ax4.barh(range(len(top_diff)), top_diff['Difference'], color=colors, alpha=0.7)
ax4.set_yticks(range(len(top_diff)))
ax4.set_yticklabels(top_diff['pair'], fontsize=9)
ax4.set_xlabel('|Tanimoto - Cosine|', fontsize=12)
ax4.set_title('Top 10 Largest Discrepancies\n(Red = Aminoglycosides)', fontsize=13)
ax4.grid(True, alpha=0.3, axis='x')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('feature_selection/tanimoto_cosine_summary.png', dpi=300, bbox_inches='tight')
print("Summary figure saved to: feature_selection/tanimoto_cosine_summary.png")
plt.close()

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nTanimoto Coefficient:")
print(f"  Mean: {df['Tanimoto'].mean():.4f}")
print(f"  Std:  {df['Tanimoto'].std():.4f}")
print(f"  Min:  {df['Tanimoto'].min():.4f}")
print(f"  Max:  {df['Tanimoto'].max():.4f}")

print(f"\nCosine Similarity:")
print(f"  Mean: {df['Cosine'].mean():.4f}")
print(f"  Std:  {df['Cosine'].std():.4f}")
print(f"  Min:  {df['Cosine'].min():.4f}")
print(f"  Max:  {df['Cosine'].max():.4f}")

print(f"\nCorrelation between Tanimoto and Cosine: {df['Tanimoto'].corr(df['Cosine']):.4f}")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
print("\n1. HIGHEST COSINE SIMILARITIES (Embedding-based):")
top5_cosine = df.nlargest(5, 'Cosine')
for idx, row in top5_cosine.iterrows():
    print(f"   {row['Drug 1']:15s} <-> {row['Drug 2']:15s}  Cosine: {row['Cosine']:.4f}")

print("\n2. HIGHEST TANIMOTO COEFFICIENTS (Structure-based):")
top5_tanimoto = df.nlargest(5, 'Tanimoto')
for idx, row in top5_tanimoto.iterrows():
    print(f"   {row['Drug 1']:15s} <-> {row['Drug 2']:15s}  Tanimoto: {row['Tanimoto']:.4f}")

print("\n3. SCAFFOLD HOLDOUT DRUGS (Amikacin, Kanamycin, Streptomycin):")
print("   These drugs have the HIGHEST cosine similarities in embeddings,")
print("   but their Tanimoto coefficients tell a different story:")
for drug1, drug2 in aminoglycoside_pairs:
    mask = ((df['Drug 1'] == drug1) & (df['Drug 2'] == drug2)) | \
           ((df['Drug 1'] == drug2) & (df['Drug 2'] == drug1))
    row = df[mask].iloc[0]
    print(f"   {row['Drug 1']:15s} <-> {row['Drug 2']:15s}  " +
          f"Tanimoto: {row['Tanimoto']:.4f}  Cosine: {row['Cosine']:.4f}")

print("\n" + "="*70)
