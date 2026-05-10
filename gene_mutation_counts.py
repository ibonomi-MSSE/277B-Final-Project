import duckdb
import matplotlib.pyplot as plt
import pandas as pd


def get_cryptic_gene_mutation_counts(min_count=500):
    """
    Get gene and mutation counts from Cryptic Consortium dataset.

    Args:
        min_count: minimum count threshold for filtering

    Returns:
        dict with keys: gene_counts, mutation_counts, total_rows, filtered_rows
    """
    con = duckdb.connect()
    con.execute("SET memory_limit='30GB'")
    con.execute("SET preserve_insertion_order=false")

    mutations_path = './data/cryptic_consortium_data/data/MUTATIONS.parquet'
    phenotypes_path = './data/cryptic_consortium_data/data/UKMYC_PHENOTYPES.parquet'

    print("Analyzing CRYPTIC CONSORTIUM dataset...")

    # Get total row count
    total_rows = con.execute(f"""
        SELECT COUNT(*) AS total
        FROM read_parquet('{mutations_path}') m
        JOIN read_parquet('{phenotypes_path}') p
            ON m.UNIQUEID = p.UNIQUEID
    """).fetchone()[0]

    print(f"Total rows in database: {total_rows:,}")

    # Get count of rows that meet all filtering criteria
    filtered_rows = con.execute(f"""
        SELECT COUNT(*) AS filtered_total
        FROM read_parquet('{mutations_path}') m
        JOIN read_parquet('{phenotypes_path}') p
            ON m.UNIQUEID = p.UNIQUEID
        WHERE
            p.LOG2MIC IS NOT NULL
            AND m.CODES_PROTEIN = TRUE
            AND p.PHENOTYPE_QUALITY IS NOT NULL
            AND m.IS_MINOR = FALSE
            AND m.REF IS NOT NULL
            AND m.ALT IS NOT NULL
    """).fetchone()[0]

    print(f"Rows meeting all filtering criteria: {filtered_rows:,}")

    # Query for gene counts
    gene_counts = con.execute(f"""
        SELECT
            m.GENE AS gene,
            COUNT(*) AS count
        FROM read_parquet('{mutations_path}') m
        JOIN read_parquet('{phenotypes_path}') p
            ON m.UNIQUEID = p.UNIQUEID
        WHERE
            p.LOG2MIC IS NOT NULL
            AND m.CODES_PROTEIN = TRUE
            AND p.PHENOTYPE_QUALITY IS NOT NULL
            AND m.IS_MINOR = FALSE
            AND m.REF IS NOT NULL
            AND m.ALT IS NOT NULL
        GROUP BY m.GENE
        HAVING COUNT(*) >= {min_count}
        ORDER BY count DESC
    """).fetchdf()

    print(f"\nGenes with at least {min_count} rows: {len(gene_counts)}")
    print(f"Total rows in genes with >= {min_count} counts: {gene_counts['count'].sum():,}")

    # Query for mutation counts
    mutation_counts = con.execute(f"""
        SELECT
            m.GENE || '_' || m.AMINO_ACID_NUMBER || m.REF || '>' || m.ALT AS mutation,
            COUNT(*) AS count
        FROM read_parquet('{mutations_path}') m
        JOIN read_parquet('{phenotypes_path}') p
            ON m.UNIQUEID = p.UNIQUEID
        WHERE
            p.LOG2MIC IS NOT NULL
            AND m.CODES_PROTEIN = TRUE
            AND p.PHENOTYPE_QUALITY IS NOT NULL
            AND m.IS_MINOR = FALSE
            AND m.REF IS NOT NULL
            AND m.ALT IS NOT NULL
        GROUP BY m.GENE, m.AMINO_ACID_NUMBER, m.REF, m.ALT
        HAVING COUNT(*) >= {min_count}
        ORDER BY count DESC
    """).fetchdf()

    print(f"\nMutations with at least {min_count} rows: {len(mutation_counts)}")
    print(f"Total rows in mutations with >= {min_count} counts: {mutation_counts['count'].sum():,}")

    # Calculate excluded rows
    rows_excluded = filtered_rows - mutation_counts['count'].sum()
    print(f"\nRows excluded (< {min_count} count threshold): {rows_excluded:,}")
    rows_no_log2mic = total_rows - filtered_rows
    print(f"Rows excluded (missing log2mic or other filters): {rows_no_log2mic:,}")

    con.close()

    return {
        'gene_counts': gene_counts,
        'mutation_counts': mutation_counts,
        'total_rows': total_rows,
        'filtered_rows': filtered_rows,
        'rows_excluded': rows_excluded,
        'rows_no_log2mic': rows_no_log2mic
    }


def get_who_gene_mutation_counts(min_count=50):
    """
    Get gene and mutation counts from WHO dataset.

    Args:
        min_count: minimum count threshold for filtering

    Returns:
        dict with keys: gene_counts, mutation_counts, total_rows
    """
    print("Analyzing WHO dataset...")

    # Read WHO catalogue file
    catalogue_path = './data/WHO_data/WHO-UCN-TB-2023.6-eng_catalogue_master_file.txt'
    df = pd.read_csv(catalogue_path, sep='\t')

    print(f"Total rows in WHO catalogue: {len(df):,}")

    # Parse gene from mutation column (format: gene_variant)
    df['gene'] = df['mutation'].str.split('_').str[0]

    # Count by gene
    gene_counts = df.groupby('gene').size().reset_index(name='count')
    gene_counts = gene_counts[gene_counts['count'] >= min_count]
    gene_counts = gene_counts.sort_values('count', ascending=False).reset_index(drop=True)

    print(f"\nGenes with at least {min_count} rows: {len(gene_counts)}")
    print(f"Total rows in genes with >= {min_count} counts: {gene_counts['count'].sum():,}")

    # Count by mutation
    mutation_counts = df.groupby('mutation').size().reset_index(name='count')
    mutation_counts = mutation_counts[mutation_counts['count'] >= min_count]
    mutation_counts = mutation_counts.sort_values('count', ascending=False).reset_index(drop=True)

    print(f"\nMutations with at least {min_count} rows: {len(mutation_counts)}")
    print(f"Total rows in mutations with >= {min_count} counts: {mutation_counts['count'].sum():,}")

    total_rows = len(df)
    rows_excluded = total_rows - mutation_counts['count'].sum()
    print(f"\nRows excluded (< {min_count} count threshold): {rows_excluded:,}")

    return {
        'gene_counts': gene_counts,
        'mutation_counts': mutation_counts,
        'total_rows': total_rows,
        'filtered_rows': total_rows,
        'rows_excluded': rows_excluded,
        'rows_no_log2mic': 0
    }


def plot_gene_mutation_counts(results, dataset_name, output_prefix, min_count=500):
    """
    Create plots and save summary for gene and mutation counts.

    Args:
        results: dict from get_gene_mutation_counts
        dataset_name: name of dataset for titles
        output_prefix: prefix for output files
        min_count: minimum count threshold (for titles)
    """
    gene_counts = results['gene_counts']
    mutation_counts = results['mutation_counts']
    total_rows = results['total_rows']
    filtered_rows = results['filtered_rows']
    rows_excluded = results['rows_excluded']
    rows_no_log2mic = results['rows_no_log2mic']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Plot 1: Gene counts
    ax1.bar(range(len(gene_counts)), gene_counts['count'], color='steelblue', alpha=0.8)
    ax1.set_xlabel('Gene Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title(f'{dataset_name} - Gene Counts (genes with ≥{min_count} rows, n={len(gene_counts)} genes)',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)

    # Add count labels on top of bars for top 20 genes
    for i in range(min(20, len(gene_counts))):
        ax1.text(i, gene_counts['count'].iloc[i], f"{gene_counts['count'].iloc[i]:,}",
                 ha='center', va='bottom', fontsize=7)

    # Plot 2: Mutation counts
    # Only show top 50 mutations for readability
    top_mutations = mutation_counts.head(50)
    ax2.bar(range(len(top_mutations)), top_mutations['count'], color='coral', alpha=0.8)
    ax2.set_xlabel('Mutation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title(f'{dataset_name} - Top 50 Mutation Counts (out of {len(mutation_counts)} mutations with ≥{min_count} rows)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(top_mutations)))
    ax2.set_xticklabels(top_mutations['mutation'], rotation=90, ha='right', fontsize=7)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plot_file = f'{output_prefix}_gene_mutation_counts.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{plot_file}'")

    # Save summary statistics
    summary_file = f'{output_prefix}_gene_mutation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"=== {dataset_name} Gene and Mutation Count Summary ===\n\n")
        f.write(f"Total rows in database: {total_rows:,}\n")
        f.write(f"Rows meeting all filtering criteria: {filtered_rows:,}\n")
        f.write(f"Rows excluded (missing log2mic or other filters): {rows_no_log2mic:,}\n\n")

        f.write(f"Genes with at least {min_count} rows: {len(gene_counts)}\n")
        f.write(f"Total rows in genes with >= {min_count} counts: {gene_counts['count'].sum():,}\n\n")

        f.write(f"Mutations with at least {min_count} rows: {len(mutation_counts)}\n")
        f.write(f"Total rows in mutations with >= {min_count} counts: {mutation_counts['count'].sum():,}\n")
        f.write(f"Rows excluded (< {min_count} count threshold): {rows_excluded:,}\n\n")

        f.write("\n=== Top 20 Genes by Count ===\n")
        for i, row in gene_counts.head(20).iterrows():
            f.write(f"{row['gene']}: {row['count']:,}\n")

        f.write("\n=== Top 20 Mutations by Count ===\n")
        for i, row in mutation_counts.head(20).iterrows():
            f.write(f"{row['mutation']}: {row['count']:,}\n")

    print(f"Summary saved as '{summary_file}'")


if __name__ == '__main__':
    # Run for cryptic consortium data
    print("=" * 80)
    print("CRYPTIC CONSORTIUM DATA")
    print("=" * 80)
    cryptic_results = get_cryptic_gene_mutation_counts(min_count=500)
    plot_gene_mutation_counts(cryptic_results, 'Cryptic Consortium', 'cryptic', min_count=500)

    print("\n" + "=" * 80)
    print("WHO DATA")
    print("=" * 80)
    # Run for WHO data (using lower threshold since WHO data is structured differently)
    who_results = get_who_gene_mutation_counts(min_count=50)
    plot_gene_mutation_counts(who_results, 'WHO', 'who', min_count=50)
