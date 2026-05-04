"""
Should be run from the root directory because the file paths are relative from there.
"""


import duckdb
import torch


BASE_QUERY = """
    FROM read_parquet('./data/MUTATIONS.parquet') m
    JOIN read_parquet('./data/UKMYC_PHENOTYPES.parquet') p
        ON m.UNIQUEID = p.UNIQUEID
    LEFT JOIN read_csv('./data/DRUG_CODES.csv.gz') d
        ON p.DRUG = d.DRUG_3_LETTER_CODE
    WHERE
        p.LOG2MIC IS NOT NULL
        AND m.CODES_PROTEIN = TRUE
        AND p.PHENOTYPE_QUALITY IS NOT NULL
        AND m.IS_MINOR = FALSE
        AND m.REF IS NOT NULL
        AND m.ALT IS NOT NULL
"""


def build_vocabs_from_duckdb(con):
    """Build gene/drug/position vocabs directly from DuckDB without pandas."""
    genes = con.execute(f"SELECT DISTINCT m.GENE {BASE_QUERY}").fetchall()
    drugs = con.execute(f"SELECT DISTINCT d.DRUG_NAME {BASE_QUERY}").fetchall()
    positions = con.execute(
        f"SELECT DISTINCT m.AMINO_ACID_NUMBER {BASE_QUERY}"
    ).fetchall()

    gene_to_idx     = {g[0]: i for i, g in enumerate(genes)}
    drug_to_idx     = {d[0]: i for i, d in enumerate(drugs)}
    position_to_idx = {p[0]: i for i, p in enumerate(positions)}
    return gene_to_idx, drug_to_idx, position_to_idx


def load_cryptic_tensors(device, vocabs=None, cache_path=None):
    """Load data into tensors. If `vocabs` is provided, use them as-is so
    indices match a previous run; otherwise build fresh from the data.
    If `cache_path` is provided and exists, load tensors from there."""

    # Fast path: load cached tensors if available
    if cache_path is not None and Path(cache_path).exists():
        print(f"Loading cached tensors from {cache_path}")
        cached = torch.load(cache_path, map_location='cpu')
        tensors = {k: v.to(device) for k, v in cached['tensors'].items()}
        return (tensors,
                cached['gene_to_idx'],
                cached['drug_to_idx'],
                cached['position_to_idx'])

    con = duckdb.connect()
    con.execute("SET memory_limit='30GB'")
    con.execute("SET preserve_insertion_order=false")

    print("Scanning data...")
    arrow_table = con.execute(f"""
        SELECT
            m.GENE              AS gene,
            m.AMINO_ACID_NUMBER AS pos,
            m.REF               AS ref,
            m.ALT               AS alt,
            d.DRUG_NAME         AS drug,
            p.LOG2MIC::FLOAT    AS log2mic
        {BASE_QUERY}
    """).to_arrow_table()
    print(f"  {arrow_table.num_rows:,} rows loaded")

    gene_col = arrow_table['gene']
    pos_col  = arrow_table['pos']
    ref_col  = arrow_table['ref']
    alt_col  = arrow_table['alt']
    drug_col = arrow_table['drug']
    log_col  = arrow_table['log2mic']

    if vocabs is None:
        import pyarrow.compute as pc
        gene_unique = pc.unique(gene_col).to_pylist()
        drug_unique = pc.unique(drug_col).to_pylist()
        pos_unique  = pc.unique(pos_col).to_pylist()
        gene_to_idx     = {g: i for i, g in enumerate(v for v in gene_unique if v is not None)}
        drug_to_idx     = {d: i for i, d in enumerate(v for v in drug_unique if v is not None)}
        position_to_idx = {p: i for i, p in enumerate(v for v in pos_unique  if v is not None)}
    else:
        gene_to_idx, drug_to_idx, position_to_idx = vocabs
        # Sanity check: warn if the new data has values the saved vocab doesn't cover
        import pyarrow.compute as pc
        new_genes = set(pc.unique(gene_col).to_pylist()) - {None}
        missing = new_genes - set(gene_to_idx.keys())
        if missing:
            print(f"WARNING: {len(missing)} genes in data not in saved vocab "
                  f"(will be mapped to 0): {list(missing)[:5]}...")

    print(f"  {len(gene_to_idx)} genes, {len(drug_to_idx)} drugs, "
          f"{len(position_to_idx)} positions")

    def map_column(col, mapping, default):
        py_list = col.to_pylist()
        return np.fromiter(
            (mapping.get(v, default) for v in py_list),
            dtype=np.int64,
            count=len(py_list),
        )

    cpu_tensors = {
        'gene':    torch.from_numpy(map_column(gene_col, gene_to_idx, 0)),
        'pos':     torch.from_numpy(map_column(pos_col,  position_to_idx, 0)),
        'wt_aa':   torch.from_numpy(map_column(ref_col,  AA_TO_IDX, 20)),
        'mut_aa':  torch.from_numpy(map_column(alt_col,  AA_TO_IDX, 20)),
        'drug':    torch.from_numpy(map_column(drug_col, drug_to_idx, 0)),
        'log2mic': torch.from_numpy(log_col.to_numpy().astype(np.float32)),
    }

    del arrow_table
    con.close()

    # Cache for next run
    if cache_path is not None:
        print(f"Caching tensors to {cache_path}")
        torch.save({
            'tensors': cpu_tensors,
            'gene_to_idx': gene_to_idx,
            'drug_to_idx': drug_to_idx,
            'position_to_idx': position_to_idx,
        }, cache_path)

    tensors = {k: v.to(device) for k, v in cpu_tensors.items()}
    return tensors, gene_to_idx, drug_to_idx, position_to_idx
