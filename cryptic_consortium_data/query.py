"""
TODO: I added the GENE_POSITION = 103 and LIMIT 20 to make the query faster. We should remove that.
"""


import duckdb

con = duckdb.connect()

query = """
COPY( SELECT
    m.UNIQUEID,
    m.GENE,
    m.MUTATION,
    m.GENE_POSITION,
    m.REF,
    m.ALT,
    m.AMINO_ACID_NUMBER,
    m.AMINO_ACID_SEQUENCE,
    m.CODES_PROTEIN,
    m.IS_NULL,
    m.IS_MINOR,
    m.COVERAGE,
    m.FRS,
    p.DRUG,
    d.DRUG_NAME,
    p.MIC,
    p.LOG2MIC,
    p.BINARY_PHENOTYPE

FROM read_parquet('./data/MUTATIONS.parquet') m
JOIN read_parquet('./data/UKMYC_PHENOTYPES.parquet') p
    ON m.UNIQUEID = p.UNIQUEID
LEFT JOIN read_csv('./data/DRUG_CODES.csv.gz') d
    ON p.DRUG = d.DRUG_3_LETTER_CODE

WHERE
    p.MIC IS NOT NULL
    AND p.LOG2MIC IS NOT NULL
    AND p.BINARY_PHENOTYPE IS NOT NULL
    AND m.CODES_PROTEIN = TRUE
    AND p.PHENOTYPE_QUALITY IS NOT NULL
    AND m.IS_MINOR = FALSE
) TO './data/cryptic_consortium_data.parquet' WITH (FORMAT PARQUET)
"""

con.execute(query)

