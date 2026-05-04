"""
Should be run from the root directory because the file paths are relative from there.
"""


import duckdb


def get_cryptic_data():
    con = duckdb.connect()

    return con.execute("""
        SELECT
            m.GENE,
            m.AMINO_ACID_NUMBER,
            m.REF,
            m.ALT,
            d.DRUG_NAME,
            p.LOG2MIC
        FROM read_parquet('./data/cryptic_consortium_data/data/MUTATIONS.parquet') m
        JOIN read_parquet('./data/cryptic_consortium_data/data/UKMYC_PHENOTYPES.parquet') p
            ON m.UNIQUEID = p.UNIQUEID
        LEFT JOIN read_csv('./data/cryptic_consortium_data/data/DRUG_CODES.csv.gz') d
            ON p.DRUG = d.DRUG_3_LETTER_CODE
        WHERE
            p.LOG2MIC IS NOT NULL
            AND m.CODES_PROTEIN = TRUE
            AND p.PHENOTYPE_QUALITY IS NOT NULL
            AND m.IS_MINOR = FALSE
            AND m.REF IS NOT NULL
            AND m.ALT IS NOT NULL
            AND d.DRUG_NAME = 'RIFAMPICIN'
    """).df()



"""
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

    FROM read_parquet('./data/cryptic_consortium_data/data/MUTATIONS.parquet') m
    JOIN read_parquet('./data/cryptic_consortium_data/data/UKMYC_PHENOTYPES.parquet') p
        ON m.UNIQUEID = p.UNIQUEID
    LEFT JOIN read_csv('./data/cryptic_consortium_data/data/DRUG_CODES.csv.gz') d
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

