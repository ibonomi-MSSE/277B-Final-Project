# create_cryptic_consortium_data.py
import duckdb

con = duckdb.connect()
con.execute("SET memory_limit='30GB'")

print("Creating cryptic_consortium_data.parquet...")

con.execute("""
    COPY (
        SELECT 
            m.GENE,
            m.MUTATION,
            m.AMINO_ACID_NUMBER AS GENE_POSITION,
            m.REF,
            m.ALT,
            p.DRUG,
            d.DRUG_NAME,
            p.LOG2MIC
        FROM read_parquet('./data/MUTATIONS.parquet') m
        JOIN read_parquet('./data/UKMYC_PHENOTYPES.parquet') p
            ON m.UNIQUEID = p.UNIQUEID
        LEFT JOIN read_csv('./data/DRUG_CODES.csv.gz') d
            ON p.DRUG = d.DRUG_3_LETTER_CODE
        WHERE 
            p.LOG2MIC IS NOT NULL
            AND m.CODES_PROTEIN = TRUE
            AND m.REF IS NOT NULL
            AND m.ALT IS NOT NULL
    ) TO './data/cryptic_consortium_data.parquet' (FORMAT PARQUET)
""")

print("Done creating cryptic_consortium_data.parquet")
con.close()