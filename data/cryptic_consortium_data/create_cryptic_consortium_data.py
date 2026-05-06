# create_cryptic_consortium_data.py
import duckdb
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "data"

mutations = DATA_DIR / "MUTATIONS.parquet"
phenotypes =  DATA_DIR / "UKMYC_PHENOTYPES.parquet"
drug_codes = DATA_DIR / "DRUG_CODES.csv.gz"
output = DATA_DIR / "cryptic_consortium_data.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='30GB'")

print("Creating cryptic_consortium_data.parquet...")

con.execute(f"""
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
        FROM read_parquet('{mutations}') m
        JOIN read_parquet('{phenotypes}') p
            ON m.UNIQUEID = p.UNIQUEID
        LEFT JOIN read_csv('{drug_codes}') d
            ON p.DRUG = d.DRUG_3_LETTER_CODE
        WHERE 
            p.LOG2MIC IS NOT NULL
            AND m.CODES_PROTEIN = TRUE
            AND m.REF IS NOT NULL
            AND m.ALT IS NOT NULL
    ) TO '{output}' (FORMAT PARQUET)
""")

print("Done creating cryptic_consortium_data.parquet")
con.close()