import duckdb
import pandas as pd

# Splitting the mutation column up to make fewer columns when one hot encoding
import re

AA1_to_3 = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
    '*': 'Ter'
}

# let's first convert the csv to parquet because it's too BIGG, then we can do the mutation splitting in pandas which will be easier than doing it in SQL
con = duckdb.connect()

con.execute("""COPY (
            SELECT
                GENE,
                MUTATION,
                GENE_POSITION,
                REF,
                ALT,
                DRUG,
                DRUG_NAME,
                AVG(LOG2MIC) AS mean_log2mic,
                MEDIAN(LOG2MIC) AS median_log2mic,
                COUNT(*) AS count
            FROM read_parquet('./data/cryptic_consortium_data.parquet')
            WHERE
                LOG2MIC IS NOT NULL
            GROUP BY
                GENE,
                MUTATION,
                GENE_POSITION,
                REF,
                ALT,
                DRUG,
                DRUG_NAME
            HAVING COUNT(*) >= 5
            )
            TO './data/cryptic_consortium_data_filtered.parquet' (FORMAT PARQUET); """) 


            
# we need to change the cryptic dataset to match the who if we want to use it for modeling
def cryptic_to_who_variant(row):
    gene = str(row['GENE']).strip()
    mutation = str(row['MUTATION']).strip().upper()

    m = re.match(r"^([A-Z*])(\d+)([A-Z*])$", mutation)

    if not m:
        return None

    ref, pos, alt = m.groups()

    ref3 = AA1_to_3.get(ref)
    alt3 = AA1_to_3.get(alt)

    if ref3 is None or alt3 is None:
        return None
    
    return f"{gene}_p.{ref3}{pos}{alt3}"

df = pd.read_parquet('./data/cryptic_consortium_data_filtered.parquet')

df['variant'] = df.apply(cryptic_to_who_variant, axis=1)
df = df.dropna(subset=['variant']).copy()

df.to_parquet('./data/cryptic_consortium_to_who.parquet', index=False)

