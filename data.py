import numpy as np
import pandas as pd
import duckdb
import re

AA3_to_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*', 'Stop': '*'
}

AA1_to_3 = {v: k for k, v in AA3_to_1.items()}

COL_TO_KEEP = [
    # original WHO features
    "mutation",
    "drug",
    "gene",
    "FINAL CONFIDENCE GRADING",
    "variant",
    "PPV_DATASET ALL",
    # Genomic position features
    "chromosome",
    "position",
    "reference_nucleotide",
    "alternative_nucleotide",
    # NEW MIC FEATURES
    "mean_log2mic_final",
    "mic_count_final",
    "has_exact_mic",
    "has_variant_mic"
]

def normalize_str(string):
    if pd.isna(string):
        return None
    return (str(string).strip().replace(" ", "").lower())


def load_data():
    who_df = pd.read_csv("WHO-UCN-TB-2023.6-eng_catalogue_master_file.txt", sep="\t")
    genomic_positions_df = pd.read_csv("WHO-UCN-TB-2023.7-eng_genomic_coordinates.txt", sep="\t")
    cryptic_df = pd.read_parquet("./cryptic_consortium_data/data/cryptic_consortium_to_who.parquet")

    return who_df, genomic_positions_df, cryptic_df


def clean_data(who_df, genomic_positions_df, cryptic_df):
    # clean the WHO data and keep only useful columns
    who_df = who_df.rename(columns={
        "PPV_DATASET ALL": "ppv",
        "MUTATION": "mutation",
        "DRUG": "drug",
        "GENE": "gene",
        "FINAL CONFIDENCE GRADING": "FINAL CONFIDENCE GRADING",
        "VARIANT": "variant"
    })
    who_df["drug_norm"] = who_df["drug"].apply(normalize_str)
    who_df = who_df[COL_TO_KEEP].copy()

    # clean the Genomic Positions data and keep only useful columns
    genomic_positions_df["variant"] = genomic_positions_df["variant"].astype(str).str.strip()
    genomic_positions_df["variant_norm"] = genomic_positions_df["variant"].apply(normalize_str)

    # clean the Genomic Positions data and keep only useful columns
    cryptic_df["variant"] = cryptic_df["variant"].astype(str).str.strip()
    cryptic_df["variant_norm"] = cryptic_df["variant"].apply(normalize_str)
    cryptic_df["drug_norm"] = cryptic_df["DRUG_NAME"].apply(normalize_str)

    return who_df, genomic_positions_df, cryptic_df


def merge_data(who_df, genomic_positions_df, cryptic_df):

    # first merge the WHO data with Genomic Positions data on the normalized variant column
    merge_genomic = who_df.merge(
        genomic_positions_df[[
        "variant_norm",
        "variant",
        "chromosome",
        "position",
        "reference_nucleotide",
        "alternative_nucleotide"
        ]], on="variant", how="left" )

    # merge the results with the cryptic consortium on variant and drug (both normalized)
    merged_cryptic = merge_genomic.merge(
    cryptic_df[[
        "variant_norm",
        "drug_norm",
        "DRUG_NAME",
        "mean_log2mic",
        "median_log2mic",
        "count"
    ]], on=["variant_norm", "drug_norm"], how="left")

    return merged_cryptic

