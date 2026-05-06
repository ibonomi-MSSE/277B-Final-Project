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

WHO_COLS_TO_KEEP = [
    # original WHO features
    "mutation",
    "drug",
    "gene",
    "FINAL CONFIDENCE GRADING",
    "variant",
    "ppv",
    "drug_norm"
]

FINAL_COLS_TO_KEEP = [
    "mutation",
    "drug",
    "ppv",
    "gene",
    "FINAL CONFIDENCE GRADING",
    "variant",
    "drug_norm",
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


def load_data(WHO: str = "./data/WHO_data/WHO-UCN-TB-2023.6-eng_catalogue_master_file.txt",
              genomic_positions: str = "./data/WHO_data/WHO-UCN-TB-2023.7-eng_genomic_coordinates.txt",
              cryptic_consortium: str = "./data/cryptic_consortium_data/cryptic_consortium_to_who.parquet") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    who_df = pd.read_csv(WHO, sep="\t", low_memory=False)
    genomic_positions_df = pd.read_csv(genomic_positions, sep="\t")
    cryptic_df = pd.read_parquet(cryptic_consortium)

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
    who_df = who_df[WHO_COLS_TO_KEEP].copy()

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


def cryptic_MIC_fallback(merged_file_cryptic, cryptic):
    """
    This logic is to fill the missing MIC values in the merged file with the average MIC values for the same variant, when available.
    If there are no MIC values for the same variant, we will fill with 0 (which is the log2MIC value for the lowest MIC value in the dataset).
    We can also add a separate feature to indicate that it's a fallback value, which might be useful for the model to know.
    """

    cryptic_variant_fallback = (
        cryptic
        .groupby("variant_norm", as_index=False)
        .agg(
            mean_log2mic_variant=("mean_log2mic", "mean"),
            median_log2mic_variant=("median_log2mic", "median"),
            count_variant=("count", "sum")
        )
    )
    merged_file_cryptic = merged_file_cryptic.merge(
        cryptic_variant_fallback,
        on="variant_norm",
        how="left"
    )
    merged_file_cryptic["has_exact_mic"] = merged_file_cryptic["mean_log2mic"].notna().astype(int)
    merged_file_cryptic["has_variant_mic"] = merged_file_cryptic["mean_log2mic_variant"].notna().astype(int)

    merged_file_cryptic["mean_log2mic_final"] = (
        merged_file_cryptic["mean_log2mic"]
        .fillna(merged_file_cryptic["mean_log2mic_variant"])
        .fillna(0)
    )
    merged_file_cryptic["mic_count_final"] = (
        merged_file_cryptic["count"]
        .fillna(merged_file_cryptic["count_variant"])
        .fillna(0)
    )

    return merged_file_cryptic


def finalize_data(merged_file_cryptic):
    # drop intermediate columns and keep only the final columns we want to use for modeling
    final_data = merged_file_cryptic[FINAL_COLS_TO_KEEP].copy()
    final_data = final_data[final_data["ppv"] > 0].copy()

    return final_data

def main():
    who_df, genomic_positions_df, cryptic_df = load_data()
    who_df, genomic_positions_df, cryptic_df = clean_data(who_df, genomic_positions_df, cryptic_df)
    merged_file_cryptic = merge_data(who_df, genomic_positions_df, cryptic_df)
    merged_file_cryptic = cryptic_MIC_fallback(merged_file_cryptic, cryptic_df)
    final_data = finalize_data(merged_file_cryptic)

    return final_data


if __name__ == "__main__":
    who_df, genomic_positions_df, cryptic_df = load_data()
    who_df, genomic_positions_df, cryptic_df = clean_data(who_df, genomic_positions_df, cryptic_df)
    merged_file_cryptic = merge_data(who_df, genomic_positions_df, cryptic_df)
    merged_file_cryptic = cryptic_MIC_fallback(merged_file_cryptic, cryptic_df)
    final_data = finalize_data(merged_file_cryptic)



