from load_data import main, load_data, clean_data, merge_data, cryptic_MIC_fallback, finalize_data
import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import ConvertToNumpyArray
import pubchempy as pcp
import numpy as np

 # keep only necessary columns
COL_TO_KEEP = [
    "gene",
    "ppv",
    "mutation",
    "drug",
    "resistant",
    "variant",
    "chromosome",
    "position",
    "reference_nucleotide",
    "alternative_nucleotide",
    "mean_log2mic_final",
    "mic_count_final",
    "has_exact_mic",
    "has_variant_mic"
]


def encode_resistance(data):

    # Drop the uncertain resistance
    df = data[data['FINAL CONFIDENCE GRADING'] != '3) Uncertain significance'].copy()

    grading_scheme = {
    "5) Not assoc w R": 0,
    "4) Not assoc w R - Interim": 1,
    "2) Assoc w R - Interim": 2,
    "1) Assoc w R": 3
    }

    df["resistant"] = df["FINAL CONFIDENCE GRADING"].map(grading_scheme)
    df_model = df.dropna(subset=["resistant"]).copy()

    df_model = df_model.drop(columns=["FINAL CONFIDENCE GRADING"])

    return df_model


def encode_mutations(data):
    # Splitting the mutation column up to make fewer columns when one hot encoding
    # Amino acid 3-letter to 1-letter code map
    AA_MAP = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
        'Ter': '*', 'Stop': '*'
    }

    def parse_aa(code):
        """Convert 3-letter amino acid code to 1-letter, or return as-is if already 1-letter or stop."""
        if code == '*':
            return '*'
        return AA_MAP.get(code, code)


    def extract_features(mutation):

        PREFIX = r'[cn]\.'

        # -------------------------
        # PROTEIN-LEVEL: p. notation
        # -------------------------

        # Stop extension: p.Ter559Glnext*? or p.Ter559ext*? or p.Ter628Serext*?
        p_ext = re.match(r'p\.Ter(\d+)([A-Z][a-z]{2})?ext\*\?$', mutation)
        if p_ext:
            pos, alt_aa = p_ext.groups()
            return {'mut_type': 'extension', 'position': int(pos), 'del_len': 0, 'ins_len': 0,
                    'ref': '*', 'alt': parse_aa(alt_aa) if alt_aa else None}

        # Stop codon unknown delins: p.Ter839delins???
        p_ter_delins = re.match(r'p\.Ter(\d+)delins\?+$', mutation)
        if p_ter_delins:
            pos = p_ter_delins.group(1)
            return {'mut_type': 'delins', 'position': int(pos), 'del_len': 1, 'ins_len': 0,
                    'ref': '*', 'alt': None}

        # Frameshift: p.Asp379fs
        p_fs = re.match(r'p\.([A-Z][a-z]{2})(\d+)fs$', mutation)
        if p_fs:
            ref_aa, pos = p_fs.groups()
            return {'mut_type': 'frameshift', 'position': int(pos), 'del_len': 0, 'ins_len': 0,
                    'ref': parse_aa(ref_aa), 'alt': None}

        # Stop/nonsense: p.Trp122*
        p_stop = re.match(r'p\.([A-Z][a-z]{2})(\d+)\*$', mutation)
        if p_stop:
            ref_aa, pos = p_stop.groups()
            return {'mut_type': 'nonsense', 'position': int(pos), 'del_len': 0, 'ins_len': 0,
                    'ref': parse_aa(ref_aa), 'alt': '*'}

        # Malformed nonsense: p.TrpLeu266*
        p_malformed_stop = re.match(r'p\.([A-Z][a-z]{2}[A-Z][a-z]{2})(\d+)\*$', mutation)
        if p_malformed_stop:
            ref_aa, pos = p_malformed_stop.groups()
            return {'mut_type': 'nonsense', 'position': int(pos), 'del_len': 0, 'ins_len': 0,
                    'ref': None, 'alt': '*'}

        # Uncertain: p.Met1?
        p_uncertain = re.match(r'p\.([A-Z][a-z]{2})(\d+)\?$', mutation)
        if p_uncertain:
            ref_aa, pos = p_uncertain.groups()
            return {'mut_type': 'uncertain', 'position': int(pos), 'del_len': 0, 'ins_len': 0,
                    'ref': parse_aa(ref_aa), 'alt': None}

        # Single AA deletion: p.Leu95del
        p_single_del = re.match(r'p\.([A-Z][a-z]{2})(\d+)del$', mutation)
        if p_single_del:
            ref_aa, pos = p_single_del.groups()
            return {'mut_type': 'del', 'position': int(pos), 'del_len': 1, 'ins_len': 0,
                    'ref': parse_aa(ref_aa), 'alt': None}

        # Range AA deletion: p.Val3_Thr4del
        p_range_del = re.match(r'p\.([A-Z][a-z]{2})(\d+)_([A-Z][a-z]{2})(\d+)del$', mutation)
        if p_range_del:
            ref_aa, start, _, end = p_range_del.groups()
            return {'mut_type': 'del', 'position': int(start), 'del_len': abs(int(end) - int(start)) + 1,
                    'ins_len': 0, 'ref': parse_aa(ref_aa), 'alt': None}

        # Range AA duplication: p.His68_Leu70dup
        p_range_dup = re.match(r'p\.([A-Z][a-z]{2})(\d+)_([A-Z][a-z]{2})(\d+)dup$', mutation)
        if p_range_dup:
            ref_aa, start, _, end = p_range_dup.groups()
            return {'mut_type': 'dup', 'position': int(start), 'del_len': 0,
                    'ins_len': abs(int(end) - int(start)) + 1, 'ref': parse_aa(ref_aa), 'alt': None}

        # Range AA insertion: p.Val389_Asp390insGly  (one or more 3-letter AA codes)
        p_range_ins = re.match(r'p\.([A-Z][a-z]{2})(\d+)_([A-Z][a-z]{2})(\d+)ins((?:[A-Z][a-z]{2})+)$', mutation)
        if p_range_ins:
            ref_aa, start, _, end, ins_aas = p_range_ins.groups()
            ins_count = len(re.findall(r'[A-Z][a-z]{2}', ins_aas))
            return {'mut_type': 'ins', 'position': int(start), 'del_len': 0,
                    'ins_len': ins_count, 'ref': parse_aa(ref_aa), 'alt': None}

        # Range AA delins: p.Pro14_Val301delinsLeu or p.Leu443_Lys446delinsProGln
        p_range_delins = re.match(r'p\.([A-Z][a-z]{2})(\d+)_([A-Z][a-z]{2})(\d+)delins((?:[A-Z][a-z]{2})+)$', mutation)
        if p_range_delins:
            ref_aa, start, _, end, ins_aas = p_range_delins.groups()
            ins_count = len(re.findall(r'[A-Z][a-z]{2}', ins_aas))
            return {'mut_type': 'delins', 'position': int(start), 'del_len': abs(int(end) - int(start)) + 1,
                    'ins_len': ins_count, 'ref': parse_aa(ref_aa), 'alt': None}

        # Missense: p.Ala152Val  (must come after all range patterns)
        p_missense = re.match(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', mutation)
        if p_missense:
            ref_aa, pos, alt_aa = p_missense.groups()
            return {'mut_type': 'missense', 'position': int(pos), 'del_len': 0, 'ins_len': 0,
                    'ref': parse_aa(ref_aa), 'alt': parse_aa(alt_aa)}

        # Duplication single AA: p.Ala285dup
        p_dup = re.match(r'p\.([A-Z][a-z]{2})(\d+)dup$', mutation)
        if p_dup:
            ref_aa, pos = p_dup.groups()
            return {'mut_type': 'dup', 'position': int(pos), 'del_len': 0, 'ins_len': 1,
                    'ref': parse_aa(ref_aa), 'alt': None}

        # -------------------------
        # DNA/RNA-LEVEL: c. or n. notation
        # -------------------------

        snv = re.match(PREFIX + r'(-?\d+)([ACGT])>([ACGT])$', mutation)
        if snv:
            pos, ref, alt = snv.groups()
            return {'mut_type': 'SNV', 'position': int(pos), 'del_len': 0, 'ins_len': 0, 'ref': ref, 'alt': alt}

        single_del = re.match(PREFIX + r'(-?\d+)del[ACGT]*$', mutation)
        if single_del:
            pos = single_del.group(1)
            return {'mut_type': 'del', 'position': int(pos), 'del_len': 1, 'ins_len': 0, 'ref': None, 'alt': None}

        range_del = re.match(PREFIX + r'(-?\d+)_(-?\d+)del[ACGT]*$', mutation)
        if range_del:
            start, end = range_del.groups()
            return {'mut_type': 'del', 'position': int(start), 'del_len': abs(int(end) - int(start)) + 1,
                    'ins_len': 0, 'ref': None, 'alt': None}

        ins = re.match(PREFIX + r'(-?\d+)_(-?\d+)ins([ACGT]+)$', mutation)
        if ins:
            start, end, bases = ins.groups()
            return {'mut_type': 'ins', 'position': int(start), 'del_len': 0, 'ins_len': len(bases), 'ref': None, 'alt': None}

        delins = re.match(PREFIX + r'(-?\d+)del[ACGT]*ins([ACGT]+)$', mutation)
        if delins:
            pos, ins_bases = delins.groups()
            return {'mut_type': 'delins', 'position': int(pos), 'del_len': 1, 'ins_len': len(ins_bases), 'ref': None, 'alt': None}

        range_delins = re.match(PREFIX + r'(-?\d+)_(-?\d+)del[ACGT]*ins[ACGT]+$', mutation)
        if range_delins:
            start, end = range_delins.groups()
            return {'mut_type': 'delins', 'position': int(start), 'del_len': abs(int(end) - int(start)) + 1,
                    'ins_len': 0, 'ref': None, 'alt': None}

        dup_single = re.match(PREFIX + r'(-?\d+)dup[ACGT]*$', mutation)
        if dup_single:
            pos = dup_single.group(1)
            return {'mut_type': 'dup', 'position': int(pos), 'del_len': 0, 'ins_len': 1, 'ref': None, 'alt': None}

        dup_range = re.match(PREFIX + r'(-?\d+)_(-?\d+)dup[ACGT]*$', mutation)
        if dup_range:
            start, end = dup_range.groups()
            return {'mut_type': 'dup', 'position': int(start), 'del_len': 0,
                    'ins_len': abs(int(end) - int(start)) + 1, 'ref': None, 'alt': None}

        # -------------------------
        # FREE TEXT
        # -------------------------
        mutation_lower = mutation.lower()
        if 'lof' in mutation_lower or 'loss of function' in mutation_lower:
            return {'mut_type': 'LoF', 'position': None, 'del_len': None, 'ins_len': None, 'ref': None, 'alt': None}
        if 'deletion' in mutation_lower:
            return {'mut_type': 'del', 'position': None, 'del_len': None, 'ins_len': None, 'ref': None, 'alt': None}

        return {'mut_type': 'unknown', 'position': None, 'del_len': None, 'ins_len': None, 'ref': None, 'alt': None}

    data["gene"] = data["gene"].str.strip()
    data["mutation"] = data["mutation"].str.strip()

    one_hot_df = data.copy()

    # --- Apply to dataframe ---
    features_df = one_hot_df['mutation'].apply(extract_features).apply(pd.Series)
    one_hot_df = pd.concat([one_hot_df, features_df], axis=1)

    # --- One-hot encode mut_type, ref, alt ---
    one_hot_df = pd.get_dummies(one_hot_df, columns=['gene', 'mut_type', 'ref', 'alt'])
    one_hot_df = one_hot_df.drop(columns=["mutation", "variant", "chromosome"])
    one_hot_df = one_hot_df.loc[:, ~one_hot_df.columns.duplicated()]

    return one_hot_df


def get_drug_smiles(data):
    drug_list = ['Amikacin', 'Bedaquiline', 'Capreomycin', 'Clofazimine', 'Delamanid',
             'Ethambutol', 'Ethionamide', 'Isoniazid', 'Kanamycin', 'Levofloxacin', 
             'Linezolid', 'Moxifloxacin', 'Pyrazinamide', 'Rifampicin', 'Streptomycin']

    # get the smiles for the drugs
    def get_smiles(drug_name):
        try:
            compounds = pcp.get_compounds(drug_name, "name")
            if len(compounds) > 0:
                return compounds[0].connectivity_smiles
        except:
            return None
        return None

    #apply to the list of TB drugs
    TB_drugs = pd.DataFrame({"drug": drug_list})
    TB_drugs['smiles'] = TB_drugs["drug"].apply(get_smiles)

    # add fingerprints to dataset
    morgan_generator = GetMorganGenerator(radius=2, fpSize=256)

    def smiles_to_fingerprint(smiles):
        if pd.isna(smiles):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fingerprint = morgan_generator.GetFingerprint(mol)
        array = np.zeros((256, ), dtype=int)
        ConvertToNumpyArray(fingerprint, array)
        
        return array

    TB_drugs["fingerprint"] = TB_drugs["smiles"].apply(smiles_to_fingerprint)

    # merge drugs to the model df
    final_data = data.merge(TB_drugs, on="drug", how="left")
    return final_data

def encode_data(data):
    data_encoded = encode_resistance(data)
    data_encoded = encode_mutations(data_encoded)
    data_encoded = get_drug_smiles(data_encoded)
    
    # keep track of mapping for later
    lookup_drugs = dict(enumerate(data_encoded["drug"].astype("category").cat.categories))
    with open("Drug_lookup.txt", "w") as f:
        f.write(f"Drug encoding map: {lookup_drugs}") # we will need this to identify the codes for the drugs we want to hold out if needed

    data_encoded["drug"] = data_encoded["drug"].astype("category").cat.codes
    
    # Now we need to convert the fingerprint arrays into separate columns for each bit, so we can use them in the model
    encoded_df = pd.DataFrame(data_encoded["fingerprint"].tolist(), index=data_encoded.index)
    encoded_df.columns = [f"fp_{i}" for i in range(encoded_df.shape[1])] # change the fingerprints from an array to columns

    final_ml = pd.concat([data_encoded.drop(columns=["fingerprint", "smiles"]), encoded_df], axis=1)
    final_ml = final_ml.loc[:, ~final_ml.columns.duplicated()].copy()
    final_ml = final_ml.drop(
        columns=[
            "mean_log2mic",
            "median_log2mic",
            "count",
            "mean_log2mic_variant",
            "median_log2mic_variant",
            "count_variant"
        ],
        errors="ignore"
    )

    return final_ml, lookup_drugs

def drop_rare_genes(data, threshold=10):
    gene_cols = [c for c in data.columns if c.startswith("gene_")]
    rare = [c for c in gene_cols if data[c].sum() < threshold]

    data = data.drop(columns=rare)
    data["ref_len"] = data["reference_nucleotide"].astype(str).str.len()
    data["alt_len"] = data["alternative_nucleotide"].astype(str).str.len()

    data["ref_base"] = data["reference_nucleotide"].where(
        data["ref_len"] == 1, "LONG"
    )
    data["alt_base"] = data["alternative_nucleotide"].where(
        data["alt_len"] == 1, "LONG"
    )
    data = data.drop(
        columns=["reference_nucleotide", "alternative_nucleotide"],
        errors="ignore"
    )

    data = pd.get_dummies(data, columns=["ref_base", "alt_base"])
    data = data.dropna(subset=["resistant"]).copy()

    return data

def genomic_positions(data):

    gene_key = data.filter(like="gene_").idxmax(axis=1)
    data["mutation_key"] = gene_key + "_" + data['position'].astype(str)

    myco = pd.read_csv("Mycobacterium_tuberculosis_H37Rv_txt_v5.txt", sep="\t")
    myco.columns = myco.columns.str.strip() # strip column names of whitespace

    myco['gene_length'] = myco['Stop'] - myco['Start'] + 1
    myco_subset = myco[['Refseq_ID', 'Start', 'Stop', 'Name','gene_length']].copy()

    new_data = data.copy()
    needed_genes = new_data.filter(like="gene_").columns.str.replace("gene_", "", regex=False)
    
    matched_myco = myco_subset[myco_subset['Name'].isin(needed_genes)].copy()
    gene_cols = [c for c in new_data.columns if c.startswith("gene_") and not c.startswith("gene_length")]

    new_data['gene_name'] = new_data[gene_cols].idxmax(axis=1).str.replace("gene_", "", regex=False).str.lower()
    matched_myco['gene_name'] = matched_myco['Name'].str.strip().str.lower()

    new_data = new_data.merge(matched_myco[['gene_name', 'gene_length', 'Start', 'Stop']], on='gene_name', how='left')
    new_data = new_data.drop(columns=['gene_name'])

    ## add relative position of mutation in the gene
    new_data['relative_position'] = (new_data['position'] - new_data['Start']) + 1
    new_data['norm_position'] = new_data['relative_position'] / new_data['gene_length']
    new_data = new_data.drop(columns=['Start', 'Stop'])

    ## change to distances
    new_data['distance_to_end'] = new_data['gene_length'] - new_data['relative_position']
    new_data['codon_position'] = new_data['relative_position'] % 3
    new_data = new_data.drop(columns=['relative_position', 'gene_length'], errors="ignore")
    
    return new_data



def full_data_pipeline():
    data = main()
    data_encoded, drug_lookup = encode_data(data)
    data_clean = drop_rare_genes(data_encoded)

    genomic_positions_data = data_clean.copy()
    data_genomic_positions = genomic_positions(genomic_positions_data)

    return data_clean, data_genomic_positions, drug_lookup


if __name__ == "__main__":
    data, data_genomic_positions, drug_lookup = full_data_pipeline()
