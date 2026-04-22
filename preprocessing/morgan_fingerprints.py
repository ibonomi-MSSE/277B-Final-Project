"""

Turns a SMILES string into either a morgan fingerprint.
This output will be concatenated with the mutation-over-loci vector to form the
input for other prediction models.

"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def smiles_to_morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES string')
    
    # radius=2 is standard (equivalent to ECFP4)
    # nBits=2048 is the most common bit vector size
    # useFeatures = use pharmacophoric atom types instead of atomic properties
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=True)
    return np.array(fp)
