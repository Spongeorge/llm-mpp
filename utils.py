import pandas as pd
import numpy as np
import faiss
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

def smiles_to_fingerprint(smiles, fingerprint_generator):
    try:
      mol = Chem.MolFromSmiles(smiles)
      return fingerprint_generator.GetFingerprint(mol)
    except:
      mol = Chem.MolFromSmiles(smiles,sanitize=False)
      return fingerprint_generator.GetFingerprint(mol)


def find_similar_molecules(target_smiles, df, k=5):
    similarities = []
    for _, row in df.iterrows():
        try:
          similarity = DataStructs.TanimotoSimilarity(smiles_to_fingerprint(target_smiles), smiles_to_fingerprint(row["smiles"]))
          similarities.append((similarity, row))
        except:
          pass
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:k]