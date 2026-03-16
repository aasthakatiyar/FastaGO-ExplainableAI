import numpy as np
import pandas as pd
import pickle
import os
from Bio import SeqIO

MAXLEN = 2000
AA_LIST = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_INDEX = {aa: i + 1 for i, aa in enumerate(AA_LIST)}

def encode_sequence(seq):
    arr = np.zeros((MAXLEN, 21), dtype=np.float32)
    for i, aa in enumerate(seq[:MAXLEN].upper()):
        if aa in AA_INDEX:
            arr[i, AA_INDEX[aa]] = 1
        else:
            arr[i, 0] = 1
    return arr

def load_go_terms(path):
    with open(path, "rb") as f:
        terms = pickle.load(f)
    return terms.iloc[:, 0].tolist() if isinstance(terms, pd.DataFrame) else list(terms)

def load_go_names(path):
    names = {}
    if os.path.exists(path):
        with open(path) as f:
            cid = ""
            for line in f:
                if line.startswith("id:"): cid = line.strip().split(": ")[1]
                elif line.startswith("name:"): names[cid] = line.strip().split(": ")[1]
    return names