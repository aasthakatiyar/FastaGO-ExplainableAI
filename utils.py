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

def load_go_metadata(path):
    """Load GO term metadata from a GO .obo file.

    Returns a mapping: GO_ID -> metadata dict.
    """
    metadata = {}
    if not os.path.exists(path):
        return metadata

    current_term = None

    def flush_term():
        nonlocal current_term
        if current_term is not None and current_term.get("id"):
            metadata[current_term["id"]] = current_term
        return

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == "[Term]":
                flush_term()
                current_term = {
                    "id": None,
                    "name": None,
                    "namespace": None,
                    "def": None,
                    "synonyms": [],
                    "alt_id": [],
                    "is_obsolete": False,
                    "replaced_by": [],
                    "comment": None,
                    "is_a": [],
                    "consider": [],
                    "relationship": [],
                }
                continue

            if current_term is None:
                continue

            if line.startswith("id:"):
                current_term["id"] = line.split(": ", 1)[1]
            elif line.startswith("name:"):
                current_term["name"] = line.split(": ", 1)[1]
            elif line.startswith("namespace:"):
                current_term["namespace"] = line.split(": ", 1)[1]
            elif line.startswith("def:"):
                # Remove surrounding quotes but keep the text
                current_term["def"] = line.split(": ", 1)[1].strip().strip('"')
            elif line.startswith("synonym:"):
                # Format: synonym: "text" SCOPE [xrefs]
                parts = line.split('"')
                if len(parts) >= 2:
                    current_term["synonyms"].append(parts[1])
            elif line.startswith("alt_id:"):
                current_term["alt_id"].append(line.split(": ", 1)[1])
            elif line.startswith("is_obsolete:"):
                current_term["is_obsolete"] = line.split(": ", 1)[1].lower() == "true"
            elif line.startswith("replaced_by:"):
                current_term["replaced_by"].append(line.split(": ", 1)[1])
            elif line.startswith("comment:"):
                current_term["comment"] = line.split(": ", 1)[1]
            elif line.startswith("is_a:"):
                # Format: is_a: GO:xxxxxxx ! name
                current_term["is_a"].append(line.split(": ", 1)[1].split(" ! ")[0])
            elif line.startswith("consider:"):
                current_term["consider"].append(line.split(": ", 1)[1])
            elif line.startswith("relationship:"):
                # Keep raw relationship line for later interpretation
                current_term["relationship"].append(line.split(": ", 1)[1])

    flush_term()
    return metadata


def load_go_names(path):
    """Legacy helper: load only GO term names."""
    meta = load_go_metadata(path)
    return {go_id: info.get("name") for go_id, info in meta.items()}