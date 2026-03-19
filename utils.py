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
    """Load comprehensive GO term metadata from a GO .obo file."""
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
                    "id": None, "name": None, "namespace": None,
                    "def": None, "comment": None,
                    "synonyms": [], "alt_id": [], "consider": [],
                    "is_obsolete": False, "replaced_by": [],
                    "is_a": [], "relationship": [],
                    "xref": [], "subset": [],
                    "created_by": None, "creation_date": None,
                    "property_value": {},
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
                parts = line.split(": ", 1)[1].split('"')
                if len(parts) >= 2: current_term["def"] = parts[1]
            elif line.startswith("synonym:"):
                parts = line.split('"')
                if len(parts) >= 2:
                    syn_text = parts[1]
                    remaining = parts[2].strip() if len(parts) > 2 else ""
                    scope = remaining.split()[0] if remaining else "UNKNOWN"
                    current_term["synonyms"].append(f"{syn_text} ({scope})")
            elif line.startswith("alt_id:"):
                current_term["alt_id"].append(line.split(": ", 1)[1])
            elif line.startswith("is_obsolete:"):
                current_term["is_obsolete"] = line.split(": ", 1)[1].lower() == "true"
            elif line.startswith("replaced_by:"):
                current_term["replaced_by"].append(line.split(": ", 1)[1])
            elif line.startswith("consider:"):
                current_term["consider"].append(line.split(": ", 1)[1])
            elif line.startswith("comment:"):
                current_term["comment"] = line.split(": ", 1)[1]
            elif line.startswith("is_a:"):
                parts = line.split(": ", 1)[1]
                if " ! " in parts:
                    go_id, name = parts.split(" ! ", 1)
                    current_term["is_a"].append(f"{go_id} ({name})")
                else:
                    current_term["is_a"].append(parts)
            elif line.startswith("relationship:"):
                rel_line = line.split(": ", 1)[1]
                parts = rel_line.split(" ", 1)
                if len(parts) == 2:
                    rel_type = parts[0]
                    go_id_and_name = parts[1]
                    name_parts = go_id_and_name.split(" (", 1)
                    if len(name_parts) == 2 and name_parts[1].endswith(")"):
                        go_id = name_parts[0]
                        name = name_parts[1][:-1]
                    else:
                        go_id = go_id_and_name
                        name = "N/A"
                    current_term["relationship"].append(f"{rel_type} {go_id} ({name})")
                else:
                    current_term["relationship"].append(rel_line)
            elif line.startswith("xref:"):
                current_term["xref"].append(line.split(": ", 1)[1])
            elif line.startswith("subset:"):
                current_term["subset"].append(line.split(": ", 1)[1])
            elif line.startswith("created_by:"):
                current_term["created_by"] = line.split(": ", 1)[1]
            elif line.startswith("creation_date:"):
                current_term["creation_date"] = line.split(": ", 1)[1]
            elif line.startswith("property_value:"):
                parts = line.split(": ", 1)[1].split('"')
                key = parts[0].strip()
                val = parts[1] if len(parts) > 1 else ""
                current_term["property_value"][key] = val

    flush_term()
    return metadata

def load_go_names(path):
    meta = load_go_metadata(path)
    return {go_id: info.get("name") for go_id, info in meta.items()}