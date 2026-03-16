import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from Bio import SeqIO

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
MODEL_PATH = "model/model.h5"
TERMS_PATH = "data/terms.pkl"
GO_OBO_PATH = "data/go.obo"
INPUT_FASTA = "input/protein.fasta"
OUTPUT_FILE = "output/go_predictions.csv"

MAXLEN = 2000
THRESHOLD = 0.1  # Filter out low-confidence background noise
TOP_K = 20       # Keep top 20 terms per protein

# OFFICIAL ALPHABET (DeepGOPlus Original)
# Alphabetical order: ARNDCQEGHILKMFPSTWYV
AA = "ARNDCQEGHILKMFPSTWYV" 
AA_INDEX = {a: i + 1 for i, a in enumerate(AA)} 

# --------------------------------------------------
# 2. ENCODING & LOADERS
# --------------------------------------------------

def encode_sequence(seq):
    """Encodes sequence to (2000, 21) matching model weights."""
    arr = np.zeros((MAXLEN, 21), dtype=np.float32)
    for i, aa in enumerate(seq[:MAXLEN].upper()):
        if aa in AA_INDEX:
            arr[i, AA_INDEX[aa]] = 1
        else:
            arr[i, 0] = 1 # Index 0 handles 'X', 'B', 'Z' or Padding
    return arr

def load_go_terms(path):
    print(f"Loading GO terms from {path}...")
    with open(path, "rb") as f:
        terms = pickle.load(f)
    if isinstance(terms, pd.DataFrame):
        return terms.iloc[:, 0].tolist()
    return list(terms)

def load_go_names(path):
    print(f"Parsing GO names from {path}...")
    names = {}
    if os.path.exists(path):
        with open(path) as f:
            cid = ""
            for line in f:
                if line.startswith("id:"): cid = line.strip().split(": ")[1]
                elif line.startswith("name:"): names[cid] = line.strip().split(": ")[1]
    return names

# --------------------------------------------------
# 3. MAIN PIPELINE
# --------------------------------------------------

def main():
    print("\n===== DeepGOPlus: Functional Prediction Pipeline =====")

    # Validate Files
    for p in [MODEL_PATH, TERMS_PATH, INPUT_FASTA]:
        if not os.path.exists(p):
            print(f"ERROR: Missing file: {p}")
            return

    # Load Components
    go_terms = load_go_terms(TERMS_PATH)
    go_names = load_go_names(GO_OBO_PATH)
    print("Loading H5 Model (TensorFlow)...")
    model = load_model(MODEL_PATH)
    print(f"Model Input Shape: {model.input_shape}")

    # Process Sequences
    ids, encoded_seqs = [], []
    for record in SeqIO.parse(INPUT_FASTA, "fasta"):
        ids.append(record.id)
        encoded_seqs.append(encode_sequence(str(record.seq)))
    
    X = np.array(encoded_seqs)
    print(f"Running predictions for {len(ids)} sequences...")
    
    # Run Inference
    preds = model.predict(X)

    # Format Results
    results = []
    for i, protein_id in enumerate(ids):
        scores = preds[i]
        # Filter by threshold and take Top K
        idx_above_thresh = np.where(scores >= THRESHOLD)[0]
        sorted_idx = idx_above_thresh[np.argsort(scores[idx_above_thresh])[::-1]]
        
        for idx in sorted_idx[:TOP_K]:
            go_id = go_terms[idx]
            results.append({
                "protein": protein_id,
                "GO_term": go_id,
                "GO_name": go_names.get(go_id, "Unknown"),
                "score": round(float(scores[idx]), 4)
            })

    # Save to CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSUCCESS: {len(df)} predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()