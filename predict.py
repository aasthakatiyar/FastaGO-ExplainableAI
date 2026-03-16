import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Bio import SeqIO
from utils import encode_sequence, load_go_terms, load_go_names, MAXLEN

# --- CONFIGURATION ---
MODEL_PATH = "model/model.h5"
TERMS_PATH = "data/terms.pkl"
GO_OBO_PATH = "data/go.obo"
INPUT_FASTA = "input/protein.fasta"
OUTPUT_FILE = "output/go_predictions.csv"

THRESHOLD = 0.3
TOP_K = 15

def main():
    print("\n--- DeepGOPlus CLI Predictor (Refactored) ---")

    # 1. Validation
    for path in [MODEL_PATH, TERMS_PATH, INPUT_FASTA]:
        if not os.path.exists(path):
            print(f"Error: Required file missing at {path}")
            return

    # 2. Load Assets
    print("Loading model and metadata...")
    model = load_model(MODEL_PATH)
    go_terms = load_go_terms(TERMS_PATH)
    go_names = load_go_names(GO_OBO_PATH)

    # 3. Process Sequences
    ids, encoded_seqs = [], []
    for record in SeqIO.parse(INPUT_FASTA, "fasta"):
        ids.append(record.id)
        encoded_seqs.append(encode_sequence(str(record.seq)))
    
    if not ids:
        print("No sequences found in FASTA file.")
        return

    # 4. Predict
    print(f"Running predictions for {len(ids)} sequence(s)...")
    preds = model.predict(np.array(encoded_seqs))

    # 5. Extract and Filter
    results = []
    for i, protein_id in enumerate(ids):
        scores = preds[i]
        # Find indices above threshold
        valid_idx = np.where(scores >= THRESHOLD)[0]
        # Sort indices by score descending
        sorted_idx = valid_idx[np.argsort(scores[valid_idx])[::-1]]
        
        for idx in sorted_idx[:TOP_K]:
            go_id = go_terms[idx]
            results.append({
                "protein": protein_id,
                "GO_term": go_id,
                "GO_name": go_names.get(go_id, "Unknown"),
                "score": round(float(scores[idx]), 4)
            })

    # 6. Save
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Success! Results saved to {OUTPUT_FILE}")
    print(df.head(5))

if __name__ == "__main__":
    main()