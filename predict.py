import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Bio import SeqIO
from utils import encode_sequence, load_go_terms, load_go_metadata, MAXLEN

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
    go_meta = load_go_metadata(GO_OBO_PATH)

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
            go_info = go_meta.get(go_id, {})
            
            # Format relationships nicely
            part_of_terms = []
            other_relationships = []
            for rel in go_info.get("relationship", []):
                if "part_of" in rel:
                    # Format: part_of GO:xxxxxxx ! name
                    go_part = rel.split()[1] if len(rel.split()) > 1 else ""
                    if go_part:
                        part_of_terms.append(go_part)
                else:
                    other_relationships.append(rel)
            
            # Format subsets
            subset_str = "; ".join(go_info.get("subset", []))
            
            # Format cross-references by database
            xref_by_db = {}
            for xref in go_info.get("xref", []):
                if ":" in xref:
                    db, ref = xref.split(":", 1)
                    if db not in xref_by_db:
                        xref_by_db[db] = []
                    xref_by_db[db].append(ref.strip())
            
            xref_str = "; ".join([f"{db}: {', '.join(refs)}" for db, refs in xref_by_db.items()])
            
            results.append({
                # Protein and Prediction
                "protein": protein_id,
                "score": round(float(scores[idx]), 4),
                
                # GO Term Core Info
                "GO_term": go_id,
                "GO_name": go_info.get("name", "Unknown"),
                "GO_namespace": go_info.get("namespace", ""),
                "GO_definition": go_info.get("def", ""),
                
                # Variations
                "GO_synonyms": "; ".join(go_info.get("synonyms", [])),
                "GO_alternate_IDs": "; ".join(go_info.get("alt_id", [])),
                
                # Hierarchy
                "GO_parent_terms": "; ".join(go_info.get("is_a", [])),
                "GO_part_of": "; ".join(part_of_terms),
                
                # Status & Alternatives
                "GO_obsolete": go_info.get("is_obsolete", False),
                "GO_replaced_by": "; ".join(go_info.get("replaced_by", [])),
                "GO_consider": "; ".join(go_info.get("consider", [])),
                
                # External References
                "GO_xref": xref_str,
                
                # Subset Information
                "GO_subset": subset_str,
                
                # Additional Notes
                "GO_comment": go_info.get("comment", ""),
                
                # Audit Information
                "GO_created_by": go_info.get("created_by", ""),
                "GO_creation_date": go_info.get("creation_date", ""),
            })

    # 6. Save
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Success! Results saved to {OUTPUT_FILE}")
    print(df.head(5))

if __name__ == "__main__":
    main()