import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from Bio import SeqIO

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "model/model.h5"
TERMS_PATH = "data/terms.pkl"
GO_OBO_PATH = "data/go.obo"
INPUT_FASTA = "input/protein.fasta"
OUTPUT_FILE = "output/go_predictions.csv"

MAXLEN = 2000
TOP_K = 10

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# --------------------------------------------------
# ENCODING FUNCTION
# --------------------------------------------------

def encode_sequence(seq):
    """
    One-hot encode amino acid sequence
    """
    arr = np.zeros((MAXLEN, 21))

    for i, aa in enumerate(seq[:MAXLEN]):
        if aa in AA_INDEX:
            arr[i, AA_INDEX[aa]] = 1
        else:
            arr[i, 20] = 1

    return arr


# --------------------------------------------------
# LOAD GO TERMS
# --------------------------------------------------

def load_go_terms(path):

    print("Loading GO term index...")

    with open(path, "rb") as f:
        terms = pickle.load(f)

    # terms.pkl may be stored as dataframe
    if isinstance(terms, pd.DataFrame):
        terms = terms.iloc[:, 0].tolist()

    elif isinstance(terms, np.ndarray):
        terms = terms.tolist()

    print("Total GO terms:", len(terms))

    return terms


# --------------------------------------------------
# LOAD GO NAMES FROM ONTOLOGY
# --------------------------------------------------

def load_go_names(path):

    print("Parsing GO ontology...")

    go_names = {}
    current_id = None

    with open(path) as f:

        for line in f:

            if line.startswith("id:"):
                current_id = line.strip().split()[1]

            elif line.startswith("name:"):
                name = line.strip()[6:]
                go_names[current_id] = name

    print("GO names loaded:", len(go_names))

    return go_names


# --------------------------------------------------
# LOAD FASTA SEQUENCES
# --------------------------------------------------

def load_sequences(fasta_path):

    print("Reading FASTA file...")

    names = []
    sequences = []

    for record in SeqIO.parse(fasta_path, "fasta"):

        names.append(record.id)
        sequences.append(encode_sequence(str(record.seq)))

    X = np.array(sequences)

    print("Sequences loaded:", len(names))
    print("Input tensor shape:", X.shape)

    return names, X


# --------------------------------------------------
# RUN MODEL
# --------------------------------------------------

def run_prediction(model, X):

    print("Running model prediction...")

    preds = model.predict(X)

    return preds


# --------------------------------------------------
# EXTRACT TOP-K GO TERMS
# --------------------------------------------------

def extract_predictions(names, preds, go_terms, go_names):

    print("Extracting Top-K predictions...")

    results = []

    for i, protein in enumerate(names):

        scores = preds[i]

        top_indices = np.argsort(scores)[::-1][:TOP_K]

        for idx in top_indices:

            go_id = go_terms[idx]
            score = float(scores[idx])
            go_name = go_names.get(go_id, "unknown")

            results.append({
                "protein": protein,
                "GO_term": go_id,
                "GO_name": go_name,
                "score": score
            })

    df = pd.DataFrame(results)

    df = df.sort_values(["protein", "score"], ascending=[True, False])

    return df


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    print("\n===== DeepGOPlus Local Prediction =====\n")

    go_terms = load_go_terms(TERMS_PATH)

    go_names = load_go_names(GO_OBO_PATH)

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    names, X = load_sequences(INPUT_FASTA)

    preds = run_prediction(model, X)

    df = extract_predictions(names, preds, go_terms, go_names)

    os.makedirs("output", exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPrediction completed successfully.")
    print("Results saved to:", OUTPUT_FILE)


# --------------------------------------------------

if __name__ == "__main__":
    main()