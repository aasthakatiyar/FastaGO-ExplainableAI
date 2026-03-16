import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Bio import SeqIO

MAXLEN = 2000

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a:i for i,a in enumerate(AA)}

# one-hot encoding
def encode(seq):
    arr = np.zeros((MAXLEN, 21))
    
    for i, aa in enumerate(seq[:MAXLEN]):
        if aa in AA_INDEX:
            arr[i, AA_INDEX[aa]] = 1
        else:
            arr[i, 20] = 1  # unknown AA
    
    return arr


print("Loading model...")
model = load_model("model/model.h5")

sequences = []
names = []

print("Reading FASTA...")

for record in SeqIO.parse("input/protein.fasta", "fasta"):
    names.append(record.id)
    sequences.append(encode(str(record.seq)))

X = np.array(sequences)

print("Input shape:", X.shape)

print("Running prediction...")

preds = model.predict(X)

df = pd.DataFrame(preds)
df.insert(0, "protein", names)

df.to_csv("output/predictions.csv", index=False)

print("Prediction completed.")