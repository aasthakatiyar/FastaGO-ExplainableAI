# CLAUDE.md

## DeepGOPlus Local Prediction Setup (Windows)

This document explains how to run **DeepGOPlus protein function prediction locally on Windows** using a lightweight inference-only setup.

The goal is:

FASTA → DeepGOPlus → GO term prediction

This setup avoids the full pipeline requirements such as DIAMOND databases and large UniProt downloads.

---

# 1. Project Structure

Create the following directory structure:

```
protein_prediction/
│
├── model/
│   └── model.h5
│
├── data/
│   └── go.obo
│
├── input/
│   └── protein.fasta
│
├── output/
│   └── predictions.csv
│
├── predict.py
└── CLAUDE.md
```

---

# 2. System Requirements

Recommended environment:

* Windows 10 or Windows 11
* Python 3.10
* 8 GB RAM minimum
* Internet connection for downloading model and ontology files

---

# 3. Create Python Virtual Environment

Open terminal in the project folder.

Create environment:

```
python -m venv venv
```

Activate environment:

```
venv\Scripts\activate
```

Upgrade pip:

```
python -m pip install --upgrade pip
```

---

# 4. Install Dependencies

Install the minimal required libraries:

```
pip install tensorflow==2.10
pip install numpy pandas scikit-learn biopython tqdm click
```

These libraries support model loading, sequence parsing, and prediction.

---

# 5. Download Pretrained DeepGOPlus Model

Download pretrained model:

https://deepgoplus.bio2vec.net/data/model.h5

Place the file in:

```
model/model.h5
```

---

# 6. Download Gene Ontology File

Download ontology file:

http://purl.obolibrary.org/obo/go.obo

Place it in:

```
data/go.obo
```

---

# 7. Prepare FASTA Input

Create input file:

```
input/protein.fasta
```

Example:

```
>Protein1
MKWVTFISLLFLFSSAYS

>Protein2
GGGSSVKVLVVVVGGGG
```

Each sequence must contain a valid amino acid sequence.

---

# 8. Prediction Script

Create `predict.py`.

```
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Bio import SeqIO

MAXLEN = 2000
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {a:i+1 for i,a in enumerate(AA)}

def encode(seq):
    arr = np.zeros(MAXLEN)
    for i,aa in enumerate(seq[:MAXLEN]):
        arr[i] = AA_INDEX.get(aa,0)
    return arr

model = load_model("model/model.h5")

sequences = []
names = []

for record in SeqIO.parse("input/protein.fasta","fasta"):
    names.append(record.id)
    sequences.append(encode(str(record.seq)))

X = np.array(sequences)

preds = model.predict(X)

df = pd.DataFrame(preds)
df.insert(0,"protein",names)

df.to_csv("output/predictions.csv",index=False)

print("Prediction completed.")
```

---

# 9. Run Prediction

Activate environment and run:

```
python predict.py
```

Output will be generated at:

```
output/predictions.csv
```

---

# 10. Output Format

Example output:

```
protein,GO_1,GO_2,GO_3,...
Protein1,0.82,0.11,0.03
Protein2,0.45,0.78,0.01
```

Each column corresponds to a GO prediction score.

Higher score = higher confidence.

---

# 11. Optional Threshold Filtering

To obtain predicted GO terms:

```
threshold = 0.5
```

Select GO terms whose score exceeds the threshold.

---

# 12. Pipeline Overview

Prediction pipeline:

```
FASTA Sequence
      ↓
Sequence Encoding
      ↓
DeepGOPlus CNN Model
      ↓
GO Prediction Scores
      ↓
Filtered GO Terms
```

---

# 13. Notes

* Maximum sequence length used by the model is 2000 amino acids.
* Longer sequences are truncated.
* Unknown amino acids are encoded as zero.
* Prediction runs on CPU by default.

---

# 14. Future Extensions

This setup can be extended with:

* GO ontology hierarchy propagation
* SHAP-based explainability
* Integration with protein language models
* Automated GO term description generation

These extensions can support research in explainable protein function prediction.
