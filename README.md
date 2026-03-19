# DeepGOPlus Protein Function Predictor

DeepGOPlus is a lightweight protein function prediction tool based on a deep learning model. Given one or more protein sequences (in FASTA format), it predicts Gene Ontology (GO) terms associated with the protein using a pre-trained neural network.

---

## ✅ Key Features

- **Predicts GO terms** from protein sequences using a deep learning model (`model/model.h5`)
- **Comprehensive GO.obo Analysis**: Captures 100% of available metadata (xrefs, subsets, hierarchy, relationships)
- **Interactive Web UI**: Streamlit-based UI with summary and detailed views (`app.py`)
- **Batch Processing**: Command-line runner for large FASTA files (`predict.py`)
- **Automated Reporting**: Generates ontology analysis reports (`analyze_ontology.py`)
- **Multi-database Integration**: Links predictions to EC, KEGG, Reactome, RHEA, and more
- Supports **multiple input modes** (file upload, manual paste, sample sequences)
- Outputs **enriched results** with full hierarchical context and external references

---

## 📁 Repository Structure

```
app.py                # Streamlit web UI
predict.py            # CLI prediction script
utils.py              # Encoding + GO metadata loader
debug_sync.py         # Model/labels sync validator
requirements.txt      # Python dependencies

data/
  go.obo              # GO ontology names (id -> name mapping)
  terms.pkl           # List of GO term IDs the model predicts

model/
  model.h5            # Trained Keras model used for prediction

input/
  protein.fasta       # Example input FASTA (used by predict.py)

output/
  go_predictions.csv  # Example output
```

---

## 🚀 Installation

1. **Create & activate a Python environment** (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

> ✅ The current requirements include `tensorflow`, `streamlit`, `biopython`, `pandas`, and `numpy`.

---

## 🧬 Usage

### 1) Run the Streamlit web app (recommended for exploration)

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

#### What you can do in the UI
- Upload a FASTA file (multi-sequence supported)
- Paste a single sequence or a FASTA block
- Load one of the built-in sample proteins
- Adjust **confidence threshold** and **max terms per protein**
- Download predictions as CSV

---

### 2) Run predictions via CLI

The CLI runner reads `input/protein.fasta` and writes `output/go_predictions.csv`.

```bash
python predict.py
```

You can customize the input/output paths and thresholds by editing the constants at the top of `predict.py`.

---

## 🔍 How It Works (High Level)

1. The model accepts protein sequences encoded as one-hot vectors (length capped at **2000** amino acids).
2. The model outputs a probability for each GO term.
3. Predictions are filtered by a **confidence threshold** and ranked.
4. The output includes:
   - Protein / sequence ID
   - GO term ID
   - GO term name (from `data/go.obo`)
   - GO term namespace (biological_process / molecular_function / cellular_component)
   - GO term definition (from `data/go.obo`)
   - GO term synonyms, alternate IDs, and obsoletion info (if available)
   - Score (model confidence)

---

## 🧪 Validating Model / Labels Sync

If you change the model or the GO term list, ensure they remain compatible:

```bash
python debug_sync.py
```

A mismatch between `model/model.h5` output size and `data/terms.pkl` length will be reported.

---

## 📝 Input Format

- Input must be in **FASTA format**.
- Sequences longer than **2000 residues** are truncated.
- Unrecognized amino acids are treated as unknown.

Example FASTA:

```fasta
>my_protein
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDT
```

---

## 📌 Notes / Tips

- You can swap the model file (e.g., train a new model) but ensure you regenerate `data/terms.pkl` accordingly.
- The GO term names come from `data/go.obo`, so keep that file consistent with your term list.
- If you want a different prediction threshold / result count, adjust the `threshold` and `top_k` values in the UI or in `predict.py`.

---

## 🧰 Extending / Customizing

- **Add more sample proteins**: edit `SAMPLE_SEQUENCES` in `app.py`.
- **Support alternate input files**: modify `predict.py` to accept CLI args with `argparse`.
- **Train your own model**: replace `model/model.h5` with your model and update `data/terms.pkl` to match.

---

## 📄 License

This repository does not include an explicit license file. If you plan to redistribute or modify it, consider adding a `LICENSE` file.
