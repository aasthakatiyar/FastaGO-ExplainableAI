# DeepGOPlus Protein Function Predictor

A Streamlit web application for predicting Gene Ontology (GO) terms from protein sequences using the DeepGOPlus model.

## Features

- 🧬 Predict protein functions from amino acid sequences
- 📁 Upload FASTA files or enter sequences directly
- 🎛️ Adjustable confidence threshold for filtering results
- 📊 Interactive results table with download option
- 🚀 Fast predictions using pre-trained DeepGOPlus model

## Installation

1. **Clone or download this repository**

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model file exists:**
   The app requires `model/model.h5` to be present. Download it from:
   https://deepgoplus.bio2vec.net/data/model.h5

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   The app will open at `http://localhost:8501`

3. **Input protein sequences:**
   - **Option 1:** Enter FASTA format text directly in the sidebar
   - **Option 2:** Upload a FASTA file (.fasta, .fa, .faa)

4. **Adjust settings:**
   - Set confidence threshold (0.0-1.0) to filter predictions

5. **Click "Predict Functions"**

6. **View and download results:**
   - See high-confidence predictions in the main table
   - Download results as CSV
   - View raw predictions in the expandable section

## Input Format

Proteins must be provided in FASTA format:

```
>Protein1
MKWVTFISLLFLFSSAYS

>Protein2
GGGSSVKVLVVVVGGGG
```

## Output

- **Protein:** Sequence identifier
- **GO_Term_ID:** Gene Ontology term identifier (GO_0, GO_1, etc.)
- **Confidence_Score:** Prediction confidence (0.0-1.0)

Only predictions above the selected threshold are shown.

## Model Details

- **Model:** DeepGOPlus (CNN-based)
- **Max sequence length:** 2000 amino acids
- **Amino acid encoding:** One-hot encoding (21 channels: 20 AA + unknown)
- **Output:** Probability scores for ~5000 GO terms

## Requirements

- Python 3.8+
- TensorFlow 2.10
- 8GB RAM recommended
- Internet connection for initial setup

## Troubleshooting

- **Model loading error:** Ensure `model/model.h5` exists and is not corrupted
- **Memory issues:** Reduce batch size or sequence length if needed
- **FASTA parsing errors:** Check that input follows proper FASTA format

## License

This project uses the DeepGOPlus model. Please refer to the original DeepGOPlus publication and licensing terms.</content>
<parameter name="filePath">d:\Fasta GO\protein_prediction\README.md