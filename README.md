# DeepGOPlus Protein Function Prediction System

A complete implementation of protein function prediction using the **DeepGOPlus** deep learning architecture. This system takes protein FASTA sequences as input and predicts Gene Ontology (GO) terms with confidence scores.

## 🎯 Features

- **DeepGOPlus CNN Architecture**: Pre-trained deep learning model for GO term prediction
- **FASTA Parsing**: Support for FASTA file input and text-based sequence input
- **Batch Processing**: Predict GO terms for multiple proteins simultaneously
- **Streamlit Web Interface**: User-friendly web application for predictions
- **CSV Export**: Save prediction results in standard CSV format
- **Configurable Threshold**: Adjust confidence thresholds for predictions
- **Reproducible Pipeline**: Complete setup for thesis demonstrations

## 📋 Project Structure

```
protein-go-prediction/
├── app/
│   └── streamlit_app.py          # Web interface
├── data/
│   ├── raw/                       # Dataset files
│   │   ├── go.obo                # GO ontology
│   │   ├── train_data.pkl        # Training data
│   │   ├── test_data.pkl         # Test data
│   │   └── terms.pkl             # GO terms list
│   └── diamond_db/               # DIAMOND database
├── models/
│   └── model.h5                  # Pre-trained CNN model
├── src/
│   ├── download_data.py          # Download dataset
│   ├── extract_data.py           # Extract archive
│   ├── create_test_model.py      # Create test model for validation
│   ├── load_model.py             # Model loading
│   ├── fasta_parser.py           # FASTA parsing
│   ├── predictor.py              # Main prediction pipeline
│   └── utils.py                  # Utility functions
├── examples/
│   └── sample.fasta              # Example sequences
├── outputs/
│   └── predictions.csv           # Output predictions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Create Python Virtual Environment

```bash
# Windows
python -m venv deepgo-env
deepgo-env\Scripts\activate

# Linux/Mac
python -m venv deepgo-env
source deepgo-env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Data and Model

```bash
python src/download_data.py
```

This downloads the dataset (834 MB). 

**For the pre-trained model** (2+ GB):
- The model must be downloaded manually from: http://deepgoplus.bio2vec.net/
- Download `model.h5` and place it in the `models/` directory
- This file is required for making predictions

Alternatively, you can use a compatible model from the official DeepGOPlus repository.

#### Testing Without Full Model

To test the pipeline before downloading the full model:

```bash
python src/create_test_model.py
```

This creates a small random test model for validation purposes.

### 4. Extract Dataset

```bash
python src/extract_data.py
```

### 5. Run Web Application

```bash
streamlit run app/streamlit_app.py
```

The application will open at `http://localhost:8501`

## 💻 Command-Line Usage

### Predict from FASTA File

```bash
python src/predictor.py --input examples/sample.fasta
```

### Advanced Options

```bash
python src/predictor.py \
  --input proteins.fasta \
  --output results.csv \
  --threshold 0.7 \
  --model models/model.h5 \
  --data data/raw
```

### Options

- `--input, -i` (required): Input FASTA file path
- `--output, -o`: Output CSV file (default: `outputs/predictions.csv`)
- `--threshold, -t`: Confidence threshold (default: `0.5`)
- `--model, -m`: Path to model.h5 (default: `models/model.h5`)
- `--data, -d`: Path to data directory (default: `data/raw`)

## 🌐 Web Application

### Features

The Streamlit web interface provides:

1. **Text Input Tab**: Paste FASTA sequences directly
2. **File Upload Tab**: Upload FASTA files
3. **Examples Tab**: Try prediction with sample proteins
4. **Adjustable Settings**:
   - Confidence threshold slider
   - Maximum results per protein
5. **Results Display**: Interactive table with predictions
6. **CSV Export**: Download results in CSV format

### Input Format

FASTA format for protein sequences:

```fasta
>protein_name
MALWMRLLPLLALLALWGPDPAAALESGERCQVQLVAGVQKARDSVRVS
VGTPGLMAALWDVQPGAPGSWQNKQLQLLQELKQVDVKKMESLGTQVPPL
F

>another_protein
MKVLFSLLTSIFSCFSACVENQSQEFYGQWKYDHTDDRVYHPHFDLSHGSAQ
```

### Output Format

Results are displayed as a table:

| Protein ID | GO Term | Confidence Score |
|-----------|---------|------------------|
| protein1 | GO:0005524 | 0.9124 |
| protein1 | GO:0004672 | 0.8756 |

And can be exported as CSV:

```csv
protein_id,GO_term,score
protein1,GO:0005524,0.9124
protein1,GO:0004672,0.8756
```

## 📦 Python Modules

### `fasta_parser.py`
- `parse_fasta()`: Parse FASTA files
- `parse_fasta_string()`: Parse FASTA from string
- `validate_sequence()`: Validate amino acid sequences

### `load_model.py`
- `DeepGOPlusModel`: Main model class
- `get_default_model()`: Load pre-configured model

### `predictor.py`
- `ProteinFunctionPredictor`: Main prediction class
- Command-line interface for batch predictions

### `utils.py`
- `one_hot_encode()`: Sequence encoding
- `filter_predictions()`: Apply confidence threshold
- `pad_sequence()`: Sequence padding/truncation

### `download_data.py`
- Download pre-trained model and dataset

### `extract_data.py`
- Extract and verify dataset files

## 🔧 Configuration

### Confidence Threshold

Adjust the threshold to control prediction specificity:

- **Lower threshold** (0.3-0.4): More predictions, lower precision
- **Default** (0.5): Balanced predictions
- **Higher threshold** (0.7+): Fewer, more confident predictions

### Sequence Length

Default maximum sequence length: 2000 amino acids

Modify in:
- `src/predictor.py`: `max_seq_length` parameter
- `src/utils.py`: `max_length` parameter in `one_hot_encode()`

## 📊 Output Files

### Predictions CSV

Location: `outputs/predictions.csv`

Format:
```
protein_id,GO_term,score
protein1,GO:0005524,0.9124
protein1,GO:0004672,0.8756
```

## 🛠️ Troubleshooting

### Model Download Failed (404 Error)

```
✗ Download failed: 404 Client Error: Not Found
```

**Solution**: The pre-trained model is large (2+ GB) and must be downloaded manually:
1. Visit: http://deepgoplus.bio2vec.net/
2. Download the `model.h5` file from the Models section
3. Place it in the `models/` directory

For testing, create a test model:
```bash
python src/create_test_model.py
```

### Model Not Loading

```
✗ Model not found: models/model.h5
```

**Solution**: Run `python src/download_data.py` or manually place the model file in `models/model.h5`.

### Missing Data Files

```
✗ GO terms file not found: data/raw/terms.pkl
```

**Solution**: Run `python src/extract_data.py` to extract the dataset.

### Sequence Validation Errors

- **Too short**: Minimum 5 amino acids required
- **Invalid characters**: Only standard amino acids (A-Z) allowed
- **Empty sequence**: Check FASTA format

### Out of Memory

For very long sequences or batch processing:
1. Reduce `max_seq_length` in configuration
2. Process proteins individually instead of batch
3. Close other applications

## 📝 FASTA File Format

Correct format:

```fasta
>protein1
MALWMRLLPLLALLALWGPDPA

>protein2
MKVLFSLLTSIFSCFSACVENQ
SQEFYGQWKYDHTDDRVYHPHF
```

Rules:
- Header starts with `>` followed by protein ID
- Sequence on following lines (can span multiple lines)
- Blank lines are ignored
- Only standard amino acids (A-Z)

## 🎓 For Thesis/Research

This system is designed for reproducible research:

1. **Fixed Models**: Uses pre-trained, fixed DeepGOPlus architecture
2. **Deterministic**: Identical input produces identical predictions
3. **Documented Pipeline**: Clear, step-by-step workflow
4. **Reproducible Environment**: `requirements.txt` for dependency versions

### Citation

When using this system, cite the original DeepGOPlus paper and model:

> Original DeepGOPlus: Bio2Vec Research Group
> Website: http://deepgoplus.bio2vec.net/

## ⚠️ Limitations

- Predictions based on pre-trained model
- Accuracy depends on sequence similarity to training data
- Requires complete model files (2+ GB)
- Predictions only for 588 GO terms in pre-trained model

## 📖 Dependencies

- **TensorFlow/Keras**: Deep learning framework
- **NumPy/Pandas**: Data processing
- **Biopython**: Sequence processing
- **Streamlit**: Web interface
- **scikit-learn**: Machine learning utilities

See `requirements.txt` for versions.

## 📄 License

This project is for educational and research purposes. 

The DeepGOPlus model and architecture are subject to their original license and terms.

## 🤝 Support

For issues or questions:

1. Check the Troubleshooting section
2. Verify all data files are present
3. Ensure TensorFlow is properly installed
4. Check console output for detailed error messages

## 🔗 References

- DeepGOPlus: http://deepgoplus.bio2vec.net/
- Gene Ontology: http://geneontology.org/
- Streamlit: https://streamlit.io/
- TensorFlow: https://www.tensorflow.org/

---

**Created for**: Protein Function Prediction in thesis work

**Last Updated**: 2026

**Status**: Implementation Complete ✓
#   F a s t a G O - E x p l a i n a b l e A I  
 