import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Bio import SeqIO
import io
import time

# Constants
MAXLEN = 2000
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# Page configuration
st.set_page_config(
    page_title="DeepGOPlus Protein Function Predictor",
    page_icon="🧬",
    layout="wide"
)

# Title and description
st.title("🧬 DeepGOPlus Protein Function Predictor")
st.markdown("""
This app uses DeepGOPlus, a deep learning model for predicting Gene Ontology (GO) terms from protein sequences.
Upload a FASTA file or enter protein sequences directly to get function predictions.
""")

# Load model (cached)
@st.cache_resource
def load_prediction_model():
    """Load the DeepGOPlus model"""
    try:
        model = load_model("model/model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Sequence encoding function
def encode_sequence(seq):
    """Encode a protein sequence into one-hot format"""
    arr = np.zeros((MAXLEN, 21))

    for i, aa in enumerate(seq[:MAXLEN]):
        if aa in AA_INDEX:
            arr[i, AA_INDEX[aa]] = 1
        else:
            arr[i, 20] = 1  # unknown AA

    return arr

# Parse FASTA from text
def parse_fasta_text(fasta_text):
    """Parse FASTA sequences from text input"""
    sequences = []
    names = []

    try:
        # Use StringIO to create a file-like object
        fasta_io = io.StringIO(fasta_text)
        for record in SeqIO.parse(fasta_io, "fasta"):
            names.append(record.id)
            sequences.append(encode_sequence(str(record.seq)))
    except Exception as e:
        st.error(f"Error parsing FASTA: {e}")
        return [], []

    return names, sequences

# Main prediction function
def predict_protein_functions(model, sequences, names):
    """Run predictions on protein sequences"""
    if not sequences:
        return None

    X = np.array(sequences)

    # Show progress
    with st.spinner("Running predictions... This may take a few moments."):
        preds = model.predict(X, verbose=0)

    # Create DataFrame
    df = pd.DataFrame(preds)
    df.insert(0, "protein", names)

    return df

# Sidebar for inputs
st.sidebar.header("Input Options")

input_method = st.sidebar.radio(
    "Choose input method:",
    ["Enter FASTA text", "Upload FASTA file"]
)

fasta_text = ""
uploaded_file = None

if input_method == "Enter FASTA text":
    fasta_text = st.sidebar.text_area(
        "Enter protein sequences in FASTA format:",
        height=200,
        placeholder=">Protein1\nMKWVTFISLLFLFSSAYS\n\n>Protein2\nGGGSSVKVLVVVVGGGG",
        help="Enter sequences in FASTA format with '>' headers"
    )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload FASTA file:",
        type=["fasta", "fa", "faa"],
        help="Upload a FASTA file containing protein sequences"
    )

# Threshold slider
threshold = st.sidebar.slider(
    "Prediction Threshold:",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Only show predictions above this confidence threshold"
)

# Load model
model = load_prediction_model()

if model is None:
    st.error("Failed to load the prediction model. Please check that model/model.h5 exists.")
    st.stop()

# Predict button
if st.sidebar.button("🔍 Predict Functions", type="primary"):
    # Get sequences
    names = []
    sequences = []

    if input_method == "Enter FASTA text" and fasta_text.strip():
        names, sequences = parse_fasta_text(fasta_text)
    elif input_method == "Upload FASTA file" and uploaded_file is not None:
        try:
            for record in SeqIO.parse(uploaded_file, "fasta"):
                names.append(record.id)
                sequences.append(encode_sequence(str(record.seq)))
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    else:
        st.sidebar.error("Please provide protein sequences using one of the input methods above.")

    if names and sequences:
        # Run predictions
        results_df = predict_protein_functions(model, sequences, names)

        if results_df is not None:
            st.success(f"✅ Predictions completed for {len(names)} protein(s)!")

            # Store results in session state
            st.session_state.results_df = results_df
            st.session_state.names = names

# Display results
if 'results_df' in st.session_state:
    results_df = st.session_state.results_df
    names = st.session_state.names

    st.header("📊 Prediction Results")

    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Proteins Analyzed", len(names))
    with col2:
        total_predictions = (results_df.iloc[:, 1:] > threshold).sum().sum()
        st.metric("High-Confidence Predictions", total_predictions)
    with col3:
        avg_predictions = total_predictions / len(names)
        st.metric("Avg Predictions per Protein", f"{avg_predictions:.1f}")

    # Results table
    st.subheader("Detailed Results")

    # Filter predictions above threshold
    filtered_df = results_df.copy()

    # For each protein, keep only GO terms above threshold
    go_columns = [col for col in filtered_df.columns if col != 'protein']

    # Create a melted version for better display
    melted_data = []
    for idx, row in filtered_df.iterrows():
        protein_name = row['protein']
        for go_col in go_columns:
            score = row[go_col]
            if score >= threshold:
                melted_data.append({
                    'Protein': protein_name,
                    'GO_Term_ID': f'GO_{go_col}',
                    'Confidence_Score': score
                })

    if melted_data:
        filtered_results = pd.DataFrame(melted_data)
        filtered_results = filtered_results.sort_values(['Protein', 'Confidence_Score'], ascending=[True, False])

        # Display as table
        st.dataframe(
            filtered_results,
            use_container_width=True,
            column_config={
                'Protein': st.column_config.TextColumn('Protein', width='medium'),
                'GO_Term_ID': st.column_config.TextColumn('GO Term', width='medium'),
                'Confidence_Score': st.column_config.NumberColumn('Confidence Score', format='%.3f')
            }
        )

        # Download button
        csv_data = filtered_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="protein_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info(f"No predictions found above the threshold of {threshold}. Try lowering the threshold.")

    # Raw predictions (collapsible)
    with st.expander("🔧 View Raw Predictions"):
        st.dataframe(results_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**About DeepGOPlus:**
DeepGOPlus is a deep learning model that predicts Gene Ontology (GO) terms for protein sequences.
It uses convolutional neural networks trained on a large dataset of protein sequences and their annotated functions.

**How to use:**
1. Provide protein sequences in FASTA format
2. Click "Predict Functions"
3. Adjust the threshold to filter predictions
4. Download results as CSV

**Note:** GO term IDs are shown as GO_0, GO_1, etc. These correspond to specific Gene Ontology terms in the model's training data.
""")</content>
<parameter name="filePath">d:\Fasta GO\protein_prediction\app.py