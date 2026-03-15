"""
Streamlit web application for protein function prediction using DeepGOPlus.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fasta_parser import parse_fasta_string, validate_sequence
from load_model import DeepGOPlusModel
from utils import one_hot_encode, filter_predictions, normalize_sequence


def initialize_session_state():
    """Initialize session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None


def load_model_cached():
    """Load model with caching."""
    if st.session_state.model is None:
        with st.spinner("Loading DeepGOPlus model..."):
            model_loader = DeepGOPlusModel()
            if model_loader.load_all():
                st.session_state.model = model_loader
            else:
                st.error("Failed to load model. Ensure model files are in place.")
                return None
    return st.session_state.model


def predict_sequence(protein_id: str, sequence: str, model, threshold: float = 0.5):
    """
    Predict GO terms for a sequence.
    
    Args:
        protein_id (str): Protein identifier
        sequence (str): Amino acid sequence
        model: DeepGOPlusModel instance
        threshold (float): Confidence threshold
        
    Returns:
        tuple: (go_terms, scores) or ([], []) if prediction fails
    """
    sequence = normalize_sequence(sequence)
    
    is_valid, error_msg = validate_sequence(sequence)
    if not is_valid:
        return [], []
    
    try:
        encoded = one_hot_encode(sequence, max_length=2000)
        probabilities = model.predict(encoded)
        
        go_terms, scores = filter_predictions(
            probabilities,
            model.go_terms,
            threshold
        )
        
        return go_terms, scores
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return [], []


def format_results(results_dict: dict) -> pd.DataFrame:
    """
    Format prediction results as DataFrame.
    
    Args:
        results_dict (dict): protein_id -> (go_terms, scores)
        
    Returns:
        pd.DataFrame: Formatted results
    """
    rows = []
    for protein_id, (go_terms, scores) in results_dict.items():
        for go_term, score in zip(go_terms, scores):
            rows.append({
                'Protein ID': protein_id,
                'GO Term': go_term,
                'Confidence': f"{score:.4f}",
                'Score_float': score  # For sorting
            })
    
    if rows:
        df = pd.DataFrame(rows)
        # Sort by score descending
        df = df.sort_values('Score_float', ascending=False)
        return df.drop('Score_float', axis=1)
    else:
        return pd.DataFrame(columns=['Protein ID', 'GO Term', 'Confidence'])


def usage_help_tab():
    """Usage and Help tab."""
    st.header("❓ Usage Guide & Help")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Quick Start")
        st.markdown("""
        1. **Enter Protein Sequences**
           - Use the "Text Input" tab to paste FASTA sequences directly
           - Or use "File Upload" to upload a FASTA file
           
        2. **Configure Settings**
           - Adjust the confidence threshold (0.1 to 1.0)
           - Set max results to display per protein
           
        3. **Click Predict**
           - The model will process your sequences
           - Results will appear in the results section
           
        4. **Review Results**
           - See summary metrics at the top
           - Expand each protein to view detailed predictions
           - Download full results as CSV
        """)
        
        st.subheader("Understanding Results")
        st.markdown("""
        **Confidence Levels:**
        - 🟢 **Very High (≥0.9)** - Highly confident prediction
        - 🟡 **High (≥0.7)** - Strong prediction
        - 🟠 **Medium (≥0.5)** - Moderate confidence
        - 🔴 **Low (<0.5)** - Lower confidence (usually filtered out)
        
        **Metrics:**
        - **Total Predictions** - Number of GO terms predicted
        - **Avg Confidence** - Average score across all predictions
        - **Max/Min Confidence** - Highest and lowest scores
        """)
    
    with col2:
        st.subheader("💡 Pro Tips")
        st.markdown("""
        • Lower threshold to see more predictions
        
        • Threshold 0.5+ recommended
        
        • Longer sequences = better predictions
        
        • 20-2000 amino acids optimal
        
        • Download CSV for batch analysis
        
        • Use expanders to keep UI clean
        
        • Max 10 sequences recommended
        """)
    
    st.markdown("---")
    
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Issue: No predictions found**
        - Try lowering the confidence threshold
        - Check sequence validity (20+ amino acids)
        - Ensure protein sequence is valid
        
        **Issue: Model not loading**
        - Restart the application
        - Check models/model.h5 exists
        - Verify data files present
        
        **Issue: Slow predictions**
        - This is normal for long sequences
        - First prediction is slower (model loading)
        - Subsequent predictions are faster
        
        **Issue: CSV download not working**
        - Try refreshing the page
        - Check browser download settings
        """)


def model_details_tab():
    """Model Details tab."""
    st.header("🤖 Model Architecture & Details")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Architecture")
        st.markdown("""
        **Model Type:** Convolutional Neural Network (CNN)
        
        **Framework:** TensorFlow/Keras
        
        **Input:**
        - Amino acid sequences
        - Max length: 2,000 residues
        - One-hot encoded (21 channels)
        
        **Output:**
        - 5,707 GO term predictions
        - Probability scores (0.0 - 1.0)
        
        **Architecture Layers:**
        - Conv1D layers with ReLU activation
        - MaxPooling for dimensionality reduction
        - Global pooling for sequence aggregation
        - Dense layers with dropout regularization
        - Sigmoid output for multi-label classification
        """)
    
    with col2:
        st.subheader("Training & Performance")
        st.markdown("""
        **Dataset:**
        - Trained on DeepGOPlus dataset
        - Proteins from UniProtKB
        - Annotated with GO terms
        
        **Training Details:**
        - Optimizer: Adam
        - Loss: Binary Crossentropy
        - Metrics: Accuracy
        
        **Characteristics:**
        - Multi-label classification
        - Each protein can have multiple GO terms
        - Sequence-to-term prediction
        - Pre-trained model (fixed weights)
        
        **Sequence Coverage:**
        - Handles variable-length sequences
        - Pads sequences < 2,000 residues
        - Truncates sequences > 2,000 residues
        """)
    
    st.markdown("---")
    
    with st.expander("📊 GO Term Distribution"):
        st.markdown("""
        **Gene Ontology Coverage:**
        
        - **Total Terms:** 5,707 unique GO terms
        - **Biological Process (BP):** ~3,500 terms
        - **Molecular Function (MF):** ~1,800 terms
        - **Cellular Component (CC):** ~400 terms
        
        **Term Characteristics:**
        - High-level terms (general functions)
        - Mid-level terms (specific functions)
        - Low-level terms (detailed annotations)
        """)
    
    with st.expander("⚙️ Input Encoding"):
        st.markdown("""
        **One-Hot Encoding:**
        
        Each amino acid is converted to a vector:
        - A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
        - Plus 1 unknown/padding channel
        - Total: 21 dimensions
        
        **Sequence Format:**
        - Shape: (sequence_length, 21)
        - Padded to 2,000 positions
        - Padding value: last channel = 1.0
        """)


def documentation_tab():
    """Documentation tab."""
    st.header("📖 Project Documentation")
    
    st.subheader("Project Overview")
    st.markdown("""
    **DeepGOPlus** is a deep learning system for predicting Gene Ontology (GO) terms 
    from protein sequences. It combines:
    - Advanced neural network architecture
    - Pre-trained deep learning model
    - Comprehensive GO term database
    - Web interface for easy access
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📂 Project Structure")
        st.markdown("""
        ```
        protein-go-prediction/
        ├── app/
        │   └── streamlit_app.py      # Web interface
        ├── src/
        │   ├── predictor.py          # Main pipeline
        │   ├── load_model.py         # Model loading
        │   ├── fasta_parser.py       # FASTA parsing
        │   ├── utils.py              # Utilities
        │   ├── download_data.py      # Data download
        │   └── extract_data.py       # Data extraction
        ├── data/
        │   ├── raw/                  # Dataset files
        │   └── diamond_db/           # DIAMOND database
        ├── models/
        │   └── model.h5              # Pre-trained model
        ├── examples/
        │   └── sample.fasta          # Example sequences
        └── outputs/
            └── predictions.csv       # Results
        ```
        """)
    
    with col2:
        st.subheader("🔄 Prediction Pipeline")
        st.markdown("""
        ```
        1. Input: FASTA sequence
                    ↓
        2. Parse: Extract protein ID & sequence
                    ↓
        3. Validate: Check sequence quality
                    ↓
        4. Encode: One-hot encode (21D)
                    ↓
        5. Pad: Normalize to 2,000 residues
                    ↓
        6. Predict: CNN forward pass
                    ↓
        7. Filter: Apply confidence threshold
                    ↓
        8. Output: GO terms + scores
        ```
        """)
    
    st.markdown("---")
    
    st.subheader("📋 Supported Formats")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Input: FASTA**
        ```
        >protein1
        MALWMRLLPLL
        ALLALWGPDPA
        
        >protein2
        MKVLFSLLTSIF
        ```
        """)
    
    with col2:
        st.markdown("""
        **Output: CSV**
        ```
        protein_id,
        GO_term,
        score
        protein1,
        GO:0005524,
        0.9124
        ```
        """)
    
    with col3:
        st.markdown("""
        **Requirements**
        - Valid amino acids (A-Z)
        - 5+ residues per sequence
        - FASTA format
        - UTF-8 encoding
        """)
    
    st.markdown("---")
    
    with st.expander("� How to Run This Project From Scratch"):
        st.markdown("""
        ### Prerequisites
        - **Python 3.8+** installed on your system
        - **Git** for cloning (optional)
        - **~2GB** disk space for model and data
        
        ### Step 1: Set Up Environment
        ```bash
        # Create project folder
        mkdir protein-go-prediction
        cd protein-go-prediction
        
        # Create virtual environment
        python -m venv .venv
        
        # Activate virtual environment
        # On Windows:
        .venv\\Scripts\\activate
        # On macOS/Linux:
        source .venv/bin/activate
        ```
        
        ### Step 2: Install Dependencies
        ```bash
        # Upgrade pip
        python -m pip install --upgrade pip
        
        # Install required packages
        pip install -r requirements.txt
        ```
        
        ### Step 3: Download & Extract Data
        ```bash
        # Download dataset and model
        python src/download_data.py
        
        # Extract and organize files
        python src/extract_data.py
        
        # This will:
        # - Extract dataset to data/raw/
        # - Organize model files to models/
        # - Set up all required data structures
        ```
        
        ### Step 4: Run the Application
        ```bash
        # Start the Streamlit app
        streamlit run app/streamlit_app.py
        ```
        
        The app will open at: **http://localhost:8501**
        
        ### Step 5: Make Predictions
        1. Go to **"🔮 Predictor"** tab
        2. Paste FASTA sequences or upload a file
        3. Adjust confidence threshold (default: 0.5)
        4. Click **"Predict"**
        5. Download results as CSV
        
        ### Troubleshooting
        
        **Issue: Model download fails**
        - Visit: http://deepgoplus.bio2vec.net/
        - Download model manually to `models/model.h5`
        
        **Issue: Missing GO terms file**
        - Go terms should be in `data/raw/terms.pkl`
        - Check data extraction completed successfully
        
        **Issue: Slow first prediction**
        - First run loads the model (normal, ~20-30 seconds)
        - Subsequent predictions are faster
        
        **Issue: Out of memory**
        - Process fewer sequences at once
        - Close other applications to free RAM
        - Model requires ~1.5GB RAM
        """)


def main_predictor():
    """Main predictor interface."""
    st.markdown("""
    Predict Gene Ontology (GO) terms for protein sequences using the **DeepGOPlus** 
    deep learning model. Simply provide protein sequences in FASTA format to get predictions.
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Only predictions above this threshold will be shown"
        )
    
    with col2:
        max_results = st.slider(
            "Max Results per Protein",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Limit number of predictions shown"
        )
    
    with col3:
        st.info(f"🎯 Threshold: {threshold:.2f} | Max: {max_results}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📤 File Upload", "📚 Examples"])
    
    with tab1:
        st.subheader("Paste FASTA Sequences")
        
        fasta_input = st.text_area(
            "Enter FASTA format protein sequences:",
            height=300,
            placeholder=">protein1\nMLALWMRLLPLLALLALWGPD\n>protein2\nMAGLSPVGPPGGVQSQEJQ",
            key="fasta_text"
        )
        
        if st.button("🔍 Predict from Text", key="predict_text"):
            if not fasta_input.strip():
                st.warning("Please enter FASTA sequences")
            else:
                model = load_model_cached()
                if model:
                    with st.spinner("Running predictions..."):
                        sequences = parse_fasta_string(fasta_input)
                        results = {}
                        
                        for protein_id, sequence in sequences.items():
                            go_terms, scores = predict_sequence(
                                protein_id, sequence, model, threshold
                            )
                            results[protein_id] = (go_terms[:max_results], scores[:max_results])
                        
                        st.session_state.predictions = results
                        st.session_state.results_df = format_results(results)
                        st.success("✓ Prediction completed!")
    
    with tab2:
        st.subheader("Upload FASTA File")
        
        uploaded_file = st.file_uploader(
            "Choose a FASTA file",
            type=['fasta', 'fa', 'faa', 'txt']
        )
        
        if uploaded_file is not None:
            if st.button("🔍 Predict from File", key="predict_file"):
                model = load_model_cached()
                if model:
                    try:
                        fasta_content = uploaded_file.read().decode('utf-8')
                        
                        with st.spinner("Running predictions..."):
                            sequences = parse_fasta_string(fasta_content)
                            results = {}
                            
                            for protein_id, sequence in sequences.items():
                                go_terms, scores = predict_sequence(
                                    protein_id, sequence, model, threshold
                                )
                                results[protein_id] = (go_terms[:max_results], scores[:max_results])
                            
                            st.session_state.predictions = results
                            st.session_state.results_df = format_results(results)
                            st.success("✓ Prediction completed!")
                            
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
    
    with tab3:
        st.subheader("Example FASTA Sequences")
        
        example_fasta = """>protein1_kinase
MALWMRLLPLLALLALWGPDPAAALESGERCQVQLVAGVQKARDSVRVS
VGTPGLMAALWDVQPGAPGSWQNKQLQLLQELKQVDVKKMESLGTQVPPL

>protein2_binding
MKVLFSLLTSIFSCFSACVENQSQEFYGQWKYDHTDDRVYHPHFDLSHGSAQ
VQGHSEELGLQAQQLQVVVLGKQVMGWDEHAQPPGNYQ

>protein3_transport
MHHHHHHGRGSSGGSGGSGGSMSLEGGGRGSCGGLQGMSGSGGTPPGCMQQ
GQQMQPSGQQQQQQQQQQQQQQQQQQQQ
"""
        
        st.code(example_fasta, language="fasta")
        
        if st.button("🔍 Predict Example Proteins"):
            model = load_model_cached()
            if model:
                with st.spinner("Running predictions..."):
                    sequences = parse_fasta_string(example_fasta)
                    results = {}
                    
                    for protein_id, sequence in sequences.items():
                        go_terms, scores = predict_sequence(
                            protein_id, sequence, model, threshold
                        )
                        results[protein_id] = (go_terms[:max_results], scores[:max_results])
                    
                    st.session_state.predictions = results
                    st.session_state.results_df = format_results(results)
                    st.success("✓ Prediction completed!")
    
    st.markdown("---")
    
    # Display results if available
    if st.session_state.results_df is not None and not st.session_state.results_df.empty:
        st.subheader("📊 Prediction Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(st.session_state.results_df))
        with col2:
            num_proteins = st.session_state.results_df['Protein ID'].nunique()
            st.metric("Proteins Analyzed", num_proteins)
        with col3:
            # Convert confidence to float for calculation
            confidence_floats = pd.to_numeric(st.session_state.results_df['Confidence'], errors='coerce')
            avg_confidence = confidence_floats.mean()
            st.metric("Avg Confidence", f"{avg_confidence:.4f}")
        with col4:
            max_confidence = confidence_floats.max()
            st.metric("Max Confidence", f"{max_confidence:.4f}")
        
        st.markdown("---")
        
        # Display by protein
        proteins = st.session_state.results_df['Protein ID'].unique()
        
        if len(proteins) == 1:
            # Single protein - show expanded view
            protein_id = proteins[0]
            protein_data = st.session_state.results_df[st.session_state.results_df['Protein ID'] == protein_id].copy()
            
            st.subheader(f"🧬 {protein_id}")
            
            # Convert confidence to floats
            protein_data['Confidence_float'] = pd.to_numeric(protein_data['Confidence'], errors='coerce')
            
            # Top predictions with visual bars
            st.write(f"**Total GO Terms Predicted:** {len(protein_data)}")
            
            # Show top 15 with bar chart
            top_n = min(15, len(protein_data))
            top_predictions = protein_data.head(top_n).copy()
            
            for idx, row in top_predictions.iterrows():
                score = float(row['Confidence_float'])
                
                # Color based on confidence
                if score >= 0.9:
                    color = "🟢"
                    level = "Very High"
                elif score >= 0.7:
                    color = "🟡"
                    level = "High"
                elif score >= 0.5:
                    color = "🟠"
                    level = "Medium"
                else:
                    color = "🔴"
                    level = "Low"
                
                col1, col2, col3 = st.columns([2, 3, 1])
                with col1:
                    st.write(f"**{row['GO Term']}**")
                with col2:
                    st.progress(score, text=f"{level}")
                with col3:
                    st.write(f"{score:.4f}")
            
            if len(protein_data) > top_n:
                st.info(f"+ {len(protein_data) - top_n} more predictions (download CSV for complete list)")
        
        else:
            # Multiple proteins - improved card layout
            st.write(f"**Analysis Results for {len(proteins)} Proteins**")
            
            # Create comparison summary table
            summary_data = []
            for p in proteins:
                pdata = st.session_state.results_df[st.session_state.results_df['Protein ID'] == p].copy()
                pdata['Score_float'] = pd.to_numeric(pdata['Confidence'], errors='coerce')
                summary_data.append({
                    'Protein': p,
                    'GO Terms': len(pdata),
                    'Avg Score': f"{pdata['Score_float'].mean():.4f}",
                    'Max Score': f"{pdata['Score_float'].max():.4f}",
                    'Min Score': f"{pdata['Score_float'].min():.4f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width='stretch', hide_index=True)
            
            st.markdown("---")
            
            # Display each protein in expandable containers
            for protein_id in proteins:
                protein_data = st.session_state.results_df[st.session_state.results_df['Protein ID'] == protein_id].copy()
                protein_data['Confidence_float'] = pd.to_numeric(protein_data['Confidence'], errors='coerce')
                
                # Create expandable section for each protein
                with st.expander(f"🧬 **{protein_id}** ({len(protein_data)} predictions)", expanded=False):
                    
                    # Mini metrics for this protein
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", len(protein_data))
                    with col2:
                        st.metric("Avg", f"{protein_data['Confidence_float'].mean():.4f}")
                    with col3:
                        st.metric("Max", f"{protein_data['Confidence_float'].max():.4f}")
                    with col4:
                        st.metric("Min", f"{protein_data['Confidence_float'].min():.4f}")
                    
                    st.markdown("**Top 12 Predictions:**")
                    
                    # Show top 12 predictions with visual bars
                    top_n = min(12, len(protein_data))
                    top_predictions = protein_data.head(top_n).copy()
                    
                    for idx, row in top_predictions.iterrows():
                        score = float(row['Confidence_float'])
                        
                        # Color based on confidence
                        if score >= 0.9:
                            color = "🟢"
                            level = "Very High"
                        elif score >= 0.7:
                            color = "🟡"
                            level = "High"
                        elif score >= 0.5:
                            color = "🟠"
                            level = "Medium"
                        else:
                            color = "🔴"
                            level = "Low"
                        
                        col1, col2, col3 = st.columns([2, 3, 1])
                        with col1:
                            st.caption(f"{color} {row['GO Term']}")
                        with col2:
                            st.progress(score, text=level)
                        with col3:
                            st.caption(f"{score:.4f}")
                    
                    if len(protein_data) > top_n:
                        st.caption(f"_...and {len(protein_data) - top_n} more predictions_")
        
        # Download section
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Download CSV
            csv_data = st.session_state.results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv_data,
                file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
        
        with col2:
            # Show full table toggle
            if st.checkbox("📋 Show Full Table"):
                st.dataframe(
                    st.session_state.results_df,
                    width='stretch',
                    height=400
                )
    
    elif st.session_state.predictions is not None:
        st.info("ℹ️ No predictions above the confidence threshold. Try adjusting the threshold slider.")



def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="DeepGOPlus - Protein Function Prediction",
        page_icon="🧬",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🧬 DeepGOPlus Protein Function Prediction")
    st.markdown("Gene Ontology Term Prediction using Deep Learning")
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictor", "📖 Documentation", "🤖 Model Details", "❓ Usage & Help"])
    
    # ===== TAB 1: PREDICTOR =====
    with tab1:
        main_predictor()
    
    # ===== TAB 2: DOCUMENTATION =====
    with tab2:
        documentation_tab()
    
    # ===== TAB 3: MODEL DETAILS =====
    with tab3:
        model_details_tab()
    
    # ===== TAB 4: USAGE & HELP =====
    with tab4:
        usage_help_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px; margin-top: 20px'>
    <b>DeepGOPlus Protein Function Prediction System</b> | Gene Ontology Term Prediction using Deep Learning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
