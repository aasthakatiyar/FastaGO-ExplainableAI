import streamlit as st
import numpy as np
import pandas as pd
import os
import io
from tensorflow.keras.models import load_model
from Bio import SeqIO
from utils import encode_sequence, load_go_terms, load_go_metadata, MAXLEN

# --- CONFIGURATION ---
MODEL_PATH = "model/model.h5"
TERMS_PATH = "data/terms.pkl"
GO_OBO_PATH = "data/go.obo"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="DeepGOPlus Predictor",
    page_icon="🧬",
    layout="wide"
)

# --- CACHE ASSETS ---
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = load_model(MODEL_PATH)
    go_terms = load_go_terms(TERMS_PATH)
    go_meta = load_go_metadata(GO_OBO_PATH)
    return model, go_terms, go_meta

# --- SAMPLE DATA ---
SAMPLE_SEQUENCES = {
    "Human p53 Tumor Suppressor": """>sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK""",
    
    "SARS-CoV-2 Spike Protein": """>sp|P0DTC2|SPIKE_SARS2 Spike glycoprotein OS=Severe acute respiratory syndrome coronavirus 2
MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFS
NVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVM""",
    
    "Insulin-like Growth Factor": """>sp|P05019|IGF1_HUMAN Insulin-like growth factor 1
MSSSVPTPSLFLPAQPLLPLLLPLLQLPAQPLLLPQPAPEVLANEPVTYSSSPWGRGPQG"""
}

# --- MAIN APP ---
def main():
    # 1. Load Model
    model, go_terms, go_meta = load_assets()
    
    if model is None:
        st.error("⚠️ **Model files not found.** Please check the following paths:")
        st.code(f"- {MODEL_PATH}\n- {TERMS_PATH}\n- {GO_OBO_PATH}")
        st.stop()

    # 2. Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )
        top_k = st.number_input(
            "Max Terms per Protein", 
            min_value=1, max_value=30, value=10
        )
        st.markdown("---")
        st.markdown("### About")
        st.info("DeepGOPlus predicts Gene Ontology terms from protein sequences using Deep Learning.")

    # 3. Main Area
    st.title("🧬 DeepGOPlus Predictor")
    st.write("Upload a file, enter a sequence, or select a sample below to predict protein functions.")

    # --- INPUT SECTION (TABS) ---
    tab1, tab2, tab3 = st.tabs(["📤 Upload File", "✍️ Input Sequence", "🧪 Sample Data"])

    input_sequences = []
    source_name = ""

    # TAB 1: UPLOAD
    with tab1:
        uploaded_file = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "txt"])
        if uploaded_file:
            content = uploaded_file.getvalue().decode("utf-8")
            for rec in SeqIO.parse(io.StringIO(content), "fasta"):
                input_sequences.append((rec.id, str(rec.seq)))
            source_name = uploaded_file.name

    # TAB 2: MANUAL INPUT
    with tab2:
        user_input = st.text_area(
            "Paste sequence (FASTA format allowed)", 
            height=150,
            placeholder=">protein_id\nMTEITAAM..."
        )
        if user_input.strip():
            # Add header if missing
            clean_text = user_input.strip()
            if not clean_text.startswith(">"):
                clean_text = ">user_input\n" + clean_text
            
            for rec in SeqIO.parse(io.StringIO(clean_text), "fasta"):
                input_sequences.append((rec.id, str(rec.seq)))
            source_name = "Manual Input"

    # TAB 3: SAMPLES
    with tab3:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_sample = st.selectbox("Choose a sample protein", list(SAMPLE_SEQUENCES.keys()))
        with col2:
            st.write("") # Align button
            load_btn = st.button("Load Sample")

        st.code(SAMPLE_SEQUENCES[selected_sample], language="text")

        if load_btn:
            for rec in SeqIO.parse(io.StringIO(SAMPLE_SEQUENCES[selected_sample]), "fasta"):
                input_sequences.append((rec.id, str(rec.seq)))
            source_name = selected_sample

    # --- PREDICTION ENGINE ---
    if input_sequences:
        st.markdown("---")
        
        # Show status
        with st.status("Processing...", expanded=True) as status:
            st.write(f"Loading {len(input_sequences)} sequence(s)...")
            
            # Preprocess
            ids, encoded_seqs = [], []
            for pid, seq in input_sequences:
                ids.append(pid)
                encoded_seqs.append(encode_sequence(seq))
            
            st.write("Running model...")
            preds = model.predict(np.array(encoded_seqs), verbose=0)
            
            status.update(label="Prediction Complete!", state="complete")

        # Process Results
        results = []
        for i, pid in enumerate(ids):
            scores = preds[i]
            valid_idx = np.where(scores >= threshold)[0]
            sorted_idx = valid_idx[np.argsort(scores[valid_idx])[::-1]]
            
            for idx in sorted_idx[:top_k]:
                go_id = go_terms[idx]
                go_info = go_meta.get(go_id, {})
                results.append({
                    "Protein": pid,
                    "GO Term": go_id,
                    "Name": go_info.get("name", "N/A"),
                    "Namespace": go_info.get("namespace", ""),
                    "Definition": go_info.get("def", ""),
                    "Synonyms": ", ".join(go_info.get("synonyms", [])),
                    "Alt IDs": ", ".join(go_info.get("alt_id", [])),
                    "Replaced By": ", ".join(go_info.get("replaced_by", [])),
                    "Obsolete": go_info.get("is_obsolete", False),
                    "Score": scores[idx]
                })

        if results:
            df = pd.DataFrame(results)
            
            # Display Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Proteins Analyzed", len(ids))
            c2.metric("Total Terms Found", len(df))
            c3.metric("Average Score", f"{df['Score'].mean():.2f}")

            # Display Table
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", format="%.2f", min_value=0, max_value=1
                    )
                }
            )

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"predictions_{source_name}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No terms found above the threshold. Try lowering the threshold in the sidebar.")

if __name__ == "__main__":
    main()