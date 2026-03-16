import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from Bio import SeqIO
import io
import plotly.express as px
from utils import encode_sequence, load_go_terms, load_go_names

# --- MODERN STYLING ---
st.set_page_config(page_title="DeepGO-Horizon", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; border-radius: 20px; background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%); color: white; border: none; font-weight: bold; }
    .stTextInput>div>div>input { background-color: #161b22; color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# --- CACHE ASSETS ---
@st.cache_resource
def get_model():
    return load_model("model/model.h5")

@st.cache_data
def get_metadata():
    return load_go_terms("data/terms.pkl"), load_go_names("data/go.obo")

# --- APP HEADER ---
st.title("🧪 DeepGO Horizon")
st.markdown("#### *Next-Generation Protein Function Annotation*")
st.divider()

# --- INPUT SECTION ---
col_in, col_set = st.columns([2, 1])

with col_set:
    st.header("⚙️ Parameters")
    with st.container(border=True):
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3)
        top_k = st.number_input("Max Terms per Protein", 5, 50, 12)
        model_version = st.caption("Engine: DeepGOPlus-v1 (5707 Sync)")

with col_in:
    st.header("🧬 Sequence Input")
    input_mode = st.tabs(["📄 Upload File", "⌨️ Manual Text", "💡 Sample Input"])
    
    input_fasta = None

    with input_mode[0]:
        uploaded_file = st.file_uploader("Drop FASTA file here", type=["fasta", "fa"])
        if uploaded_file:
            input_fasta = uploaded_file.getvalue().decode("utf-8")

    with input_mode[1]:
        text_input = st.text_area("Paste Amino Acid Sequence", height=150, placeholder=">Protein_ID\nMGDVEKGKKIFIM...")
        if text_input:
            input_fasta = text_input

    with input_mode[2]:
        if st.button("Load Cytochrome C (Human) Sample"):
            input_fasta = ">sp|P99999|CYC_HUMAN\nMGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE"
            st.info("Sample Loaded!")

# --- EXECUTION ---
if input_fasta:
    if st.button("🚀 ANALYZE SEQUENCE"):
        with st.spinner("Decoding biological patterns..."):
            # Prepare Data
            fasta_io = io.StringIO(input_fasta)
            records = list(SeqIO.parse(fasta_io, "fasta"))
            
            if not records:
                st.error("Format Error: Ensure input is in FASTA format (starts with '>')")
            else:
                # Load Assets
                model = get_model()
                go_terms, go_names = get_metadata()
                
                # Predict
                ids = [r.id for r in records]
                X = np.array([encode_sequence(str(r.seq)) for r in records])
                preds = model.predict(X)
                
                # Process
                all_results = []
                for i, protein_id in enumerate(ids):
                    scores = preds[i]
                    valid_idx = np.where(scores >= threshold)[0]
                    sorted_idx = valid_idx[np.argsort(scores[valid_idx])[::-1]]
                    
                    for idx in sorted_idx[:top_k]:
                        go_id = go_terms[idx]
                        all_results.append({
                            "Protein": protein_id,
                            "GO_ID": go_id,
                            "Function_Name": go_names.get(go_id, "Unknown"),
                            "Confidence": round(float(scores[idx]), 4)
                        })
                
                res_df = pd.DataFrame(all_results)

                if not res_df.empty:
                    st.divider()
                    st.success("Analysis Complete")
                    
                    # Layout Results
                    res_col, viz_col = st.columns([1, 1])
                    
                    with res_col:
                        st.subheader("📋 Prediction Table")
                        st.dataframe(res_df, use_container_width=True, hide_index=True)
                        st.download_button("📥 Export Results (CSV)", res_df.to_csv(index=False), "deepgo_results.csv")
                    
                    with viz_col:
                        st.subheader("📊 Functional Distribution")
                        # Modern chart styling
                        fig = px.bar(res_df.head(top_k), x="Confidence", y="Function_Name", 
                                     orientation='h', color='Confidence',
                                     color_continuous_scale='Turbo')
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color="#58a6ff",
                            yaxis={'categoryorder':'total ascending'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No significant functions detected above the threshold.")