import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from Bio import SeqIO
from io import StringIO

# ------------------------------
# CONFIG
# ------------------------------

MAXLEN = 2000
TOP_K = 10

MODEL_PATH = "model/model.h5"
TERMS_PATH = "data/terms.pkl"
GO_OBO_PATH = "data/go.obo"

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="Protein Function Predictor",
    page_icon="🧬",
    layout="wide"
)

# ------------------------------
# STYLE
# ------------------------------

st.markdown("""
<style>
.title{
font-size:40px;
font-weight:700;
}
.subtitle{
font-size:18px;
color:gray;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# LOAD MODEL
# ------------------------------

@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

# ------------------------------
# LOAD GO TERMS
# ------------------------------

@st.cache_data
def load_go_terms():

    with open(TERMS_PATH, "rb") as f:
        terms = pickle.load(f)

    if isinstance(terms, pd.DataFrame):
        terms = terms.iloc[:,0].tolist()

    return terms

# ------------------------------
# LOAD GO NAMES
# ------------------------------

@st.cache_data
def load_go_names():

    go_names = {}
    current = None

    with open(GO_OBO_PATH) as f:

        for line in f:

            if line.startswith("id:"):
                current = line.strip().split()[1]

            elif line.startswith("name:"):
                name = line.strip()[6:]
                go_names[current] = name

    return go_names

# ------------------------------
# ENCODE SEQUENCE
# ------------------------------

def encode(seq):

    arr = np.zeros((MAXLEN,21))

    for i,aa in enumerate(seq[:MAXLEN]):

        if aa in AA_INDEX:
            arr[i,AA_INDEX[aa]] = 1
        else:
            arr[i,20] = 1

    return arr

# ------------------------------
# RUN PREDICTION
# ------------------------------

def run_prediction(names, seqs):

    model = load_model_cached()
    go_terms = load_go_terms()
    go_names = load_go_names()

    encoded = [encode(s) for s in seqs]

    X = np.array(encoded)

    preds = model.predict(X)

    rows = []

    for i,name in enumerate(names):

        scores = preds[i]
        top = np.argsort(scores)[::-1][:TOP_K]

        for idx in top:

            go_id = go_terms[idx]

            rows.append({
                "Protein": name,
                "GO Term": go_id,
                "GO Name": go_names.get(go_id,"Unknown"),
                "Score": float(scores[idx])
            })

    return pd.DataFrame(rows)

# ------------------------------
# SAMPLE SEQUENCES
# ------------------------------

SAMPLES = {
"Insulin": "MALWMRLLPLLALLALWGPDPAAA",
"Cytochrome C": "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNE",
"Hemoglobin": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF",
"Myoglobin": "GLSDGEWQQVLNVWGKVEADIPGHGQEVLIRLF",
"Lysozyme": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
"Albumin": "MKWVTFISLLFLFSSAYS"
}

# ------------------------------
# HEADER
# ------------------------------

st.markdown('<div class="title">🧬 Protein Function Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning based Gene Ontology Annotation</div>', unsafe_allow_html=True)

st.write("")

# ------------------------------
# TABS
# ------------------------------

tab1, tab2, tab3 = st.tabs(["Upload FASTA", "Paste Sequence", "Sample Proteins"])

# ------------------------------
# TAB 1 — FASTA UPLOAD
# ------------------------------

with tab1:

    uploaded = st.file_uploader("Upload FASTA file", type=["fa","fasta","txt"])

    if uploaded:

        fasta_string = uploaded.getvalue().decode()

        names = []
        seqs = []

        for record in SeqIO.parse(StringIO(fasta_string),"fasta"):

            names.append(record.id)
            seqs.append(str(record.seq))

        st.success(f"{len(names)} sequences loaded")

        if st.button("Predict Functions"):

            with st.spinner("Running model..."):

                df = run_prediction(names,seqs)

            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                "predictions.csv"
            )

# ------------------------------
# TAB 2 — MANUAL INPUT
# ------------------------------

with tab2:

    seq = st.text_area("Paste amino acid sequence")

    if st.button("Predict Sequence"):

        if seq:

            names = ["Manual_Sequence"]
            seqs = [seq.replace("\n","").strip()]

            with st.spinner("Running model..."):

                df = run_prediction(names,seqs)

            st.dataframe(df, use_container_width=True)

# ------------------------------
# TAB 3 — SAMPLE PROTEINS
# ------------------------------

with tab3:

    sample = st.selectbox("Choose sample protein", list(SAMPLES.keys()))

    st.code(SAMPLES[sample])

    if st.button("Run Sample Prediction"):

        names = [sample]
        seqs = [SAMPLES[sample]]

        with st.spinner("Running model..."):

            df = run_prediction(names,seqs)

        st.dataframe(df, use_container_width=True)