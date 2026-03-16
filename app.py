import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from Bio import SeqIO
from io import StringIO

# -----------------------------------
# CONFIG
# -----------------------------------

MAXLEN = 2000
TOP_K = 10

MODEL_PATH = "model/model.h5"
TERMS_PATH = "data/terms.pkl"
GO_OBO_PATH = "data/go.obo"

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a:i for i,a in enumerate(AA)}

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Protein Function Predictor",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------------
# STYLING
# -----------------------------------

st.markdown("""
<style>
.big-title {
    font-size:38px;
    font-weight:700;
}
.subtitle {
    font-size:18px;
    color:gray;
}
.result-box {
    background-color:#f5f7fa;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

# -----------------------------------
# LOAD GO TERMS
# -----------------------------------

@st.cache_data
def load_go_terms():

    with open(TERMS_PATH, "rb") as f:
        terms = pickle.load(f)

    if isinstance(terms, pd.DataFrame):
        terms = terms.iloc[:,0].tolist()

    return terms


# -----------------------------------
# LOAD GO NAMES
# -----------------------------------

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


# -----------------------------------
# ENCODING
# -----------------------------------

def encode(seq):

    arr = np.zeros((MAXLEN,21))

    for i,aa in enumerate(seq[:MAXLEN]):

        if aa in AA_INDEX:
            arr[i,AA_INDEX[aa]] = 1
        else:
            arr[i,20] = 1

    return arr


# -----------------------------------
# PARSE FASTA
# -----------------------------------

def read_fasta(uploaded_file):

    fasta_string = uploaded_file.getvalue().decode()

    sequences = []
    names = []

    for record in SeqIO.parse(StringIO(fasta_string),"fasta"):

        names.append(record.id)
        sequences.append(str(record.seq))

    return names, sequences


# -----------------------------------
# PREDICTION
# -----------------------------------

def run_prediction(names, seqs):

    model = load_model_cached()
    go_terms = load_go_terms()
    go_names = load_go_names()

    encoded = [encode(s) for s in seqs]

    X = np.array(encoded)

    preds = model.predict(X)

    results = []

    for i,name in enumerate(names):

        scores = preds[i]

        top = np.argsort(scores)[::-1][:TOP_K]

        for idx in top:

            go_id = go_terms[idx]

            results.append({

                "Protein":name,
                "GO ID":go_id,
                "GO Name":go_names.get(go_id,"Unknown"),
                "Score":float(scores[idx])

            })

    return pd.DataFrame(results)


# -----------------------------------
# UI
# -----------------------------------

st.markdown('<div class="big-title">🧬 Protein Function Prediction</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">Deep learning based Gene Ontology prediction</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded = st.file_uploader(
    "Upload FASTA file",
    type=["fa","fasta","txt"]
)

if uploaded:

    names, seqs = read_fasta(uploaded)

    st.success(f"{len(names)} sequences loaded")

    with st.expander("Preview sequences"):

        for i in range(min(5,len(names))):

            st.write(names[i])
            st.code(seqs[i][:120]+"...")

    if st.button("Run Prediction 🚀"):

        with st.spinner("Running DeepGOPlus model..."):

            df = run_prediction(names,seqs)

        st.success("Prediction completed!")

        # -----------------------------------
        # DISPLAY RESULTS
        # -----------------------------------

        st.subheader("Predicted Gene Ontology Terms")

        st.dataframe(df, use_container_width=True)

        # -----------------------------------
        # DOWNLOAD
        # -----------------------------------

        csv = df.to_csv(index=False).encode()

        st.download_button(
            "Download Results CSV",
            csv,
            "go_predictions.csv",
            "text/csv"
        )