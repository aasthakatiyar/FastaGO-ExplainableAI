import streamlit as st
import numpy as np
import pandas as pd
import os
import io
from tensorflow.keras.models import load_model
from Bio import SeqIO
from utils import encode_sequence, load_go_terms, load_go_metadata, MAXLEN
from genai_manager import load_stored_key, save_key_locally, delete_stored_key, GeminiExplainer

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
        
        # API Key Section with Persistence (using genai_manager)
        st.markdown("### 🔑 GenAI Configuration")
        stored_key = load_stored_key()
        api_key = st.text_input(
            "Gemini API Key", 
            value=stored_key,
            type="password", 
            help="Enter your Google Gemini API key. You can save it permanently below."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Key", help="Save key to a local file"):
                if api_key:
                    save_key_locally(api_key)
                    st.success("Saved!")
                    st.rerun()
        with col2:
            if st.button("🗑️ Clear", help="Delete the local key file"):
                delete_stored_key()
                st.warning("Deleted!")
                st.rerun()

        if api_key:
            st.caption("✅ Key is active")
        else:
            st.info("💡 Get a key at [Google AI Studio](https://aistudio.google.com/)")
        
        st.markdown("---")
        
        threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )
        top_k = st.number_input(
            "Max Terms per Protein", 
            min_value=1, max_value=1000, value=10
        )
        st.markdown("---")
        st.markdown("### About")
        st.info("DeepGOPlus predicts Gene Ontology terms from protein sequences using Deep Learning.")

    # 3. Main Area
    st.title("🧬 DeepGOPlus Predictor")
    st.write("Upload a file, enter a sequence, or select a sample below to predict protein functions.")

    # --- INPUT SECTION (TABS) ---
    tab1, tab2, tab3 = st.tabs(["📤 Upload File", "✍️ Input Sequence", "🧪 Sample Data"])

    # Initialize session state for persistence
    if "input_sequences" not in st.session_state:
        st.session_state.input_sequences = []
    if "source_name" not in st.session_state:
        st.session_state.source_name = ""
    if "preds" not in st.session_state:
        st.session_state.preds = None
    if "ids" not in st.session_state:
        st.session_state.ids = []

    # TAB 1: UPLOAD
    with tab1:
        uploaded_file = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "txt"])
        if uploaded_file:
            content = uploaded_file.getvalue().decode("utf-8")
            seqs = []
            for rec in SeqIO.parse(io.StringIO(content), "fasta"):
                seqs.append((rec.id, str(rec.seq)))
            st.session_state.input_sequences = seqs
            st.session_state.source_name = uploaded_file.name
            st.session_state.preds = None # Reset preds for new data

    # TAB 2: MANUAL INPUT
    with tab2:
        user_input = st.text_area(
            "Paste sequence (FASTA format allowed)", 
            height=150,
            placeholder=">protein_id\nMTEITAAM..."
        )
        if st.button("🔍 Predict from Manual Input"):
            if user_input.strip():
                clean_text = user_input.strip()
                if not clean_text.startswith(">"):
                    clean_text = ">user_input\n" + clean_text
                
                seqs = []
                for rec in SeqIO.parse(io.StringIO(clean_text), "fasta"):
                    seqs.append((rec.id, str(rec.seq)))
                st.session_state.input_sequences = seqs
                st.session_state.source_name = "Manual Input"
                st.session_state.preds = None # Reset preds for new data

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
            seqs = []
            for rec in SeqIO.parse(io.StringIO(SAMPLE_SEQUENCES[selected_sample]), "fasta"):
                seqs.append((rec.id, str(rec.seq)))
            st.session_state.input_sequences = seqs
            st.session_state.source_name = selected_sample
            st.session_state.preds = None # Reset preds for new data

    # Clear results button
    if st.session_state.input_sequences:
        if st.sidebar.button("🧹 Clear All Results"):
            st.session_state.input_sequences = []
            st.session_state.source_name = ""
            st.session_state.preds = None
            st.rerun()

    # --- PREDICTION ENGINE ---
    if st.session_state.input_sequences:
        input_sequences = st.session_state.input_sequences
        source_name = st.session_state.source_name
        
        # Only run prediction if not already cached
        if st.session_state.preds is None:
            st.markdown("---")
            with st.status("Processing...", expanded=True) as status:
                st.write(f"Loading {len(input_sequences)} sequence(s)...")
                
                # Preprocess
                ids, encoded_seqs = [], []
                for pid, seq in input_sequences:
                    ids.append(pid)
                    encoded_seqs.append(encode_sequence(seq))
                
                st.write("Running model...")
                st.session_state.preds = model.predict(np.array(encoded_seqs), verbose=0)
                st.session_state.ids = ids
                
                status.update(label="Prediction Complete!", state="complete")

        preds = st.session_state.preds
        ids = st.session_state.ids
        
        # Create a mapping for full sequences for AI insights
        protein_seqs = {pid: seq for pid, seq in input_sequences}
        
        # Process Results
        results = []
        for i, pid in enumerate(ids):
            scores = preds[i]
            valid_idx = np.where(scores >= threshold)[0]
            sorted_idx = valid_idx[np.argsort(scores[valid_idx])[::-1]]
            
            for idx in sorted_idx[:top_k]:
                go_id = go_terms[idx]
                go_info = go_meta.get(go_id, {})
                
                # Format relationships
                part_of_terms = []
                for rel in go_info.get("relationship", []):
                    if "part_of" in rel:
                        go_part = rel.split()[1] if len(rel.split()) > 1 else ""
                        if go_part:
                            part_of_terms.append(go_part)
                
                # Format xref by database
                xref_by_db = {}
                for xref in go_info.get("xref", []):
                    if ":" in xref:
                        db, ref = xref.split(":", 1)
                        if db not in xref_by_db:
                            xref_by_db[db] = []
                        xref_by_db[db].append(ref.strip())
                
                results.append({
                    "Protein": pid,
                    "GO Term": go_id,
                    "Name": go_info.get("name", "N/A"),
                    "Namespace": go_info.get("namespace", ""),
                    "Score": scores[idx],
                    # Expanded fields
                    "Definition": go_info.get("def", ""),
                    "Synonyms": "; ".join(go_info.get("synonyms", [])),
                    "Alt IDs": "; ".join(go_info.get("alt_id", [])),
                    "Parent Terms": "; ".join(go_info.get("is_a", [])),
                    "Part Of": "; ".join(part_of_terms),
                    "Replaced By": "; ".join(go_info.get("replaced_by", [])),
                    "Consider": "; ".join(go_info.get("consider", [])),
                    "Subsets": "; ".join(go_info.get("subset", [])),
                    "Comment": go_info.get("comment", ""),
                    "Obsolete": go_info.get("is_obsolete", False),
                    "_xref_dict": xref_by_db,  # For detail expansion
                    "_go_info": go_info  # Store full info for expandable details
                })

        if results:
            df_display = pd.DataFrame(results)
            
            # Display Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Proteins Analyzed", len(ids))
            c2.metric("Total Terms Found", len(df_display))
            c3.metric("Avg Score", f"{df_display['Score'].mean():.2f}")
            c4.metric("High Confidence (>0.5)", len(df_display[df_display['Score'] > 0.5]))

            # Tabs for different views
            view_tab1, view_tab2, view_tab3 = st.tabs(["📊 Summary Table", "🔬 Detailed View", "📥 Export"])
            
            with view_tab1:
                # Grouped summary table view
                st.markdown("### Predictions by Namespace")
                namespaces = df_display['Namespace'].unique()
                
                for ns in namespaces:
                    with st.expander(f"📦 {ns.replace('_', ' ').title()} ({len(df_display[df_display['Namespace'] == ns])} terms)", expanded=True):
                        df_ns = df_display[df_display['Namespace'] == ns][["Protein", "GO Term", "Name", "Score", "Definition"]]
                        st.dataframe(
                            df_ns,
                            use_container_width=True,
                            column_config={
                                "Score": st.column_config.ProgressColumn(
                                    "Score", format="%.3f", min_value=0, max_value=1
                                )
                            }
                        )
            
            with view_tab2:
                # Detailed view with grouped sections and better visual structure
                for idx, row in df_display.iterrows():
                    # Header with prominent Score and Term ID
                    with st.container(border=True):
                        h_col1, h_col2, h_col3 = st.columns([3, 1, 1])
                        with h_col1:
                            st.subheader(f"{row['Name']}")
                            st.caption(f"**Protein ID:** {row['Protein']} | **GO ID:** {row['GO Term']}")
                        with h_col2:
                            st.metric("Score", f"{row['Score']:.3f}")
                        with h_col3:
                            st.markdown("<br>", unsafe_allow_html=True) # Vertical alignment
                            if row['Obsolete']:
                                st.error("🔴 OBSOLETE")
                            else:
                                st.success("🟢 ACTIVE")

                        # Main Info Grid
                        info_tab1, info_tab2, info_tab3, info_tab4 = st.tabs(["📖 Definition & Notes", "🌳 Hierarchy & Relationships", "🔗 External References", "🧠 AI Insights"])
                        
                        with info_tab1:
                            # Left column: Definition, Right column: Terminology/Status
                            d_col1, d_col2 = st.columns([2, 1])
                            with d_col1:
                                st.markdown("##### 📝 Definition")
                                st.write(row['Definition'] if row['Definition'] else "No definition available.")
                                if row['Comment']:
                                    st.markdown("##### 💡 Curator Comments")
                                    st.info(row['Comment'])
                            
                            with d_col2:
                                st.markdown("##### 🏷️ Metadata")
                                st.write(f"**Namespace:** {row['Namespace']}")
                                if row['Synonyms']:
                                    st.write(f"**Synonyms:** {row['Synonyms']}")
                                if row['Alt IDs']:
                                    st.write(f"**Alternate IDs:** {row['Alt IDs']}")
                                if row['Subsets']:
                                    st.write(f"**Subsets:** {row['Subsets']}")
                        
                        with info_tab2:
                            h_col1, h_col2 = st.columns(2)
                            with h_col1:
                                st.markdown("##### ⬆️ Parent Terms (is_a)")
                                if row['Parent Terms']:
                                    # Parent Terms are already formatted as "GO_ID (Name)" from utils.py
                                    for parent in row['Parent Terms'].split("; "):
                                        st.markdown(f"- {parent}")
                                else:
                                    st.write("None listed.")
                                    
                                if row['Replaced By']:
                                    st.warning(f"**Replaced by:** {row['Replaced By']}")
                                if row['Consider']:
                                    st.info(f"**Consider instead:** {row['Consider']}")
                                    
                            with h_col2:
                                st.markdown("##### 🔗 GO Relationships")
                                all_relationships = row['_go_info'].get('relationship', [])
                                
                                if all_relationships:
                                    # Process each relationship string to extract type, ID, and name
                                    for rel_str in all_relationships:
                                        parts = rel_str.split(" ", 1)
                                        if len(parts) == 2:
                                            rel_type = parts[0]
                                            go_id_and_name = parts[1]
                                            
                                            go_id = ""
                                            name = ""
                                            
                                            # Try to extract ID and Name from "GO_ID (Name)" format
                                            name_parts = go_id_and_name.split(" (", 1)
                                            if len(name_parts) == 2 and name_parts[1].endswith(")"):
                                                go_id = name_parts[0]
                                                name = name_parts[1][:-1] # Remove trailing ')'
                                            else:
                                                go_id = go_id_and_name # Assume the whole string is the GO ID if no name format
                                                name = "N/A"

                                            # Display relationship clearly
                                            st.markdown(f"- **{rel_type.replace('_', ' ').title()}:** `{go_id}` ({name})")
                                else:
                                    st.write("No GO relationships defined for this term.")
                                
                                if not all_relationships: # This condition is redundant if all_relationships is empty.
                                    st.write("No other relationships defined.") # This line might be removed or combined.

                        with info_tab3:
                            if row['_xref_dict']:
                                # Create a grid for xrefs
                                x_cols = st.columns(3)
                                for i, (db, refs) in enumerate(row['_xref_dict'].items()):
                                    with x_cols[i % 3]:
                                        st.markdown(f"**{db}**")
                                        for ref in refs:
                                            # Truncate long xref names if needed or format as list
                                            st.caption(f"• {ref}")
                            else:
                                st.write("No external cross-references available.")
                        
                        with info_tab4:
                            st.markdown("##### 🧠 AI Functional Insight")
                            st.write("Generate a scientific explanation for why this protein is associated with this GO term.")
                            
                            if st.button(f"Generate Explanation for {row['GO Term']}", key=f"ai_btn_{row['Protein']}_{row['GO Term']}_{idx}"):
                                with st.spinner("Analyzing sequence and ontology..."):
                                    # Use the new modular GeminiExplainer
                                    explainer = GeminiExplainer(api_key)
                                    explanation = explainer.get_explanation(
                                        row['Protein'], 
                                        protein_seqs.get(row['Protein'], ""), 
                                        row['GO Term'], 
                                        row['Name'], 
                                        row['Definition'], 
                                        row['Score']
                                    )
                                    st.markdown("---")
                                    st.markdown(explanation)
                                    st.caption("Generated by Gemini AI via genai_manager. Please verify with primary literature.")
                                
                    st.markdown("<br>", unsafe_allow_html=True) # Spacer between terms
            
            with view_tab3:
                # Export options
                st.markdown("### Export Options")
                
                # Full CSV export
                df_export = df_display.drop(columns=["_xref_dict", "_go_info"])
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name=f"predictions_full_{source_name}.csv",
                    mime="text/csv"
                )
                
                # Summary CSV export
                df_summary_export = df_display[["Protein", "GO Term", "Name", "Namespace", "Score"]].copy()
                csv_summary = df_summary_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Summary (CSV)",
                    data=csv_summary,
                    file_name=f"predictions_summary_{source_name}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No terms found above the threshold. Try lowering the threshold in the sidebar.")

if __name__ == "__main__":
    main()