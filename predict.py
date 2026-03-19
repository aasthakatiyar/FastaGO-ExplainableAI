import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Bio import SeqIO
from utils import encode_sequence, load_go_terms, load_go_metadata, MAXLEN

# --- CONFIGURATION ---
MODEL_PATH = "model/model.h5"
TERMS_PATH = "data/terms.pkl"
GO_OBO_PATH = "data/go.obo"
INPUT_FASTA = "input/protein.fasta"
OUTPUT_FILE = "output/go_predictions.csv"

THRESHOLD = 0.3
TOP_K_PER_NAMESPACE = 10
GO_ID_PATTERN = re.compile(r"GO:\d{7}")
ROOT_GO_TERMS = {"GO:0003674", "GO:0005575", "GO:0008150"}


def _extract_go_ids(text_values):
    """Extract GO IDs from metadata fields that may include labels."""
    go_ids = []
    for value in text_values:
        go_ids.extend(GO_ID_PATTERN.findall(value))
    return go_ids


def _build_parent_lookup(go_meta):
    """Build GO parent lookup from is_a and part_of relationships."""
    parent_lookup = {}
    for go_id, info in go_meta.items():
        parents = set(_extract_go_ids(info.get("is_a", [])))

        for rel in info.get("relationship", []):
            # Keep only hierarchical relationships that imply a parent term.
            if rel.startswith("part_of "):
                parents.update(_extract_go_ids([rel]))

        parent_lookup[go_id] = parents
    return parent_lookup


def _build_child_lookup(parent_lookup):
    child_lookup = {go_id: set() for go_id in parent_lookup}
    for child, parents in parent_lookup.items():
        for parent in parents:
            child_lookup.setdefault(parent, set()).add(child)
    return child_lookup


def _format_go_terms(go_ids, go_meta):
    formatted = []
    for go_id in sorted(go_ids):
        go_name = go_meta.get(go_id, {}).get("name", "Unknown")
        formatted.append(f"{go_id} ({go_name})")
    return "; ".join(formatted)


def _get_ancestors(go_id, parent_lookup, cache):
    if go_id in cache:
        return cache[go_id]

    seen = set()
    stack = list(parent_lookup.get(go_id, set()))

    while stack:
        parent = stack.pop()
        if parent in seen:
            continue
        seen.add(parent)
        stack.extend(parent_lookup.get(parent, set()) - seen)

    cache[go_id] = seen
    return seen


def _leaf_terms_within_predictions(predicted_go_ids, parent_lookup):
    """
    Keep only most specific terms among predicted terms.

    A term is treated as non-leaf if it is an ancestor of any other predicted term.
    """
    predicted_set = set(predicted_go_ids)
    non_leaf = set()
    cache = {}

    for go_id in predicted_go_ids:
        ancestors = _get_ancestors(go_id, parent_lookup, cache)
        non_leaf.update(ancestors & predicted_set)

    return [go_id for go_id in predicted_go_ids if go_id not in non_leaf]


def _get_depth(go_id, parent_lookup, cache):
    if go_id in cache:
        return cache[go_id]

    parents = parent_lookup.get(go_id, set())
    if not parents:
        cache[go_id] = 0
        return 0

    parent_depths = [_get_depth(parent, parent_lookup, cache) for parent in parents if parent != go_id]
    depth = 1 + (max(parent_depths) if parent_depths else 0)
    cache[go_id] = depth
    return depth


def _select_deepest_by_namespace(indices, go_terms, go_meta, parent_lookup, k_per_namespace=1):
    """Pick deepest terms per namespace (MF/CC/BP), with score-order tie-breaking."""
    if not indices:
        return []

    depth_cache = {}
    namespace_to_indices = {}

    for idx in indices:
        go_id = go_terms[idx]
        namespace = go_meta.get(go_id, {}).get("namespace", "")
        namespace_to_indices.setdefault(namespace, []).append(idx)

    selected = []
    for ns_indices in namespace_to_indices.values():
        max_depth = max(_get_depth(go_terms[idx], parent_lookup, depth_cache) for idx in ns_indices)
        deepest = [idx for idx in ns_indices if _get_depth(go_terms[idx], parent_lookup, depth_cache) == max_depth]
        selected.extend(deepest[:k_per_namespace])

    # Preserve global score order from original sorted indices.
    selected_set = set(selected)
    return [idx for idx in indices if idx in selected_set]


def _select_deepest_global(indices, go_terms, parent_lookup, k=1):
    """Pick globally deepest terms, preserving existing score-order tie-breaking."""
    if not indices:
        return []

    depth_cache = {}
    max_depth = max(_get_depth(go_terms[idx], parent_lookup, depth_cache) for idx in indices)
    deepest = [idx for idx in indices if _get_depth(go_terms[idx], parent_lookup, depth_cache) == max_depth]
    return deepest[:k]


def _select_top_k_per_namespace(sorted_idx, go_terms, go_meta, k_per_namespace):
    """Select top-k terms for each GO namespace from score-sorted indices."""
    selected = []
    ns_counts = {
        "biological_process": 0,
        "molecular_function": 0,
        "cellular_component": 0,
    }

    for idx in sorted_idx:
        go_id = go_terms[idx]
        namespace = go_meta.get(go_id, {}).get("namespace", "")
        if namespace not in ns_counts:
            continue
        if ns_counts[namespace] < k_per_namespace:
            selected.append(idx)
            ns_counts[namespace] += 1

        if all(count >= k_per_namespace for count in ns_counts.values()):
            break

    return selected

def main():
    print("\n--- DeepGOPlus CLI Predictor (Refactored) ---")

    # 1. Validation
    for path in [MODEL_PATH, TERMS_PATH, INPUT_FASTA]:
        if not os.path.exists(path):
            print(f"Error: Required file missing at {path}")
            return

    # 2. Load Assets
    print("Loading model and metadata...")
    model = load_model(MODEL_PATH)
    go_terms = load_go_terms(TERMS_PATH)
    go_meta = load_go_metadata(GO_OBO_PATH)
    parent_lookup = _build_parent_lookup(go_meta)
    child_lookup = _build_child_lookup(parent_lookup)

    # 3. Process Sequences
    ids, encoded_seqs = [], []
    for record in SeqIO.parse(INPUT_FASTA, "fasta"):
        ids.append(record.id)
        encoded_seqs.append(encode_sequence(str(record.seq)))
    
    if not ids:
        print("No sequences found in FASTA file.")
        return

    # 4. Predict
    print(f"Running predictions for {len(ids)} sequence(s)...")
    preds = model.predict(np.array(encoded_seqs))

    # 5. Extract and Filter
    results = []
    for i, protein_id in enumerate(ids):
        scores = preds[i]
        # Find indices above threshold
        valid_idx = np.where(scores >= THRESHOLD)[0]
        # Sort indices by score descending
        sorted_idx = valid_idx[np.argsort(scores[valid_idx])[::-1]]
        sorted_idx = [idx for idx in sorted_idx if go_terms[idx] not in ROOT_GO_TERMS]
        chosen_idx = _select_top_k_per_namespace(
            sorted_idx,
            go_terms,
            go_meta,
            TOP_K_PER_NAMESPACE,
        )

        # Recursively prune parent terms within selected candidates to keep only leaf nodes.
        chosen_go_ids = [go_terms[idx] for idx in chosen_idx]
        leaf_go_ids = set(_leaf_terms_within_predictions(chosen_go_ids, parent_lookup))
        chosen_idx = [idx for idx in chosen_idx if go_terms[idx] in leaf_go_ids]
        
        for idx in chosen_idx:
            go_id = go_terms[idx]
            go_info = go_meta.get(go_id, {})
            child_terms = child_lookup.get(go_id, set())
            
            # Format relationships nicely
            part_of_terms = []
            other_relationships = []
            for rel in go_info.get("relationship", []):
                if "part_of" in rel:
                    # Format: part_of GO:xxxxxxx ! name
                    go_part = rel.split()[1] if len(rel.split()) > 1 else ""
                    if go_part:
                        part_of_terms.append(go_part)
                else:
                    other_relationships.append(rel)
            
            # Format subsets
            subset_str = "; ".join(go_info.get("subset", []))
            
            # Format cross-references by database
            xref_by_db = {}
            for xref in go_info.get("xref", []):
                if ":" in xref:
                    db, ref = xref.split(":", 1)
                    if db not in xref_by_db:
                        xref_by_db[db] = []
                    xref_by_db[db].append(ref.strip())
            
            xref_str = "; ".join([f"{db}: {', '.join(refs)}" for db, refs in xref_by_db.items()])
            
            results.append({
                # Protein and Prediction
                "protein": protein_id,
                "score": round(float(scores[idx]), 4),
                
                # GO Term Core Info
                "GO_term": go_id,
                "GO_name": go_info.get("name", "Unknown"),
                "GO_namespace": go_info.get("namespace", ""),
                "GO_definition": go_info.get("def", ""),
                
                # Variations
                "GO_synonyms": "; ".join(go_info.get("synonyms", [])),
                "GO_alternate_IDs": "; ".join(go_info.get("alt_id", [])),
                
                # Hierarchy
                "GO_parent_terms": "; ".join(go_info.get("is_a", [])),
                "GO_part_of": "; ".join(part_of_terms),
                "GO_child_terms": _format_go_terms(child_terms, go_meta),
                
                # Status & Alternatives
                "GO_obsolete": go_info.get("is_obsolete", False),
                "GO_replaced_by": "; ".join(go_info.get("replaced_by", [])),
                "GO_consider": "; ".join(go_info.get("consider", [])),
                
                # External References
                "GO_xref": xref_str,
                
                # Subset Information
                "GO_subset": subset_str,
                
                # Additional Notes
                "GO_comment": go_info.get("comment", ""),
                
                # Audit Information
                "GO_created_by": go_info.get("created_by", ""),
                "GO_creation_date": go_info.get("creation_date", ""),
            })

    # 6. Save
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output_path = OUTPUT_FILE
    try:
        df.to_csv(output_path, index=False)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/go_predictions_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        print(f"Warning: {OUTPUT_FILE} was locked. Saved to {output_path} instead.")
    
    print(f"Success! Results saved to {output_path}")
    print(df.head(5))

if __name__ == "__main__":
    main()