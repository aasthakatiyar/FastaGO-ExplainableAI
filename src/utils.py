"""
Utility functions for the DeepGOPlus protein function prediction system.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple


def one_hot_encode(sequence: str, max_length: int = 2000) -> np.ndarray:
    """
    One-hot encode a protein sequence.
    
    Args:
        sequence (str): Amino acid sequence
        max_length (int): Maximum sequence length for padding
        
    Returns:
        np.ndarray: One-hot encoded sequence in shape (max_length, 21)
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    
    sequence = sequence.upper()
    # Shape: (max_length, 21) - 21 for 20 amino acids + 1 padding/unknown channel
    encoded = np.zeros((max_length, 21), dtype=np.float32)
    
    for i, aa in enumerate(sequence[:max_length]):
        if aa in aa_to_idx:
            encoded[i, aa_to_idx[aa]] = 1.0
        else:
            # Unknown amino acid gets padding channel (index 20)
            encoded[i, 20] = 1.0
    
    # Ensure remaining positions (if sequence shorter than max_length) use padding channel
    for i in range(len(sequence), max_length):
        encoded[i, 20] = 1.0
    
    return encoded


def pad_sequence(sequence: str, max_length: int = 2000) -> str:
    """
    Pad or truncate sequence to max_length.
    
    Args:
        sequence (str): Input sequence
        max_length (int): Target length
        
    Returns:
        str: Padded/truncated sequence
    """
    if len(sequence) >= max_length:
        return sequence[:max_length]
    else:
        return sequence + 'X' * (max_length - len(sequence))


def filter_predictions(probabilities: np.ndarray, 
                      go_terms: List[str], 
                      threshold: float = 0.5) -> Tuple[List[str], List[float]]:
    """
    Filter predictions based on confidence threshold.
    
    Args:
        probabilities (np.ndarray): Prediction probabilities
        go_terms (List[str]): List of GO term identifiers
        threshold (float): Confidence threshold
        
    Returns:
        Tuple: (filtered_go_terms, filtered_scores)
    """
    predictions = []
    scores = []
    
    # Ensure probabilities has same length as go_terms
    if len(probabilities) != len(go_terms):
        print(f"⚠ Warning: probabilities length ({len(probabilities)}) != go_terms length ({len(go_terms)})")
        # Trim to minimum length
        min_len = min(len(probabilities), len(go_terms))
        probabilities = probabilities[:min_len]
        go_terms = go_terms[:min_len]
    
    for i, (score, go_term) in enumerate(zip(probabilities, go_terms)):
        score = float(score)
        if score >= threshold:
            # Extract GO term - handle if it's wrapped in something
            go_term_str = str(go_term).strip()
            # Remove brackets if present
            go_term_str = go_term_str.strip("[]'\"")
            if not go_term_str:
                go_term_str = f"GO:UNKNOWN_{i}"
            predictions.append(go_term_str)
            scores.append(round(score, 4))
    
    # Sort by score descending
    sorted_pairs = sorted(zip(predictions, scores), 
                         key=lambda x: x[1], reverse=True)
    
    if sorted_pairs:
        predictions, scores = zip(*sorted_pairs)
        return list(predictions), list(scores)
    
    return [], []


def ensure_path(path: Path) -> Path:
    """
    Create parent directories if they don't exist.
    
    Args:
        path (Path): File path
        
    Returns:
        Path: The same path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_sequence(sequence: str) -> str:
    """
    Normalize protein sequence to uppercase.
    
    Args:
        sequence (str): Input sequence
        
    Returns:
        str: Normalized sequence
    """
    return sequence.upper().strip()
