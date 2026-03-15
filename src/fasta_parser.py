"""
FASTA file parser for protein sequences.
Extracts protein IDs and amino acid sequences from FASTA format files.
"""

from pathlib import Path
from typing import Dict, Tuple


def parse_fasta(fasta_file: Path) -> Dict[str, str]:
    """
    Parse a FASTA file and extract protein sequences.
    
    Args:
        fasta_file (Path): Path to the FASTA file
        
    Returns:
        Dict[str, str]: Dictionary mapping protein IDs to sequences
        
    Example:
        >>> sequences = parse_fasta(Path("example.fasta"))
        >>> sequences["protein1"]
        'MALWMRLLPLLALLALWGPDPA'
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                    
                if line.startswith('>'):
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)
                    
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
                
    except FileNotFoundError:
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
    except Exception as e:
        raise Exception(f"Error parsing FASTA file: {str(e)}")
    
    return sequences


def parse_fasta_string(fasta_content: str) -> Dict[str, str]:
    """
    Parse FASTA content from a string.
    
    Args:
        fasta_content (str): FASTA format string
        
    Returns:
        Dict[str, str]: Dictionary mapping protein IDs to sequences
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    for line in fasta_content.strip().split('\n'):
        line = line.strip()
        
        if not line:
            continue
            
        if line.startswith('>'):
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            
            current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)
    
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    
    return sequences


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate amino acid sequence.
    
    Args:
        sequence (str): Amino acid sequence
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    
    if not sequence:
        return False, "Sequence is empty"
    
    if len(sequence) < 5:
        return False, "Sequence too short (minimum 5 amino acids)"
    
    invalid_chars = set(sequence.upper()) - valid_amino_acids
    if invalid_chars:
        return False, f"Invalid amino acids: {', '.join(invalid_chars)}"
    
    return True, ""
