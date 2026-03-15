"""
Main prediction module for protein function prediction using DeepGOPlus.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from fasta_parser import parse_fasta, parse_fasta_string, validate_sequence
from load_model import DeepGOPlusModel
from utils import one_hot_encode, filter_predictions, normalize_sequence, ensure_path


class ProteinFunctionPredictor:
    """
    Main predictor class combining FASTA parsing and model inference.
    """
    
    def __init__(self, model_path: str = None, data_dir: str = None, 
                 threshold: float = 0.5, max_seq_length: int = 2000):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to model.h5
            data_dir (str): Path to data directory
            threshold (float): Confidence threshold for predictions
            max_seq_length (int): Maximum sequence length
        """
        self.model = DeepGOPlusModel(model_path, data_dir)
        self.threshold = threshold
        self.max_seq_length = max_seq_length
        self.model_ready = False
    
    def initialize(self) -> bool:
        """
        Load model and data.
        
        Returns:
            bool: True if initialization successful
        """
        print("Initializing predictor...")
        self.model_ready = self.model.load_all()
        return self.model_ready
    
    def predict_sequence(self, protein_id: str, sequence: str) -> Tuple[List[str], List[float]]:
        """
        Predict GO terms for a single sequence.
        
        Args:
            protein_id (str): Protein identifier
            sequence (str): Amino acid sequence
            
        Returns:
            Tuple: (go_terms, confidence_scores)
        """
        if not self.model_ready:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        sequence = normalize_sequence(sequence)
        
        is_valid, error_msg = validate_sequence(sequence)
        if not is_valid:
            print(f"⚠ Sequence validation failed for {protein_id}: {error_msg}")
            return [], []
        
        try:
            encoded = one_hot_encode(sequence, self.max_seq_length)
            probabilities = self.model.predict(encoded)
            
            go_terms, scores = filter_predictions(
                probabilities, 
                self.model.go_terms, 
                self.threshold
            )
            
            return go_terms, scores
            
        except Exception as e:
            print(f"✗ Prediction failed for {protein_id}: {str(e)}")
            return [], []
    
    def predict_sequences(self, sequences: Dict[str, str]) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Predict GO terms for multiple sequences.
        
        Args:
            sequences (Dict[str, str]): Dictionary of protein_id -> sequence
            
        Returns:
            Dict: protein_id -> (go_terms, scores)
        """
        results = {}
        
        for protein_id, sequence in sequences.items():
            go_terms, scores = self.predict_sequence(protein_id, sequence)
            results[protein_id] = (go_terms, scores)
        
        return results
    
    def predict_fasta_file(self, fasta_path: Path) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Predict GO terms from FASTA file.
        
        Args:
            fasta_path (Path): Path to FASTA file
            
        Returns:
            Dict: protein_id -> (go_terms, scores)
        """
        sequences = parse_fasta(Path(fasta_path))
        return self.predict_sequences(sequences)
    
    def predict_fasta_string(self, fasta_content: str) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Predict GO terms from FASTA format string.
        
        Args:
            fasta_content (str): FASTA format string
            
        Returns:
            Dict: protein_id -> (go_terms, scores)
        """
        sequences = parse_fasta_string(fasta_content)
        return self.predict_sequences(sequences)
    
    def save_results(self, results: Dict[str, Tuple[List[str], List[float]]], 
                    output_file: Path = None) -> bool:
        """
        Save predictions to CSV file.
        
        Args:
            results (Dict): Prediction results
            output_file (Path): Output CSV file path
            
        Returns:
            bool: True if successful
        """
        if output_file is None:
            output_file = Path("outputs/predictions.csv")
        
        ensure_path(output_file)
        
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['protein_id', 'GO_term', 'score'])
                
                for protein_id, (go_terms, scores) in results.items():
                    for go_term, score in zip(go_terms, scores):
                        writer.writerow([protein_id, go_term, score])
            
            print(f"✓ Results saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save results: {str(e)}")
            return False


def main():
    """
    Command-line interface for protein function prediction.
    """
    parser = argparse.ArgumentParser(
        description='Predict protein functions using DeepGOPlus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predictor.py --input proteins.fasta
  python predictor.py --input proteins.fasta --output results.csv
  python predictor.py --input proteins.fasta --threshold 0.7
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input FASTA file path')
    parser.add_argument('--output', '-o', default='outputs/predictions.csv',
                       help='Output CSV file path (default: outputs/predictions.csv)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--model', '-m', default='models/model.h5',
                       help='Path to model.h5 (default: models/model.h5)')
    parser.add_argument('--data', '-d', default='data/raw',
                       help='Path to data directory (default: data/raw)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        return False
    
    print("=" * 70)
    print("DeepGOPlus Protein Function Predictor")
    print("=" * 70)
    
    predictor = ProteinFunctionPredictor(
        model_path=args.model,
        data_dir=args.data,
        threshold=args.threshold
    )
    
    if not predictor.initialize():
        print("✗ Failed to initialize predictor")
        return False
    
    print(f"\nProcessing: {input_path}")
    results = predictor.predict_fasta_file(input_path)
    
    print(f"\nPredicted {len(results)} proteins")
    for protein_id, (go_terms, scores) in results.items():
        print(f"\n{protein_id}: {len(go_terms)} GO terms")
        for go_term, score in list(zip(go_terms, scores))[:5]:
            print(f"  {go_term}: {score:.4f}")
        if len(go_terms) > 5:
            print(f"  + {len(go_terms) - 5} more...")
    
    predictor.save_results(results, Path(args.output))
    
    print("\n" + "=" * 70)
    print("Prediction completed!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
