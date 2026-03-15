"""
Load pretrained DeepGOPlus model and required data.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    from tensorflow import keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None


class DeepGOPlusModel:
    """
    Wrapper for loading and using the pretrained DeepGOPlus model.
    """
    
    def __init__(self, model_path: str = None, data_dir: str = None):
        """
        Initialize the model loader.
        
        Args:
            model_path (str): Path to model.h5 file
            data_dir (str): Path to data directory containing pickle files
        """
        self.model_path = Path(model_path) if model_path else Path("models/model.h5")
        self.data_dir = Path(data_dir) if data_dir else Path("data/raw")
        
        self.model = None
        self.go_terms = None
        self.train_data = None
        self.test_data = None
        self.ontology = None
    
    def load_model(self) -> bool:
        """
        Load the Keras model.
        
        Returns:
            bool: True if successful
        """
        if not self.model_path.exists():
            print(f"✗ Model not found: {self.model_path}")
            print("\nTo use predictions, please:")
            print("1. Visit: http://deepgoplus.bio2vec.net/")
            print("2. Download the pre-trained DeepGOPlus model")
            print("3. Save it as: models/model.h5")
            print("\nFor testing without the model, use example data.")
            return False
        
        try:
            if keras is None:
                print("✗ TensorFlow/Keras not installed")
                print("Install with: pip install tensorflow")
                return False
            
            self.model = keras.models.load_model(str(self.model_path))
            print(f"✓ Model loaded: {self.model_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model: {str(e)}")
            return False
    
    def load_go_terms(self) -> bool:
        """
        Load GO terms list.
        
        Returns:
            bool: True if successful
        """
        terms_file = self.data_dir / "terms.pkl"
        
        if not terms_file.exists():
            print(f"✗ GO terms file not found: {terms_file}")
            return False
        
        try:
            with open(terms_file, 'rb') as f:
                terms_data = pickle.load(f)
            
            # Handle pandas Series or DataFrame
            if hasattr(terms_data, 'values'):
                self.go_terms = list(terms_data.values)
            elif isinstance(terms_data, list):
                self.go_terms = terms_data
            else:
                self.go_terms = list(terms_data)
            
            print(f"✓ Loaded {len(self.go_terms)} GO terms")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load GO terms: {str(e)}")
            return False
    
    def load_train_data(self) -> bool:
        """
        Load training data (optional).
        
        Returns:
            bool: True if successful
        """
        train_file = self.data_dir / "train_data.pkl"
        
        if not train_file.exists():
            return False
        
        try:
            with open(train_file, 'rb') as f:
                self.train_data = pickle.load(f)
            
            return True
            
        except Exception as e:
            # Silently fail - training data is optional
            return False
    
    def load_test_data(self) -> bool:
        """
        Load test data (optional).
        
        Returns:
            bool: True if successful
        """
        test_file = self.data_dir / "test_data.pkl"
        
        if not test_file.exists():
            return False
        
        try:
            with open(test_file, 'rb') as f:
                self.test_data = pickle.load(f)
            
            return True
            
        except Exception as e:
            # Silently fail - test data is optional
            return False
    
    def load_ontology(self) -> bool:
        """
        Load GO ontology file.
        
        Returns:
            bool: True if successful
        """
        ontology_file = self.data_dir / "go.obo"
        
        if not ontology_file.exists():
            print(f"! Ontology file not found: {ontology_file}")
            return False
        
        try:
            self.ontology = self._parse_obo(ontology_file)
            print(f"✓ Loaded GO ontology")
            return True
            
        except Exception as e:
            print(f"⚠ Failed to load ontology: {str(e)}")
            return False
    
    def _parse_obo(self, obo_file: Path) -> Dict[str, Dict[str, Any]]:
        """
        Parse GO ontology OBO file.
        
        Args:
            obo_file (Path): Path to go.obo file
            
        Returns:
            Dict: Ontology structure
        """
        ontology = {}
        current_term = None
        
        with open(obo_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line == '[Term]':
                    current_term = {}
                    ontology[current_term.get('id')] = current_term
                elif line.startswith('id:'):
                    term_id = line.split(':', 1)[1].strip()
                    current_term = {'id': term_id}
                    ontology[term_id] = current_term
                elif current_term is not None:
                    if line.startswith('name:'):
                        current_term['name'] = line.split(':', 1)[1].strip()
                    elif line.startswith('namespace:'):
                        current_term['namespace'] = line.split(':', 1)[1].strip()
        
        return ontology
    
    def load_all(self) -> bool:
        """
        Load all model and data files. Training/test data and ontology are optional.
        
        Returns:
            bool: True if critical files loaded successfully
        """
        print("Loading DeepGOPlus model and data...")
        print("-" * 60)
        
        model_loaded = self.load_model()
        terms_loaded = self.load_go_terms()
        
        # Optional data files - silently skip if they fail
        self.load_train_data()
        self.load_test_data()
        self.load_ontology()
        
        print("-" * 60)
        
        if model_loaded and terms_loaded:
            print("✓ Model ready for predictions!")
            return True
        else:
            print("✗ Critical files missing. Cannot proceed.")
            return False
    
    def predict(self, sequence_encoded: np.ndarray) -> np.ndarray:
        """
        Run prediction on encoded sequence.
        
        Args:
            sequence_encoded (np.ndarray): One-hot encoded sequence (2000, 21)
            
        Returns:
            np.ndarray: Prediction probabilities for GO terms
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure proper shape: (batch_size, max_length, channels)
        if len(sequence_encoded.shape) == 2:
            # Input is (max_length, channels), add batch dimension
            sequence_encoded = np.expand_dims(sequence_encoded, axis=0)
        
        predictions = self.model.predict(sequence_encoded, verbose=0)
        return predictions[0] if len(predictions.shape) > 1 else predictions
    
    def get_go_term_name(self, go_id: str) -> str:
        """
        Get the name of a GO term.
        
        Args:
            go_id (str): GO term identifier (e.g., "GO:0005524")
            
        Returns:
            str: GO term name or the ID if name not found
        """
        if self.ontology is None:
            return go_id
        
        term = self.ontology.get(go_id, {})
        return term.get('name', go_id)


def get_default_model() -> DeepGOPlusModel:
    """
    Get default model instance.
    
    Returns:
        DeepGOPlusModel: Loaded model instance
    """
    model = DeepGOPlusModel()
    model.load_all()
    return model
