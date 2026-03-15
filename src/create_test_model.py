"""
Create a mock/test model for system validation without the full pre-trained model.
Useful for testing the pipeline before downloading the actual model.
"""

import sys
from pathlib import Path

try:
    from tensorflow import keras
    import numpy as np
except ImportError:
    print("✗ TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)


def create_test_model(output_path: Path = None):
    """
    Create a simple test CNN model compatible with DeepGOPlus pipeline.
    
    Args:
        output_path (Path): Where to save the model
    """
    if output_path is None:
        output_path = Path("models/model.h5")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Creating test CNN model for DeepGOPlus...")
    print(f"Output: {output_path}")
    
    # Input: one-hot encoded sequence (20 amino acids x 2000 length)
    inputs = keras.Input(shape=(20, 2000))
    
    # Convolutional layers
    x = keras.layers.Conv1D(32, 8, activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling1D(4)(x)
    
    x = keras.layers.Conv1D(64, 8, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling1D(4)(x)
    
    x = keras.layers.Conv1D(128, 8, activation='relu', padding='same')(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    
    # Dense layers
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Output: 588 GO terms (standard DeepGOPlus size)
    outputs = keras.layers.Dense(588, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Save
    model.save(str(output_path))
    print(f"\n✓ Test model saved to: {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return True


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create a test model for DeepGOPlus pipeline validation'
    )
    parser.add_argument(
        '--output', '-o',
        default='models/model.h5',
        help='Output model path (default: models/model.h5)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DeepGOPlus Test Model Creator")
    print("=" * 70)
    print()
    print("This creates a small CNN model for testing the prediction pipeline")
    print("without downloading the full pre-trained model (2+ GB).")
    print()
    
    try:
        success = create_test_model(Path(args.output))
        
        print("\n" + "=" * 70)
        if success:
            print("✓ Test model created successfully!")
            print("\nYou can now:")
            print(f"  1. Test predictions: python src/predictor.py --input examples/sample.fasta")
            print(f"  2. Run web app: streamlit run app/streamlit_app.py")
            print("\nNote: This test model generates random predictions.")
            print("For accurate predictions, download the actual pre-trained model from:")
            print("http://deepgoplus.bio2vec.net/")
        print("=" * 70)
        
        return success
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
