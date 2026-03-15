import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from data_preparation import load_dataset
from sequence_encoder import encode_sequences_batch, MAX_SEQUENCE_LENGTH
import pickle

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
MODEL_PATH = "models/deepgoplus_model.h5"

def build_model(num_go_terms, input_shape=(MAX_SEQUENCE_LENGTH, 20)):
    """Build CNN model for GO term prediction"""
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # 1D Convolution blocks
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Global pooling
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer with sigmoid for multi-label classification
        layers.Dense(num_go_terms, activation='sigmoid')
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model

def prepare_training_data(dataset):
    """Prepare sequences and labels for training"""
    print("Encoding sequences...")
    
    sequences = [sample['sequence'] for sample in dataset]
    labels = np.array([sample['labels'] for sample in dataset])
    
    # Encode sequences
    X = encode_sequences_batch(sequences)
    y = labels
    
    print(f"Data shape - X: {X.shape}, y: {y.shape}")
    
    return X, y

def train_model():
    """Train the model"""
    
    # Load dataset
    print("Loading dataset...")
    dataset, encoder, go_terms = load_dataset()
    
    num_go_terms = len(go_terms)
    print(f"Number of GO terms: {num_go_terms}")
    
    # Prepare data
    X, y = prepare_training_data(dataset)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    print("Building model...")
    model = build_model(num_go_terms)
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.00001
            )
        ]
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Save training history and test results
    with open("outputs/training_history.pkl", 'wb') as f:
        pickle.dump(history.history, f)
    
    with open("outputs/test_results.txt", 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
    
    return model, history

def main():
    train_model()

if __name__ == "__main__":
    main()
