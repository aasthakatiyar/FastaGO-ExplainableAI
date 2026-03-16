import pickle
import pandas as pd
from tensorflow.keras.models import load_model

MODEL_PATH = "model/model.h5"
TERMS_PATH = "data/terms.pkl"

print("--- DEEPGOPLUS SYNC DIAGNOSTICS ---")

# 1. Check Model Output Layer
model = load_model(MODEL_PATH)
model_output_size = model.output_shape[1]
print(f"Neural Network Output Classes: {model_output_size}")

# 2. Check terms.pkl Size
with open(TERMS_PATH, "rb") as f:
    terms = pickle.load(f)
    if isinstance(terms, pd.DataFrame):
        terms_list = terms.iloc[:, 0].tolist()
    else:
        terms_list = list(terms)
terms_size = len(terms_list)
print(f"Terms.pkl Label Count:        {terms_size}")

# 3. Validation
if model_output_size == terms_size:
    print("\nSUCCESS: Model and Terms are SYNCED.")
else:
    print("\nCRITICAL ERROR: Mismatch detected!")
    print(f"The model predicts {model_output_size} things, but you only have names for {terms_size}.")
    print("This is why your results are biologically incorrect.")