import os
from google import genai
from google.genai import types

# --- CONFIGURATION ---
KEY_FILE = ".gemini_key"

def load_stored_key():
    """Load API key from the local hidden file."""
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""

def save_key_locally(api_key):
    """Persist the API key to a local hidden file."""
    with open(KEY_FILE, "w") as f:
        f.write(api_key)

def delete_stored_key():
    """Remove the local key file."""
    if os.path.exists(KEY_FILE):
        os.remove(KEY_FILE)

class GeminiExplainer:
    """Manager for Gemini API interactions."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = None
        if api_key:
            # Initialize the client using the API Key
            self.client = genai.Client(api_key=api_key)

    def list_my_models(self):
        """Helper to see which models your API key can access."""
        if not self.client:
            return []
        print("Checking accessible models...")
        return [m.name for m in self.client.models.list()]

    def get_explanation(self, protein_id, sequence, go_term, go_name, go_def, score):
        """Generate a scientific explanation for a GO term prediction."""
        if not self.client:
            return "⚠️ Gemini client not initialized. Please provide an API key."
        
        prompt = f"""
        As a bioinformatics expert, explain the functional relevance of a Gene Ontology (GO) term prediction.
        
        Protein ID: {protein_id}
        Protein Sequence (first 100 aa): {sequence[:100]}...
        Predicted GO Term: {go_term} ({go_name})
        GO Definition: {go_def}
        Prediction Confidence Score: {score:.4f}
        
        Please provide:
        1. A brief explanation of what this function means for the protein.
        2. Why a deep learning model (DeepGOPlus) might associate this sequence with this function.
        3. The biological significance of this finding.
        """
        
        try:
            # Using 'gemini-2.5-flash' for stability on the Free Tier
            response = self.client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt
            )
            return response.text
        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg:
                # Fallback to 1.5 if 2.5 is not yet enabled for your key
                try:
                    response = self.client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=prompt
                    )
                    return response.text
                except:
                    return f"❌ Model not found. Available models: {self.list_my_models()}"
            
            if "401" in err_msg:
                return "❌ Unauthorized: Your API key appears to be invalid."
            if "429" in err_msg:
                return "❌ Quota Exceeded: Free tier limit reached."
            return f"❌ Error: {err_msg}"

if __name__ == "__main__":
    api_key = load_stored_key()
    if not api_key:
        print("No API key found. Please set your Gemini API key.")
        exit(1)
    
    explainer = GeminiExplainer(api_key)
    
    # Optional: Uncomment to see all models you can use:
    # print("Available Models:", explainer.list_my_models())
    
    protein_id = "test_protein_001"
    sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    go_term = "GO:0005506"
    go_name = "iron ion binding"
    go_def = "Binding to an iron ion."
    score = 0.85
    
    print("\nRequesting AI Explanation...\n")
    explanation = explainer.get_explanation(protein_id, sequence, go_term, go_name, go_def, score)
    print(explanation)