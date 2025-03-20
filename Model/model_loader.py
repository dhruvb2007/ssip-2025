import os
import pickle
from io import BytesIO

chunk_folder = os.path.join(os.getcwd(), "ModelChunks")  # Absolute path
_model = None  # Cached model variable

def load_model():
    """Load model lazily and cache it."""
    global _model
    if _model is None:  # Load only if not already loaded
        print("🔄 Loading model from chunks...")
        model_chunks = sorted(os.listdir(chunk_folder))  # Sort to maintain order
        model_data = b""

        for chunk in model_chunks:
            chunk_path = os.path.join(chunk_folder, chunk)
            with open(chunk_path, "rb") as chunk_file:
                model_data += chunk_file.read()

        model_buffer = BytesIO(model_data)
        _model = pickle.load(model_buffer)
        print("✅ Model successfully loaded!")
    return _model
