import os
import pickle
from io import BytesIO

chunk_folder = os.path.join(os.getcwd(), "ModelChunks")  # Absolute path

def load_model():
    """Reassemble the model from multiple chunks without storing a full file."""
    model_chunks = sorted(os.listdir(chunk_folder))  # Sort to maintain order
    model_data = b""

    # Read chunks and reconstruct model data
    for chunk in model_chunks:
        chunk_path = os.path.join(chunk_folder, chunk)
        with open(chunk_path, "rb") as chunk_file:
            model_data += chunk_file.read()

    # Load Model from Bytes
    model_buffer = BytesIO(model_data)
    model = pickle.load(model_buffer)
    
    print("✅ Model successfully loaded from chunks!")
    return model
