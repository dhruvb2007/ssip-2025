import pandas as pd
import joblib
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from io import BytesIO

# Load dataset
df = pd.read_csv("../Data/gujarat_students_data.csv")

# Define max values for normalization
max_values = {
    "Gujarati_PAT": 20, "Gujarati_SAT": 80,
    "Mathematics_PAT": 20, "Mathematics_SAT": 80,
    "Science_PAT": 20, "Science_SAT": 80,
    "SocialScience_PAT": 20, "SocialScience_SAT": 80,
    "English_PAT": 20, "English_SAT": 80,
    "Hindi_PAT": 20, "Hindi_SAT": 80,
    "Attendance": 100, "Assessment": 100,
    "Final_Score": 100
}

# Normalize the data
for col in max_values:
    if col in df.columns:
        df[col] = df[col] / max_values[col]

# Select Features & Target
X = df.drop(columns=["Final_Score"])
y = df["Final_Score"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Convert Model to Bytes (Without Writing a Large File)
model_buffer = BytesIO()
pickle.dump(model, model_buffer)
model_bytes = model_buffer.getvalue()

# Define Chunk Size (<25MB)
chunk_size = 25 * 1024 * 1024  # 25MB
chunk_folder = "../ModelChunks"
os.makedirs(chunk_folder, exist_ok=True)

# Save Model in Chunks
for i in range(0, len(model_bytes), chunk_size):
    chunk_data = model_bytes[i:i+chunk_size]
    chunk_path = os.path.join(chunk_folder, f"model_chunk_{i // chunk_size}.pkl")
    
    with open(chunk_path, "wb") as chunk_file:
        chunk_file.write(chunk_data)
    
    print(f"✅ Saved chunk {i // chunk_size}: {chunk_path}")

print("✅ Model training complete! Model saved in chunks without storing a large file.")
