import os
from fastembed import TextEmbedding

# Define cache directory relative to this script or project root
# Using a local folder ensures it's part of the build artifact
CACHE_DIR = os.path.join(os.getcwd(), "model_cache")

def download_model():
    print(f"Downloading FastEmbed model to {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Initialize TextEmbedding with cache_dir set to our local folder
    # This triggers the download
    TextEmbedding(cache_dir=CACHE_DIR)
    
    print("Model download complete.")

if __name__ == "__main__":
    download_model()
