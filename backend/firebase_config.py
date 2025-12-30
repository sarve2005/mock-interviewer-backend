import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase Admin
# Expects GOOGLE_APPLICATION_CREDENTIALS to be set or use default
if not firebase_admin._apps:
    try:
        # Check if we have specific creds file or use default
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app()
    except Exception as e:
        print(f"Warning: Firebase Admin init failed: {e}")

def get_db():
    if not firebase_admin._apps:
        return None
    try:
        return firestore.client()
    except Exception as e:
        print(f"Firestore client error: {e}")
        return None
