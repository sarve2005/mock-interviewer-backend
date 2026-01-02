import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin
# Attempt to use default credentials (GOOGLE_APPLICATION_CREDENTIALS)
# If running locally without env var, this might fail unless authenticated via gcloud CLI
try:
    if not firebase_admin._apps:
        app = firebase_admin.initialize_app()
    else:
        app = firebase_admin.get_app()
except Exception as e:
    print(f"Warning: Firebase Admin initialization failed: {e}")
    print("Ensure GOOGLE_APPLICATION_CREDENTIALS is set or you are logged in via gcloud.")
    # For now, we'll let it fail hard on usage if not initialized, 
    # or you could try a fallback if you have a serviceAccountKey.json explicitly.
    db = None
else:
    db = firestore.client()
