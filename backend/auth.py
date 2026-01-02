from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verifies the Firebase ID token and returns the user's UID.
    """
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        return uid
    except Exception as e:
        print(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )
