import os
from fastapi import Header, HTTPException

ADMIN_API_KEY = (os.getenv("ADMIN_API_KEY") or "").strip()

def require_admin(x_api_key: str = Header(default="")):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not set on server")
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
