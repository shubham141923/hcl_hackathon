"""
API Key Authentication
"""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from app.config import settings

api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key from header"""
    if api_key is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "API key is missing. Provide x-api-key header."
            }
        )
    
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )
    
    return api_key
