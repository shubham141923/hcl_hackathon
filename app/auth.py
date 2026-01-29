"""
API Authentication middleware and security utilities
"""
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

from app.config import settings


# Define API Key header
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify the API key from request headers.
    
    Args:
        api_key: API key extracted from request header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "API key is missing. Please provide a valid API key in the x-api-key header."
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


def get_api_key_dependency():
    """Returns the API key verification dependency"""
    return Depends(verify_api_key)
