"""
Application Configuration
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # API Settings
    API_KEY: str = "sk_test_123456789"
    API_KEY_HEADER: str = "x-api-key"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Supported Languages
    SUPPORTED_LANGUAGES: List[str] = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
