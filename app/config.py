"""
Configuration settings for the Voice Detection API
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    API_KEY: str = "sk_test_123456789"
    API_KEY_HEADER: str = "x-api-key"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Model Configuration
    MODEL_PATH: str = "models/voice_classifier.joblib"
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Supported Languages
    SUPPORTED_LANGUAGES: List[str] = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    # Audio Configuration
    SAMPLE_RATE: int = 22050
    MAX_AUDIO_DURATION: int = 300  # 5 minutes max
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
