"""
Pydantic schemas for request and response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from enum import Enum


class LanguageEnum(str, Enum):
    """Supported languages for voice detection"""
    TAMIL = "Tamil"
    ENGLISH = "English"
    HINDI = "Hindi"
    MALAYALAM = "Malayalam"
    TELUGU = "Telugu"


class AudioFormatEnum(str, Enum):
    """Supported audio formats"""
    MP3 = "mp3"


class ClassificationEnum(str, Enum):
    """Voice classification types"""
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"


class VoiceDetectionRequest(BaseModel):
    """Request schema for voice detection endpoint"""
    language: LanguageEnum = Field(
        ...,
        description="Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)"
    )
    audioFormat: AudioFormatEnum = Field(
        default=AudioFormatEnum.MP3,
        description="Audio format (always mp3)"
    )
    audioBase64: str = Field(
        ...,
        min_length=100,
        description="Base64-encoded MP3 audio data"
    )
    
    @field_validator('audioBase64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that the audioBase64 field contains valid base64 data"""
        import base64
        try:
            # Try to decode the base64 string
            decoded = base64.b64decode(v)
            if len(decoded) < 100:
                raise ValueError("Audio data too short")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
    
    class Config:
        json_schema_extra = {
            "example": {
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
            }
        }


class VoiceDetectionSuccessResponse(BaseModel):
    """Success response schema for voice detection"""
    status: Literal["success"] = "success"
    language: LanguageEnum
    classification: ClassificationEnum
    confidenceScore: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    explanation: str = Field(
        ...,
        description="Short reason for the classification decision"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "language": "Tamil",
                "classification": "AI_GENERATED",
                "confidenceScore": 0.91,
                "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
            }
        }


class VoiceDetectionErrorResponse(BaseModel):
    """Error response schema for voice detection"""
    status: Literal["error"] = "error"
    message: str = Field(
        ...,
        description="Error message describing what went wrong"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    status: str = "healthy"
    version: str = "1.0.0"
    supported_languages: list = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
