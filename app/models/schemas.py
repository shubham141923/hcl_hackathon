"""
Request/Response Schemas
"""
from pydantic import BaseModel, Field, field_validator
from typing import Literal
import base64


class VoiceDetectionRequest(BaseModel):
    """Request schema for voice detection"""
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ...,
        description="Language of the audio"
    )
    audioFormat: Literal["mp3"] = Field(
        ...,
        description="Audio format (mp3 only)"
    )
    audioBase64: str = Field(
        ...,
        description="Base64 encoded MP3 audio",
        min_length=100
    )
    
    @field_validator('audioBase64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            # Add padding if needed
            padding = 4 - len(v) % 4
            if padding != 4:
                v += '=' * padding
            base64.b64decode(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}")


class VoiceDetectionResponse(BaseModel):
    """Response schema for voice detection"""
    status: Literal["success", "error"] = "success"
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class ErrorResponse(BaseModel):
    """Error response schema"""
    status: Literal["error"] = "error"
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str = "1.0.0"
    supported_languages: list = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
