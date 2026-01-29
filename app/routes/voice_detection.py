"""
Voice Detection API Routes
"""
from fastapi import APIRouter, Depends, HTTPException
from starlette.status import HTTP_200_OK, HTTP_500_INTERNAL_SERVER_ERROR

from app.models import VoiceDetectionRequest, VoiceDetectionResponse, HealthResponse
from app.services import voice_detector
from app.auth import verify_api_key
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        supported_languages=settings.SUPPORTED_LANGUAGES
    )


@router.post("/voice-detection", response_model=VoiceDetectionResponse)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect if voice is AI-generated or Human
    
    - **language**: Tamil, English, Hindi, Malayalam, or Telugu
    - **audioFormat**: mp3
    - **audioBase64**: Base64 encoded MP3 audio
    """
    try:
        result = voice_detector.detect(
            audio_base64=request.audioBase64,
            language=request.language
        )
        
        return VoiceDetectionResponse(
            status="success",
            language=result["language"],
            classification=result["classification"],
            confidenceScore=result["confidenceScore"],
            explanation=result["explanation"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": str(e)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Processing error: {str(e)}"}
        )
