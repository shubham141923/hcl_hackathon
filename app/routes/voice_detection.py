"""
Voice Detection API Routes
Main API endpoints for voice detection functionality
"""
from fastapi import APIRouter, Depends, HTTPException
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
import traceback

from app.models.schemas import (
    VoiceDetectionRequest,
    VoiceDetectionSuccessResponse,
    VoiceDetectionErrorResponse,
    HealthCheckResponse
)
from app.auth import verify_api_key
from app.services.voice_detector import voice_detector
from app.config import settings


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check if the API is running and healthy"
)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        supported_languages=settings.SUPPORTED_LANGUAGES
    )


@router.post(
    "/voice-detection",
    response_model=VoiceDetectionSuccessResponse,
    responses={
        200: {"model": VoiceDetectionSuccessResponse, "description": "Successful voice detection"},
        400: {"model": VoiceDetectionErrorResponse, "description": "Bad request"},
        401: {"model": VoiceDetectionErrorResponse, "description": "Unauthorized - Missing API key"},
        403: {"model": VoiceDetectionErrorResponse, "description": "Forbidden - Invalid API key"},
        500: {"model": VoiceDetectionErrorResponse, "description": "Internal server error"}
    },
    summary="Detect AI-Generated Voice",
    description="""
    Analyze a voice audio sample to determine if it is AI-generated or spoken by a human.
    
    **Supported Languages:** Tamil, English, Hindi, Malayalam, Telugu
    
    **Input:** Base64-encoded MP3 audio
    
    **Output:** Classification (AI_GENERATED or HUMAN) with confidence score and explanation
    """
)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Main voice detection endpoint.
    
    Accepts a Base64-encoded MP3 audio file and returns whether the voice
    is AI-generated or human, along with a confidence score and explanation.
    """
    try:
        # Validate language
        language = request.language.value
        
        if language not in settings.SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error",
                    "message": f"Unsupported language: {language}. Supported languages are: {', '.join(settings.SUPPORTED_LANGUAGES)}"
                }
            )
        
        # Perform voice detection
        result = voice_detector.detect(
            audio_base64=request.audioBase64,
            language=language
        )
        
        # Return success response
        return VoiceDetectionSuccessResponse(
            status="success",
            language=request.language,
            classification=result["classification"],
            confidenceScore=result["confidenceScore"],
            explanation=result["explanation"]
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        # Handle validation errors from audio processing
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": str(e)
            }
        )
    except Exception as e:
        # Log the error for debugging
        print(f"Error in voice detection: {str(e)}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "An internal error occurred while processing the audio"
            }
        )
