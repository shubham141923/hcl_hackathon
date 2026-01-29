"""
AI-Generated Voice Detection API
Main FastAPI Application

This API detects whether a voice audio sample is AI-generated or 
spoken by a real human across 5 supported languages:
- Tamil
- English
- Hindi
- Malayalam
- Telugu
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
import time

from app.routes import voice_detection
from app.config import settings


# Create FastAPI application
app = FastAPI(
    title="AI Voice Detection API",
    description="""
## AI-Generated Voice Detection API

This API analyzes voice audio samples to determine whether they are:
- **AI_GENERATED** - Voice created using AI or synthetic systems (TTS, voice cloning, etc.)
- **HUMAN** - Voice spoken by a real human

### Supported Languages
- Tamil
- English  
- Hindi
- Malayalam
- Telugu

### Authentication
All requests must include a valid API key in the `x-api-key` header.

### How It Works
The API uses advanced audio signal processing and machine learning to analyze:
- Pitch patterns and consistency
- Spectral characteristics
- Voice timbre and quality
- Speech rhythm and dynamics
- Natural micro-variations

""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response


@app.exception_handler(HTTP_422_UNPROCESSABLE_ENTITY)
async def validation_exception_handler(request: Request, exc):
    """Custom handler for validation errors"""
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Invalid request format. Please check your request body."
        }
    )


# Include routers
app.include_router(
    voice_detection.router,
    prefix="/api",
    tags=["Voice Detection"]
)


@app.on_event("startup")
async def startup_event():
    """Preload heavy libraries in background"""
    import threading
    
    def preload_libs():
        try:
            print("Preloading libraries in background...")
            import librosa
            import numpy as np
            dummy_signal = np.zeros(22050, dtype=np.float32)
            _ = librosa.feature.mfcc(y=dummy_signal, sr=22050, n_mfcc=13)
            print("Libraries preloaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to preload: {e}")
    
    # Run in background thread
    thread = threading.Thread(target=preload_libs, daemon=True)
    thread.start()


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Voice Detection API",
        "version": "1.0.0",
        "description": "Detect AI-generated voices in Tamil, English, Hindi, Malayalam, and Telugu",
        "endpoints": {
            "voice_detection": "/api/voice-detection",
            "health": "/api/health",
            "docs": "/docs"
        },
        "supported_languages": settings.SUPPORTED_LANGUAGES
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
