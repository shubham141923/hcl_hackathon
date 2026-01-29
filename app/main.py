"""
AI Voice Detection API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router
from app.config import settings

# Create app
app = FastAPI(
    title="AI Voice Detection API",
    description="Detect AI-generated vs Human voices in Tamil, English, Hindi, Malayalam, Telugu",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api", tags=["Voice Detection"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": "AI Voice Detection API",
        "version": "1.0.0",
        "endpoints": {
            "voice_detection": "/api/voice-detection",
            "health": "/api/health",
            "docs": "/docs"
        },
        "supported_languages": settings.SUPPORTED_LANGUAGES
    }


@app.on_event("startup")
async def startup():
    """Preload libraries on startup"""
    import threading
    
    def preload():
        try:
            print("Preloading libraries...")
            import librosa
            import torch
            import numpy as np
            # Quick warmup
            _ = librosa.feature.mfcc(y=np.zeros(16000), sr=16000, n_mfcc=13)
            print("Libraries ready!")
        except Exception as e:
            print(f"Preload warning: {e}")
    
    thread = threading.Thread(target=preload, daemon=True)
    thread.start()
