"""
Run the Voice Detection API Server
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           AI Voice Detection API Server                      ║
╠══════════════════════════════════════════════════════════════╣
║  Starting server at: http://{settings.HOST}:{settings.PORT}              ║
║  API Documentation:  http://localhost:{settings.PORT}/docs              ║
║  Supported Languages: Tamil, English, Hindi, Malayalam, Telugu ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
