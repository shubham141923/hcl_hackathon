# AI-Generated Voice Detection API

A REST API that detects whether a voice audio sample is **AI-generated** or spoken by a **real human** across 5 supported languages.

## 🎯 Features

- **Multi-language Support**: Tamil, English, Hindi, Malayalam, Telugu
- **Secure API**: Protected with API key authentication
- **High Accuracy**: Uses advanced audio analysis and machine learning
- **Fast Processing**: Optimized for quick response times
- **Clear Explanations**: Provides reasoning for each classification

## 📋 Supported Languages

| Language   | Code      |
|------------|-----------|
| Tamil      | Tamil     |
| English    | English   |
| Hindi      | Hindi     |
| Malayalam  | Malayalam |
| Telugu     | Telugu    |

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   cd "hcl hackathon"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Copy example env file
   copy .env.example .env
   
   # Edit .env with your settings
   # Change API_KEY to a secure value for production
   ```

5. **Run the server**
   ```bash
   python run.py
   ```

The API will be available at `http://localhost:5000`

## 📖 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Authentication

All requests must include an API key in the header:

```
x-api-key: YOUR_SECRET_API_KEY
```

### Endpoints

#### Health Check
```bash
GET /api/health
```

#### Voice Detection
```bash
POST /api/voice-detection
```

### Request Format

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_ENCODED_MP3_DATA..."
}
```

### Response Format (Success)

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

### Response Format (Error)

```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

## 🧪 Testing

### Using cURL

```bash
# Health check
curl http://localhost:8000/api/health

# Voice detection
curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_ENCODED_AUDIO"
  }'
```

### Using Test Client

```bash
python test_client.py path/to/audio.mp3 English
```

## 🏗️ Project Structure

```
hcl hackathon/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── auth.py              # API key authentication
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── routes/
│   │   ├── __init__.py
│   │   └── voice_detection.py  # API endpoints
│   └── services/
│       ├── __init__.py
│       ├── audio_processor.py  # Audio feature extraction
│       └── voice_detector.py   # AI/Human classification
├── models/
│   └── README.md            # Trained model storage
├── .env                     # Environment variables
├── .env.example             # Example environment file
├── requirements.txt         # Python dependencies
├── run.py                   # Server entry point
├── test_client.py           # Test utilities
├── Dockerfile               # Docker container
└── README.md
```

## 🔧 Configuration

Environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | Secret API key for authentication | `sk_test_123456789` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Enable debug mode | `true` |

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t voice-detection-api .
```

### Run Container

```bash
docker run -p 8000:8000 -e API_KEY=your_secret_key voice-detection-api
```

## 📊 Classification Details

The API analyzes multiple audio features to determine voice authenticity:

### Features Analyzed

1. **MFCC (Mel-frequency cepstral coefficients)** - Voice timbre analysis
2. **Pitch Patterns** - Natural vs synthetic pitch variations
3. **Spectral Features** - Frequency distribution characteristics
4. **Zero Crossing Rate** - Voice texture indicator
5. **RMS Energy** - Dynamic range patterns
6. **Tempo & Rhythm** - Speech rhythm naturalness

### Classification Output

| Value | Meaning |
|-------|---------|
| `AI_GENERATED` | Voice created using AI/TTS/synthetic systems |
| `HUMAN` | Voice spoken by a real human being |

### Confidence Score

- **0.0 - 0.5**: Low confidence (uncertain)
- **0.5 - 0.7**: Moderate confidence
- **0.7 - 0.9**: High confidence
- **0.9 - 1.0**: Very high confidence

## ⚠️ Important Notes

- Only MP3 format is supported
- Audio must be Base64 encoded
- One audio file per request
- Maximum audio duration: 5 minutes
- API key is required for all detection requests

## 📝 License

This project is created for the HCL Hackathon.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
