# AI Voice Detection API

REST API that detects whether a voice audio sample is **AI-generated** or **Human** across 5 languages.

## Supported Languages
- Tamil
- English
- Hindi
- Malayalam
- Telugu

## API Endpoint

**POST** `/api/voice-detection`

### Headers
```
Content-Type: application/json
x-api-key: YOUR_API_KEY
```

### Request Body
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_ENCODED_MP3_AUDIO"
}
```

### Response
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency detected"
}
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker Deployment

```bash
docker build -t voice-detection-api .
docker run -p 8000:8000 -e API_KEY=your_key voice-detection-api
```

## cURL Example

```bash
curl -X POST https://your-domain.com/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{"language": "Tamil", "audioFormat": "mp3", "audioBase64": "..."}'
```
