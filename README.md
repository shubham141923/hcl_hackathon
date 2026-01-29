# AI Voice Detection API

Detects AI-generated vs Human voices across 5 languages.

## Languages
Tamil, English, Hindi, Malayalam, Telugu

## API

**POST** `/api/voice-detection`

### Headers
```
Content-Type: application/json
x-api-key: YOUR_API_KEY
```

### Request
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_MP3_DATA"
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

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t voice-api .
docker run -p 8000:8000 -e API_KEY=your_key voice-api
```
