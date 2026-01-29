"""
Test client for the Voice Detection API
Provides utility functions to test the API with sample audio
"""
import base64
import requests
import json
import sys
import os

# API Configuration
BASE_URL = "http://127.0.0.1:5000"
API_KEY = "sk_test_123456789"


def encode_audio_file(file_path: str) -> str:
    """
    Encode an MP3 audio file to base64
    
    Args:
        file_path: Path to the MP3 file
        
    Returns:
        Base64 encoded string
    """
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        return base64.b64encode(audio_bytes).decode("utf-8")


def test_voice_detection(audio_path: str, language: str = "English"):
    """
    Test the voice detection API with a sample audio file
    
    Args:
        audio_path: Path to the MP3 audio file
        language: Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing Voice Detection API")
    print(f"{'='*60}")
    print(f"Audio File: {audio_path}")
    print(f"Language: {language}")
    print(f"{'='*60}\n")
    
    # Encode audio to base64
    print("Encoding audio to base64...")
    audio_base64 = encode_audio_file(audio_path)
    print(f"Audio size: {len(audio_base64)} characters (base64)")
    
    # Prepare request
    url = f"{BASE_URL}/api/voice-detection"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    print(f"\nSending request to {url}...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"\nResponse Body:")
        print(json.dumps(response.json(), indent=2))
        
        return response.json()
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API server.")
        print("Make sure the server is running at", BASE_URL)
    except Exception as e:
        print(f"\nError: {str(e)}")


def test_health_check():
    """Test the health check endpoint"""
    print("\nTesting Health Check...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.json()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
    except Exception as e:
        print(f"Error: {str(e)}")


def test_without_api_key(audio_path: str, language: str = "English"):
    """Test the API without an API key (should fail)"""
    print("\nTesting without API key (should return 401)...")
    
    audio_base64 = encode_audio_file(audio_path)
    
    url = f"{BASE_URL}/api/voice-detection"
    headers = {
        "Content-Type": "application/json"
        # No API key header
    }
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")


def test_invalid_api_key(audio_path: str, language: str = "English"):
    """Test the API with an invalid API key (should fail)"""
    print("\nTesting with invalid API key (should return 403)...")
    
    audio_base64 = encode_audio_file(audio_path)
    
    url = f"{BASE_URL}/api/voice-detection"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "invalid_key_12345"
    }
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <path_to_audio.mp3> [language]")
        print("\nSupported languages: Tamil, English, Hindi, Malayalam, Telugu")
        print("\nExample: python test_client.py sample_voice.mp3 Tamil")
        print("\nRunning health check only...")
        test_health_check()
    else:
        audio_path = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else "English"
        
        # Run all tests
        test_health_check()
        test_voice_detection(audio_path, language)
        
        if os.path.exists(audio_path):
            test_without_api_key(audio_path, language)
            test_invalid_api_key(audio_path, language)
