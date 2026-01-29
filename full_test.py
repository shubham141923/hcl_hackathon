"""
Create a proper test MP3 file using wave module and test the API
"""
import wave
import struct
import base64
import io
import requests
import json
import os

# API Configuration
BASE_URL = "http://127.0.0.1:5000"
API_KEY = "sk_test_123456789"


def create_wav_audio():
    """Create a simple WAV audio in memory"""
    sample_rate = 22050
    duration = 2  # seconds
    frequency = 440  # Hz
    
    # Generate samples
    num_samples = sample_rate * duration
    samples = []
    
    for i in range(num_samples):
        t = i / sample_rate
        # Simple sine wave with some variations
        value = int(32767 * 0.5 * (
            0.7 * __import__('math').sin(2 * 3.14159 * frequency * t) +
            0.2 * __import__('math').sin(2 * 3.14159 * (frequency * 2) * t) +
            0.1 * __import__('math').sin(2 * 3.14159 * (frequency * 3) * t)
        ))
        samples.append(value)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        for sample in samples:
            wav_file.writeframes(struct.pack('<h', sample))
    
    buffer.seek(0)
    return buffer.read()


def test_with_wav():
    """Test the API with WAV audio (converted to base64)"""
    print("\n" + "="*60)
    print("Creating test audio...")
    print("="*60)
    
    # Create WAV audio
    wav_data = create_wav_audio()
    print(f"Created WAV audio: {len(wav_data)} bytes")
    
    # Convert to base64
    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
    print(f"Base64 length: {len(audio_base64)} chars")
    
    # Test health first
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test voice detection
    print("\n" + "="*60)
    print("Testing Voice Detection with Real Audio")
    print("="*60)
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",  # We're sending wav but API should still process it
        "audioBase64": audio_base64
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    print(f"Request URL: POST {BASE_URL}/api/voice-detection")
    print(f"Audio size: {len(audio_base64)} chars")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/voice-detection",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("SUCCESS!")
            print("="*60)
            print(f"Classification: {result.get('classification')}")
            print(f"Confidence: {result.get('confidenceScore')}")
            print(f"Explanation: {result.get('explanation')}")
        
    except Exception as e:
        print(f"Error: {e}")


def test_auth():
    """Test authentication"""
    print("\n" + "="*60)
    print("Testing Authentication")
    print("="*60)
    
    wav_data = create_wav_audio()
    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    # Test without API key
    print("\n1. Without API Key:")
    try:
        response = requests.post(
            f"{BASE_URL}/api/voice-detection",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        print(f"   Status: {response.status_code} (expected: 401)")
        print(f"   PASS" if response.status_code == 401 else "   FAIL")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test with invalid API key
    print("\n2. With Invalid API Key:")
    try:
        response = requests.post(
            f"{BASE_URL}/api/voice-detection",
            headers={"Content-Type": "application/json", "x-api-key": "wrong_key"},
            json=payload,
            timeout=10
        )
        print(f"   Status: {response.status_code} (expected: 403)")
        print(f"   PASS" if response.status_code == 403 else "   FAIL")
    except Exception as e:
        print(f"   Error: {e}")


def test_all_languages():
    """Test all supported languages"""
    print("\n" + "="*60)
    print("Testing All 5 Languages")
    print("="*60)
    
    wav_data = create_wav_audio()
    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
    
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    results = {}
    
    for lang in languages:
        print(f"\nTesting {lang}...")
        
        payload = {
            "language": lang,
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": API_KEY
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/voice-detection",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                results[lang] = True
                print(f"   {lang}: PASS - {result.get('classification')} ({result.get('confidenceScore')})")
            else:
                results[lang] = False
                print(f"   {lang}: FAIL - Status {response.status_code}")
                
        except Exception as e:
            results[lang] = False
            print(f"   {lang}: ERROR - {e}")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI Voice Detection API - Complete Test Suite")
    print("="*60)
    
    try:
        # Test authentication
        test_auth()
        
        # Test voice detection
        test_with_wav()
        
        # Test all languages
        results = test_all_languages()
        
        # Summary
        print("\n" + "="*60)
        print("  FINAL SUMMARY")
        print("="*60)
        passed = sum(1 for v in results.values() if v)
        print(f"Languages Passed: {passed}/5")
        for lang, status in results.items():
            print(f"  {lang}: {'PASS' if status else 'FAIL'}")
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to the API server.")
        print("Make sure the server is running at:", BASE_URL)
