"""
Simple API test for the Voice Detection API
Uses pre-encoded base64 audio sample
"""
import requests
import json

# API Configuration
BASE_URL = "http://127.0.0.1:5000"
API_KEY = "sk_test_123456789"

# A minimal valid MP3 file encoded in base64 (properly padded)
# This is a very short valid MP3 frame
SAMPLE_AUDIO_BASE64 = (
    "//uQxAAAAAANIAAAAAExBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUx"
    "BTUUzLjEwMFZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZW"
    "VlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZW"
    "VlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZW//uQxEsAAADSAAAAAEFBQUFBQUFB"
    "RERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERER"
    "ERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERER"
    "ERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERER"
    "ERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERGMARox"
    "AAAAAAAAAAAAAAAAAAA="
)


def test_health():
    """Test the health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_voice_detection(language="English"):
    """Test the voice detection endpoint"""
    print("\n" + "="*60)
    print(f"Testing Voice Detection - {language}")
    print("="*60)
    
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": SAMPLE_AUDIO_BASE64
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    print(f"Request URL: POST {BASE_URL}/api/voice-detection")
    print(f"Language: {language}")
    print(f"Audio Base64 Length: {len(SAMPLE_AUDIO_BASE64)} chars")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/voice-detection",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_without_api_key():
    """Test that requests without API key are rejected"""
    print("\n" + "="*60)
    print("Testing Without API Key (Should Fail)")
    print("="*60)
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": SAMPLE_AUDIO_BASE64
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/voice-detection",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.status_code in [401, 403]
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_invalid_api_key():
    """Test that requests with invalid API key are rejected"""
    print("\n" + "="*60)
    print("Testing With Invalid API Key (Should Fail)")
    print("="*60)
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": SAMPLE_AUDIO_BASE64
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "invalid_api_key_12345"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/voice-detection",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.status_code == 403
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_all_languages():
    """Test detection for all supported languages"""
    print("\n" + "="*60)
    print("Testing All Supported Languages")
    print("="*60)
    
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    results = {}
    
    for lang in languages:
        try:
            result = test_voice_detection(lang)
            if result:
                results[lang] = result.get('status') == 'success'
            else:
                results[lang] = False
            print(f"\n{lang}: {'PASS' if results[lang] else 'FAIL'}")
        except Exception as e:
            results[lang] = False
            print(f"\n{lang}: FAIL - {str(e)}")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI Voice Detection API - Test Suite")
    print("="*60)
    
    try:
        # Run tests
        health_ok = test_health()
        print(f"\n>>> Health Check: {'PASS' if health_ok else 'FAIL'}")
        
        if health_ok:
            # Test authentication
            no_key_ok = test_without_api_key()
            print(f"\n>>> No API Key Test: {'PASS' if no_key_ok else 'FAIL'}")
            
            invalid_key_ok = test_invalid_api_key()
            print(f"\n>>> Invalid API Key Test: {'PASS' if invalid_key_ok else 'FAIL'}")
            
            # Test voice detection for all languages
            lang_results = test_all_languages()
            
            print("\n" + "="*60)
            print("  FINAL SUMMARY")
            print("="*60)
            print(f"Health Check: {'PASS' if health_ok else 'FAIL'}")
            print(f"Auth (No Key): {'PASS' if no_key_ok else 'FAIL'}")
            print(f"Auth (Invalid Key): {'PASS' if invalid_key_ok else 'FAIL'}")
            for lang, passed in lang_results.items():
                print(f"Voice Detection ({lang}): {'PASS' if passed else 'FAIL'}")
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to the API server.")
        print("Make sure the server is running at:", BASE_URL)
        print("\nStart the server with: python run.py")
