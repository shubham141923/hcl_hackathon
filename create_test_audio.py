"""
Generate a sample audio file for testing the Voice Detection API
Uses Python to create a simple sine wave audio
"""
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import io
import base64
import os


def create_test_audio(output_path: str = "test_audio.mp3", duration_seconds: float = 3.0):
    """
    Create a simple test audio file with a sine wave.
    This is for testing purposes - it won't represent AI or human voice,
    but can be used to test the API mechanics.
    
    Args:
        output_path: Path to save the MP3 file
        duration_seconds: Duration of the audio in seconds
    """
    # Audio parameters
    sample_rate = 22050  # Hz
    frequency = 440  # Hz (A4 note)
    
    # Generate time array
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    
    # Generate sine wave with some frequency modulation to simulate voice-like characteristics
    # Add multiple harmonics and some amplitude modulation
    signal = np.zeros_like(t)
    
    # Fundamental frequency with vibrato
    vibrato_rate = 5  # Hz
    vibrato_depth = 20  # Hz
    freq_modulated = frequency + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    
    # Generate harmonics (like human voice formants)
    for i, (harmonic, amplitude) in enumerate([(1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15), (5, 0.08)]):
        signal += amplitude * np.sin(2 * np.pi * (freq_modulated * harmonic) * t)
    
    # Add amplitude modulation (like natural speech rhythm)
    am_rate = 3  # Hz
    amplitude_modulation = 0.7 + 0.3 * np.sin(2 * np.pi * am_rate * t)
    signal *= amplitude_modulation
    
    # Add some noise (like breath and environment)
    noise = np.random.normal(0, 0.02, len(signal))
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    signal = (signal * 32767).astype(np.int16)
    
    # Save as WAV first
    wav_path = output_path.replace('.mp3', '.wav')
    wavfile.write(wav_path, sample_rate, signal)
    
    # Convert to MP3
    audio = AudioSegment.from_wav(wav_path)
    audio.export(output_path, format='mp3')
    
    # Clean up WAV file
    os.remove(wav_path)
    
    print(f"Created test audio file: {output_path}")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    
    return output_path


def audio_to_base64(file_path: str) -> str:
    """Convert audio file to base64 string"""
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode('utf-8')


def create_sample_request(audio_path: str = None, language: str = "English"):
    """
    Create a sample API request with a test audio file.
    
    Args:
        audio_path: Path to existing audio file, or None to create test audio
        language: Language to use in the request
    """
    import json
    
    if audio_path is None:
        audio_path = create_test_audio()
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    # Get base64 encoded audio
    audio_base64 = audio_to_base64(audio_path)
    
    # Create request payload
    request = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64[:100] + "..."  # Truncated for display
    }
    
    full_request = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    print("\n=== Sample API Request ===")
    print(f"POST http://localhost:8000/api/voice-detection")
    print(f"Headers: {{ 'Content-Type': 'application/json', 'x-api-key': 'sk_test_123456789' }}")
    print(f"\nRequest Body (truncated):")
    print(json.dumps(request, indent=2))
    print(f"\nFull base64 length: {len(audio_base64)} characters")
    
    # Save full request to file for easy testing
    request_file = "sample_request.json"
    with open(request_file, 'w') as f:
        json.dump(full_request, f)
    
    print(f"\nFull request saved to: {request_file}")
    print(f"\nTo test with curl:")
    print(f"  curl -X POST http://localhost:8000/api/voice-detection \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -H 'x-api-key: sk_test_123456789' \\")
    print(f"    -d @{request_file}")
    
    return audio_base64


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use provided audio file
        audio_path = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else "English"
        create_sample_request(audio_path, language)
    else:
        # Create test audio and sample request
        print("Creating test audio file...")
        create_sample_request(None, "English")
