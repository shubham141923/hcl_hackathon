"""
Audio Processing Service
Handles audio decoding, feature extraction, and preprocessing for voice analysis
"""
import base64
import io
import tempfile
import os
from typing import Dict, Any, Tuple, Optional
import numpy as np


class AudioProcessor:
    """
    Processes audio files for voice detection analysis.
    Extracts various audio features used for AI vs Human classification.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self._librosa = None
        self._sf = None
    
    def _get_librosa(self):
        """Lazy load librosa to improve startup time"""
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    def _get_soundfile(self):
        """Lazy load soundfile"""
        if self._sf is None:
            import soundfile as sf
            self._sf = sf
        return self._sf
    
    def decode_base64_audio(self, audio_base64: str) -> bytes:
        """
        Decode base64 encoded audio data.
        
        Args:
            audio_base64: Base64 encoded audio string
            
        Returns:
            Raw audio bytes
            
        Raises:
            ValueError: If base64 decoding fails
        """
        try:
            return base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {str(e)}")
    
    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes and return as numpy array.
        
        Args:
            audio_bytes: Raw audio bytes (MP3 format)
            
        Returns:
            Tuple of (audio signal as numpy array, sample rate)
            
        Raises:
            ValueError: If audio loading fails
        """
        librosa = self._get_librosa()
        
        # Create a temporary file to handle MP3 format
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Load audio using librosa
            audio_signal, sr = librosa.load(
                tmp_path,
                sr=self.sample_rate,
                mono=True
            )
            return audio_signal, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def extract_features(self, audio_signal: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract comprehensive audio features for voice analysis.
        
        Features extracted:
        - MFCC (Mel-frequency cepstral coefficients)
        - Spectral features (centroid, bandwidth, contrast, rolloff)
        - Zero crossing rate
        - RMS energy
        - Pitch features
        - Tempo-based features
        
        Args:
            audio_signal: Audio signal as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        librosa = self._get_librosa()
        features = {}
        
        try:
            # 1. MFCC Features (most important for voice analysis)
            mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=20)
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
            features['mfcc_delta_mean'] = np.mean(librosa.feature.delta(mfccs), axis=1).tolist()
            
            # 2. Spectral Centroid (brightness of sound)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_signal, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_std'] = float(np.std(spectral_centroid))
            
            # 3. Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # 4. Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_signal, sr=sr)
            features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1).tolist()
            
            # 5. Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # 6. Zero Crossing Rate (voice texture indicator)
            zcr = librosa.feature.zero_crossing_rate(audio_signal)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # 7. RMS Energy (loudness patterns)
            rms = librosa.feature.rms(y=audio_signal)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            
            # 8. Pitch/F0 Features using piptrack
            pitches, magnitudes = librosa.piptrack(y=audio_signal, sr=sr)
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0
            
            # 9. Tempo and Beat Features
            tempo, _ = librosa.beat.beat_track(y=audio_signal, sr=sr)
            features['tempo'] = float(tempo) if isinstance(tempo, (int, float, np.floating, np.integer)) else float(tempo[0]) if len(tempo) > 0 else 0.0
            
            # 10. Mel Spectrogram Statistics
            mel_spec = librosa.feature.melspectrogram(y=audio_signal, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spec_mean'] = float(np.mean(mel_spec_db))
            features['mel_spec_std'] = float(np.std(mel_spec_db))
            
            # 11. Chroma Features (harmonic content)
            chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
            features['chroma_std'] = np.mean(np.std(chroma, axis=1)).tolist() if hasattr(np.std(chroma, axis=1), 'tolist') else float(np.mean(np.std(chroma, axis=1)))
            
            # 12. Duration
            features['duration'] = float(len(audio_signal) / sr)
            
        except Exception as e:
            raise ValueError(f"Feature extraction failed: {str(e)}")
        
        return features
    
    def process_audio(self, audio_base64: str) -> Dict[str, Any]:
        """
        Complete audio processing pipeline.
        
        Args:
            audio_base64: Base64 encoded MP3 audio
            
        Returns:
            Dictionary containing all extracted features
        """
        # Decode base64
        audio_bytes = self.decode_base64_audio(audio_base64)
        
        # Load audio
        audio_signal, sr = self.load_audio_from_bytes(audio_bytes)
        
        # Extract features
        features = self.extract_features(audio_signal, sr)
        
        return features


# Global processor instance
audio_processor = AudioProcessor()
