"""
Audio Processing Service
Uses librosa, pydub, torchaudio for audio processing
"""
import base64
import io
import tempfile
import os
import numpy as np
from typing import Dict, Any, Tuple


class AudioProcessor:
    """Process audio files for voice detection"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._librosa = None
        self._torch = None
        self._torchaudio = None
        self._noisereduce = None
    
    @property
    def librosa(self):
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    @property
    def torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch
    
    @property
    def torchaudio(self):
        if self._torchaudio is None:
            import torchaudio
            self._torchaudio = torchaudio
        return self._torchaudio
    
    @property
    def noisereduce(self):
        if self._noisereduce is None:
            import noisereduce as nr
            self._noisereduce = nr
        return self._noisereduce
    
    def decode_base64(self, audio_base64: str) -> bytes:
        """Decode base64 audio"""
        try:
            return base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64: {e}")
    
    def load_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load audio from bytes"""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            # Load with librosa
            audio, sr = self.librosa.load(temp_path, sr=self.sample_rate, mono=True)
            return audio, sr
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction"""
        try:
            return self.noisereduce.reduce_noise(y=audio, sr=sr)
        except:
            return audio
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive audio features"""
        features = {}
        librosa = self.librosa
        
        # 1. MFCCs - Most important for voice
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
        features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
        features['mfcc_var'] = np.var(mfccs, axis=1).tolist()
        
        # 2. Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spec_cent))
        features['spectral_centroid_std'] = float(np.std(spec_cent))
        
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
        features['spectral_bandwidth_std'] = float(np.std(spec_bw))
        
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spec_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spec_rolloff))
        
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spec_contrast, axis=1).tolist()
        
        # 3. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # 4. RMS Energy
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # 5. Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[pitches > 0]
        if len(pitch_values) > 0:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0
        
        # 6. Mel spectrogram stats
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spec_mean'] = float(np.mean(mel_spec_db))
        features['mel_spec_std'] = float(np.std(mel_spec_db))
        
        # 7. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
        
        # 8. Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        
        # 9. Duration
        features['duration'] = float(len(audio) / sr)
        
        return features
    
    def get_mel_spectrogram_tensor(self, audio: np.ndarray, sr: int):
        """Get mel spectrogram as PyTorch tensor for model input"""
        torch = self.torch
        torchaudio = self.torchaudio
        
        # Convert to tensor
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        
        # Create mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )
        
        mel_spec = mel_transform(waveform)
        
        # Convert to dB
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        mel_spec_db = amplitude_to_db(mel_spec)
        
        return mel_spec_db
    
    def process(self, audio_base64: str) -> Dict[str, Any]:
        """Full processing pipeline"""
        # Decode
        audio_bytes = self.decode_base64(audio_base64)
        
        # Load
        audio, sr = self.load_audio(audio_bytes)
        
        # Noise reduction
        audio = self.reduce_noise(audio, sr)
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Add raw audio for model
        features['_audio'] = audio
        features['_sr'] = sr
        
        return features


# Global instance
audio_processor = AudioProcessor()
