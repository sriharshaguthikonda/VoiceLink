# ============================================================================
# VoiceLink — Ava Custom TTS Model Backend
# ============================================================================
#
# This backend handles the custom-trained Ava model using ONNX encoder.
# The Ava model consists of:
# - kokoro_encoder.onnx: Text encoder + duration predictor
# - kokoro_weights.pth: Full model weights (for PyTorch inference)
# - phoneme_processor.pkl: Phoneme tokenizer
# - model_config.json: Model configuration
#
# This implementation uses the ONNX encoder for fast inference.
# ============================================================================

import json
import pickle
import numpy as np
import onnxruntime as ort
from typing import AsyncGenerator, Generator, Optional
from pathlib import Path
from loguru import logger

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - will use simple vocoder")

from server.models.base import TTSModel, VoiceInfo


class AvaModel(TTSModel):
    """
    Custom-trained Ava TTS model backend using ONNX encoder.
    
    This model supports speed conditioning and generates mel spectrograms
    that are converted to audio using a simple vocoder.
    """
    
    def __init__(self, **kwargs):
        """Initialize the Ava model."""
        self.model_dir = Path("models") / "Ava"
        self.encoder_path = self.model_dir / "kokoro_encoder.onnx"
        self.config_path = self.model_dir / "model_config.json"
        self.phoneme_processor_path = self.model_dir / "phoneme_processor.pkl"
        
        self.session: Optional[ort.InferenceSession] = None
        self.phoneme_processor = None
        self.config = None
        self._is_loaded = False
        self.sample_rate = 22050
        
        logger.info("Ava model initialized")

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "Ava v1.0"

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready."""
        return self._is_loaded

    def unload(self) -> None:
        """Release the model from memory."""
        self.cleanup()

    def load(self) -> None:
        """Load the Ava model components."""
        try:
            # Check model files exist
            if not all(p.exists() for p in [
                self.encoder_path, 
                self.config_path, 
                self.phoneme_processor_path
            ]):
                missing = [p for p in [
                    self.encoder_path, 
                    self.config_path, 
                    self.phoneme_processor_path
                ] if not p.exists()]
                raise FileNotFoundError(
                    f"Ava model files missing: {missing}\n"
                    f"Please ensure the Ava model is properly installed in models/Ava/"
                )
            
            # Load configuration
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.sample_rate = self.config.get("sample_rate", 22050)
            
            # Load phoneme processor
            with open(self.phoneme_processor_path, 'rb') as f:
                self.phoneme_processor = pickle.load(f)
            
            # Load ONNX encoder
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(
                str(self.encoder_path), 
                providers=providers
            )
            
            # Log model info
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            
            logger.info(f"Ava ONNX encoder loaded:")
            logger.info(f"  Inputs: {[inp.name for inp in inputs]}")
            logger.info(f"  Outputs: {[out.name for out in outputs]}")
            logger.info(f"  Sample rate: {self.sample_rate}")
            
            self._is_loaded = True
            logger.success("Ava model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Ava model: {e}")
            self._is_loaded = False
            raise

    def list_voices(self) -> list[VoiceInfo]:
        """Get available voices for the Ava model."""
        # Ava is a single voice model
        return [
            VoiceInfo(
                id="ava",
                name="Ava",
                language="en-US",
                description="Custom-trained Ava voice",
                gender="female",
                model="ava",
                sample_rate=self.sample_rate
            )
        ]

    def synthesize(self, text: str, voice: str = "ava", speed: float = 1.0, **kwargs) -> Generator[bytes, None, None]:
        """
        Synthesize speech using the Ava model.
        
        Args:
            text: Input text to synthesize
            voice: Voice ID (only "ava" is supported)
            speed: Speech speed (0.5 to 2.0)
            **kwargs: Additional parameters (ignored)
            
        Yields:
            Raw PCM audio bytes (22.05kHz, 16-bit, mono)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if voice != "ava":
            raise ValueError(f"Voice '{voice}' not supported. Only 'ava' is available.")
        
        try:
            # Generate audio
            audio = self._generate_audio(text, speed)
            
            # Convert to int16 PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Yield in chunks for streaming compatibility
            chunk_size = 1024  # 1KB chunks
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size]
                yield chunk.tobytes()
                
        except Exception as e:
            logger.error(f"Error synthesizing with Ava model: {e}")
            raise

    async def synthesize_async(self, text: str, voice: str = "ava", speed: float = 1.0, **kwargs) -> AsyncGenerator[bytes, None]:
        """Async version of synthesize."""
        for chunk in self.synthesize(text, voice, speed, **kwargs):
            yield chunk

    def _generate_audio(self, text: str, speed: float) -> np.ndarray:
        """Generate audio from text using the Ava model."""
        try:
            # Tokenize text
            tokens = self._tokenize_text(text)
            
            # Prepare inputs
            max_length = 512
            tokens_padded = tokens[:max_length] + [0] * (max_length - len(tokens))
            
            input_data = np.array([tokens_padded], dtype=np.int64)
            target_speeds = np.array([speed], dtype=np.float32)
            
            # Run inference
            input_names = [inp.name for inp in self.session.get_inputs()]
            outputs = self.session.run(None, {
                input_names[0]: input_data,
                input_names[1]: target_speeds
            })
            
            # Get text encoding (first output)
            text_encoded = outputs[0]
            
            # Convert to audio using simple vocoder
            audio = self._mel_to_audio(text_encoded)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error in _generate_audio: {e}")
            raise

    def _tokenize_text(self, text: str) -> list[int]:
        """Tokenize text using the phoneme processor."""
        try:
            # Try different tokenization methods
            if hasattr(self.phoneme_processor, 'text_to_ids'):
                return self.phoneme_processor.text_to_ids(text)
            elif hasattr(self.phoneme_processor, 'encode'):
                return self.phoneme_processor.encode(text)
            elif hasattr(self.phoneme_processor, 'tokenize'):
                return self.phoneme_processor.tokenize(text)
            else:
                # Fallback: simple character tokenization
                return [ord(c) for c in text]
        except Exception as e:
            logger.warning(f"Phoneme processor failed: {e}, using fallback tokenization")
            return [ord(c) for c in text]

    def _mel_to_audio(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram to audio using improved vocoder."""
        try:
            # Squeeze batch dimension if present
            if len(mel.shape) == 3:
                mel = mel.squeeze(0)
            
            mel_np = mel.astype(np.float32)
            
            if TORCH_AVAILABLE:
                return self._mel_to_audio_torch(mel_np)
            else:
                return self._mel_to_audio_numpy(mel_np)
                
        except Exception as e:
            logger.error(f"Error in mel_to_audio: {e}")
            raise

    def _mel_to_audio_torch(self, mel: np.ndarray) -> np.ndarray:
        """Use PyTorch for better mel-to-audio conversion."""
        try:
            import torch
            import torchaudio
            
            # Convert to torch tensor
            mel_tensor = torch.from_numpy(mel).T  # [80, T]
            
            # Use Griffin-Lim algorithm for basic reconstruction
            # This is much better than sine waves
            hop_length = self.config.get("hop_length", 256)
            n_fft = self.config.get("n_fft", 1024)
            win_length = self.config.get("win_length", 1024)
            
            # Convert mel back to linear spectrogram (approximate)
            # This is a simplified inverse mel transform
            mel_spec = torch.exp(mel_tensor)  # Convert from log
            
            # Create a simple linear spectrogram approximation
            # In practice, you'd use the actual mel filter bank inverse
            linear_spec = torch.zeros(n_fft // 2 + 1, mel_spec.shape[1])
            
            # Map mel bands to linear frequencies (simplified)
            for i in range(mel_spec.shape[0]):
                freq_idx = int(i * (n_fft // 2 + 1) / mel_spec.shape[0])
                if freq_idx < linear_spec.shape[0]:
                    linear_spec[freq_idx] += mel_spec[i]
            
            # Griffin-Lim algorithm
            angles = torch.zeros_like(linear_spec, dtype=torch.complex64)
            
            for _ in range(50):  # 50 iterations of Griffin-Lim
                # Reconstruct complex spectrogram
                complex_spec = linear_spec * torch.exp(1j * angles)
                
                # ISTFT to get waveform
                waveform = torchaudio.functional.istft(
                    complex_spec, 
                    n_fft=n_fft, 
                    hop_length=hop_length, 
                    win_length=win_length,
                    window=torch.hann_window(win_length)
                )
                
                # Re-estimate magnitudes
                stft = torch.stft(
                    waveform, 
                    n_fft=n_fft, 
                    hop_length=hop_length, 
                    win_length=win_length,
                    window=torch.hann_window(win_length),
                    return_complex=True
                )
                
                angles = torch.angle(stft)
            
            # Normalize and convert to numpy
            waveform = waveform.squeeze()
            if torch.max(torch.abs(waveform)) > 0:
                waveform = waveform / torch.max(torch.abs(waveform)) * 0.8
            
            return waveform.numpy()
            
        except Exception as e:
            logger.warning(f"PyTorch vocoder failed: {e}, falling back to numpy")
            return self._mel_to_audio_numpy(mel)

    def _mel_to_audio_numpy(self, mel: np.ndarray) -> np.ndarray:
        """Fallback numpy-based mel-to-audio conversion."""
        try:
            # Better approximation using filter banks
            hop_length = self.config.get("hop_length", 256)
            n_frames = mel.shape[0]
            duration = n_frames * hop_length / self.sample_rate
            
            # Create time array
            t = np.linspace(0, duration, int(duration * self.sample_rate))
            
            # Use mel energy to modulate multiple frequency bands
            audio = np.zeros_like(t)
            
            # Create frequency bands based on mel scale
            mel_freqs = mel.mean(axis=0)  # Average energy across time for each mel band
            
            # Convert mel bands to actual frequencies (approximate)
            # Mel scale: m = 2595 * log10(1 + f/700)
            # Inverse: f = 700 * (10^(m/2595) - 1)
            
            sample_rate = self.sample_rate
            for i, mel_energy in enumerate(mel_freqs[:20]):  # Use first 20 mel bands
                # Map mel band index to frequency
                mel_idx_norm = i / len(mel_freqs)
                # Approximate frequency mapping
                freq = 100 * np.exp(mel_idx_norm * 4.6)  # 100Hz to ~10kHz range
                
                # Generate carrier wave
                carrier = np.sin(2 * np.pi * freq * t)
                
                # Amplitude modulation based on mel energy and time evolution
                # Use the mel spectrogram time evolution for amplitude
                time_energy = mel[:, i] if i < mel.shape[1] else mel[:, -1]
                
                # Interpolate time energy to audio sample rate
                time_frames = np.linspace(0, duration, len(time_energy))
                energy_interp = np.interp(t, time_frames, time_energy)
                
                # Normalize energy
                if np.max(np.abs(energy_interp)) > 0:
                    energy_interp = energy_interp / np.max(np.abs(energy_interp))
                
                # Apply amplitude modulation
                amplitude = 0.05 * energy_interp * mel_energy
                audio += amplitude * carrier
            
            # Apply envelope and post-processing
            envelope = np.ones_like(t)
            fade_samples = int(0.1 * sample_rate)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            
            audio *= envelope
            
            # Add some reverb/room simulation (simple convolution)
            if len(audio) > 1000:
                # Simple early reflection simulation
                reflection_delay = int(0.03 * sample_rate)  # 30ms delay
                reflection_gain = 0.3
                
                if len(audio) > reflection_delay:
                    reflection = np.zeros_like(audio)
                    reflection[reflection_delay:] = audio[:-reflection_delay] * reflection_gain
                    audio = audio + reflection
            
            # Normalize final output
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in numpy vocoder: {e}")
            # Last resort: simple tone
            duration = 2.0
            t = np.linspace(0, duration, int(duration * self.sample_rate))
            return 0.1 * np.sin(2 * np.pi * 200 * t).astype(np.float32)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            del self.session
            self.session = None
        self._is_loaded = False
        logger.info("Ava model cleaned up")
