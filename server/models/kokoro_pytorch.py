# ============================================================================
# VoiceLink — PyTorch Kokoro Model Backend with Jessica Voice
# ============================================================================
#
# This backend uses the PyTorch version of Kokoro which supports 67+ voices
# including the af_jessica voice. It uses the kokoro PyPI package.
#
# Key advantages over ONNX version:
# - Access to all 67+ voices (including af_jessica)
# - Latest voice models from hexgrad/Kokoro-82M
# - Better voice quality and variety
# ============================================================================

import io
import wave
from pathlib import Path
import numpy as np
from typing import Generator, Optional
from loguru import logger

try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    logger.warning("Kokoro PyTorch package not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from server.models.base import TTSModel, VoiceInfo


class KokoroPyTorchModel(TTSModel):
    """
    PyTorch-based Kokoro TTS model backend with full voice library.
    
    Supports 67+ voices including af_jessica, af_sky, and many others.
    Uses the kokoro PyPI package which downloads voices from HuggingFace.
    """
    
    def __init__(self, **kwargs):
        """Initialize the PyTorch Kokoro model."""
        if not KOKORO_AVAILABLE:
            raise ImportError("Kokoro PyTorch package not installed. Install with: pip install kokoro")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch torchaudio")
        
        self.pipeline: Optional[KPipeline] = None
        self._is_loaded = False
        self.sample_rate = 24000
        self.lang_code = 'a'  # American English
        
        # Voice metadata mapping
        # Directory for user-supplied custom voice .pt files
        self.custom_voices_dir = Path("models/custom_voices")

        # Optional voices bundle (.bin NPZ) — overrides HF downloads for all voices it contains.
        # Server prefers voices-v1.0-with-ava.bin if present, then voices-v1.0.bin.
        self.voices_bundle: Path | None = self._find_voices_bundle()

        self.voice_info_map = {
            # American English Female
            'af_heart': VoiceInfo('af_heart', 'Heart', 'en-US', 'female', 
                                'Warm, expressive female voice. Great for audiobooks.', 
                                'kokoro_pytorch', ['warm', 'expressive', 'default'], self.sample_rate),
            'af_bella': VoiceInfo('af_bella', 'Bella', 'en-US', 'female',
                                'Clear, professional female voice.',
                                'kokoro_pytorch', ['clear', 'professional'], self.sample_rate),
            'af_nicole': VoiceInfo('af_nicole', 'Nicole', 'en-US', 'female',
                                  'Smooth, calm female voice.',
                                  'kokoro_pytorch', ['smooth', 'calm'], self.sample_rate),
            'af_sarah': VoiceInfo('af_sarah', 'Sarah', 'en-US', 'female',
                                'Friendly, conversational female voice.',
                                'kokoro_pytorch', ['friendly', 'conversational'], self.sample_rate),
            'af_sky': VoiceInfo('af_sky', 'Sky', 'en-US', 'female',
                              'Light, youthful female voice.',
                              'kokoro_pytorch', ['light', 'youthful'], self.sample_rate),
            'af_jessica': VoiceInfo('af_jessica', 'Jessica', 'en-US', 'female',
                                   'Natural, pleasant female voice with clear articulation.',
                                   'kokoro_pytorch', ['natural', 'pleasant', 'clear'], self.sample_rate),
            
            # American English Male
            'am_adam': VoiceInfo('am_adam', 'Adam', 'en-US', 'male',
                               'Natural, conversational male voice.',
                               'kokoro_pytorch', ['natural', 'conversational'], self.sample_rate),
            'am_michael': VoiceInfo('am_michael', 'Michael', 'en-US', 'male',
                                  'Deep, authoritative male voice.',
                                  'kokoro_pytorch', ['deep', 'authoritative'], self.sample_rate),
            
            # British English Female
            'bf_emma': VoiceInfo('bf_emma', 'Emma', 'en-GB', 'female',
                               'Classic British English female voice.',
                               'kokoro_pytorch', ['british', 'classic'], self.sample_rate),
            'bf_isabella': VoiceInfo('bf_isabella', 'Isabella', 'en-GB', 'female',
                                    'Elegant British female voice.',
                                    'kokoro_pytorch', ['british', 'elegant'], self.sample_rate),
            
            # British English Male
            'bm_george': VoiceInfo('bm_george', 'George', 'en-GB', 'male',
                                 'Traditional British English male voice.',
                                 'kokoro_pytorch', ['british', 'traditional'], self.sample_rate),
            'bm_lewis': VoiceInfo('bm_lewis', 'Lewis', 'en-GB', 'male',
                                'Warm British male voice.',
                                'kokoro_pytorch', ['british', 'warm'], self.sample_rate),
        }
        
        logger.info("Kokoro PyTorch model initialized")

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "Kokoro PyTorch v0.9.4 (67+ voices)"

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready."""
        return self._is_loaded

    def load(self) -> None:
        """Load the Kokoro PyTorch pipeline."""
        try:
            logger.info("Loading Kokoro PyTorch pipeline...")
            
            # Initialize pipeline with American English
            self.pipeline = KPipeline(
                lang_code=self.lang_code,
                repo_id='hexgrad/Kokoro-82M',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Pre-load a popular voice to initialize the model
            self.pipeline.load_voice('af_heart')
            
            # Test that the pipeline works
            test_text = "Hello, this is a test."
            generator = self.pipeline(test_text, voice='af_heart', speed=1.0)
            
            # Get first chunk to verify it works
            first_chunk = next(generator)
            if first_chunk is not None:
                logger.success("Kokoro PyTorch pipeline loaded successfully")
                self._is_loaded = True
            else:
                raise RuntimeError("Pipeline test failed")

            # Load all voices from a .bin bundle (e.g. voices-v1.0-with-ava.bin)
            self._load_voices_bundle()

            # Load any custom .pt voice files from models/custom_voices/
            self._load_custom_voices()
                
        except Exception as e:
            logger.error(f"Failed to load Kokoro PyTorch model: {e}")
            self._is_loaded = False
            raise

    @staticmethod
    def _find_voices_bundle() -> "Path | None":
        """Return the best available voices .bin bundle, or None."""
        candidates = [
            Path("models/voices-v1.0-with-ava.bin"),
            Path("models/voices-v1.0.bin"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_voices_bundle(self) -> None:
        """
        Load all voices from a kokoro-onnx-format NPZ bundle (.bin file) into
        the pipeline's voice cache.  Each key in the NPZ becomes a voice ID.

        A bundle like voices-v1.0-with-ava.bin contains all 54 built-in voices
        plus any custom voices (e.g. 'ava') injected by the notebook.
        """
        if self.voices_bundle is None:
            return

        import numpy as np
        import torch

        logger.info(f"Loading voices bundle: {self.voices_bundle}")
        bundle = np.load(self.voices_bundle, allow_pickle=False)

        loaded = 0
        for voice_id, arr in bundle.items():
            tensor = torch.from_numpy(arr.copy())
            self.pipeline.voices[voice_id] = tensor

            if voice_id not in self.voice_info_map:
                name = voice_id.split("_", 1)[-1].title()
                gender = "male" if voice_id.startswith(("am_", "bm_")) else "female"
                language = "en-GB" if voice_id.startswith(("bf_", "bm_")) else "en-US"
                self.voice_info_map[voice_id] = VoiceInfo(
                    id=voice_id,
                    name=name,
                    language=language,
                    gender=gender,
                    description=f"Kokoro {name} voice",
                    model="kokoro_pytorch",
                    tags=[],
                    sample_rate=self.sample_rate,
                )
            loaded += 1

        logger.success(f"Loaded {loaded} voices from bundle ({self.voices_bundle.name})")

    def _load_custom_voices(self) -> None:
        """
        Scan models/custom_voices/ for .pt files and inject each one into the
        pipeline's voice cache so they can be referenced by filename stem.

        A file named af_ava.pt becomes the voice ID "af_ava".
        """
        if not self.custom_voices_dir.exists():
            return

        import torch

        for pt_file in sorted(self.custom_voices_dir.glob("*.pt")):
            voice_id = pt_file.stem  # e.g. "af_ava"
            try:
                tensor = torch.load(pt_file, weights_only=True)
                self.pipeline.voices[voice_id] = tensor

                # Register metadata if not already present
                if voice_id not in self.voice_info_map:
                    name = voice_id.split("_", 1)[-1].title()  # "af_ava" → "Ava"
                    gender = "male" if voice_id.startswith(("am_", "bm_")) else "female"
                    language = "en-GB" if voice_id.startswith(("bf_", "bm_")) else "en-US"
                    self.voice_info_map[voice_id] = VoiceInfo(
                        id=voice_id,
                        name=name,
                        language=language,
                        gender=gender,
                        description=f"Custom voice: {name}",
                        model="kokoro_pytorch",
                        tags=["custom"],
                        sample_rate=self.sample_rate,
                    )

                logger.success(f"Loaded custom voice '{voice_id}' from {pt_file}")
            except Exception as e:
                logger.warning(f"Could not load custom voice '{pt_file.name}': {e}")

    def unload(self) -> None:
        """Release the model from memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        self._is_loaded = False
        logger.info("Kokoro PyTorch model unloaded")

    def list_voices(self) -> list[VoiceInfo]:
        """Get all available voices for the PyTorch Kokoro model."""
        if not self._is_loaded:
            logger.warning("Model not loaded, returning cached voice list")
        
        voices = list(self.voice_info_map.values())
        
        # Try to discover additional voices that might be available
        try:
            if self.pipeline and hasattr(self.pipeline, 'voices'):
                # Check for voices we don't have metadata for
                known_voices = set(self.voice_info_map.keys())
                available_voices = set(self.pipeline.voices.keys())
                unknown_voices = available_voices - known_voices
                
                for voice_id in unknown_voices:
                    # Create basic voice info for unknown voices
                    if voice_id.startswith('af_'):
                        gender = 'female'
                        name = voice_id[3:].title()
                    elif voice_id.startswith('am_'):
                        gender = 'male'
                        name = voice_id[3:].title()
                    elif voice_id.startswith('bf_'):
                        gender = 'female'
                        name = voice_id[3:].title()
                        lang = 'en-GB'
                    elif voice_id.startswith('bm_'):
                        gender = 'male'
                        name = voice_id[3:].title()
                        lang = 'en-GB'
                    else:
                        gender = 'female'
                        name = voice_id.title()
                        lang = 'en-US'
                    
                    voices.append(VoiceInfo(
                        id=voice_id,
                        name=name,
                        language=lang if 'lang' in locals() else 'en-US',
                        gender=gender,
                        description=f"Kokoro {name} voice",
                        model='kokoro_pytorch',
                        sample_rate=self.sample_rate
                    ))
                    
        except Exception as e:
            logger.warning(f"Could not discover additional voices: {e}")
        
        return voices

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> Generator[bytes, None, None]:
        """
        Synthesize speech using PyTorch Kokoro.
        
        Args:
            text: Text to synthesize
            voice: Voice ID (e.g. "af_jessica")
            speed: Speech speed (0.5 to 2.0)
            
        Yields:
            Raw PCM audio bytes (24kHz, 16-bit, mono)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        # Use default voice if none specified
        if voice is None:
            voice = 'af_heart'
        
        # Validate voice exists
        if voice not in self.voice_info_map:
            logger.warning(f"Voice '{voice}' not in known list, trying anyway...")
        
        try:
            # Load the voice if not already loaded
            self.pipeline.load_voice(voice)
            
            # Generate audio using Kokoro
            logger.info(f"Synthesizing with voice '{voice}' at speed {speed}")
            
            # Kokoro returns a generator of KPipeline.Result objects
            audio_generator = self.pipeline(text, voice=voice, speed=speed)
            
            # Process each result
            for result in audio_generator:
                if result is not None:
                    # Extract audio tensor from result.output.audio
                    if hasattr(result, 'output') and hasattr(result.output, 'audio'):
                        audio_tensor = result.output.audio
                        
                        # Convert to numpy array
                        if hasattr(audio_tensor, 'cpu'):
                            audio_np = audio_tensor.cpu().numpy()
                        else:
                            audio_np = audio_tensor
                        
                        # Convert to int16 PCM
                        audio_int16 = (audio_np * 32767).astype(np.int16)
                        
                        # Yield in chunks for streaming compatibility
                        chunk_size = 1024 * 4  # 4KB chunks
                        for i in range(0, len(audio_int16), chunk_size):
                            chunk = audio_int16[i:i+chunk_size]
                            yield chunk.tobytes()
                    else:
                        logger.warning(f"Unexpected result format: {type(result)}")
            
        except Exception as e:
            logger.error(f"Error synthesizing with Kokoro PyTorch: {e}")
            raise

    def _audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio numpy array to WAV bytes."""
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
