# ============================================================================
# VoiceLink — Official Kokoro ONNX TTS Model Backend
# ============================================================================
#
# HOW THIS WORKS:
# 1. We use the official kokoro-onnx package which handles all optimization
# 2. This package is designed for maximum performance and compatibility
# 3. It handles tokenization, voice embeddings, and ONNX inference automatically
# 4. We wrap it in the TTSModel interface for VoiceLink compatibility
#
# PERFORMANCE ADVANTAGES:
# - Uses optimized ONNX Runtime configuration
# - Proper tokenization with phonemizer-fork
# - Efficient voice embedding handling
# - Designed for production use
#
# WHY THIS APPROACH:
# The official kokoro-onnx package is battle-tested and optimized
# It handles all the complexity we were struggling with manually
# ============================================================================

import numpy as np
from typing import Generator, Optional
from pathlib import Path
from loguru import logger

from server.models.base import TTSModel, VoiceInfo
from server.models.kokoro_model import KOKORO_VOICES

# Import the official kokoro-onnx package
try:
    from kokoro_onnx import Kokoro
    KOKORO_ONNX_AVAILABLE = True
    logger.info("Using official kokoro-onnx package")
except ImportError as e:
    logger.error(f"Official kokoro-onnx package not available: {e}")
    KOKORO_ONNX_AVAILABLE = False


class KokoroONNXOfficialModel(TTSModel):
    """
    Production-ready ONNX-optimized Kokoro TTS model backend.
    
    Uses the official kokoro-onnx package for maximum performance
    and compatibility.
    """

    def __init__(self, lang_code: str = "a", device: str = "auto", **kwargs):
        """
        Initialize the ONNX model.
        
        Args:
            lang_code: Language code ('a' = American English, 'b' = British English)
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.lang_code = lang_code
        self.device = device
        self.model_path: Optional[Path] = None
        self.voices_path: Optional[Path] = None
        self.kokoro: Optional[Kokoro] = None
        self._is_loaded = False
        self.sample_rate = 24000
        
        logger.info(f"KokoroONNX Official initialized: lang={lang_code}, device={device}")

    def load(self) -> None:
        """
        Load the ONNX model using the official kokoro-onnx package.
        """
        if not KOKORO_ONNX_AVAILABLE:
            raise RuntimeError("kokoro-onnx package not available. Install with: pip install kokoro-onnx")
        
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Model file paths - try int8 quantized version first for better CPU performance
            self.model_path = models_dir / "kokoro-v1.0.int8.onnx"
            self.voices_path = models_dir / "voices-v1.0.bin"
            
            if not self.model_path.exists():
                # Fallback to full model if int8 not available
                self.model_path = models_dir / "kokoro-v1.0.onnx"
            
            if not self.model_path.exists() or not self.voices_path.exists():
                raise FileNotFoundError(
                    f"Model files not found. Need: {self.model_path} and {self.voices_path}\n"
                    f"Download them from: https://github.com/thewh1teagle/kokoro-onnx/releases"
                )
            
            model_type = "INT8 Quantized" if "int8" in self.model_path.name else "Full"
            logger.info(f"Loading kokoro-onnx {model_type} model from {self.model_path}")
            
            # Initialize the official Kokoro model
            self.kokoro = Kokoro(str(self.model_path), str(self.voices_path))
            
            logger.info("Official kokoro-onnx model loaded successfully")
            self._is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load kokoro-onnx model: {e}")
            self._is_loaded = False
            raise

    def unload(self) -> None:
        """Release the ONNX model from memory."""
        if self.kokoro is not None:
            del self.kokoro
            self.kokoro = None
        
        self._is_loaded = False
        logger.info("kokoro-onnx model unloaded")

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> Generator[bytes, None, None]:
        """
        Convert text to speech using the official kokoro-onnx package.
        
        Args:
            text: Text to synthesize
            voice: Voice ID (e.g., 'af_heart', 'am_adam')
            speed: Speaking speed multiplier
            
        Yields:
            PCM audio chunks
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self.kokoro is None:
            raise RuntimeError("kokoro-onnx model not initialized.")
        
        try:
            # Map VoiceLink voice names to kokoro-onnx voice names
            kokoro_voice = self._map_voice_name(voice)
            
            # Determine language from lang_code
            lang = "en-us" if self.lang_code == "a" else "en-gb"
            
            # Clamp speed to valid range (0.5 to 2.0)
            valid_speed = max(0.5, min(2.0, speed))
            if speed != valid_speed:
                logger.warning(f"Speed {speed} clamped to {valid_speed} (valid range: 0.5-2.0)")
            
            logger.debug(f"Synthesizing with kokoro-onnx: voice={kokoro_voice}, lang={lang}, speed={valid_speed}")
            
            # Use the official kokoro-onnx package for synthesis
            samples, sample_rate = self.kokoro.create(
                text=text,
                voice=kokoro_voice,
                speed=valid_speed,
                lang=lang
            )
            
            # Convert to PCM bytes
            pcm_data = self._samples_to_pcm(samples, sample_rate)
            
            # Yield in chunks for streaming
            chunk_size = 1024 * 8  # 8KB chunks
            for i in range(0, len(pcm_data), chunk_size):
                yield pcm_data[i:i + chunk_size]
                
        except Exception as e:
            logger.error(f"kokoro-onnx synthesis failed: {e}")
            raise

    def _map_voice_name(self, voice: str | None) -> str:
        """
        Map VoiceLink voice names to kokoro-onnx voice names.
        
        Args:
            voice: VoiceLink voice ID
            
        Returns:
            kokoro-onnx compatible voice name
        """
        if voice is None:
            return "af_heart"
        
        # Direct mapping for most voices
        voice_mapping = {
            "af_heart": "af_heart",
            "af_bella": "af_bella", 
            "af_nicole": "af_nicole",
            "af_sarah": "af_sarah",
            "af_sky": "af_sky",
            "am_adam": "am_adam",
            "am_michael": "am_michael",
            "bf_emma": "bf_emma",
            "bf_isabella": "bf_isabella", 
            "bm_george": "bm_george",
            "bm_lewis": "bm_lewis"
        }
        
        return voice_mapping.get(voice, "af_heart")

    def _samples_to_pcm(self, samples: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert audio samples to PCM bytes.
        
        Args:
            samples: Audio samples from kokoro-onnx
            sample_rate: Sample rate of the audio
            
        Returns:
            PCM byte data (16-bit signed, little-endian)
        """
        # Ensure samples are in the right format
        if samples.dtype != np.int16:
            # Convert float32 [-1, 1] to int16 [-32768, 32767]
            if samples.dtype in [np.float32, np.float64]:
                samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            else:
                samples = samples.astype(np.int16)
        
        # Flatten and convert to bytes
        return samples.flatten().tobytes()

    def list_voices(self) -> list[VoiceInfo]:
        """Return available voices with updated info for ONNX model."""
        # Return the standard Kokoro voices but note they're ONNX-optimized
        onnx_voices = []
        for voice in KOKORO_VOICES:
            onnx_voice = VoiceInfo(
                id=voice.id,
                name=voice.name,
                language=voice.language,
                gender=voice.gender,
                description=f"{voice.description} (Official ONNX)",
                model="kokoro_onnx",
                tags=voice.tags + ["onnx", "official", "optimized"],
                sample_rate=self.sample_rate
            )
            onnx_voices.append(onnx_voice)
        
        return onnx_voices

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "Kokoro ONNX Official v1.0"

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready."""
        return self._is_loaded
