# ============================================================================
# VoiceLink — TTS Model Registry
# ============================================================================
#
# This package contains the abstract model interface and all model backends.
# Models are loaded by name via the registry pattern below.
# ============================================================================

from server.models.base import TTSModel, VoiceInfo
from server.models.kokoro_model import KokoroModel

# --- Model Registry ---
# Maps model name → class. Add new models here.
MODEL_REGISTRY: dict[str, type[TTSModel]] = {
    "kokoro": KokoroModel,
}


def load_model(model_name: str, **kwargs) -> TTSModel:
    """
    Factory function: load a TTS model by name.

    Usage:
        model = load_model("kokoro")
        for chunk in model.synthesize("Hello world", voice="af_heart"):
            # chunk is raw PCM bytes (24kHz, 16-bit, mono)
            ...

    Raises:
        ValueError: if model_name is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    model_class = MODEL_REGISTRY[model_name]
    model = model_class(**kwargs)
    model.load()
    return model


__all__ = ["TTSModel", "VoiceInfo", "load_model", "MODEL_REGISTRY"]
