# Kokoro TTS Export

## Files
- `kokoro_encoder.onnx`   — Text encoder + duration predictor (fast, portable)
- `kokoro_weights.pth`    — Full model weights for Python inference
- `phoneme_processor.pkl` — Phoneme tokenizer
- `model_config.json`     — Model config

## Quick Inference (Python)
```python
import sys, torch, pickle
sys.path.append("kokoro-ava-speed-training")  # needs the training repo

from training.config_english import EnglishTrainingConfig
torch.serialization.add_safe_globals([EnglishTrainingConfig])
from kokoro.model import KokoroModel

with open("phoneme_processor.pkl", "rb") as f:
    pp = pickle.load(f)

ckpt = torch.load("kokoro_weights.pth", map_location="cpu", weights_only=False)
cfg  = ckpt["config"]

model = KokoroModel(
    vocab_size=ckpt["vocab_size"], mel_dim=cfg.n_mels, hidden_dim=cfg.hidden_dim,
    n_encoder_layers=cfg.n_encoder_layers, n_heads=cfg.n_heads,
    encoder_ff_dim=cfg.encoder_ff_dim, encoder_dropout=cfg.encoder_dropout,
    n_decoder_layers=cfg.n_decoder_layers, decoder_ff_dim=cfg.decoder_ff_dim,
    max_decoder_seq_len=cfg.max_decoder_seq_len, gradient_checkpointing=False,
    enable_speed_conditioning=cfg.enable_speed_conditioning,
    default_inference_speed=cfg.default_target_speed,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Tokenize and run
phonemes = pp.text_to_ids("Hello world")   # adjust method name to your processor
indices  = torch.tensor([phonemes])
with torch.no_grad():
    mel = model(indices)   # (1, T, 80)
print("Mel shape:", mel.shape)
```
