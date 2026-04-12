"""
Convert a Kokoro style vector file to a .pt file for use with the
kokoro PyTorch pipeline (KPipeline).

Supported input formats:
  .npy  — raw numpy array saved with np.save()
  .bin  — NPZ archive produced by the notebook (np.savez(..., ava=vec))

Usage:
    python tools/convert_voice.py <input> [output.pt]

Examples:
    python tools/convert_voice.py ava_voice.bin models/custom_voices/af_ava.pt
    python tools/convert_voice.py ava_flat_vec_best.npy models/custom_voices/af_ava.pt

The style vector must be float32 of shape (510, 1, 256).
The output .pt file will be auto-discovered by the VoiceLink server on next startup.
"""

import sys
import argparse
from pathlib import Path


def convert(npy_path: Path, pt_path: Path) -> None:
    import numpy as np
    import torch

    raw = np.load(npy_path, allow_pickle=False)

    # NPZ (e.g. ava_voice.bin): pick the first array key
    if hasattr(raw, 'keys'):
        keys = list(raw.keys())
        arr = raw[keys[0]]
        print(f"  NPZ key used: '{keys[0]}'")
    else:
        arr = raw

    if arr.dtype != np.float32:
        print(f"  Converting dtype {arr.dtype} → float32")
        arr = arr.astype(np.float32)

    if arr.shape != (510, 1, 256):
        print(f"  Warning: expected shape (510, 1, 256), got {arr.shape}")
        print("  The voice may not work correctly.")

    tensor = torch.from_numpy(arr)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, pt_path)

    print(f"  Saved: {pt_path}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert kokoro style vector (.npy/.bin) to .pt")
    parser.add_argument("input", help="Path to input .npy or .bin file (e.g. ava_voice.bin)")
    parser.add_argument(
        "output",
        nargs="?",
        help="Path to output .pt file (default: models/custom_voices/<stem>.pt)",
    )
    args = parser.parse_args()

    npy_path = Path(args.input)
    if not npy_path.exists():
        print(f"Error: input file not found: {npy_path}")
        sys.exit(1)
    if npy_path.suffix.lower() != ".npy":
        print(f"Warning: expected a .npy file, got '{npy_path.suffix}'")

    if args.output:
        pt_path = Path(args.output)
    else:
        # Default: models/custom_voices/<original_stem>.pt
        pt_path = Path("models/custom_voices") / (npy_path.stem + ".pt")

    print(f"Converting: {npy_path} → {pt_path}")
    convert(npy_path, pt_path)
    print(f"\nDone. Restart the VoiceLink server and the voice '{pt_path.stem}' will be available.")


if __name__ == "__main__":
    main()
