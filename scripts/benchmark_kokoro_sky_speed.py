#!/usr/bin/env python3
"""
Benchmark Kokoro ONNX variants for high inference throughput and high speaking rate.

What this measures:
1. Inference speed (wall-clock generation time)
2. Effective speaking-rate acceleration at speed={1.0, 1.5, 2.0}
3. Rough af_sky voice timbre similarity vs baseline

Outputs:
- JSON report with all raw metrics
- WAV files for quick subjective listening
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

try:
    import soundfile as sf

    SOUND_FILE_AVAILABLE = True
except Exception:
    SOUND_FILE_AVAILABLE = False


@dataclass
class BenchmarkResult:
    model_file: str
    provider: str
    speed: float
    success: bool
    error: Optional[str]
    load_seconds: Optional[float]
    gen_seconds: Optional[float]
    audio_seconds: Optional[float]
    realtime_factor: Optional[float]
    effective_rate_vs_speed1: Optional[float]
    similarity_vs_baseline: Optional[float]
    wav_path: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kokoro af_sky speed benchmark")
    parser.add_argument(
        "--models-dir",
        default=r"C:\Windows_software\VoiceLink\models",
        help="Absolute path to model directory.",
    )
    parser.add_argument(
        "--voice",
        default="af_sky",
        help="Voice ID to benchmark.",
    )
    parser.add_argument(
        "--speeds",
        default="1.0,1.5,2.0",
        help="Comma-separated speaking speeds (kokoro range is 0.5-2.0).",
    )
    parser.add_argument(
        "--providers",
        default="cuda,cpu",
        help="Comma-separated providers to test: cuda,cpu",
    )
    parser.add_argument(
        "--text",
        default=(
            "Sky voice high speed stress test. "
            "Please keep clarity, prosody, and natural tone while speaking quickly. "
            "We are benchmarking practical real-time TTS performance."
        ),
        help="Benchmark text.",
    )
    parser.add_argument(
        "--warmup-text",
        default="Warmup run for benchmark.",
        help="Short warmup text.",
    )
    parser.add_argument(
        "--output-json",
        default=r"C:\Windows_software\VoiceLink\research\kokoro_sky_speed_benchmark_results.json",
        help="Absolute path for JSON report.",
    )
    parser.add_argument(
        "--output-audio-dir",
        default=r"C:\Windows_software\VoiceLink\research\kokoro_sky_samples",
        help="Absolute path for generated WAV samples.",
    )
    parser.add_argument(
        "--lang",
        default="en-us",
        help="Language code passed to kokoro_onnx.create().",
    )
    return parser.parse_args()


def model_candidates(models_dir: Path) -> List[Path]:
    preferred = [
        "kokoro-v1.0.onnx",
        "kokoro-v1.0.fp16.onnx",
        "kokoro-v1.0.int8.onnx",
    ]
    candidates = []
    for name in preferred:
        p = models_dir / name
        if p.exists():
            candidates.append(p)
    return candidates


def provider_names(requested: str) -> List[str]:
    available = set(ort.get_available_providers())
    out = []
    for item in [x.strip().lower() for x in requested.split(",") if x.strip()]:
        if item == "cuda" and "CUDAExecutionProvider" in available:
            out.append("CUDAExecutionProvider")
        if item == "cpu" and "CPUExecutionProvider" in available:
            out.append("CPUExecutionProvider")
    return out


def _average_log_spectrum(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft).astype(np.float32)
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    frames = 0
    for i in range(0, len(x) - n_fft + 1, hop):
        frame = x[i : i + n_fft] * win
        spec = np.abs(np.fft.rfft(frame))
        acc += spec
        frames += 1
    if frames == 0:
        return acc
    avg = np.log1p(acc / frames)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


def voice_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = _average_log_spectrum(a)
    vb = _average_log_spectrum(b)
    return float(np.clip(np.dot(va, vb), -1.0, 1.0))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def print_header(msg: str) -> None:
    print("\n" + "=" * 90)
    print(msg)
    print("=" * 90)


def main() -> int:
    args = parse_args()

    models_dir = Path(args.models_dir)
    voices_path = models_dir / "voices-v1.0.bin"
    if not models_dir.exists():
        raise FileNotFoundError(f"models dir not found: {models_dir}")
    if not voices_path.exists():
        raise FileNotFoundError(f"voices file not found: {voices_path}")

    speeds = [float(x.strip()) for x in args.speeds.split(",") if x.strip()]
    if any(s < 0.5 or s > 2.0 for s in speeds):
        raise ValueError("All speeds must be in [0.5, 2.0] for kokoro_onnx")

    providers = provider_names(args.providers)
    if not providers:
        raise RuntimeError("No requested providers are available on this machine.")

    models = model_candidates(models_dir)
    if not models:
        raise RuntimeError("No supported Kokoro ONNX model files found.")

    output_json = Path(args.output_json)
    output_audio_dir = Path(args.output_audio_dir)
    ensure_parent(output_json)
    ensure_dir(output_audio_dir)

    print_header("Kokoro af_sky speed benchmark")
    print(f"Models: {[m.name for m in models]}")
    print(f"Providers: {providers}")
    print(f"Speeds: {speeds}")
    print(f"Voice: {args.voice}")
    print(f"Audio export: {SOUND_FILE_AVAILABLE}")

    total_cases = len(models) * len(providers) * len(speeds)
    case_index = 0
    results: List[BenchmarkResult] = []
    audio_cache: Dict[Tuple[str, str, float], Tuple[np.ndarray, int]] = {}
    speed1_duration: Dict[Tuple[str, str], float] = {}

    old_provider_env = os.environ.get("ONNX_PROVIDER")

    try:
        for model_path in models:
            for provider in providers:
                combo_key = (model_path.name, provider)
                print_header(f"Loading {model_path.name} with {provider}")

                os.environ["ONNX_PROVIDER"] = provider

                kokoro: Optional[Kokoro] = None
                load_seconds: Optional[float] = None
                load_error: Optional[str] = None
                try:
                    t0 = time.perf_counter()
                    kokoro = Kokoro(str(model_path), str(voices_path))
                    load_seconds = time.perf_counter() - t0
                    # warmup
                    kokoro.create(
                        text=args.warmup_text,
                        voice=args.voice,
                        speed=1.0,
                        lang=args.lang,
                    )
                    print(f"Loaded in {load_seconds:.3f}s")
                except Exception as e:
                    load_error = str(e)
                    print(f"Load failed: {load_error}")

                for speed in speeds:
                    case_index += 1
                    print(f"[{case_index}/{total_cases}] model={model_path.name} provider={provider} speed={speed}")

                    if kokoro is None:
                        results.append(
                            BenchmarkResult(
                                model_file=model_path.name,
                                provider=provider,
                                speed=speed,
                                success=False,
                                error=f"load failed: {load_error}",
                                load_seconds=load_seconds,
                                gen_seconds=None,
                                audio_seconds=None,
                                realtime_factor=None,
                                effective_rate_vs_speed1=None,
                                similarity_vs_baseline=None,
                                wav_path=None,
                            )
                        )
                        continue

                    try:
                        t1 = time.perf_counter()
                        samples, sample_rate = kokoro.create(
                            text=args.text,
                            voice=args.voice,
                            speed=speed,
                            lang=args.lang,
                        )
                        gen_seconds = time.perf_counter() - t1
                        audio_seconds = len(samples) / float(sample_rate)
                        rtf = audio_seconds / gen_seconds if gen_seconds > 0 else None

                        key = (model_path.name, provider, speed)
                        audio_cache[key] = (samples, sample_rate)

                        wav_path = None
                        if SOUND_FILE_AVAILABLE:
                            wav_file = output_audio_dir / f"{model_path.stem}_{provider.replace('ExecutionProvider','').lower()}_s{speed:.1f}.wav"
                            sf.write(wav_file, samples, sample_rate)
                            wav_path = str(wav_file)

                        if speed == 1.0:
                            speed1_duration[combo_key] = audio_seconds

                        results.append(
                            BenchmarkResult(
                                model_file=model_path.name,
                                provider=provider,
                                speed=speed,
                                success=True,
                                error=None,
                                load_seconds=load_seconds,
                                gen_seconds=gen_seconds,
                                audio_seconds=audio_seconds,
                                realtime_factor=rtf,
                                effective_rate_vs_speed1=None,
                                similarity_vs_baseline=None,
                                wav_path=wav_path,
                            )
                        )
                        print(
                            f"  OK  gen={gen_seconds:.3f}s audio={audio_seconds:.3f}s "
                            f"rtf={rtf:.2f}x"
                        )
                    except Exception as e:
                        results.append(
                            BenchmarkResult(
                                model_file=model_path.name,
                                provider=provider,
                                speed=speed,
                                success=False,
                                error=str(e),
                                load_seconds=load_seconds,
                                gen_seconds=None,
                                audio_seconds=None,
                                realtime_factor=None,
                                effective_rate_vs_speed1=None,
                                similarity_vs_baseline=None,
                                wav_path=None,
                            )
                        )
                        print(f"  FAIL {e}")

        # Effective speaking rate vs speed=1.0 for each model+provider
        for row in results:
            if not row.success or row.audio_seconds is None:
                continue
            d1 = speed1_duration.get((row.model_file, row.provider))
            if d1 and row.audio_seconds > 0:
                row.effective_rate_vs_speed1 = d1 / row.audio_seconds

        # Baseline for voice similarity: prefer fp32 + CUDA + speed 1.0
        preferred_baseline = ("kokoro-v1.0.onnx", "CUDAExecutionProvider", 1.0)
        baseline_key = None
        if preferred_baseline in audio_cache:
            baseline_key = preferred_baseline
        else:
            for key in audio_cache:
                if key[2] == 1.0:
                    baseline_key = key
                    break

        if baseline_key is not None:
            base_audio, _ = audio_cache[baseline_key]
            for row in results:
                if not row.success:
                    continue
                key = (row.model_file, row.provider, row.speed)
                if key in audio_cache:
                    this_audio, _ = audio_cache[key]
                    row.similarity_vs_baseline = voice_similarity(base_audio, this_audio)

        # Sort: fastest first (high speaking rate priority -> speed desc, then rtf desc)
        sortable = [r for r in results if r.success and r.realtime_factor is not None]
        sortable.sort(
            key=lambda r: (
                r.speed,
                r.realtime_factor if r.realtime_factor is not None else -1.0,
            ),
            reverse=True,
        )

        print_header("Top results (prioritizing high speaking speed then throughput)")
        for idx, row in enumerate(sortable[:10], start=1):
            sim_txt = (
                f"{row.similarity_vs_baseline:.4f}"
                if row.similarity_vs_baseline is not None
                else "n/a"
            )
            eff_txt = (
                f"{row.effective_rate_vs_speed1:.2f}x"
                if row.effective_rate_vs_speed1 is not None
                else "n/a"
            )
            print(
                f"{idx:2d}. {row.model_file:22s} | {row.provider:22s} | "
                f"speed={row.speed:.1f} | rtf={row.realtime_factor:.2f}x | "
                f"eff_rate={eff_txt:>6s} | sim={sim_txt}"
            )

        # Persist JSON
        payload = {
            "timestamp_unix": time.time(),
            "voice": args.voice,
            "text": args.text,
            "models_dir": str(models_dir),
            "providers_available": ort.get_available_providers(),
            "providers_tested": providers,
            "speeds_tested": speeds,
            "baseline_key": baseline_key,
            "results": [asdict(r) for r in results],
            "best_overall": asdict(sortable[0]) if sortable else None,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print_header("Report written")
        print(f"JSON: {output_json}")
        print(f"WAVs: {output_audio_dir}")

        return 0
    finally:
        if old_provider_env is None:
            os.environ.pop("ONNX_PROVIDER", None)
        else:
            os.environ["ONNX_PROVIDER"] = old_provider_env


if __name__ == "__main__":
    raise SystemExit(main())

