# VoiceLink Agent Notes (High-Level)

This file captures practical runtime guidance discovered from local benchmarking on `2026-03-31`.

## Git Workflow Discipline

- Use small, focused commits.
- After each significant piece of work, commit and push immediately instead of batching many unrelated changes together.
- Keep each commit message specific to that one completed unit of work.

## Kokoro ONNX Performance Baseline

- Machine tested: Windows + NVIDIA GTX 1660
- Voice tested: `af_sky`
- Speaking speeds tested: `1.0`, `1.5`, `2.0`
- Best high-speed result: `kokoro-v1.0.onnx` + `CUDAExecutionProvider` at `speed=2.0` (`~1.60x` realtime, `~1.83x` faster speaking rate than speed `1.0`)
- Important: `kokoro-v1.0.int8.onnx` was slower on this GPU and should not be the default fast path.

## Current Runtime Behavior (after patch)

- `server/models/kokoro_onnx_official.py` now:
  - selects ONNX provider deterministically (`CUDA` if available and requested, otherwise `CPU`)
  - prefers `kokoro-v1.0.onnx` first (no longer int8-first)
  - supports explicit model override via `VOICELINK_KOKORO_ONNX_MODEL_FILE`
  - supports explicit provider override via `ONNX_PROVIDER`

## Recommended Defaults for “High Speaking Rate”

1. Provider: `CUDAExecutionProvider`
2. Model file: `C:\Windows_software\VoiceLink\models\kokoro-v1.0.onnx`
3. Request parameter: `speed=2.0` (max supported by `kokoro_onnx`)

## Benchmark Command

```powershell
& "C:\Windows_software\VoiceLink\gpu_env\Scripts\python.exe" "C:\Windows_software\VoiceLink\scripts\benchmark_kokoro_sky_speed.py" --models-dir "C:\Windows_software\VoiceLink\models" --output-json "C:\Windows_software\VoiceLink\research\kokoro_sky_speed_benchmark_results.json" --output-audio-dir "C:\Windows_software\VoiceLink\research\kokoro_sky_samples"
```

## Benchmark Artifacts

- Metrics JSON: `C:\Windows_software\VoiceLink\research\kokoro_sky_speed_benchmark_results.json`
- Audio samples: `C:\Windows_software\VoiceLink\research\kokoro_sky_samples\`
