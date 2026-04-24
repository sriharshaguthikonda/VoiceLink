# TODO — VoiceLink

Index of GitHub Issues. Source of truth: [GitHub Issues](https://github.com/sriharshaguthikonda/VoiceLink/issues).

## P0 — Critical (security)

- [#2](https://github.com/sriharshaguthikonda/VoiceLink/issues/2) Unsafe `pickle.load` of phoneme_processor → RCE risk — `type:security`
- [#3](https://github.com/sriharshaguthikonda/VoiceLink/issues/3) Voice `.pt` load path not validated → symlink/traversal RCE risk — `type:security`
- [#4](https://github.com/sriharshaguthikonda/VoiceLink/issues/4) Unauthenticated voice-cloning upload endpoint = disk/DoS + PII store — `type:security`

## P1 — Important

- [#5](https://github.com/sriharshaguthikonda/VoiceLink/issues/5) No CORS / auth on TTS server — open on `0.0.0.0` if env var set — `type:security`
- [#6](https://github.com/sriharshaguthikonda/VoiceLink/issues/6) Voice profile path traversal race (mkdir before resolve check) — `type:security`
- [#7](https://github.com/sriharshaguthikonda/VoiceLink/issues/7) 50k-char text input + no queue = GPU DoS via concurrent requests — `type:security, type:perf`
- [#8](https://github.com/sriharshaguthikonda/VoiceLink/issues/8) Tauri CSP disabled + potential innerHTML sinks — `type:security`
- [#9](https://github.com/sriharshaguthikonda/VoiceLink/issues/9) Streaming TTS: exception after response headers sent → client hangs — `type:bug`
- [#10](https://github.com/sriharshaguthikonda/VoiceLink/issues/10) COM RefCount imbalance in `SetObjectToken` (SAPI bridge) — `type:bug`
- [#11](https://github.com/sriharshaguthikonda/VoiceLink/issues/11) TODO: incomplete error handling on stream failure (SAPI bridge) — `type:todo, type:enhancement`

## P2 — Deferred

_Style, deps, docs — next audit sweep._

---

_Generated 2026-04-24 by cross-repo audit. Branch: `codex/matcha-tts-server` (at `origin/main` HEAD)._
