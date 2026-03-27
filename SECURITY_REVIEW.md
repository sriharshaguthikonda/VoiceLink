# Security Review (March 27, 2026)

Scope reviewed:
- `server/` (FastAPI inference service)
- `gui/src-tauri/src/lib.rs` (Tauri privileged backend)
- `.github/workflows/*.yml` (CI/release workflow hygiene)

Methods used:
- Manual code audit of security-sensitive paths (process execution, filesystem writes, HTTP surfaces, elevation paths, and input handling).
- Pattern search for risky primitives (`Command::new`, `subprocess`, `eval/exec`, `shell=True`, dynamic HTML).
- Focused review of voice toggle/elevation, model clone uploads, and streaming endpoints.

## Findings

### 1) Potential PowerShell command injection in elevated registry path construction (High)
**Where:** `gui/src-tauri/src/lib.rs` (`toggle_voice` + `run_elevated_powershell`)

The elevated path builds PowerShell commands by string concatenation and interpolates `token_name`/`reg_path` directly into single-quoted command fragments.
`voice_id` is not constrained to a strict allowlist in `toggle_voice`, so crafted input containing quote/control characters could break quoting and alter elevated commands.

**Why this matters:** this path runs with `-Verb RunAs` (admin elevation), so successful injection is high impact.

**Recommended fix:**
- Enforce strict `voice_id` validation before use (e.g., `^[A-Za-z0-9_\-]+$`).
- Escape *all* interpolated fields used in command strings (including path fragments), or avoid script concatenation entirely.
- Prefer invoking PowerShell with structured arguments or using direct Win32/registry APIs under elevation helpers instead of script text.

---

### 2) Predictable temp script path for elevated PowerShell (Medium)
**Where:** `gui/src-tauri/src/lib.rs` (`run_elevated_powershell`)

The code always writes to `%TEMP%\\voicelink_elevate.ps1` before running elevated.
A fixed filename in a shared temp location increases tampering/race risk (TOCTOU), especially on multi-user systems.

**Recommended fix:**
- Use a unique randomized temp filename per invocation.
- Open/write with exclusive creation semantics.
- Restrict ACLs on the temporary script file where possible.

---

### 3) Unbounded upload read in Qwen3 clone endpoint can cause memory exhaustion (Medium)
**Where:** `server/routers/qwen3.py` (`qwen3_clone_voice`)

The endpoint performs `contents = await audio.read()` with no explicit max size guard.
Large payloads can force high memory usage and possible service instability.

**Recommended fix:**
- Enforce request body limits at ASGI/server level and endpoint level.
- Stream to disk in bounded chunks instead of reading full body into memory.
- Reject files exceeding a strict maximum (e.g., 10–20 MB for intended use case).

---

### 4) “Streaming” TTS route buffers full audio in memory before responding (Medium)
**Where:** `server/routers/tts.py` (`synthesize`)

`_run_synthesis()` appends all audio chunks to a list and only then responds.
For long texts, this can consume significant memory and enables easy local DoS.

**Recommended fix:**
- Use a producer-consumer queue and yield chunks as generated.
- Add per-request text/audio duration caps and concurrency limits.

---

### 5) Startup/shutdown robustness bug can crash during shutdown after model-load failure (Low reliability bug)
**Where:** `server/main.py` (`lifespan`)

`model` is defined only inside the `try` block but referenced on shutdown.
If model load fails, shutdown can hit `if model is not None:` with an unbound variable.

**Recommended fix:**
- Initialize `model = None` before `try`.

## Positive notes

- GitHub workflows are already on non-deprecated action major versions (`checkout@v4`, `setup-node@v4`, `upload-artifact@v4`).
- Voice profile delete path includes traversal defense using `resolve().relative_to(...)`.
- FastAPI/Pydantic input models enforce several sensible bounds (e.g., text length and field sizes).

## Suggested next steps

1. Patch finding #1 (input validation + safe elevated command strategy) first.
2. Patch findings #3 and #4 to improve resilience under abusive inputs.
3. Patch #2 and #5 as hardening/quality follow-ups.
4. Add security regression tests for `voice_id` validation and large upload rejection.
