# AGENTS.md — VoiceLink

## Active work

- **Open issues:** see [`TODO.md`](./TODO.md) for P0/P1-sorted index, or [GitHub Issues](https://github.com/sriharshaguthikonda/VoiceLink/issues).
- **Working branch:** `codex/matcha-tts-server` (currently at `origin/main` HEAD).

## Architecture

Three components:

| Component | Language | Role |
|---|---|---|
| `server/` | Python (FastAPI + torch + kokoro + matcha) | HTTP TTS inference server, port 7860 |
| `gui/` | Tauri + Vite + TypeScript | Desktop UI client |
| `sapi_bridge/` | C++ (Win32 COM DLL) | Windows SAPI 5 voice adapter that relays to HTTP server |

## Entry points

- Server: `server/main.py` → `uvicorn server.main:app --host 127.0.0.1 --port 7860`
- GUI dev: `cd gui && pnpm dev`
- SAPI bridge: build with CMake → `sapi_bridge/build/`; registered as COM via `regsvr32`

## See also

- [`CLIENT_API_REFERENCE.md`](./CLIENT_API_REFERENCE.md)
- [`DEEP_DIVE.md`](./DEEP_DIVE.md)
- [`JESSICA_VOICE_INTEGRATION.md`](./JESSICA_VOICE_INTEGRATION.md)
- [`TASKS.md`](./TASKS.md)
