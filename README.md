<p align="center">
  <h1 align="center">🔗 VoiceLink</h1>
  <p align="center">
    <strong>A bridge between neural TTS models and the Windows system voice API.</strong>
  </p>
  <p align="center">
    Use AI-powered voices in any Windows app — Thorium Reader, Microsoft Edge, Narrator, Balabolka, and more.
  </p>
</p>

---

## The Problem

Windows apps that support text-to-speech (like Thorium Reader for ebooks) rely on **Windows SAPI** (Speech API) to discover and use voices. The built-in Microsoft voices sound robotic and unnatural — fine for notifications, painful for listening to a full book.

Meanwhile, incredible open-source neural TTS models exist (Kokoro, Piper, Qwen-3 TTS, etc.) that sound nearly human — but **no Windows app can use them** because they don't speak SAPI.

## The Solution

**VoiceLink** is a SAPI-compliant COM driver that acts as a bridge:

```
┌──────────────────┐       SAPI/COM        ┌──────────────────┐      HTTP/WS       ┌──────────────────┐
│   Any Windows    │ ───────────────────►  │    VoiceLink     │ ────────────────►  │  TTS Inference   │
│   App (SAPI)     │   ISpTTSEngine        │   COM Driver     │   localhost:7860   │  Server (Python) │
│                  │                       │                  │                    │                  │
│  Thorium Reader  │ ◄─────────────────── │                  │ ◄──────────────── │  Kokoro / Piper  │
│  Edge Read Aloud │     PCM audio          │                  │    PCM/WAV audio   │  Qwen-3 TTS      │
│  Narrator        │     streamed back      │                  │    streamed back   │  Any ONNX model  │
└──────────────────┘                       └──────────────────┘                    └──────────────────┘
```

When Thorium Reader (or any app) asks Windows for a voice, VoiceLink shows up as a system voice. When it receives text, it forwards it to a local inference server running the neural model, receives the audio, and streams it back — all transparently.

## Project Goals

1. **Learning-first** — This project is built to deeply understand every layer: COM, SAPI, audio pipelines, TTS models, streaming, and system integration.
2. **Production-ready** — Clean architecture, proper error handling, installers, CI/CD — not a hacky prototype.
3. **Universal** — Any Windows SAPI app gets upgraded voices for free.
4. **Extensible** — Swap TTS models easily. Add new voices without recompiling the driver.

## Architecture

The project has three main components:

### 1. TTS Inference Server (`/server`)
- Python-based HTTP/WebSocket server
- Loads neural TTS models (Kokoro, Piper, Qwen-3 TTS, etc.)
- Accepts text → returns streaming PCM audio
- Can run on CPU or GPU

### 2. SAPI COM Bridge (`/bridge`)
- C++ or Rust COM DLL implementing `ISpTTSEngine` and related interfaces
- Registers as a Windows system voice
- Forwards speech requests to the inference server
- Streams audio back to the SAPI audio sink

### 3. System Tray App / Installer (`/app`)
- Windows installer (NSIS/WiX/MSIX)
- System tray app for configuration
- Voice management (download, enable, disable)
- Health monitoring (is the server running? GPU available?)

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Inference Server | Python, FastAPI/WebSocket | Fast prototyping, ML ecosystem |
| TTS Models | Kokoro, Piper, ONNX Runtime | Open-source, high quality |
| COM Bridge | C++ or Rust | COM interop, performance |
| Installer | WiX / MSIX | Windows-native packaging |
| Tray App | C# WPF or Tauri | Lightweight system UI |
| CI/CD | GitHub Actions | Automated builds and releases |

## Current Status

🟡 **Phase 0 — Research and Specification**

We are currently in the research phase, understanding the Windows SAPI interface, surveying TTS models, and defining the architecture. See [TASKS.md](TASKS.md) for detailed progress.

## Getting Started

> 🚧 The project is in early development. Setup instructions will be added as components are built.

### Prerequisites
- Windows 10/11
- Python 3.10+ (for inference server)
- Visual Studio 2022 or Rust toolchain (for COM bridge)
- CUDA toolkit (optional, for GPU inference)

## Learning Journal

This project is built with a learning-first approach. Key concepts we are exploring:

- **COM (Component Object Model)** — How Windows plugins work at the binary level
- **SAPI (Speech API)** — Microsoft's speech interface standard
- **Neural TTS** — How modern text-to-speech models work (transformers, vocoders, mel spectrograms)
- **Audio Pipelines** — PCM, sample rates, streaming, buffering
- **System Integration** — COM registration, Windows services, installers
- **Production Engineering** — CI/CD, error handling, monitoring, auto-updates

## Contributing

This is an open project built for learning. Contributions, questions, and discussions are welcome. Open an issue or start a discussion!

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built by <a href="https://github.com/ManveerAnand">Manveer Anand</a> — learning in public, one component at a time.</em>
</p>
