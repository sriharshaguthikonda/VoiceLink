# VoiceLink API Reference - Dynamic Voice Selection

## Quick Start

```python
import requests

# Get all available voices
response = requests.get("http://127.0.0.1:7860/v1/voices")
voices = response.json()

# Synthesize with a specific voice
data = {
    "text": "Hello world",
    "voice": "af_jessica",
    "speed": 1.0,
    "format": "pcm_24k_16bit"
}
response = requests.post("http://127.0.0.1:7860/v1/tts", json=data)
audio_data = response.content
```

## API Endpoints

### 1. Get Available Voices

**Endpoint**: `GET /v1/voices`

**Response**: JSON array of voice objects

```json
[
  {
    "id": "af_jessica",
    "name": "Jessica",
    "language": "en-US",
    "gender": "female",
    "description": "Natural, pleasant female voice with clear articulation.",
    "model": "kokoro_pytorch",
    "tags": ["natural", "pleasant", "clear"],
    "sample_rate": 24000
  }
]
```

**Voice Fields**:
- `id`: Unique identifier for synthesis requests
- `name`: Human-readable display name
- `language`: BCP 47 language code (en-US, en-GB)
- `gender`: "male" or "female"
- `description`: Voice characteristics
- `model`: Backend model used
- `tags`: Descriptive tags for filtering
- `sample_rate`: Audio sample rate (Hz)

### 2. Synthesize Speech

**Endpoint**: `POST /v1/tts`

**Request Body**:
```json
{
  "text": "Text to synthesize",
  "voice": "af_jessica",
  "speed": 1.0,
  "format": "pcm_24k_16bit"
}
```

**Parameters**:
- `text`: Text to speak (required)
- `voice`: Voice ID from `/v1/voices` (required)
- `speed`: Speaking rate 0.5-2.0 (optional, default 1.0)
- `format`: Audio format (optional, default "pcm_24k_16bit")

**Response**: Raw PCM audio bytes (24kHz, 16-bit, mono)

### 3. Health Check

**Endpoint**: `GET /v1/health`

**Response**: Server status
```json
{
  "status": "healthy",
  "model": "kokoro_pytorch",
  "voices_loaded": 12
}
```

## Voice Selection Strategies

### 1. By Language and Gender

```python
def get_voices_by_criteria(voices, language="en-US", gender="female"):
    return [v for v in voices 
            if v['language'] == language and v['gender'] == gender]

# Usage
voices = requests.get("http://127.0.0.1:7860/v1/voices").json()
american_female = get_voices_by_criteria(voices, "en-US", "female")
```

### 2. By Tags

```python
def get_voices_by_tags(voices, required_tags):
    return [v for v in voices 
            if any(tag in v['tags'] for tag in required_tags)]

# Usage
professional_voices = get_voices_by_tags(voices, ["professional", "clear"])
natural_voices = get_voices_by_tags(voices, ["natural"])
```

### 3. Dynamic Context-Based Selection

```python
def select_voice_for_context(voices, context, user_preference):
    if user_preference == "professional":
        candidates = get_voices_by_tags(voices, ["professional", "clear"])
    elif user_preference == "friendly":
        candidates = get_voices_by_tags(voices, ["friendly", "conversational"])
    elif user_preference == "natural":
        candidates = get_voices_by_tags(voices, ["natural"])
    else:
        candidates = voices
    
    # Return first match or fallback to Jessica
    return candidates[0]['id'] if candidates else "af_jessica"
```

## Available Voices (Current)

### American English (en-US)
**Female**:
- `af_heart` - Heart (warm, expressive)
- `af_bella` - Bella (clear, professional)
- `af_nicole` - Nicole (smooth, calm)
- `af_sarah` - Sarah (friendly, conversational)
- `af_sky` - Sky (light, youthful)
- `af_jessica` - Jessica (natural, pleasant, clear)

**Male**:
- `am_adam` - Adam (natural, conversational)
- `am_michael` - Michael (deep, authoritative)

### British English (en-GB)
**Female**:
- `bf_emma` - Emma (classic, british)
- `bf_isabella` - Isabella (elegant, british)

**Male**:
- `bm_george` - George (traditional, british)
- `bm_lewis` - Lewis (warm, british)

## Client Implementation Examples

### JavaScript/TypeScript

```javascript
class VoiceLinkClient {
    constructor(baseUrl = 'http://127.0.0.1:7860') {
        this.baseUrl = baseUrl;
    }
    
    async getVoices() {
        const response = await fetch(`${this.baseUrl}/v1/voices`);
        return response.json();
    }
    
    async synthesize(text, voiceId, speed = 1.0) {
        const response = await fetch(`${this.baseUrl}/v1/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                voice: voiceId,
                speed,
                format: 'pcm_24k_16bit'
            })
        });
        return response.arrayBuffer();
    }
}

// Usage
const client = new VoiceLinkClient();
const voices = await client.getVoices();
const jessica = voices.find(v => v.id === 'af_jessica');
const audio = await client.synthesize('Hello world', jessica.id);
```

### Python

```python
import requests

class VoiceLinkClient:
    def __init__(self, base_url="http://127.0.0.1:7860"):
        self.base_url = base_url
    
    def get_voices(self):
        response = requests.get(f"{self.base_url}/v1/voices")
        return response.json()
    
    def synthesize(self, text, voice_id, speed=1.0):
        data = {
            "text": text,
            "voice": voice_id,
            "speed": speed,
            "format": "pcm_24k_16bit"
        }
        response = requests.post(f"{self.base_url}/v1/tts", json=data)
        return response.content
```

## Error Handling

```python
try:
    voices = requests.get("http://127.0.0.1:7860/v1/voices", timeout=5)
    voices.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Server error: {e}")
    # Fallback to default voice
    voice_id = "af_jessica"
```

## Performance Tips

1. **Cache voice list**: Voice list doesn't change often, cache it
2. **Use streaming**: For long text, use streaming responses
3. **Batch requests**: Group multiple TTS requests
4. **Voice pre-loading**: First request to a voice may be slower

## Audio Format

- **Format**: PCM (raw audio)
- **Sample Rate**: 24,000 Hz
- **Bit Depth**: 16-bit
- **Channels**: Mono (1 channel)
- **Byte Order**: Little-endian

Convert to WAV if needed:
```python
import wave
import numpy as np

def pcm_to_wav(pcm_data, sample_rate=24000):
    audio = np.frombuffer(pcm_data, dtype=np.int16)
    with wave.open('output.wav', 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())
```
