#!/usr/bin/env python3
"""
VoiceLink Client - Dynamic Voice Selection Example

This script shows how to:
1. Get the list of available voices from the server
2. Filter voices by language, gender, tags
3. Select a voice dynamically
4. Synthesize speech with the selected voice
"""

import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Voice:
    """Voice information from VoiceLink server"""
    id: str
    name: str
    language: str
    gender: str
    description: str
    model: str
    tags: List[str]
    sample_rate: int

class VoiceLinkClient:
    """Client for VoiceLink TTS server"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def get_voices(self) -> List[Voice]:
        """Get all available voices from the server"""
        try:
            response = self.session.get(f"{self.base_url}/v1/voices")
            response.raise_for_status()
            
            voices_data = response.json()
            voices = []
            
            for voice_data in voices_data:
                voice = Voice(
                    id=voice_data['id'],
                    name=voice_data['name'],
                    language=voice_data['language'],
                    gender=voice_data['gender'],
                    description=voice_data['description'],
                    model=voice_data['model'],
                    tags=voice_data.get('tags', []),
                    sample_rate=voice_data['sample_rate']
                )
                voices.append(voice)
            
            return voices
            
        except Exception as e:
            print(f"Error getting voices: {e}")
            return []
    
    def filter_voices(self, voices: List[Voice], 
                     language: Optional[str] = None,
                     gender: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> List[Voice]:
        """Filter voices by criteria"""
        filtered = voices
        
        if language:
            filtered = [v for v in filtered if v.language == language]
        
        if gender:
            filtered = [v for v in filtered if v.gender == gender]
        
        if tags:
            filtered = [v for v in filtered 
                       if any(tag in v.tags for tag in tags)]
        
        return filtered
    
    def get_voice_by_id(self, voice_id: str) -> Optional[Voice]:
        """Get a specific voice by ID"""
        voices = self.get_voices()
        for voice in voices:
            if voice.id == voice_id:
                return voice
        return None
    
    def synthesize(self, text: str, voice_id: str, 
                  speed: float = 1.0, 
                  output_file: Optional[str] = None) -> bytes:
        """Synthesize speech with a specific voice"""
        try:
            data = {
                "text": text,
                "voice": voice_id,
                "speed": speed,
                "format": "pcm_24k_16bit"
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/tts",
                json=data,
                stream=True
            )
            response.raise_for_status()
            
            # Get audio data
            audio_data = b''
            for chunk in response.iter_content(chunk_size=8192):
                audio_data += chunk
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                print(f"Audio saved to: {output_file}")
            
            return audio_data
            
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            return b''

def main():
    """Example usage of VoiceLink client"""
    
    print("🎤 VoiceLink Client - Dynamic Voice Selection Demo")
    print("=" * 50)
    
    # Initialize client
    client = VoiceLinkClient()
    
    # Get all available voices
    print("\n1. Getting all available voices...")
    voices = client.get_voices()
    print(f"   Found {len(voices)} voices")
    
    # Display all voices
    print("\n📋 Available Voices:")
    for i, voice in enumerate(voices, 1):
        print(f"   {i:2d}. {voice.name:12s} ({voice.id}) - {voice.language} {voice.gender}")
        print(f"       {voice.description}")
        print(f"       Tags: {', '.join(voice.tags)}")
        print()
    
    # Example 1: Filter by language and gender
    print("2. Filtering voices (American English, Female)...")
    american_female = client.filter_voices(voices, language="en-US", gender="female")
    print(f"   Found {len(american_female)} American female voices:")
    for voice in american_female:
        print(f"   - {voice.name} ({voice.id})")
    
    # Example 2: Find voices with specific tags
    print("\n3. Finding voices with 'natural' tag...")
    natural_voices = client.filter_voices(voices, tags=["natural"])
    print(f"   Found {len(natural_voices)} voices with 'natural' tag:")
    for voice in natural_voices:
        print(f"   - {voice.name} ({voice.id})")
    
    # Example 3: Get Jessica voice specifically
    print("\n4. Getting Jessica voice...")
    jessica = client.get_voice_by_id("af_jessica")
    if jessica:
        print(f"   ✅ Found: {jessica.name}")
        print(f"   Description: {jessica.description}")
        print(f"   Language: {jessica.language}, Gender: {jessica.gender}")
    else:
        print("   ❌ Jessica voice not found")
    
    # Example 4: Synthesize with different voices
    print("\n5. Testing synthesis with different voices...")
    test_text = "Hello, this is a test of dynamic voice selection."
    
    # Test Jessica voice
    print("   Testing Jessica voice...")
    jessica_audio = client.synthesize(test_text, "af_jessica", 
                                     output_file="jessica_demo.pcm")
    if jessica_audio:
        print(f"   ✅ Jessica: {len(jessica_audio)} bytes generated")
    
    # Test Sky voice
    print("   Testing Sky voice...")
    sky_audio = client.synthesize(test_text, "af_sky", 
                                output_file="sky_demo.pcm")
    if sky_audio:
        print(f"   ✅ Sky: {len(sky_audio)} bytes generated")
    
    # Example 5: Dynamic voice selection based on user preference
    print("\n6. Dynamic voice selection example...")
    
    def select_voice_for_context(text: str, preference: str) -> str:
        """Select voice based on text context and user preference"""
        voices = client.get_voices()
        
        if preference == "professional":
            # Look for voices with 'professional' or 'clear' tags
            professional = client.filter_voices(voices, tags=["professional", "clear"])
            if professional:
                return professional[0].id
        
        elif preference == "friendly":
            # Look for voices with 'friendly' or 'conversational' tags
            friendly = client.filter_voices(voices, tags=["friendly", "conversational"])
            if friendly:
                return friendly[0].id
        
        elif preference == "natural":
            # Look for voices with 'natural' tag
            natural = client.filter_voices(voices, tags=["natural"])
            if natural:
                return natural[0].id
        
        # Default to Jessica
        return "af_jessica"
    
    # Test dynamic selection
    contexts = [
        ("Welcome to our business presentation", "professional"),
        ("Hey there! How are you doing today?", "friendly"),
        ("This is a natural reading of the text", "natural")
    ]
    
    for text, preference in contexts:
        selected_voice = select_voice_for_context(text, preference)
        voice_info = client.get_voice_by_id(selected_voice)
        print(f"   Context: '{preference}' -> Selected: {voice_info.name}")
    
    print("\n✅ Demo completed!")
    print("\n📝 Key APIs for dynamic voice selection:")
    print("   GET  /v1/voices          - Get all available voices")
    print("   POST /v1/tts             - Synthesize with specific voice")
    print("   Voice selection criteria: id, language, gender, tags")

if __name__ == "__main__":
    main()
