#!/usr/bin/env python3
"""
Test script to debug TTS output
"""

import requests
import json

def test_tts():
    url = "http://127.0.0.1:7860/v1/tts"
    
    # Test data
    data = {
        "text": "Hello world, this is a test of the VoiceLink ONNX server.",
        "voice": "af_heart",
        "speed": 1.0
    }
    
    try:
        print("Sending TTS request...")
        response = requests.post(url, json=data)
        
        print(f"Status code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            # Save audio to file
            with open("test_output.wav", "wb") as f:
                f.write(response.content)
            
            print(f"Audio saved to test_output.wav ({len(response.content)} bytes)")
            
            # Check if it's valid audio data
            if len(response.content) > 100:
                print("✅ Received audio data")
            else:
                print("❌ Audio data seems too small")
                print(f"Content preview: {response.content[:100]}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_tts()
