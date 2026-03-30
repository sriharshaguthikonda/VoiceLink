#!/usr/bin/env python3
"""
Simple TTS test to measure performance
"""

import requests
import time

def test_performance():
    url = "http://127.0.0.1:7860/v1/tts"
    
    # Test data
    data = {
        "text": "Hello world, this is a performance test of the INT8 quantized model.",
        "voice": "af_heart", 
        "speed": 1.0  # Valid speed range
    }
    
    try:
        print("Testing INT8 Quantized Model Performance...")
        start_time = time.time()
        
        response = requests.post(url, json=data)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if response.status_code == 200:
            audio_length = int(response.headers.get('x-audio-length', '0'))
            sample_rate = 24000
            sample_width = 2  # 16-bit
            channels = 1
            
            # Calculate audio duration
            bytes_per_second = sample_rate * sample_width * channels
            audio_duration = audio_length / bytes_per_second
            
            realtime_factor = audio_duration / total_time
            
            print(f"✅ Success!")
            print(f"Total time: {total_time:.2f}s")
            print(f"Audio duration: {audio_duration:.2f}s") 
            print(f"Realtime factor: {realtime_factor:.2f}x")
            print(f"Audio size: {len(response.content):,} bytes")
            
            if realtime_factor >= 1.0:
                print("🚀 FASTER than real-time!")
            elif realtime_factor >= 0.5:
                print("⚡ Near real-time performance")
            else:
                print("🐌 Slower than real-time")
                
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_performance()
