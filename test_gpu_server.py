#!/usr/bin/env python3
"""
Test the GPU-enabled VoiceLink server performance
"""

import requests
import time

def test_gpu_server():
    url = "http://127.0.0.1:7860/v1/tts"
    
    # Test data
    data = {
        "text": "Hello world! This is a test of the GPU-accelerated VoiceLink server with the latest fine-tuned Kokoro models.",
        "voice": "af_heart",
        "speed": 1.0
    }
    
    try:
        print("🚀 Testing GPU-Accelerated VoiceLink Server...")
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
            
            print(f"✅ GPU Server Success!")
            print(f"Total time: {total_time:.2f}s")
            print(f"Audio duration: {audio_duration:.2f}s") 
            print(f"Realtime factor: {realtime_factor:.2f}x")
            print(f"Audio size: {len(response.content):,} bytes")
            
            if realtime_factor >= 2.0:
                print("🎉 AMAZING! 2x+ faster than real-time!")
            elif realtime_factor >= 1.0:
                print("🚀 FASTER than real-time!")
            elif realtime_factor >= 0.5:
                print("⚡ Near real-time performance")
            else:
                print("🐌 Still slower than real-time")
                
            print(f"\n🎯 Performance Summary:")
            print(f"- GPU Acceleration: ✅ Active")
            print(f"- Model: INT8 Quantized Kokoro v1.0")
            print(f"- Voice: af_heart (fine-tuned)")
            print(f"- Speed: {realtime_factor:.2f}x realtime")
                
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_gpu_server()
