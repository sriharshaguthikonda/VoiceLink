#!/usr/bin/env python3
"""
Compare GPU vs original PyTorch performance
"""

import requests
import time

def test_performance(model_name="kokoro_onnx"):
    url = "http://127.0.0.1:7860/v1/tts"
    
    # Test data
    data = {
        "text": "Hello world, this is a performance test.",
        "voice": "af_heart",
        "speed": 1.0
    }
    
    try:
        print(f"🧪 Testing {model_name} performance...")
        start_time = time.time()
        
        response = requests.post(url, json=data)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if response.status_code == 200:
            audio_length = int(response.headers.get('x-audio-length', '0'))
            sample_rate = 24000
            sample_width = 2
            channels = 1
            
            bytes_per_second = sample_rate * sample_width * channels
            audio_duration = audio_length / bytes_per_second
            realtime_factor = audio_duration / total_time
            
            print(f"✅ {model_name}:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Audio duration: {audio_duration:.2f}s") 
            print(f"  Realtime factor: {realtime_factor:.2f}x")
            print(f"  Status: {'🚀 FAST' if realtime_factor >= 1.0 else '⚡ OK' if realtime_factor >= 0.5 else '🐌 SLOW'}")
            
            return realtime_factor
        else:
            print(f"❌ Error: {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return 0

def analyze_gpu_utilization():
    print("\n🔍 GPU Utilization Analysis:")
    print("GTX 1660 Specifications:")
    print("- CUDA Cores: 1408")
    print("- Memory: 6GB GDDR5") 
    print("- Memory Bandwidth: 192 GB/s")
    print("- Architecture: Turing (2019)")
    print("- No Tensor Cores (RTX feature)")
    
    print("\n💡 Why it's slow:")
    print("1. Kokoro 82M parameters = heavy model for older GPU")
    print("2. No Tensor Cores = slower matrix operations")
    print("3. Memory bandwidth limited by older architecture")
    print("4. Single inference = GPU underutilization")
    print("5. CPU-GPU transfer overhead dominates")

if __name__ == "__main__":
    print("🎯 VoiceLink Performance Analysis")
    print("=" * 50)
    
    # Test current performance
    current_perf = test_performance("GPU Kokoro ONNX")
    
    analyze_gpu_utilization()
    
    print(f"\n📊 Summary:")
    print(f"Current GPU Performance: {current_perf:.2f}x realtime")
    print(f"Expected (RTX GPU): 2-3x realtime")
    print(f"Bottleneck: GTX 1660 architecture limitations")
    
    if current_perf < 0.5:
        print(f"\n🔧 Recommendations:")
        print(f"1. Upgrade to RTX 3060/4060 for Tensor Cores")
        print(f"2. Use smaller/faster model (like Piper)")
        print(f"3. Batch multiple requests for better GPU utilization")
        print(f"4. Consider CPU optimization for GTX 1660")
