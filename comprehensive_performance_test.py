#!/usr/bin/env python3
"""
Comprehensive performance test for all available models
"""

import requests
import time
import json

def test_model_combination(model_name, text="Hello world, this is a performance test.", voice="af_heart", speed=1.0):
    """Test a specific model combination"""
    url = "http://127.0.0.1:7860/v1/tts"
    
    data = {
        "text": text,
        "voice": voice,
        "speed": speed
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=data, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            total_time = end_time - start_time
            audio_length = int(response.headers.get('x-audio-length', '0'))
            sample_rate = 24000
            sample_width = 2
            channels = 1
            
            bytes_per_second = sample_rate * sample_width * channels
            audio_duration = audio_length / bytes_per_second
            realtime_factor = audio_duration / total_time
            
            return {
                "success": True,
                "total_time": total_time,
                "audio_duration": audio_duration,
                "realtime_factor": realtime_factor,
                "audio_size": len(response.content),
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "total_time": end_time - start_time
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "total_time": 0
        }

def test_available_voices():
    """Get list of available voices"""
    try:
        response = requests.get("http://127.0.0.1:7860/v1/voices", timeout=10)
        if response.status_code == 200:
            voices = response.json()
            return [v["id"] for v in voices]
        return []
    except:
        return []

def run_comprehensive_test():
    """Run comprehensive performance comparison"""
    
    print("🎯 Comprehensive VoiceLink Performance Test")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        # Different text lengths
        {"text": "Hello world.", "label": "Short text"},
        {"text": "Hello world, this is a medium length test sentence.", "label": "Medium text"},
        {"text": "Hello world, this is a much longer test sentence with multiple words to test performance under different text lengths.", "label": "Long text"},
    ]
    
    # Get available voices
    voices = test_available_voices()
    if not voices:
        print("❌ Could not get available voices")
        return
    
    # Use a subset of voices for testing
    test_voices = voices[:3] if len(voices) >= 3 else voices
    
    print(f"📊 Testing with voices: {test_voices}")
    print(f"📝 Text variations: {[c['label'] for c in test_configs]}")
    print()
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"🧪 Test {i+1}: {config['label']}")
        print("-" * 40)
        
        for voice in test_voices:
            result = test_model_combination(
                model_name="kokoro_onnx",
                text=config["text"],
                voice=voice,
                speed=1.0
            )
            
            if result["success"]:
                status = "🚀 FAST" if result["realtime_factor"] >= 1.0 else "⚡ OK" if result["realtime_factor"] >= 0.5 else "🐌 SLOW"
                
                print(f"  {voice}: {result['realtime_factor']:.2f}x realtime ({result['total_time']:.2f}s) [{status}]")
                
                results.append({
                    "text_length": config["label"],
                    "voice": voice,
                    "realtime_factor": result["realtime_factor"],
                    "total_time": result["total_time"],
                    "audio_duration": result["audio_duration"],
                    "status": status
                })
            else:
                print(f"  {voice}: ERROR - {result['error']}")
        
        print()
    
    # Analysis
    print("📈 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if results:
        # Find best performing combination
        best_result = max(results, key=lambda x: x["realtime_factor"])
        worst_result = min(results, key=lambda x: x["realtime_factor"])
        
        # Calculate averages
        avg_realtime = sum(r["realtime_factor"] for r in results) / len(results)
        
        print(f"🏆 Best Performance:")
        print(f"   Voice: {best_result['voice']}")
        print(f"   Text: {best_result['text_length']}")
        print(f"   Speed: {best_result['realtime_factor']:.2f}x realtime")
        print(f"   Time: {best_result['total_time']:.2f}s")
        
        print(f"\n⚠️  Worst Performance:")
        print(f"   Voice: {worst_result['voice']}")
        print(f"   Text: {worst_result['text_length']}")
        print(f"   Speed: {worst_result['realtime_factor']:.2f}x realtime")
        print(f"   Time: {worst_result['total_time']:.2f}s")
        
        print(f"\n📊 Average Performance: {avg_realtime:.2f}x realtime")
        
        # Voice performance comparison
        voice_performance = {}
        for result in results:
            if result["voice"] not in voice_performance:
                voice_performance[result["voice"]] = []
            voice_performance[result["voice"]].append(result["realtime_factor"])
        
        print(f"\n🎤 Voice Performance Ranking:")
        voice_avg = {voice: sum(perf) / len(perf) for voice, perf in voice_performance.items()}
        sorted_voices = sorted(voice_avg.items(), key=lambda x: x[1], reverse=True)
        
        for i, (voice, avg_perf) in enumerate(sorted_voices, 1):
            status = "🚀" if avg_perf >= 1.0 else "⚡" if avg_perf >= 0.5 else "🐌"
            print(f"   {i}. {voice}: {avg_perf:.2f}x realtime {status}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if best_result["realtime_factor"] >= 0.5:
            print(f"   ✅ Use voice '{best_result['voice']}' for best performance")
            print(f"   ✅ Current setup is usable for real-time applications")
        else:
            print(f"   ⚠️  All combinations are slower than real-time")
            print(f"   💡 Consider: CPU optimization or hardware upgrade")
        
        print(f"\n🔧 OPTIMIZATION SUGGESTIONS:")
        print(f"   1. Use voice '{best_result['voice']}' (fastest)")
        print(f"   2. Keep text short for better responsiveness")
        print(f"   3. Consider CPU version for GTX 1660")
        print(f"   4. Batch multiple requests when possible")

if __name__ == "__main__":
    run_comprehensive_test()
