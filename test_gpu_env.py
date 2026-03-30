#!/usr/bin/env python3
"""
Test GPU in the new virtual environment
"""

import torch
import onnxruntime as ort
# from kokoro_onnx import Kokoro  # Commented out to avoid import issues

def test_gpu_env():
    print("=== GPU Environment Test ===")
    
    # Test PyTorch CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test GPU memory
        torch.cuda.set_device(0)
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Test ONNX Runtime providers
    try:
        print(f"\nONNX Runtime version: {ort.__version__}")
    except AttributeError:
        print("\nONNX Runtime version: Unable to get version")
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    # Test CUDA provider
    if 'CUDAExecutionProvider' in providers:
        print("✅ CUDA provider available!")
        
        try:
            session = ort.InferenceSession(
                "models/kokoro-v1.0.int8.onnx",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            active_providers = session.get_providers()
            print(f"Active providers: {active_providers}")
            
            if 'CUDAExecutionProvider' in active_providers:
                print("🚀 GPU acceleration is WORKING!")
                
                # Test kokoro-onnx
                print("\nTesting kokoro-onnx with GPU...")
                try:
                    from kokoro_onnx import Kokoro
                    kokoro = Kokoro("models/kokoro-v1.0.int8.onnx", "models/voices-v1.0.bin")
                    
                    import time
                    start_time = time.time()
                    samples, sample_rate = kokoro.create(
                        "Hello world, this is a GPU test!", 
                        voice="af_heart", 
                        speed=1.0, 
                        lang="en-us"
                    )
                    synthesis_time = time.time() - start_time
                    audio_duration = len(samples) / sample_rate
                    realtime_factor = audio_duration / synthesis_time
                    
                    print(f"Synthesis: {synthesis_time:.2f}s for {audio_duration:.2f}s audio")
                    print(f"Realtime factor: {realtime_factor:.2f}x")
                    
                    if realtime_factor >= 1.0:
                        print("🎉 FASTER than real-time!")
                    elif realtime_factor >= 0.5:
                        print("⚡ Near real-time!")
                    else:
                        print("🐌 Still slower than real-time")
                        
                except ImportError as e:
                    print(f"kokoro-onnx not available: {e}")
                    print("But GPU acceleration for ONNX is working!")
                    return True
                
                return True
            else:
                print("⚠️ CUDA provider available but not being used")
                
        except Exception as e:
            print(f"❌ Error testing GPU: {e}")
            return False
    else:
        print("❌ CUDA provider not available")
        return False

if __name__ == "__main__":
    success = test_gpu_env()
    if success:
        print("\n🎉 GPU acceleration is ready!")
    else:
        print("\n❌ GPU acceleration not working")
