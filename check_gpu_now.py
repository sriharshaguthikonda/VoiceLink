#!/usr/bin/env python3
"""
Check if GPU is now available with onnxruntime-gpu
"""

import onnxruntime as ort

def check_gpu():
    print("=== GPU Check After Installing onnxruntime-gpu ===")
    
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("✅ CUDA provider available!")
        
        # Test with the actual model
        try:
            session = ort.InferenceSession(
                "models/kokoro-v1.0.onnx",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            active_providers = session.get_providers()
            print(f"Active providers: {active_providers}")
            
            if 'CUDAExecutionProvider' in active_providers:
                print("🚀 GPU acceleration is working!")
            else:
                print("⚠️ CUDA provider available but not being used")
                
        except Exception as e:
            print(f"❌ Error testing GPU: {e}")
    else:
        print("❌ CUDA provider not available")
        print("You may need to:")
        print("1. Install CUDA 12.x from NVIDIA")
        print("2. Install cuDNN 9.x")
        print("3. Add CUDA to PATH")

if __name__ == "__main__":
    check_gpu()
