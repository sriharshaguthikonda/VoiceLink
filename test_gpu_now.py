#!/usr/bin/env python3
"""
Test GPU availability with CUDA 12.6
"""

import onnxruntime as ort
import torch

def test_gpu():
    print("=== GPU Test with CUDA 12.6 ===")
    
    # Check PyTorch CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check ONNX Runtime providers
    print(f"\nONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    # Test CUDA provider
    if 'CUDAExecutionProvider' in providers:
        print("✅ CUDA provider available!")
        
        try:
            # Test with the actual model
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3  # Show all logs
            
            session = ort.InferenceSession(
                "models/kokoro-v1.0.int8.onnx",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                sess_options=session_options
            )
            
            active_providers = session.get_providers()
            print(f"Active providers: {active_providers}")
            
            if 'CUDAExecutionProvider' in active_providers:
                print("🚀 GPU acceleration is WORKING!")
                return True
            else:
                print("⚠️ CUDA provider available but not being used")
                return False
                
        except Exception as e:
            print(f"❌ Error testing GPU: {e}")
            return False
    else:
        print("❌ CUDA provider not available")
        return False

if __name__ == "__main__":
    success = test_gpu()
    if success:
        print("\n🎉 GPU acceleration is ready!")
    else:
        print("\n❌ GPU acceleration not working")
