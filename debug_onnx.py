#!/usr/bin/env python3
"""
Debug script to examine ONNX model inputs and fix tokenization
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from loguru import logger

def debug_onnx_model():
    """Examine the ONNX model structure and fix the implementation"""
    
    model_path = Path("models/kokoro-v1.0.onnx")
    voices_path = Path("models/voices-v1.0.bin")
    
    if not model_path.exists():
        print("❌ ONNX model not found")
        return
    
    # Load the model
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    
    # Examine inputs
    print("=== ONNX Model Inputs ===")
    for inp in session.get_inputs():
        print(f"Name: {inp.name}")
        print(f"Shape: {inp.shape}")
        print(f"Type: {inp.type}")
        print()
    
    # Examine outputs
    print("=== ONNX Model Outputs ===")
    for out in session.get_outputs():
        print(f"Name: {out.name}")
        print(f"Shape: {out.shape}")
        print(f"Type: {out.type}")
        print()
    
    # Load voices data
    print("=== Voice Data ===")
    voices_data = np.load(voices_path, allow_pickle=True)
    voice_keys = list(voices_data.keys())
    print(f"Available voices: {voice_keys[:10]}... (showing first 10)")
    
    # Check a sample voice embedding
    sample_voice = voices_data[voice_keys[0]]
    print(f"Sample voice '{voice_keys[0]}' shape: {sample_voice.shape}")
    print(f"Sample voice dtype: {sample_voice.dtype}")
    print(f"Sample voice min/max: {sample_voice.min():.3f} / {sample_voice.max():.3f}")
    
    # Test with ttstokenizer
    try:
        from ttstokenizer import IPATokenizer
        tokenizer = IPATokenizer()
        
        test_text = "Hello world"
        tokens = tokenizer(test_text)
        print(f"\n=== Tokenization Test ===")
        print(f"Text: '{test_text}'")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Token sample: {tokens[:20]}...")
        
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
    
    # Test model inference with dummy data
    print("\n=== Model Inference Test ===")
    try:
        # Create dummy inputs based on the model's expected format
        inputs = {}
        
        for inp in session.get_inputs():
            if inp.name == "tokens":
                # Create dummy token sequence
                dummy_tokens = np.array([[0, 1, 2, 3, 4, 5, 0]], dtype=np.int64)
                inputs[inp.name] = dummy_tokens
            elif inp.name == "style":
                # Use a sample voice embedding - need (1, 256) not (1, 1, 256)
                dummy_style = sample_voice[0, 0, :].reshape(1, 256)  # Shape: (1, 256)
                inputs[inp.name] = dummy_style
            elif inp.name == "speed":
                dummy_speed = np.array([1.0], dtype=np.float32)
                inputs[inp.name] = dummy_speed
            else:
                # Create dummy data for other inputs
                shape = [s if isinstance(s, int) else 1 for s in inp.shape]
                dummy_data = np.zeros(shape, dtype=np.float32)
                inputs[inp.name] = dummy_data
        
        print("Inputs prepared:")
        for name, data in inputs.items():
            print(f"  {name}: {data.shape} {data.dtype}")
        
        # Run inference
        outputs = session.run(None, inputs)
        
        print("\nOutputs:")
        for i, output in enumerate(outputs):
            print(f"  Output {i}: {output.shape} {output.dtype}")
            print(f"  Min/Max: {output.min():.3f} / {output.max():.3f}")
            
            # Check if it looks like audio
            if output.dtype == np.float32 and len(output.shape) >= 2:
                audio_sample = output[0] if output.shape[0] == 1 else output
                if len(audio_sample) > 100:
                    print(f"  Audio sample (first 20): {audio_sample[:20]}")
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_onnx_model()
