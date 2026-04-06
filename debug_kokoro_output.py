#!/usr/bin/env python3
"""
Debug script to understand Kokoro PyTorch output format.
"""

from kokoro import KPipeline

def debug_kokoro_output():
    """Debug the Kokoro pipeline output format."""
    
    print("Creating Kokoro pipeline...")
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    
    print("Loading af_jessica voice...")
    pipeline.load_voice('af_jessica')
    
    print("Testing synthesis...")
    test_text = "Hello world"
    
    # Test the generator directly
    generator = pipeline(test_text, voice='af_jessica', speed=1.0)
    
    print(f"Generator type: {type(generator)}")
    
    # Collect all outputs
    outputs = list(generator)
    print(f"Number of outputs: {len(outputs)}")
    
    for i, output in enumerate(outputs):
        print(f"Output {i}:")
        print(f"  Type: {type(output)}")
        print(f"  Value: {output}")
        
        if isinstance(output, tuple):
            print(f"  Tuple length: {len(output)}")
            for j, item in enumerate(output):
                print(f"    Item {j}: {type(item)} - {item if not hasattr(item, 'shape') else f'shape={item.shape}'}")
        elif hasattr(output, 'shape'):
            print(f"  Shape: {output.shape}")
            print(f"  Dtype: {output.dtype}")
            print(f"  Min/Max: {output.min():.3f} / {output.max():.3f}")
        
        print()

if __name__ == "__main__":
    debug_kokoro_output()
