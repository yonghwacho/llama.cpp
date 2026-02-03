#!/usr/bin/env python3
"""
Simple test script for gear_decode
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gear_decode.gear_generate import GearGenerator

def test_basic():
    """Test basic functionality without a real model"""
    print("Testing gear_decode library loading...")
    
    try:
        generator = GearGenerator()
        print("✓ Library loaded successfully")
        print(f"  Library path: {generator._find_library()}")
        return True
    except Exception as e:
        print(f"✗ Failed to load library: {e}")
        return False

def test_with_model(model_path):
    """Test with an actual model"""
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return False
    
    print(f"\nTesting generation with model: {model_path}")
    
    try:
        generator = GearGenerator()
        
        print("Generating text...")
        result = generator.generate(
            model_path=model_path,
            prompt="What is 2+2?",
            n_predict=10,
            use_instruct=True,
            n_threads=4,
            enable_flash_attn=False,
            n_gpu_layers=0  # Use CPU only for testing
        )
        
        if not result.is_success:
            print(f"✗ Generation failed with error code: {result.error_code}")
            return False
        
        print("✓ Generation successful!")
        print(f"  Tokens generated: {result.n_tokens_generated}")
        print(f"  Total time: {result.total_time_ms:.2f} ms")
        print(f"  Tokens/sec: {result.tokens_per_second:.2f}")
        print(f"  Output: {result.output_text[:100]}...")
        print(f"  Per-token timing samples: {len(result.time_per_token)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("gear_decode Test Suite")
    print("=" * 80)
    
    # Test 1: Library loading
    if not test_basic():
        sys.exit(1)
    
    # Test 2: With model (if provided)
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not test_with_model(model_path):
            sys.exit(1)
    else:
        print("\nSkipping model test (no model path provided)")
        print("Usage: python test_gear_decode.py [model_path]")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
