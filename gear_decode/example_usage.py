#!/usr/bin/env python3
"""
Example script demonstrating gear_decode usage with detailed analysis
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gear_decode.gear_generate import GearGenerator

def analyze_generation(result):
    """Detailed analysis of generation results"""
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    times = result.time_per_token
    if not times:
        print("No timing data available")
        return
    
    # Statistics
    min_time = min(times)
    max_time = max(times)
    avg_time = sum(times) / len(times)
    
    print(f"\nTiming Statistics:")
    print(f"  Minimum time/token: {min_time:.2f} ms ({1000.0/min_time:.2f} tok/sec)")
    print(f"  Maximum time/token: {max_time:.2f} ms ({1000.0/max_time:.2f} tok/sec)")
    print(f"  Average time/token: {avg_time:.2f} ms ({1000.0/avg_time:.2f} tok/sec)")
    print(f"  Variance: {max_time - min_time:.2f} ms")
    
    # Percentiles
    sorted_times = sorted(times)
    p50_idx = len(sorted_times) // 2
    p90_idx = int(len(sorted_times) * 0.9)
    p99_idx = int(len(sorted_times) * 0.99)
    
    print(f"\nPercentiles:")
    if len(sorted_times) > p50_idx:
        print(f"  P50 (median): {sorted_times[p50_idx]:.2f} ms")
    if len(sorted_times) > p90_idx:
        print(f"  P90: {sorted_times[p90_idx]:.2f} ms")
    if len(sorted_times) > p99_idx:
        print(f"  P99: {sorted_times[p99_idx]:.2f} ms")
    
    # Throughput analysis
    print(f"\nThroughput:")
    print(f"  Overall: {result.tokens_per_second:.2f} tok/sec")
    print(f"  Total tokens: {result.n_tokens_generated}")
    print(f"  Total time: {result.total_time_ms:.2f} ms ({result.total_time_ms/1000:.2f} sec)")

def example_simple():
    """Simple usage example"""
    print("=" * 80)
    print("Example 1: Simple Generation")
    print("=" * 80)
    
    # This would require a real model
    print("""
generator = GearGenerator()
result = generator.generate(
    model_path="/path/to/model.gguf",
    prompt="What is 2+2?",
    n_predict=20
)

print(f"Output: {result.output_text}")
print(f"Tokens/sec: {result.tokens_per_second:.2f}")
""")

def example_benchmarking():
    """Benchmarking example"""
    print("\n" + "=" * 80)
    print("Example 2: Benchmarking Different Configurations")
    print("=" * 80)
    
    print("""
generator = GearGenerator()

configs = [
    {"n_threads": 1, "name": "1 thread"},
    {"n_threads": 4, "name": "4 threads"},
    {"n_threads": 8, "name": "8 threads"},
]

for config in configs:
    result = generator.generate(
        model_path="/path/to/model.gguf",
        prompt="Write a short story",
        n_predict=100,
        n_threads=config["n_threads"]
    )
    
    print(f"{config['name']}: {result.tokens_per_second:.2f} tok/sec")
""")

def example_analysis():
    """Detailed analysis example"""
    print("\n" + "=" * 80)
    print("Example 3: Detailed Performance Analysis")
    print("=" * 80)
    
    print("""
generator = GearGenerator()
result = generator.generate(
    model_path="/path/to/model.gguf",
    prompt="Explain quantum computing",
    n_predict=200
)

# Analyze per-token timing
times = result.time_per_token
print(f"Total tokens: {len(times)}")
print(f"Average time/token: {sum(times)/len(times):.2f} ms")
print(f"Min/Max: {min(times):.2f} / {max(times):.2f} ms")

# Find slowest tokens
slowest_indices = sorted(range(len(times)), key=lambda i: times[i], reverse=True)[:5]
print(f"\\nSlowest 5 tokens:")
for i in slowest_indices:
    print(f"  Token {i+1}: {times[i]:.2f} ms")
""")

def example_with_real_model(model_path):
    """Run a real example if model is provided"""
    print("\n" + "=" * 80)
    print("Example 4: Real Generation with Analysis")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    print(f"\nUsing model: {model_path}")
    print("Generating text...")
    
    generator = GearGenerator()
    result = generator.generate(
        model_path=model_path,
        prompt="What are the benefits of Python programming language?",
        n_predict=30,
        use_instruct=True,
        n_threads=4,
        enable_flash_attn=False,
        n_gpu_layers=0
    )
    
    if not result.is_success:
        print(f"Generation failed with error code: {result.error_code}")
        return
    
    print("\n" + "-" * 80)
    print("OUTPUT:")
    print("-" * 80)
    print(result.output_text)
    print("-" * 80)
    
    analyze_generation(result)

def main():
    print("=" * 80)
    print("gear_decode Usage Examples")
    print("=" * 80)
    
    # Show code examples
    example_simple()
    example_benchmarking()
    example_analysis()
    
    # Run real example if model provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        example_with_real_model(model_path)
    else:
        print("\n" + "=" * 80)
        print("To run a real example, provide a model path:")
        print(f"  python {sys.argv[0]} /path/to/model.gguf")
        print("=" * 80)

if __name__ == "__main__":
    main()
