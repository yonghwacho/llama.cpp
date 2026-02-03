# gear_decode

A Python-wrapped baseline llama.cpp text generation library for benchmarking and analysis.

## Overview

This library provides a simple interface to llama.cpp text generation with detailed per-token timing information. Unlike the streaming approach in `custom_gen`, this implementation runs the complete decode loop in C++ and returns all results at once, including timing data for each generated token.

## Features

- Complete decode loop execution in C++
- Per-token timing measurements (decode tok/sec)
- Python wrapper using ctypes
- Support for instruct mode templates
- Configurable thread count and flash attention
- GPU layer offloading support

## Building

The library is built automatically when building llama.cpp with examples enabled:

```bash
cd llama.cpp
mkdir -p build && cd build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON
make gear_decode
```

The shared library will be output to `build/lib/libgear_decode.so` (or `.dylib` on macOS, `.dll` on Windows).

## Usage

### Python API

```python
from gear_decode.gear_generate import GearGenerator

# Initialize generator
generator = GearGenerator()

# Generate text
result = generator.generate(
    model_path="/path/to/model.gguf",
    prompt="What is Python?",
    n_predict=50,
    use_instruct=True,
    n_threads=4,
    enable_flash_attn=False,
    n_gpu_layers=99
)

# Check results
if result.is_success:
    print(f"Output: {result.output_text}")
    print(f"Tokens generated: {result.n_tokens_generated}")
    print(f"Total time: {result.total_time_ms:.2f} ms")
    print(f"Tokens/second: {result.tokens_per_second:.2f}")
    print(f"Average time per token: {result.average_time_per_token:.2f} ms")
    
    # Per-token timing
    for i, time_ms in enumerate(result.time_per_token):
        print(f"Token {i+1}: {time_ms:.2f} ms ({1000.0/time_ms:.2f} tok/sec)")
```

### Command Line

```bash
python gear_decode/gear_generate.py \
    -m /path/to/model.gguf \
    -p "What is the capital of France?" \
    -n 100 \
    -t 4 \
    --use-instruct \
    -ngl 99
```

## API Reference

### GearGenerator

**`generate(model_path, prompt, n_predict=32, use_instruct=True, n_threads=4, enable_flash_attn=False, n_gpu_layers=99)`**

Generates text using the llama.cpp model.

**Parameters:**
- `model_path` (str): Path to the GGUF model file
- `prompt` (str): Input prompt text
- `n_predict` (int): Maximum number of tokens to generate (default: 32)
- `use_instruct` (bool): Whether to use instruct mode template (default: True)
- `n_threads` (int): Number of threads to use (default: 4)
- `enable_flash_attn` (bool): Enable flash attention (default: False)
- `n_gpu_layers` (int): Number of layers to offload to GPU (default: 99)

**Returns:**
- `GenerationResult`: Object containing generation results and timing data

### GenerationResult

**Properties:**
- `output_text` (str): Complete generated text
- `n_tokens_generated` (int): Number of tokens generated
- `total_time_ms` (float): Total generation time in milliseconds
- `time_per_token` (List[float]): List of time per token in milliseconds
- `average_time_per_token` (float): Average time per token
- `tokens_per_second` (float): Calculated tokens per second
- `error_code` (int): 0 for success, non-zero for error
- `is_success` (bool): Whether generation was successful

## Differences from custom_gen

| Feature | gear_decode | custom_gen |
|---------|-------------|------------|
| Decode loop | Complete in C++ | Token-by-token from Python |
| Return style | All at once | Streaming |
| Timing data | Per-token list | Per-token in result |
| Use case | Benchmarking, batch analysis | Interactive streaming |

## Example Output

```
================================================================================
GENERATION RESULTS
================================================================================

Output Text:
Python is a high-level, interpreted programming language known for its simplicity and readability.

--------------------------------------------------------------------------------
Statistics:
  Tokens Generated: 15
  Total Time: 3421.45 ms
  Tokens/Second: 4.38
  Average Time/Token: 228.10 ms

Per-Token Timing (ms):
  Token   1:  245.32 ms (  4.08 tok/sec)
  Token   2:  223.15 ms (  4.48 tok/sec)
  Token   3:  219.87 ms (  4.55 tok/sec)
  ...
================================================================================
```

## License

Same as llama.cpp (MIT License)
