"""
Python wrapper for the gear_decode library
This module provides a Python interface to the C++ text generation functions
"""

import ctypes
import os
from pathlib import Path
from typing import List, Optional


class GenerationResult:
    """Python wrapper for the C++ GenerationResult structure"""
    
    def __init__(self, ptr, lib):
        self._ptr = ptr
        self._lib = lib
        
    @property
    def output_text(self) -> str:
        """Get the complete generated output text"""
        text_ptr = self._lib.get_output_text(self._ptr)
        if text_ptr:
            return text_ptr.decode('utf-8', errors='ignore')
        return ""
    
    @property
    def n_tokens_generated(self) -> int:
        """Get the total number of tokens generated"""
        return self._lib.get_n_tokens_generated(self._ptr)
    
    @property
    def total_time_ms(self) -> float:
        """Get the total generation time in milliseconds"""
        return self._lib.get_total_time_ms(self._ptr)
    
    @property
    def error_code(self) -> int:
        """Get the error code (0 = success)"""
        return self._lib.get_error_code(self._ptr)
    
    @property
    def is_success(self) -> bool:
        """Check if generation was successful"""
        return self.error_code == 0
    
    @property
    def time_per_token(self) -> List[float]:
        """Get the list of time per token in milliseconds"""
        count = self._lib.get_time_per_token_count(self._ptr)
        times = []
        for i in range(count):
            times.append(self._lib.get_time_per_token_at(self._ptr, i))
        return times
    
    @property
    def token_start_time(self) -> List[float]:
        """Get the list of token generation start times in seconds since epoch"""
        count = self._lib.get_token_start_time_count(self._ptr)
        times = []
        for i in range(count):
            times.append(self._lib.get_token_start_time_at(self._ptr, i))
        return times
    
    @property
    def average_time_per_token(self) -> float:
        """Calculate average time per token in milliseconds"""
        times = self.time_per_token
        if not times:
            return 0.0
        return sum(times) / len(times)
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second"""
        if self.total_time_ms <= 0:
            return 0.0
        return (self.n_tokens_generated * 1000.0) / self.total_time_ms
    
    def __del__(self):
        """Free the C++ result structure"""
        if self._ptr and self._lib:
            self._lib.free_generation_result(self._ptr)
    
    def __str__(self):
        return f"GenerationResult(tokens={self.n_tokens_generated}, time={self.total_time_ms:.2f}ms, tps={self.tokens_per_second:.2f}, success={self.is_success})"


class GearGenerator:
    """Python interface to the gear_decode generation library"""
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the GearGenerator
        
        Args:
            lib_path: Path to the shared library. If None, will search in common locations.
        """
        if lib_path is None:
            lib_path = self._find_library()
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found at: {lib_path}")
        
        # Load the shared library
        self._lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        
        # GenerationResult* generate_text(const char*, const char*, int, bool, int, bool)
        self._lib.generate_text.argtypes = [
            ctypes.c_char_p,  # model_path
            ctypes.c_char_p,  # prompt_text
            ctypes.c_int,     # n_predict
            ctypes.c_bool,    # use_instruct
            ctypes.c_int,     # n_threads
            ctypes.c_bool     # enable_flash_attn
        ]
        self._lib.generate_text.restype = ctypes.c_void_p
        
        # const char* get_output_text(GenerationResult*)
        self._lib.get_output_text.argtypes = [ctypes.c_void_p]
        self._lib.get_output_text.restype = ctypes.c_char_p
        
        # int get_n_tokens_generated(GenerationResult*)
        self._lib.get_n_tokens_generated.argtypes = [ctypes.c_void_p]
        self._lib.get_n_tokens_generated.restype = ctypes.c_int
        
        # float get_total_time_ms(GenerationResult*)
        self._lib.get_total_time_ms.argtypes = [ctypes.c_void_p]
        self._lib.get_total_time_ms.restype = ctypes.c_float
        
        # int get_error_code(GenerationResult*)
        self._lib.get_error_code.argtypes = [ctypes.c_void_p]
        self._lib.get_error_code.restype = ctypes.c_int
        
        # int get_time_per_token_count(GenerationResult*)
        self._lib.get_time_per_token_count.argtypes = [ctypes.c_void_p]
        self._lib.get_time_per_token_count.restype = ctypes.c_int
        
        # float get_time_per_token_at(GenerationResult*, int)
        self._lib.get_time_per_token_at.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.get_time_per_token_at.restype = ctypes.c_float
        
        # int get_token_start_time_count(GenerationResult*)
        self._lib.get_token_start_time_count.argtypes = [ctypes.c_void_p]
        self._lib.get_token_start_time_count.restype = ctypes.c_int
        
        # double get_token_start_time_at(GenerationResult*, int)
        self._lib.get_token_start_time_at.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.get_token_start_time_at.restype = ctypes.c_double
        
        # void free_generation_result(GenerationResult*)
        self._lib.free_generation_result.argtypes = [ctypes.c_void_p]
        self._lib.free_generation_result.restype = None
    
    def _find_library(self) -> str:
        """Find the shared library in common locations"""
        # Common library names on different platforms
        lib_names = [
            "libgear_decode.so",      # Linux
            "libgear_decode.dylib",   # macOS
            "gear_decode.dll",        # Windows
        ]
        
        # Search paths
        search_paths = [
            # Build directory
            Path(__file__).parent.parent / "build" / "lib",
            Path(__file__).parent.parent / "build" / "bin",
            # Install directory
            Path("/usr/local/lib"),
            Path("/usr/lib"),
            # Current directory
            Path.cwd(),
        ]
        
        for search_path in search_paths:
            for lib_name in lib_names:
                lib_path = search_path / lib_name
                if lib_path.exists():
                    return str(lib_path)
        
        # Default to build directory
        default_path = Path(__file__).parent.parent / "build" / "lib" / "libgear_decode.so"
        return str(default_path)
    
    def generate(
        self,
        model_path: str,
        prompt: str,
        n_predict: int = 32,
        use_instruct: bool = True,
        n_threads: int = 4,
        enable_flash_attn: bool = False
    ) -> GenerationResult:
        """
        Generate text using the llama.cpp model
        
        This function runs the complete decode loop in C++ and returns all results.
        
        Args:
            model_path: Path to the GGUF model file
            prompt: Input prompt text
            n_predict: Maximum number of tokens to generate (default: 32)
            use_instruct: Whether to use instruct mode template (default: True)
            n_threads: Number of threads to use (default: 4)
            enable_flash_attn: Enable flash attention (default: False)
        
        Returns:
            GenerationResult object containing:
            - output_text: Complete generated text
            - n_tokens_generated: Number of tokens generated
            - total_time_ms: Total generation time in milliseconds
            - time_per_token: List of time per token in milliseconds
            - tokens_per_second: Calculated tokens per second
            - error_code: 0 for success, non-zero for error
        
        Example:
            >>> generator = GearGenerator()
            >>> result = generator.generate(
            ...     model_path="/path/to/model.gguf",
            ...     prompt="What is Python?",
            ...     n_predict=50,
            ...     use_instruct=True
            ... )
            >>> if result.is_success:
            ...     print(f"Output: {result.output_text}")
            ...     print(f"Tokens/sec: {result.tokens_per_second:.2f}")
            ...     print(f"Time per token: {result.time_per_token}")
        """
        # Convert Python strings to C strings
        model_path_bytes = model_path.encode('utf-8')
        prompt_bytes = prompt.encode('utf-8')
        
        # Call the C function
        result_ptr = self._lib.generate_text(
            model_path_bytes,
            prompt_bytes,
            n_predict,
            use_instruct,
            n_threads,
            enable_flash_attn
        )
        
        if not result_ptr:
            raise RuntimeError("Failed to create GenerationResult")
        
        # Wrap in Python object
        return GenerationResult(result_ptr, self._lib)


def main():
    """Example usage of the GearGenerator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text using llama.cpp with gear_decode")
    parser.add_argument("-m", "--model", required=True, help="Path to GGUF model file")
    parser.add_argument("-p", "--prompt", default="Hello my name is", help="Input prompt")
    parser.add_argument("-n", "--n-predict", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--use-instruct", action="store_true", default=True, help="Use instruct mode")
    parser.add_argument("--no-instruct", dest="use_instruct", action="store_false", help="Disable instruct mode")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--flash-attn", action="store_true", default=False, help="Enable flash attention")
    parser.add_argument("--lib", help="Path to gear_decode shared library")
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = GearGenerator(lib_path=args.lib)
        
        print(f"Model: {args.model}")
        print(f"Prompt: {args.prompt}")
        print(f"Generating up to {args.n_predict} tokens with {args.threads} threads...")
        print(f"Flash Attention: {args.flash_attn}")
        print("-" * 80)
        
        # Generate text
        result = generator.generate(
            model_path=args.model,
            prompt=args.prompt,
            n_predict=args.n_predict,
            use_instruct=args.use_instruct,
            n_threads=args.threads,
            enable_flash_attn=args.flash_attn
        )
        
        # Check for errors
        if not result.is_success:
            print(f"Error: Generation failed with code {result.error_code}")
            return 1
        
        # Display results
        print("\n" + "=" * 80)
        print("GENERATION RESULTS")
        print("=" * 80)
        print(f"\nOutput Text:\n{result.output_text}")
        print("\n" + "-" * 80)
        print(f"Statistics:")
        print(f"  Tokens Generated: {result.n_tokens_generated}")
        print(f"  Total Time: {result.total_time_ms:.2f} ms")
        print(f"  Tokens/Second: {result.tokens_per_second:.2f}")
        print(f"  Average Time/Token: {result.average_time_per_token:.2f} ms")
        
        # Show per-token timing
        print(f"\nPer-Token Timing (ms):")
        times = result.time_per_token
        for i, time_ms in enumerate(times):
            tps = 1000.0 / time_ms if time_ms > 0 else 0
            print(f"  Token {i+1:3d}: {time_ms:7.2f} ms ({tps:6.2f} tok/sec)")
        
        print("\n" + "=" * 80)
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
