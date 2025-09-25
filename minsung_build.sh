#!/usr/bin/env bash
set -euo pipefail

# 1) CMake configure
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_C_FLAGS="-march=armv8.7a" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a" \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DLLAMA_CURL=OFF \
  -B build-android

# 2) Build
cmake --build build-android -j24

# 3) Install
cmake --install build-android --prefix binary_install

# 4) Push to device
adb push binary /data/local/tmp/mskim/

# 5) Set LD_LIBRARY_PATH 
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/mskim/binary/lib:\$LD_LIBRARY_PATH"

echo "== Done =="
