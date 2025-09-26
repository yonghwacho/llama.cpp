#!/usr/bin/env bash
set -euo pipefail

# 첫 번째 인자가 "new"인지 확인
if [[ "${1:-}" == "new" ]]; then
  echo "== CMake configure =="
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
else
  echo "== Skip CMake configure (use existing build dir) =="
fi

# 2) Build
echo "== Build =="
cmake --build build-android -j24

# 3) Install
echo "== Install =="
cmake --install build-android --prefix binary_install

# 4) Push to device
echo "== Push binaries =="
adb push binary_install /data/local/tmp/mskim/

# 5) Set LD_LIBRARY_PATH
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/mskim/binary_install/lib:\$LD_LIBRARY_PATH"

echo "== Done =="

