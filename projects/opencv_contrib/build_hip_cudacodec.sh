#!/usr/bin/env bash
# opencv_contrib HIP build for the cudacodec rocDecode VideoReader (gfx1100).
# FFmpeg ON (the container demuxer feeding the rocDecode parser) and
# WITH_ROCDECODE ON (the hardware-decode back end). Run every build/test step
# with: LIBVA_DRIVER_NAME=radeonsi HIP_VISIBLE_DEVICES=0
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_cudacodec"
SRC_CORE="$SCRIPT_DIR/src-core"
SRC_CONTRIB="$SCRIPT_DIR/src"

BUILD_LIST="${BUILD_LIST:-core,cudev,cudaarithm,cudawarping,cudacodec,imgproc,imgcodecs,videoio,highgui,ts}"
BUILD_TARGET="${BUILD_TARGET:-opencv_test_cudacodec}"
HIP_ARCH="${HIP_ARCH:-gfx1100}"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -G Ninja "$SRC_CORE" \
  -DWITH_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
  -DOPENCV_EXTRA_MODULES_PATH="$SRC_CONTRIB/modules" \
  -DBUILD_LIST="$BUILD_LIST" \
  -DBUILD_TESTS=ON \
  -DWITH_CUDA=OFF \
  -DWITH_OPENCL=OFF \
  -DWITH_PYTHON=OFF \
  -DWITH_FFMPEG=ON \
  -DWITH_ROCDECODE=ON \
  -DWITH_GSTREAMER=OFF \
  "$@"

cmake --build . --target $BUILD_TARGET -j$(nproc)
