#!/bin/bash
# build.sh <srcdir> <builddir>
set -e
R=$(hipconfig --rocmpath)
cmake -S "$1" -B "$2" -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DHAVE_TESTS=1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ > "$2.configure.txt" 2>&1
cmake --build "$2" -j64 --target mmseqs > "$2.build.txt" 2>&1
