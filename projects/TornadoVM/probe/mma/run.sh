#!/usr/bin/env bash
# MMA validation for the TornadoVM HIP backend on the GPU of this host, in one command:
#
#   source <TornadoVM checkout>/setvars.sh      # a HIP SDK: make jdk21 BACKEND=hip
#   ulimit -c 0
#   bash projects/TornadoVM/probe/mma/run.sh
#
# 1. Emits the HIP MMA helpers with the SDK's TC (helpers.tcir), compiles harness.hip for this GPU,
#    lists the matrix instructions in its code object, and runs it (then with --mutate, which must fail).
# 2. Runs tornado-selftest and every KernelContext MMA unit-test class.
# Needs TORNADOVM_HOME (setvars.sh) and ROCM_PATH. ARCH overrides the GPU architecture, OUT the
# scratch directory. Exit status 0 only if every step passed.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${TORNADOVM_HOME:?source setvars.sh of a HIP SDK first}"
: "${ROCM_PATH:?set ROCM_PATH to the ROCm root}"
arch="${ARCH:-$("$ROCM_PATH/lib/llvm/bin/amdgpu-arch" 2>/dev/null | head -1)}"
[ -n "$arch" ] || { echo "run.sh: no AMD GPU found (set ARCH)"; exit 2; }
out="${OUT:-$(mktemp -d)}"
mkdir -p "$out"
failed=()

echo "== harness ($arch, scratch $out)"
cp="$(ls "$TORNADOVM_HOME"/share/java/tornado/tc-*.jar | tr '\n' ':')"
preview=$(grep -q 'tornado.jdk.preview=true' "$TORNADOVM_HOME/etc/tornado.jdk" 2>/dev/null && echo --enable-preview || true)
if ! "${JAVA_HOME:+$JAVA_HOME/bin/}java" $preview -cp "$cp" uk.ac.manchester.tornado.tc.codegen.tools.EmitTool --target=hip "$here/helpers.tcir" \
        < /dev/null > "$out/tc_helpers.hip"; then
    echo "run.sh: TC could not emit the helpers"; exit 1
fi
if ! "$ROCM_PATH/bin/hipcc" --offload-arch="$arch" -O2 -std=c++17 -I"$out" "$here/harness.hip" -o "$out/harness"; then
    echo "run.sh: harness.hip does not compile for $arch"; exit 1
fi
"$ROCM_PATH/bin/hipcc" --offload-arch="$arch" --cuda-device-only -S -O2 -std=c++17 -I"$out" "$here/harness.hip" -o "$out/harness-$arch.s"
echo "matrix instructions in the code object:"
grep -oE 'v_(mfma|wmma)[a-z0-9_]*' "$out/harness-$arch.s" | sort | uniq -c
"$out/harness" || failed+=("harness")
"$out/harness" --mutate > "$out/mutate.log" || failed+=("harness --mutate (a wrong reference went unnoticed)")
tail -1 "$out/mutate.log"

echo "== unit tests"
tornado-selftest || failed+=("tornado-selftest")
for class in TestMatrixMultiplicationMMA TestMatrixMultiplicationMMABF16 TestMatrixMultiplicationMMAInt8 TestMatrixMultiplicationMMAFP8 \
        TestMatrixMultiplicationMMACpAsync TestMMAGlobalFragmentLoad TestMMAInt8FragmentElementRead TestMMAFragmentHelper; do
    log="$out/$class.log"
    tornado-test -V "uk.ac.manchester.tornado.unittests.kernelcontext.matrices.$class" > "$log" 2>&1
    summary="$(grep -E '^Test ran' "$log" | tail -1)"
    echo "$class: ${summary:-no summary (see $log)}"
    case "$summary" in
    *"Failed: 0,"*) ;;
    *) failed+=("$class") ;;
    esac
done

if [ ${#failed[@]} -eq 0 ]; then
    echo "== PASS ($arch)"
else
    printf '== FAIL (%s): %s\n' "$arch" "${failed[*]}"
    exit 1
fi
