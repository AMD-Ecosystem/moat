#!/usr/bin/env bash
# Detect this host's AMD GPU arch. Emits KEY=VALUE lines (GFX_ARCH, GFX_TRIPLE,
# PLATFORM) to stdout for `eval`. Exits nonzero if no AMD GPU. Never hardcodes
# the arch; arch varies per container/host.
#
# OS comes from `uname` -- Git Bash, MSYS and Cygwin all identify themselves, so
# a Windows host needs no configuration to be recognized as one. WSL reports
# Linux, which is correct: it runs the Linux ROCm stack. MOAT_OS overrides.
#
# Linux: rocm_agent_enumerator / rocminfo.
# Windows: hipInfo, falling back to amdgpu-arch. Both are searched on PATH and
#   then under HIP_PATH/ROCM_PATH/ROCM_HOME, an active venv's pip-packaged SDK,
#   beside any reachable hipcc, and the standard installer roots. MOAT_HIPINFO
#   names one outright. A Windows host may expose multiple GPUs of different
#   archs; pin one with HIP_VISIBLE_DEVICES (hipInfo honors the mask, so the
#   first reported gcnArchName is the selected device).
# Host-local settings: `.moat.local` in the repo root is sourced when it exists
#   and is gitignored. It is the place to record where a non-standard SDK lives
#   on one machine, so detection stays automatic without putting a host path in
#   the shared repo. The caller's environment still wins over the file.
# Override: set MOAT_PLATFORM (e.g. windows-gfx1101) to bypass detection entirely.
set -uo pipefail

# Sourced before anything reads the environment, so a host whose SDK is not in
# any standard place configures itself once instead of every agent rediscovering
# it. Explicit environment beats the file: an operator overriding a variable for
# one command should not be silently undone by a checked-out-of-band default.
_env_os="${MOAT_OS:-}"; _env_platform="${MOAT_PLATFORM:-}"; _env_hipinfo="${MOAT_HIPINFO:-}"
_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)
_local_env="${MOAT_LOCAL_ENV:-${_repo_root:-.}/.moat.local}"
if [ -f "$_local_env" ]; then
  # shellcheck disable=SC1090
  . "$_local_env" || echo "detect_arch: failed to source $_local_env" >&2
fi
[ -n "$_env_os" ] && MOAT_OS="$_env_os"
[ -n "$_env_platform" ] && MOAT_PLATFORM="$_env_platform"
[ -n "$_env_hipinfo" ] && MOAT_HIPINFO="$_env_hipinfo"

# Explicit override wins (useful on hosts where tooling is awkward to invoke).
if [ -n "${MOAT_PLATFORM:-}" ]; then
  arch="${MOAT_PLATFORM##*-gfx}"; arch="gfx${arch}"
  echo "GFX_ARCH=${arch}"
  echo "GFX_TRIPLE=${arch}"
  echo "PLATFORM=${MOAT_PLATFORM}"
  exit 0
fi

os="${MOAT_OS:-}"
if [ -z "$os" ]; then
  case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT*) os=windows ;;
    *) os=linux ;;
  esac
  # Belt and braces for a bash whose uname reports something unexpected: Windows
  # sets OS=Windows_NT for every process, and WSL -- the one Linux that could see
  # it -- does not inherit it.
  [ "$os" = "linux" ] && [ "${OS:-}" = "Windows_NT" ] && os=windows
fi

arch=""
triple=""

# Directories worth searching for a Windows ROCm tool, most specific first. Each
# is a guess that costs a stat; none is a hardcoded fleet fact.
win_bin_dirs() {
  local root d
  for root in "${HIP_PATH:-}" "${ROCM_PATH:-}" "${ROCM_HOME:-}"; do
    [ -n "$root" ] || continue
    printf '%s\n' "$root/bin" "$root/lib/llvm/bin"
  done
  # TheRock / pip-packaged SDK in an active virtualenv.
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    printf '%s\n' "$VIRTUAL_ENV/Lib/site-packages/_rocm_sdk_core/bin" \
                  "$VIRTUAL_ENV/Lib/site-packages/_rocm_sdk_core/lib/llvm/bin"
  fi
  # Beside a compiler driver that is already reachable.
  for d in hipcc hipconfig; do
    d=$(command -v "$d" 2>/dev/null) || continue
    [ -n "$d" ] && printf '%s\n' "$(dirname "$d")"
  done
  # Standard installer layout.
  for d in "/c/Program Files/AMD/ROCm"/*/bin "${PROGRAMFILES:-}/AMD/ROCm"/*/bin; do
    [ -d "$d" ] && printf '%s\n' "$d"
  done
}

# PATH first, then the candidates. Prints the resolved path; nonzero if absent.
find_win_tool() {
  local name="$1" cand dir
  for cand in "$name" "$name.exe"; do
    cand=$(command -v "$cand" 2>/dev/null) && [ -n "$cand" ] && { printf '%s\n' "$cand"; return 0; }
  done
  while IFS= read -r dir; do
    [ -n "$dir" ] || continue
    for cand in "$dir/$name" "$dir/$name.exe"; do
      [ -x "$cand" ] && { printf '%s\n' "$cand"; return 0; }
    done
  done < <(win_bin_dirs)
  return 1
}

if [ "$os" = "windows" ]; then
  tool="${MOAT_HIPINFO:-}"
  if [ -n "$tool" ]; then
    if [ ! -x "$tool" ] && ! command -v "$tool" >/dev/null 2>&1; then
      echo "detect_arch: MOAT_HIPINFO=$tool is not an executable" >&2
      exit 1
    fi
  else
    # hipInfo is preferred because it honors HIP_VISIBLE_DEVICES, which is how a
    # multi-GPU Windows host pins the arch it means to validate. amdgpu-arch is
    # the fallback: it ships with the LLVM toolchain when hipInfo is absent, and
    # reports one arch per line rather than a masked view.
    tool=$(find_win_tool hipInfo) || tool=$(find_win_tool amdgpu-arch) || tool=""
  fi
  if [ -z "$tool" ]; then
    echo "detect_arch: no hipInfo or amdgpu-arch found on PATH, under" >&2
    echo "  HIP_PATH/ROCM_PATH/ROCM_HOME, in an active venv's ROCm SDK, or in" >&2
    echo "  the standard install roots. Point MOAT_HIPINFO at hipInfo.exe --" >&2
    echo "  record it in ${_local_env} to make it stick on this host." >&2
    exit 1
  fi
  arch=$("$tool" 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | grep -v '^gfx000$' | head -1)
  if [ -z "$arch" ]; then
    echo "detect_arch: $tool reported no AMD GPU (no visible device?)" >&2
    exit 1
  fi
else
  if command -v rocm_agent_enumerator >/dev/null 2>&1; then
    # -o keeps only the arch token: an enumerator that appends target features
    # (gfx90a:sramecc+:xnack-) would otherwise yield a PLATFORM moatlib refuses.
    arch=$(rocm_agent_enumerator 2>/dev/null \
           | grep -oE '^gfx[0-9a-f]+' | grep -v '^gfx000$' | sort -u | head -1)
  fi
  if [ -z "$arch" ] && command -v rocminfo >/dev/null 2>&1; then
    arch=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | grep -v '^gfx000$' | sort -u | head -1)
  fi
  if [ -z "$arch" ]; then
    echo "detect_arch: no AMD GPU found (rocm_agent_enumerator/rocminfo)" >&2
    exit 1
  fi
  distinct=$(rocm_agent_enumerator 2>/dev/null | grep -oE '^gfx[0-9a-f]+' | grep -v '^gfx000$' | sort -u | wc -l)
  if [ "${distinct:-1}" -gt 1 ]; then
    echo "detect_arch: multiple GPU archs present; using $arch (set MOAT_PLATFORM=linux-<gfx> to pin -- HIP_VISIBLE_DEVICES does not mask the HSA tools this reads)" >&2
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    triple=$(rocminfo 2>/dev/null | grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-f:+-]+' | head -1)
  fi
fi

echo "GFX_ARCH=${arch}"
echo "GFX_TRIPLE=${triple:-${arch}}"
echo "PLATFORM=${os}-${arch}"
