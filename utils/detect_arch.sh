#!/usr/bin/env bash
# Detect this host's AMD GPU arch. Emits KEY=VALUE lines (GFX_ARCH, GFX_TRIPLE,
# PLATFORM) to stdout for `eval`. Exits nonzero if no AMD GPU. Never hardcodes
# the arch; arch varies per container/host.
#
# Linux: rocm_agent_enumerator / rocminfo.
# Windows: set MOAT_OS=windows; detection uses hipInfo (the ROCm `hipInfo[.exe]`
#   tool) found on PATH or pointed to by MOAT_HIPINFO. A Windows host may expose
#   multiple GPUs of different archs; pin one with HIP_VISIBLE_DEVICES (hipInfo
#   honors the mask, so the first reported gcnArchName is the selected device).
# Override: set MOAT_PLATFORM (e.g. windows-gfx1101) to bypass detection entirely.
set -uo pipefail

os="${MOAT_OS:-linux}"

# Explicit override wins (useful on hosts where tooling is awkward to invoke).
if [ -n "${MOAT_PLATFORM:-}" ]; then
  arch="${MOAT_PLATFORM##*-gfx}"; arch="gfx${arch}"
  echo "GFX_ARCH=${arch}"
  echo "GFX_TRIPLE=${arch}"
  echo "PLATFORM=${MOAT_PLATFORM}"
  exit 0
fi

arch=""
triple=""

if [ "$os" = "windows" ]; then
  hipinfo="${MOAT_HIPINFO:-}"
  if [ -z "$hipinfo" ]; then
    if command -v hipInfo >/dev/null 2>&1; then hipinfo="hipInfo"
    elif command -v hipInfo.exe >/dev/null 2>&1; then hipinfo="hipInfo.exe"
    fi
  fi
  if [ -z "$hipinfo" ] || { [ "$hipinfo" != "hipInfo" ] && [ "$hipinfo" != "hipInfo.exe" ] && [ ! -x "$hipinfo" ]; }; then
    echo "detect_arch: hipInfo not found (set MOAT_HIPINFO to hipInfo.exe)" >&2
    exit 1
  fi
  # HIP_VISIBLE_DEVICES masks hipInfo; first gcnArchName is the selected device.
  arch=$("$hipinfo" 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' )
  if [ -z "$arch" ]; then
    echo "detect_arch: hipInfo reported no gcnArchName (no visible AMD GPU?)" >&2
    exit 1
  fi
else
  if command -v rocm_agent_enumerator >/dev/null 2>&1; then
    arch=$(rocm_agent_enumerator 2>/dev/null \
           | grep -E '^gfx[0-9a-f]+' | grep -v '^gfx000$' | sort -u | head -1)
  fi
  if [ -z "$arch" ] && command -v rocminfo >/dev/null 2>&1; then
    arch=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | grep -v '^gfx000$' | sort -u | head -1)
  fi
  if [ -z "$arch" ]; then
    echo "detect_arch: no AMD GPU found (rocm_agent_enumerator/rocminfo)" >&2
    exit 1
  fi
  distinct=$(rocm_agent_enumerator 2>/dev/null | grep -E '^gfx[0-9a-f]+' | grep -v '^gfx000$' | sort -u | wc -l)
  if [ "${distinct:-1}" -gt 1 ]; then
    echo "detect_arch: multiple GPU archs present; using $arch (set HIP_VISIBLE_DEVICES to pin)" >&2
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    triple=$(rocminfo 2>/dev/null | grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-f:+-]+' | head -1)
  fi
fi

echo "GFX_ARCH=${arch}"
echo "GFX_TRIPLE=${triple:-${arch}}"
echo "PLATFORM=${os}-${arch}"
