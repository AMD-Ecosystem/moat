<!--
repo: GPUOpen-LibrariesAndSDKs/HIPRT
title: Kernel cache, offload arch, comgr source name and library search order
note: Four runtime-compiler and library-loading defects
-->

## Summary

Four independent defects in the runtime-compilation and library-loading paths, all hit while building against a ROCm SDK. They are unrelated to each other and to the wavefront-width issues reported separately; each is small and self-contained.

All four were checked against HIP RT HEAD on 2026-08-13 and are present there.

## 1. The kernel cache path is built from an unsanitized device name

`Compiler::getCacheFilename` uses the raw device name as a path component. Any device whose name contains a slash writes into a directory that does not exist and the cache write throws.

`AMD Instinct MI250X / MI250` is such a name, so this reproduces on any MI250X.

Fix: sanitize `/` and `\` to `_` before using the name as a path component.

## 2. `addCommonOpts` passes no `--offload-arch`

`Compiler::addCommonOpts` does not pass `--offload-arch`, so the runtime compiler targets its default architecture set. Loading the BVH kernels then fails with `hipErrorInvalidImage` (200) on any architecture outside that set.

Observed on gfx1201.

Fix: pass `--offload-arch=<gcnArch>` for the device actually in use.

## 3. `hiprtcCreateProgram` is given an absolute path as the source name

`Compiler::buildProgram` passes the full absolute path as the `hiprtcCreateProgram` source name. On Windows, comgr fails **silently** when that name starts with a drive letter: the return code is 6 and the log is empty, which makes this expensive to diagnose.

Fix: pass only the file name.

## 4. `LoadLibraryA` picks the display-driver runtime over the ROCm SDK

`hiprt.cpp` and `hiprt_libpath.h` load `amdhip64` and `hiprtc` by bare name through `LoadLibraryA`, which follows the legacy search order and finds the System32 display-driver copy in preference to the ROCm SDK one that the application was built against. Mixing the two produces failures far from this call.

Fix: build full library paths from `ROCM_PATH` / `HIP_PATH` and load those.

Note: the same pattern appears in `contrib/Orochi/contrib/hipew/src/hipew.cpp`, so one hunk of our local fix lands in vendored Orochi rather than in `hiprt/`. We are happy to split that half into an Orochi issue if you would prefer.

Separately, `hiprt_libpath.h`'s `g_hiprtc_paths` list stops at `hiprtc0707.dll`, so newer ROCm runtimes are not found by name at all. Extending the list works, but it is the same enumeration-instead-of-query pattern noted in the wavefront-width report.

## Scope note

We are only reporting defects we found still present upstream. Three other fixes we carry locally -- the `Compiler` `init()` split, `oroSetRawDevice`, and some member-initializer-order and uninitialized-loop-variable warnings -- are **already fixed on HIP RT main** at `e3c01fc` and are backported on our side only because the tag we pinned predates them. Those need nothing from you.

We are happy to open a pull request for any or all of the four above.
