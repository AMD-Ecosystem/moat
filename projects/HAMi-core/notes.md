# HAMi-core notes

## What it is

HAMi-core (Project-HAMi/HAMi-core) is a `dlsym`/`LD_PRELOAD`-style interposer library
that sits between `libcudart.so` and `libcuda.so`. It intercepts CUDA driver API calls
(`cuInit`, `cuDeviceGet*`, `cuCtxCreate_v2`, memory-allocation entry points, etc; see
`src/cuda/hook.c`) to enforce per-container device memory limits and time-sliced
compute-utilization limits inside Kubernetes pods, without changing the application or
the driver. It is consumed by the parent HAMi project (a Kubernetes GPU-sharing
scheduler) and by volcano-sh/devices. There is no GPU compute of its own: the two `.cu`
files in the repo (`test/test_multi_gpu_utilization.cu`, `test/test_runtime_launch.cu`)
are trivial test harnesses, not kernels to translate. This is symbol interception of a
vendor runtime's interception surface, not a kernels-and-libraries port.

## Licence

No LICENSE file exists anywhere in the repository (checked the full git tree, 95
entries, README, README_CN, README_JA, CONTRIBUTING.md -- none carry licence text, and
no source file carries a header). GitHub's `/license` endpoint returns 404 and
`licenseInfo` is null, which here reflects a genuine absence rather than a parse
failure (contrast the sibling parent repo `Project-HAMi/HAMi`, which IS Apache-2.0 and
parses cleanly). Recorded as `status.json.license_spdx = "no licence file"`. Per
`config/licenses.toml`, "not open source at all... unlicensed" is tier 4: it always
waits for a person before anything is offered upstream. Moot here given the duplicate
finding below, but recorded as a fact regardless.

## Duplicate effort -- this is the deciding finding

`Project-HAMi/amd-hami-core` already exists, in the SAME upstream org, and is a
functioning AMD-native equivalent:

- Description: "hami-core for amd-device". Builds an `LD_AUDIT` library
  (`libamvgpu.so`) that intercepts HIP API calls (`hipMalloc`, `hipFree`,
  `hipMemGetInfo`) for per-pod memory limiting, and uses `ROC_GLOBAL_CU_MASK` (a native
  ROCm mechanism) for compute-unit partitioning instead of re-implementing HAMi-core's
  own time-slice scheduler.
- 4 commits, 2026-03-26 through 2026-04-02 (~4 months old as of this screen, no
  further commits since -- looks parked but complete for its stated scope, not
  abandoned mid-effort).
- README states "Verified on AMD Instinct MI300X (192GB HBM3e), ROCm 6.2, 7.0, 7.1,
  7.2" -- i.e. already validated on real GPU across four ROCm releases, which is
  further along than a MOAT screen would normally get before the planner even opens.
- Not linked from HAMi-core's own README (the cheap grep-for-amd/rocm/hip check on
  THIS repo is negative), which is exactly the pattern
  `assess-existing-support.md` warns about: "the AMD port is frequently a
  separately-named project, not a fork-of." It only turns up via an org-level search
  for sibling repos, not a fork check or a README grep on the candidate itself.
- It also carries no LICENSE file (same as HAMi-core), so it inherits the identical
  licence gap -- not a reason to treat it as less authoritative, since it is written
  and owned by the Project-HAMi org itself, the same org that owns HAMi-core.

Authoritativeness: this is about as authoritative as a non-linked companion project
gets -- same GitHub org, same functional scope (per-pod memory limit + compute
partitioning for the HAMi scheduler), explicit "Verified on MI300X" claim across
multiple ROCm versions. Not a one-off community hack.

## Viability verdict (the underlying question, now moot but worth recording)

Does an equivalent exist / is it even possible on ROCm? Yes, and amd-hami-core proves
it two ways at once:
1. The interception mechanism does not translate 1:1 -- ROCm has no direct driver-level
   analogue of CUDA's `cuDevice*`/`cuCtx*` surface that HAMi-core hooks via `dlsym`;
   amd-hami-core instead hooks the HIP runtime layer (`hipMalloc`/`hipFree`/
   `hipMemGetInfo`) via `LD_AUDIT` and reuses a native ROCm primitive
   (`ROC_GLOBAL_CU_MASK`) for the compute-limiting half, rather than re-implementing
   HAMi-core's own userspace time-slice scheduler. So a mechanical CUDA-to-HIP port of
   HAMi-core's actual hook tables would not be the right shape of solution even if
   nothing else existed.
2. Because it already exists, does its own compute, and is already validated on real
   MI300X hardware, there is nothing left to port here for MOAT to add value on.

## Recommendation

Decline. `already-supported`: a working, GPU-verified AMD-native port already lives at
`Project-HAMi/amd-hami-core` in the same upstream org. Any residual gap (parity
features, freshness) is a coordination question with that project, not a MOAT porting
task, and its unlicensed status is a separate, pre-existing fact about it that a MOAT
port would inherit unchanged.

## What could not be determined

- Whether amd-hami-core is feature-complete relative to HAMi-core (e.g. does it cover
  every hook HAMi-core has, such as the NVML-style utilization monitor in
  `src/nvml/`?) -- out of scope for a cheap intake screen; a person deciding the queue
  row can weigh whether a coordination follow-up is worth opening rather than a
  fork-and-port here.
