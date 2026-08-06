# fp32 element-read linear-texture probe

Establishes the per-arch truth on the HARDWARE texture path (software fallback
OFF) for the TIGRE / popsift texture class: an fp32, `cudaReadModeElementType`
3D texture created with `cudaFilterModeLinear`.

Three possible verdicts:
- `REJECTED` -- `hipCreateTextureObject` returns an error (the safe failure).
- `LINEAR` -- creation OK and the sampler interpolates (hardware filtering works).
- `POINT` -- creation OK but the sampler silently point-samples (wrong result,
  NO error -- the dangerous case that a software fallback must guard against).

## Run it (each host records its own line)

Linux (gfx90a / gfx1100):
```
hipcc --offload-arch=<arch> -o tex_filter_probe tex_filter_probe.cpp
HIP_VISIBLE_DEVICES=<dev> ./tex_filter_probe
```

Windows (gfx1201, TheRock ROCm; pick the single discrete GPU via hipInfo):
```
<rocm>/lib/llvm/bin/clang++.exe -x hip --offload-arch=gfx1201 --hip-path=<rocm> -fms-runtime-lib=dll -o tex_filter_probe.exe tex_filter_probe.cpp
tex_filter_probe.exe
```

## Results

| arch | ROCm | verdict |
| --- | --- | --- |
| gfx90a (CDNA2) | 7.2.1 | REJECTED ("operation not supported") |
| gfx1100 (RDNA3) | 7.2.1 | LINEAR (sample@2.0 = 1.5000; hardware trilinear works) |
| gfx1201 (RDNA4) | 7.14 | LINEAR (sample@2.0 = 1.5000; hardware trilinear works) |

Record your host's verdict in this table (commit on top), then the TIGRE PR's
texture claim can be scoped precisely and PORTING_GUIDE's "hardware linear-filter
texture" entry can state arch scope instead of "NOT confirmed beyond gfx90a".
