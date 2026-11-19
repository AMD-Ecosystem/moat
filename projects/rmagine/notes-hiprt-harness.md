# rmagine: out-of-tree HIPRT validation harness

The fork has no in-tree test for `rmagine_hiprt` (the backend is opt-in and has no install rules), so its GPU check is this out-of-tree harness. The 2026-06-05 harness was never committed and was lost with its host's `agent_space`; this one was rewritten on 2026-11-19 (linux-gfx90a validation of `moat-fix-25` at `e1b7ac5`) and is kept here so the next validator can rebuild it. (The `hiprtSetLogLevel` call is diagnostic only.) It is stricter than the original: instead of hit counts plus one centre range, every ray of every simulator is compared with a CPU ray-triangle reference that composes the pose with rmagine's own `Transform`/`Quaternion` math (hit/miss must agree; range within 1e-4 relative; a miss must carry a value above `range.max`). Scene: a 2x2 quad at z=2 (two triangles). Three poses (identity; translated; rotated R_y(-pi/2) so sensor +X looks along map +Z) and a non-identity `Tsb`, so `Nposes > 1` and the Tbm*Tsb composition in the kernels are exercised. Pinhole follows the HIPRT kernel's convention (optical axis +Z, `x=(u-cx)/fx`, `y=(v-cy)/fy`).

## HIPRT SDK (job-local)

```
git clone https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT <job>/deps/HIPRT
cd <job>/deps/HIPRT && git checkout 243d376 && git submodule update --init --recursive   # "Update to 3.1.1.edfd561", untagged
PATH=$ROCM/bin:$PATH cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBITCODE=OFF \
  -DNO_UNITTEST=ON -DHIP_PATH=$ROCM -DFORCE_DISABLE_CUDA=ON
cmake --build build -j 42        # -> dist/bin/Release/libhiprt0300164.so
```

Version: the port's CMake looks for `libhiprt0300164.so` (or `hiprt03001`): HIPRT names its library `hiprt` + zero-padded `major*1000+minor` + `64`, so only 3.1.x matches; 3.2 (`hiprt03002`) would make CMake skip `rmagine_hiprt` with a warning.

Use 3.1.1, not the 3.1.0 tag (`3.1.0.cb09c56`), on MI250X. HIPRT 3.1.0 puts the device's marketing name into the kernel-cache file name, and on MI250X that is `AMD Instinct MI250X / MI250`: the `/` makes `cache/<hash>.v.AMD Instinct MI250X / MI250.<drv>_64.bin` a path into a directory that does not exist, `Compiler::cacheBinaryToFile` throws `Unable to open`, and `hiprtBuildGeometry` returns `hiprtErrorInternal` (2) with nothing logged even at `hiprtLogLevelInfo`. Found with `gdb -ex 'catch throw'` and `strace -e openat`. HIPRT fixed it in 3.1.1 (243d376, 2026-10-10: unsafe characters in the name become `_`), which keeps the `hiprt0300164` library name, so no patch is needed. A one-line local patch on 3.1.0 (`std::replace(deviceName.begin(), deviceName.end(), '/', '_')` in `Compiler::getCacheFilename`) gave identical results before 3.1.1 was found. Earlier gfx90a runs (2026-06, ROCm 7.2.1) used an unrecorded HIPRT build; the error-2 failure they blamed on a missing `HIPRT_PATH` may have had this cause or another one.

Runtime: HIPRT dlopens `libhiprtc.so.7` / `libamdhip64.so.7` by bare name (or `/opt/rocm/lib`), so with a pip ROCm SDK `$ROCM/lib` must be on `LD_LIBRARY_PATH`; without it Orochi's hiprtc pointers stay null and the BVH build segfaults at address 0 in `CreateAndCompileProgram`. HIPRT writes its kernel cache to `./cache` relative to the working directory: run from a scratch directory, never the repository root (`utils/timeit.sh` changes to the repository root, so delete the `cache/` it leaves there).

## build.sh

```
#!/bin/bash
# usage: build.sh <rmagine src> <rmagine build> <HIPRT_PATH> <ROCM>
set -e
S=$1/src; B=$2; HP=$3; ROCM=$4; H=$(dirname "$0")
g++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__=1 -DUSE_HIP=1 -I$S/rmagine_core/include -I$S/rmagine_cuda/include \
  -I$S/rmagine_hiprt/include -I$B/include -I/usr/include/eigen3 -I$ROCM/include -I$HP \
  $H/test_all_simulators.cpp -o $H/test_all_simulators -L$B/lib -lrmagine_hiprt -lrmagine-cuda -lrmagine-core \
  -L$HP/dist/bin/Release -lhiprt0300164 -L$ROCM/lib -lamdhip64 -Wl,-rpath,$B/lib:$ROCM/lib:$HP/dist/bin/Release
```

Run: `HIPRT_PATH=<hiprt> LD_LIBRARY_PATH=<hiprt>/dist/bin/Release:$ROCM/lib ./test_all_simulators` (exit 0 = pass; `HIPRT_INFO=1` adds HIPRT info logging).

## test_all_simulators.cpp

```cpp
// Out-of-tree validation harness for rmagine_hiprt (not part of the fork).
// Scene: 2x2 quad at z=2 (x,y in [-1,1]), two triangles.
// Every simulator's per-ray hit/range is compared against a CPU ray-triangle
// reference built from rmagine's own Transform/Quaternion math.
#include <rmagine/util/hiprt/HiprtContext.hpp>
#include <rmagine/map/hiprt/HiprtMesh.hpp>
#include <rmagine/map/hiprt/HiprtScene.hpp>
#include <rmagine/simulation/PinholeSimulatorHiprt.hpp>
#include <rmagine/simulation/SphericalSimulatorHiprt.hpp>
#include <rmagine/simulation/O1DnSimulatorHiprt.hpp>
#include <rmagine/simulation/OnDnSimulatorHiprt.hpp>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/sensor_models.h>
#include <rmagine/math/types.h>

#include <hiprt/hiprt.h>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace rmagine;

static const Vector V[4] = {{-1, -1, 2}, {1, -1, 2}, {1, 1, 2}, {-1, 1, 2}};
static const unsigned F[2][3] = {{0, 1, 2}, {0, 2, 3}};

static bool tri(const Vector& o, const Vector& d, const Vector& a, const Vector& b,
                const Vector& c, float& t)
{
  Vector e1 = b - a, e2 = c - a, p = d.cross(e2);
  float det = e1.dot(p);
  if (std::fabs(det) < 1e-12f) return false;
  float inv = 1.0f / det;
  Vector s = o - a;
  float u = s.dot(p) * inv;
  if (u < 0.0f || u > 1.0f) return false;
  Vector q = s.cross(e1);
  float v = d.dot(q) * inv;
  if (v < 0.0f || u + v > 1.0f) return false;
  t = e2.dot(q) * inv;
  return true;
}

// returns range or -1 for miss; also flags rays near an edge (ambiguous)
static float cpu_trace(const Vector& o, const Vector& d, float rmin, float rmax, bool& near_edge)
{
  near_edge = false;
  float best = -1.0f;
  for (auto& f : F) {
    float t;
    if (tri(o, d, V[f[0]], V[f[1]], V[f[2]], t) && t >= rmin && t <= rmax)
      if (best < 0 || t < best) best = t;
  }
  if (std::fabs(d.z) > 1e-6f) {
    float t = (2.0f - o.z) / d.z;
    if (t > 0) {
      float x = o.x + d.x * t, y = o.y + d.y * t;
      if (std::fabs(std::fabs(x) - 1.0f) < 1e-4f && std::fabs(y) <= 1.0f + 1e-4f) near_edge = true;
      if (std::fabs(std::fabs(y) - 1.0f) < 1e-4f && std::fabs(x) <= 1.0f + 1e-4f) near_edge = true;
    }
  }
  return best;
}

struct Res { unsigned rays = 0, hits = 0, mism = 0, skipped = 0; double sum = 0; float maxerr = 0; };

static Res compare(const char* name, const std::vector<float>& rg, const std::vector<uint8_t>& ht,
                   unsigned nposes, unsigned nper, float rmax,
                   const std::function<void(unsigned, unsigned, Vector&, Vector&)>& ray_m,
                   float rmin)
{
  Res r;
  for (unsigned p = 0; p < nposes; ++p)
    for (unsigned i = 0; i < nper; ++i) {
      unsigned g = p * nper + i;
      Vector o, d;
      ray_m(p, i, o, d);
      bool edge;
      float ref = cpu_trace(o, d, rmin, rmax, edge);
      bool gpu_hit = ht[g] != 0;
      r.rays++;
      if (gpu_hit) { r.hits++; r.sum += rg[g]; }
      if (edge) { r.skipped++; continue; }
      bool ref_hit = ref >= 0;
      if (ref_hit != gpu_hit) { r.mism++; continue; }
      if (ref_hit) {
        float e = std::fabs(rg[g] - ref);
        if (e > r.maxerr) r.maxerr = e;
        if (e > 1e-4f * std::max(1.0f, ref)) r.mism++;
      } else if (rg[g] <= rmax) {
        r.mism++;  // a miss must carry the invalid value (> range max)
      }
    }
  printf("  %-28s rays=%u hits=%u mean_range=%.6f max_abs_err=%.3g edge_skipped=%u mismatches=%u -> %s\n",
         name, r.rays, r.hits, r.hits ? r.sum / r.hits : 0.0, r.maxerr, r.skipped, r.mism,
         r.mism ? "FAIL" : "PASS");
  return r;
}

template <typename T>
static std::vector<T> download(const Memory<T, VRAM_CUDA>& m)
{
  Memory<T, RAM> h = m;
  return std::vector<T>(h.raw(), h.raw() + h.size());
}

int main()
{
  auto ctx = std::make_shared<HiprtContext>();
  if (!ctx->valid()) { printf("HiprtContext invalid\n"); return 1; }

  hiprtSetLogLevel(ctx->handle(), hiprtLogLevelWarn | hiprtLogLevelError | (getenv("HIPRT_INFO") ? hiprtLogLevelInfo : 0));
  auto mesh = std::make_shared<HiprtMesh>();
  Memory<Point, RAM> vh(4);
  for (int i = 0; i < 4; ++i) vh[i] = V[i];
  Memory<Face, RAM> fh(2);
  for (int i = 0; i < 2; ++i) fh[i] = Face{F[i][0], F[i][1], F[i][2]};
  mesh->vertices = vh;
  mesh->faces = fh;
  mesh->computeFaceNormals();
  mesh->commit();

  auto scene = std::make_shared<HiprtScene>(ctx);
  scene->add(mesh);
  scene->commit();

  // Poses: identity, a translated pose, and a pose rotated so sensor +X looks along map +Z.
  Quaternion qy;  // R_y(-pi/2): x -> +z
  qy.x = 0; qy.y = -std::sin(M_PI / 4); qy.z = 0; qy.w = std::cos(M_PI / 4);
  Transform T0 = Transform::Identity();
  Transform T1 = Transform::Identity(); T1.t = Vector{0.3f, -0.2f, 0.5f};
  Transform T2 = Transform::Identity(); T2.R = qy; T2.t = Vector{0.1f, 0.1f, 0.0f};
  Memory<Transform, RAM> Tbm_h(3);
  Tbm_h[0] = T0; Tbm_h[1] = T1; Tbm_h[2] = T2;
  Memory<Transform, VRAM_CUDA> Tbm = Tbm_h;
  Transform Tsb = Transform::Identity();
  Tsb.t = Vector{0.0f, 0.0f, 0.05f};

  auto to_map = [&](unsigned p, const Vector& os, const Vector& ds, Vector& o, Vector& d) {
    Transform Tsm = Tbm_h[p] * Tsb;
    o = Tsm * os;
    d = Tsm.R * ds;
  };

  unsigned fails = 0;
  const float rmin = 0.0f, rmax = 100.0f;

  {  // Pinhole (kernel convention: optical axis +Z, x=(u-cx)/fx, y=(v-cy)/fy)
    PinholeModel m;
    m.width = 16; m.height = 12; m.range = {rmin, rmax};
    m.f[0] = 6.0f; m.f[1] = 6.0f; m.c[0] = 7.5f; m.c[1] = 5.5f;
    PinholeSimulatorHiprt sim(scene);
    sim.setModel(m); sim.setTsb(Tsb);
    auto rg = download(sim.simulateRanges(Tbm));
    auto ht = download(sim.simulateHits(Tbm));
    Res r = compare("PinholeSimulatorHiprt", rg, ht, 3, m.width * m.height, rmax,
      [&](unsigned p, unsigned i, Vector& o, Vector& d) {
        unsigned u = i % m.width, v = i / m.width;
        Vector ds{(float(u) - m.c[0]) / m.f[0], (float(v) - m.c[1]) / m.f[1], 1.0f};
        to_map(p, Vector{0, 0, 0}, ds.normalize(), o, d);
      }, rmin);
    fails += r.mism != 0 || r.hits == 0;
  }
  {  // Spherical
    SphericalModel m;
    m.phi.min = -0.6f; m.phi.inc = 0.05f; m.phi.size = 25;
    m.theta.min = -0.8f; m.theta.inc = 0.04f; m.theta.size = 41;
    m.range = {rmin, rmax};
    SphericalSimulatorHiprt sim(scene);
    sim.setModel(m); sim.setTsb(Tsb);
    auto rg = download(sim.simulateRanges(Tbm));
    auto ht = download(sim.simulateHits(Tbm));
    Res r = compare("SphericalSimulatorHiprt", rg, ht, 3, m.phi.size * m.theta.size, rmax,
      [&](unsigned p, unsigned i, Vector& o, Vector& d) {
        unsigned th = i % m.theta.size, ph = i / m.theta.size;
        float phi = m.phi.min + ph * m.phi.inc, theta = m.theta.min + th * m.theta.inc;
        Vector ds{std::cos(phi) * std::cos(theta), std::cos(phi) * std::sin(theta), std::sin(phi)};
        to_map(p, Vector{0, 0, 0}, ds, o, d);
      }, rmin);
    fails += r.mism != 0 || r.hits == 0;
  }
  {  // O1Dn
    O1DnModel m;
    m.width = 20; m.height = 10; m.range = {rmin, rmax};
    m.orig = Vector{0.02f, -0.03f, 0.1f};
    m.dirs.resize(m.width * m.height);
    for (unsigned i = 0; i < m.width * m.height; ++i) {
      float a = -0.9f + 1.8f * (i % m.width) / (m.width - 1);
      float b = -0.7f + 1.4f * (i / m.width) / (m.height - 1);
      m.dirs[i] = Vector{a, b, 1.0f}.normalize();
    }
    O1DnSimulatorHiprt sim(scene);
    sim.setModel(m); sim.setTsb(Tsb);
    auto rg = download(sim.simulateRanges(Tbm));
    auto ht = download(sim.simulateHits(Tbm));
    Res r = compare("O1DnSimulatorHiprt", rg, ht, 3, m.width * m.height, rmax,
      [&](unsigned p, unsigned i, Vector& o, Vector& d) { to_map(p, m.orig, m.dirs[i], o, d); }, rmin);
    fails += r.mism != 0 || r.hits == 0;
  }
  {  // OnDn
    OnDnModel m;
    m.width = 12; m.height = 9; m.range = {rmin, rmax};
    m.origs.resize(m.width * m.height);
    m.dirs.resize(m.width * m.height);
    for (unsigned i = 0; i < m.width * m.height; ++i) {
      float x = -1.5f + 3.0f * (i % m.width) / (m.width - 1);
      float y = -1.3f + 2.6f * (i / m.width) / (m.height - 1);
      m.origs[i] = Vector{x, y, 0.0f};
      m.dirs[i] = Vector{0.1f * x, -0.05f * y, 1.0f}.normalize();
    }
    OnDnSimulatorHiprt sim(scene);
    sim.setModel(m); sim.setTsb(Tsb);
    auto rg = download(sim.simulateRanges(Tbm));
    auto ht = download(sim.simulateHits(Tbm));
    Res r = compare("OnDnSimulatorHiprt", rg, ht, 3, m.width * m.height, rmax,
      [&](unsigned p, unsigned i, Vector& o, Vector& d) { to_map(p, m.origs[i], m.dirs[i], o, d); }, rmin);
    fails += r.mism != 0 || r.hits == 0;
  }

  printf(fails ? "=== %u simulator(s) FAILED ===\n" : "=== ALL 4 HIPRT SIMULATORS PASSED ===\n", fails);
  return fails ? 1 : 0;
}
```
