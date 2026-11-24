// Headless ANARI validation harness for the visionaray devices.
// Scenes: triangle surface with one directional light; 32^3 structuredRegular
// radial blob (nearest, linear); NanoVDB fog sphere r=50 (nearest, linear);
// 216 Gaussian splats. 256x256 UFIXED8_RGBA_SRGB.
// Each scene is rendered RUNS times per device, each run with a fresh world and
// frame; PASS = every run renders an image that is non-empty and has more than
// one distinct colour. Run-to-run identity is reported, not gated. When two
// devices are given, the first device's image is compared against the second's
// (mean abs diff per channel, 0..255).
// env: RUNS (default 8), ONLY (substring of scene names), RENDERER (renderer
// subtype, default "default" = pathtrace), PPM_PREFIX (write the first run's
// image per device to <prefix>-<device>.ppm), NO_LIGHT=1 (surface scene
// without its directional light), MODE (renderer "mode" parameter, e.g. Ng).
#include <anari/anari.h>
#include <chrono>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/GridHandle.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const int W = 256, H = 256;

static void statusFunc(const void *, ANARIDevice, ANARIObject, ANARIDataType,
    ANARIStatusSeverity sev, ANARIStatusCode, const char *msg)
{
  if (sev <= ANARI_SEVERITY_WARNING)
    fprintf(stderr, "[anari %d] %s\n", (int)sev, msg);
}

struct Img { std::vector<uint32_t> px; };

static ANARIWorld buildTriangle(ANARIDevice d)
{
  float verts[] = {-1.f, -1.f, 0.f, 1.f, -1.f, 0.f, 0.f, 1.f, 0.f,
                   -0.8f, 0.6f, -0.5f, -0.2f, 0.9f, -0.5f, -0.6f, 0.2f, -0.5f};
  float colors[] = {1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
                    1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1};
  ANARIGeometry g = anariNewGeometry(d, "triangle");
  ANARIArray1D va = anariNewArray1D(d, verts, nullptr, nullptr, ANARI_FLOAT32_VEC3, 6);
  ANARIArray1D ca = anariNewArray1D(d, colors, nullptr, nullptr, ANARI_FLOAT32_VEC4, 6);
  anariSetParameter(d, g, "vertex.position", ANARI_ARRAY1D, &va);
  anariSetParameter(d, g, "vertex.color", ANARI_ARRAY1D, &ca);
  anariCommitParameters(d, g);
  ANARIMaterial m = anariNewMaterial(d, "matte");
  anariSetParameter(d, m, "color", ANARI_STRING, "color");
  anariCommitParameters(d, m);
  ANARISurface s = anariNewSurface(d);
  anariSetParameter(d, s, "geometry", ANARI_GEOMETRY, &g);
  anariSetParameter(d, s, "material", ANARI_MATERIAL, &m);
  anariCommitParameters(d, s);
  ANARIWorld w = anariNewWorld(d);
  ANARIArray1D sa = anariNewArray1D(d, &s, nullptr, nullptr, ANARI_SURFACE, 1);
  anariSetParameter(d, w, "surface", ANARI_ARRAY1D, &sa);
  if (!getenv("NO_LIGHT")) {
    ANARILight l = anariNewLight(d, "directional");
    float dir[] = {0.f, 0.f, -1.f};
    anariSetParameter(d, l, "direction", ANARI_FLOAT32_VEC3, dir);
    anariCommitParameters(d, l);
    ANARIArray1D la = anariNewArray1D(d, &l, nullptr, nullptr, ANARI_LIGHT, 1);
    anariSetParameter(d, w, "light", ANARI_ARRAY1D, &la);
    anariRelease(d, l); anariRelease(d, la);
  }
  anariCommitParameters(d, w);
  anariRelease(d, va); anariRelease(d, ca); anariRelease(d, g); anariRelease(d, m);
  anariRelease(d, s); anariRelease(d, sa);
  return w;
}

static ANARIWorld wrapVolume(ANARIDevice d, ANARISpatialField f, float lo, float hi)
{
  float tfc[] = {0, 0, 1, 0, 1, 0, 1, 0, 0};
  float tfo[] = {0.f, 0.5f, 1.f};
  ANARIArray1D ca = anariNewArray1D(d, tfc, nullptr, nullptr, ANARI_FLOAT32_VEC3, 3);
  ANARIArray1D oa = anariNewArray1D(d, tfo, nullptr, nullptr, ANARI_FLOAT32, 3);
  ANARIVolume v = anariNewVolume(d, "transferFunction1D");
  anariSetParameter(d, v, "value", ANARI_SPATIAL_FIELD, &f);
  anariSetParameter(d, v, "color", ANARI_ARRAY1D, &ca);
  anariSetParameter(d, v, "opacity", ANARI_ARRAY1D, &oa);
  float vr[] = {lo, hi};
  anariSetParameter(d, v, "valueRange", ANARI_FLOAT32_BOX1, vr);
  anariCommitParameters(d, v);
  ANARIWorld w = anariNewWorld(d);
  ANARIArray1D va = anariNewArray1D(d, &v, nullptr, nullptr, ANARI_VOLUME, 1);
  anariSetParameter(d, w, "volume", ANARI_ARRAY1D, &va);
  anariCommitParameters(d, w);
  anariRelease(d, ca); anariRelease(d, oa); anariRelease(d, v); anariRelease(d, va);
  return w;
}

static ANARIWorld buildStructured(ANARIDevice d, const char *filter, float box[6])
{
  const int N = 32;
  static std::vector<float> data(N * N * N);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        float dx = x - N / 2.f, dy = y - N / 2.f, dz = z - N / 2.f;
        float r = std::sqrt(dx * dx + dy * dy + dz * dz) / (N / 2.f);
        data[(z * N + y) * N + x] = r < 1.f ? 1.f - r : 0.f;
      }
  ANARIArray3D a = anariNewArray3D(d, data.data(), nullptr, nullptr, ANARI_FLOAT32, N, N, N);
  ANARISpatialField f = anariNewSpatialField(d, "structuredRegular");
  anariSetParameter(d, f, "data", ANARI_ARRAY3D, &a);
  float origin[] = {-1.f, -1.f, -1.f}, spacing[] = {2.f / N, 2.f / N, 2.f / N};
  anariSetParameter(d, f, "origin", ANARI_FLOAT32_VEC3, origin);
  anariSetParameter(d, f, "spacing", ANARI_FLOAT32_VEC3, spacing);
  anariSetParameter(d, f, "filter", ANARI_STRING, filter);
  anariCommitParameters(d, f);
  ANARIWorld w = wrapVolume(d, f, 0.f, 1.f);
  anariRelease(d, a); anariRelease(d, f);
  float b[6] = {-1, -1, -1, 1, 1, 1}; memcpy(box, b, sizeof(b));
  return w;
}

static ANARIWorld buildVDB(ANARIDevice d, const char *filter, float box[6])
{
  static nanovdb::GridHandle<nanovdb::HostBuffer> handle =
      nanovdb::tools::createFogVolumeSphere<float>(50.f);
  ANARIArray1D a = anariNewArray1D(d, handle.data(), nullptr, nullptr, ANARI_UINT8, handle.size());
  ANARISpatialField f = anariNewSpatialField(d, "nanovdb");
  anariSetParameter(d, f, "data", ANARI_ARRAY1D, &a);
  anariSetParameter(d, f, "filter", ANARI_STRING, filter);
  anariCommitParameters(d, f);
  ANARIWorld w = wrapVolume(d, f, 0.f, 1.f);
  anariRelease(d, a); anariRelease(d, f);
  float b[6] = {-50, -50, -50, 51, 51, 51}; memcpy(box, b, sizeof(b));
  return w;
}

static ANARIWorld buildSplats(ANARIDevice d, float box[6])
{
  const int G = 6, N = G * G * G;
  static std::vector<float> pos(3 * N), scale(3 * N), rot(4 * N), op(N), fdc(3 * N);
  for (int i = 0; i < N; ++i) {
    int x = i % G, y = (i / G) % G, z = i / (G * G);
    pos[3 * i + 0] = -1.f + 2.f * (x + 0.5f) / G;
    pos[3 * i + 1] = -1.f + 2.f * (y + 0.5f) / G;
    pos[3 * i + 2] = -1.f + 2.f * (z + 0.5f) / G;
    // The device takes scale and opacity as linear values (no exp/sigmoid).
    for (int k = 0; k < 3; ++k) scale[3 * i + k] = 0.05f;
    rot[4 * i + 0] = 1.f; rot[4 * i + 1] = rot[4 * i + 2] = rot[4 * i + 3] = 0.f;
    op[i] = 0.9f;
    fdc[3 * i + 0] = (x / float(G)) - 0.5f;
    fdc[3 * i + 1] = (y / float(G)) - 0.5f;
    fdc[3 * i + 2] = (z / float(G)) - 0.5f;
  }
  ANARIArray1D pa = anariNewArray1D(d, pos.data(), nullptr, nullptr, ANARI_FLOAT32_VEC3, N);
  ANARIArray1D sa = anariNewArray1D(d, scale.data(), nullptr, nullptr, ANARI_FLOAT32_VEC3, N);
  ANARIArray1D ra = anariNewArray1D(d, rot.data(), nullptr, nullptr, ANARI_FLOAT32_VEC4, N);
  ANARIArray1D oa = anariNewArray1D(d, op.data(), nullptr, nullptr, ANARI_FLOAT32, N);
  ANARIArray1D fa = anariNewArray1D(d, fdc.data(), nullptr, nullptr, ANARI_FLOAT32_VEC3, N);
  ANARIVolume v = anariNewVolume(d, "gaussianSplat");
  anariSetParameter(d, v, "vertex.position", ANARI_ARRAY1D, &pa);
  anariSetParameter(d, v, "vertex.scale", ANARI_ARRAY1D, &sa);
  anariSetParameter(d, v, "vertex.rotation", ANARI_ARRAY1D, &ra);
  anariSetParameter(d, v, "vertex.opacity", ANARI_ARRAY1D, &oa);
  anariSetParameter(d, v, "vertex.f_dc", ANARI_ARRAY1D, &fa);
  anariCommitParameters(d, v);
  ANARIWorld w = anariNewWorld(d);
  ANARIArray1D va = anariNewArray1D(d, &v, nullptr, nullptr, ANARI_VOLUME, 1);
  anariSetParameter(d, w, "volume", ANARI_ARRAY1D, &va);
  anariCommitParameters(d, w);
  anariRelease(d, pa); anariRelease(d, sa); anariRelease(d, ra); anariRelease(d, oa);
  anariRelease(d, fa); anariRelease(d, v); anariRelease(d, va);
  float b[6] = {-1, -1, -1, 1, 1, 1}; memcpy(box, b, sizeof(b));
  return w;
}

static bool render(ANARIDevice d, ANARIWorld w, const float box[6], Img &out)
{
  float c[3] = {(box[0] + box[3]) / 2, (box[1] + box[4]) / 2, (box[2] + box[5]) / 2};
  float ext = box[3] - box[0];
  float pos[3] = {c[0] + 0.3f * ext, c[1] + 0.2f * ext, c[2] + 1.6f * ext};
  float dir[3] = {c[0] - pos[0], c[1] - pos[1], c[2] - pos[2]};
  float up[3] = {0, 1, 0};
  float aspect = float(W) / H;
  ANARICamera cam = anariNewCamera(d, "perspective");
  anariSetParameter(d, cam, "position", ANARI_FLOAT32_VEC3, pos);
  anariSetParameter(d, cam, "direction", ANARI_FLOAT32_VEC3, dir);
  anariSetParameter(d, cam, "up", ANARI_FLOAT32_VEC3, up);
  anariSetParameter(d, cam, "aspect", ANARI_FLOAT32, &aspect);
  anariCommitParameters(d, cam);
  ANARIRenderer r = anariNewRenderer(d, getenv("RENDERER") ? getenv("RENDERER") : "default");
  float bg[4] = {0, 0, 0, 1};
  anariSetParameter(d, r, "background", ANARI_FLOAT32_VEC4, bg);
  if (const char *mode = getenv("MODE"))
    anariSetParameter(d, r, "mode", ANARI_STRING, mode);
  anariCommitParameters(d, r);
  ANARIFrame f = anariNewFrame(d);
  unsigned size[2] = {W, H};
  ANARIDataType fmt = ANARI_UFIXED8_RGBA_SRGB;
  anariSetParameter(d, f, "size", ANARI_UINT32_VEC2, size);
  anariSetParameter(d, f, "channel.color", ANARI_DATA_TYPE, &fmt);
  anariSetParameter(d, f, "world", ANARI_WORLD, &w);
  anariSetParameter(d, f, "camera", ANARI_CAMERA, &cam);
  anariSetParameter(d, f, "renderer", ANARI_RENDERER, &r);
  anariCommitParameters(d, f);
  anariRenderFrame(d, f);
  anariFrameReady(d, f, ANARI_WAIT);
  uint32_t fw = 0, fh = 0;
  ANARIDataType t = ANARI_UNKNOWN;
  const uint32_t *px = (const uint32_t *)anariMapFrame(d, f, "channel.color", &fw, &fh, &t);
  bool ok = px && fw == W && fh == H;
  if (ok) out.px.assign(px, px + W * H);
  anariUnmapFrame(d, f, "channel.color");
  anariRelease(d, f); anariRelease(d, r); anariRelease(d, cam);
  return ok;
}

static void writePPM(const char *path, const Img &img)
{
  FILE *fp = fopen(path, "wb");
  if (!fp) return;
  fprintf(fp, "P6\n%d %d\n255\n", W, H);
  for (int y = H - 1; y >= 0; --y)
    for (int x = 0; x < W; ++x) {
      uint32_t p = img.px[y * W + x];
      unsigned char c[3] = {(unsigned char)(p & 0xff), (unsigned char)((p >> 8) & 0xff),
          (unsigned char)((p >> 16) & 0xff)};
      fwrite(c, 1, 3, fp);
    }
  fclose(fp);
}

static void stats(const Img &img, size_t &nonzero, size_t &distinct)
{
  nonzero = 0;
  std::map<uint32_t, int> vals;
  for (uint32_t p : img.px) {
    if (p & 0x00ffffff) ++nonzero;
    if (vals.size() < 100000) vals[p]++;
  }
  distinct = vals.size();
}

static double meanAbsDiff(const Img &a, const Img &b)
{
  double s = 0;
  for (size_t i = 0; i < a.px.size(); ++i)
    for (int k = 0; k < 3; ++k)
      s += std::abs(int((a.px[i] >> (8 * k)) & 0xff) - int((b.px[i] >> (8 * k)) & 0xff));
  return s / (a.px.size() * 3.0);
}

int main(int argc, char **argv)
{
  if (argc < 2) {
    fprintf(stderr, "usage: %s <device> [<reference device>]\n", argv[0]);
    return 2;
  }
  int runs = getenv("RUNS") ? atoi(getenv("RUNS")) : 8;
  const char *only = getenv("ONLY");
  std::vector<std::string> scenes = {"surface", "volume-nearest", "volume-linear",
      "nanovdb-nearest", "nanovdb-linear", "gaussian-splat"};
  std::map<std::string, std::map<std::string, Img>> first;
  int fails = 0;
  for (int di = 1; di < argc; ++di) {
    const char *libName = argv[di];
    ANARILibrary lib = anariLoadLibrary(libName, statusFunc, nullptr);
    if (!lib) { printf("[%s] cannot load library\n", libName); return 1; }
    ANARIDevice d = anariNewDevice(lib, "default");
    anariCommitParameters(d, d);
    for (auto &sc : scenes) {
      if (only && sc.find(only) == std::string::npos) continue;
      bool pass = true;
      size_t nz = 0, dist = 0;
      int stable = 0;
      auto t0 = std::chrono::steady_clock::now();
      for (int run = 0; run < runs; ++run) {
        float box[6];
        ANARIWorld w = nullptr;
        if (sc == "surface") { w = buildTriangle(d); float b[6] = {-1, -1, -0.5f, 1, 1, 0}; memcpy(box, b, sizeof(b)); }
        else if (sc == "volume-nearest") w = buildStructured(d, "nearest", box);
        else if (sc == "volume-linear") w = buildStructured(d, "linear", box);
        else if (sc == "nanovdb-nearest") w = buildVDB(d, "nearest", box);
        else if (sc == "nanovdb-linear") w = buildVDB(d, "linear", box);
        else w = buildSplats(d, box);
        Img img;
        bool ok = render(d, w, box, img);
        anariRelease(d, w);
        if (!ok) { pass = false; break; }
        stats(img, nz, dist);
        if (nz == 0 || dist < 2) pass = false;
        if (run == 0) {
          first[libName][sc] = img;
          if (const char *pre = getenv("PPM_PREFIX"))
            writePPM((std::string(pre) + "-" + libName + ".ppm").c_str(), img);
        }
        else if (img.px == first[libName][sc].px) ++stable;
      }
      double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      printf("[%s] %-16s nonzero=%zu distinct=%zu identical=%d/%d %.1fs %s\n", libName,
          sc.c_str(), nz, dist, stable + 1, runs, secs, pass ? "PASS" : "FAIL");
      fflush(stdout);
      if (!pass) ++fails;
    }
    anariRelease(d, d);
    anariUnloadLibrary(lib);
  }
  if (argc >= 3) {
    for (auto &sc : scenes) {
      auto &a = first[argv[1]], &b = first[argv[2]];
      if (a.count(sc) && b.count(sc) && !a[sc].px.empty() && !b[sc].px.empty())
        printf("[diff %s vs %s] %-16s meanAbsDiff=%.3f (0..255)\n", argv[1], argv[2],
            sc.c_str(), meanAbsDiff(a[sc], b[sc]));
    }
  }
  printf("RESULT: %d FAIL\n", fails);
  return fails ? 1 : 0;
}
