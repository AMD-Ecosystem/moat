// Ground segmentation check: synthetic multi-sector scene through CudaScanGroundSegmentationFilter,
// compared point by point with a sequential CPU replica of the same algorithm.
#include "autoware/cuda_scan_ground_segmentation/cuda_scan_ground_segmentation_filter.hpp"

#include <cuda_blackboard/cuda_pointcloud2.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace autoware::cuda_ground_segmentation;

static FilterParameters make_params()
{
  FilterParameters p{};
  p.max_x = 150.0f; p.min_x = -100.0f; p.max_y = 70.0f; p.min_y = -70.0f;
  p.max_z = 2.5f; p.min_z = -2.5f; p.center_x = 0.0f; p.center_y = 0.0f;
  p.max_radius = std::max(std::max(std::hypot(p.max_x, p.max_y), std::hypot(p.min_x, p.min_y)),
                          std::max(std::hypot(p.max_x, p.min_y), std::hypot(p.min_x, p.max_y)));
  p.sector_angle_rad = static_cast<float>(1.0 * M_PI / 180.0);
  p.inv_sector_angle_rad = 1.0f / p.sector_angle_rad;
  p.num_sectors = static_cast<uint32_t>(std::ceil(2.0 * M_PI * p.inv_sector_angle_rad));
  p.global_slope_max_angle_rad = static_cast<float>(10.0 * M_PI / 180.0);
  p.local_slope_max_angle_rad = static_cast<float>(25.0 * M_PI / 180.0);
  p.global_slope_max_ratio = std::tan(p.global_slope_max_angle_rad);
  p.local_slope_max_ratio = std::tan(p.local_slope_max_angle_rad);
  p.use_recheck_ground_cluster = 1;
  p.recheck_start_distance = 20.0f;
  p.detection_range_z_max = 3.2f;
  p.non_ground_height_threshold = 0.20f;
  p.cell_divider_size_m = 0.5f;
  p.max_num_cells_per_sector = static_cast<uint32_t>(p.max_radius / p.cell_divider_size_m);
  p.max_num_cells = p.max_num_cells_per_sector * p.num_sectors;
  p.gnd_cell_buffer_size = 5;
  return p;
}

// Terrain differs between neighbouring sectors, so two sectors sharing a 64-wide wavefront
// (sector 2k and 2k+1) need different ground references.
static float ground_z(int sector, float r)
{
  switch (sector % 4) {
    case 0: return -0.05f;                       // flat
    case 1: return 0.07f * r - 0.05f;            // climbing slope
    case 2: return -0.05f - 0.03f * r;           // descending slope
    default: return (r < 30.0f) ? 0.0f : 0.05f * (r - 30.0f);  // flat, then a ramp
  }
}

static std::vector<PointTypeStruct> make_scene(uint32_t seed, const FilterParameters & prm)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  std::vector<PointTypeStruct> pts;
  const int nsec = static_cast<int>(prm.num_sectors);
  for (int s = 0; s < nsec; ++s) {
    // stay clear of sector boundaries
    const float a0 = (s + 0.25f) * prm.sector_angle_rad, a1 = (s + 0.75f) * prm.sector_angle_rad;
    // gaps (empty cells) in some sectors make DISCONTINUOUS and BREAK modes happen
    const bool gappy = (s % 7) == 3;
    for (int c = 4; c < 120; ++c) {  // r from 2 m to 60 m
      if (gappy && (c % 13) >= 9) continue;
      if (c > 90 && (c % 3) != 0) continue;  // sparse far points
      const float r0 = c * prm.cell_divider_size_m + 0.1f, r1 = (c + 1) * prm.cell_divider_size_m - 0.1f;
      const int n = (c < 60) ? 6 : 3;
      for (int k = 0; k < n; ++k) {
        float a = a0 + (a1 - a0) * u01(rng), r = r0 + (r1 - r0) * u01(rng);
        float z = ground_z(s, r) + 0.02f * (u01(rng) - 0.5f);
        pts.push_back({r * std::cos(a), r * std::sin(a), z, 10, 1, static_cast<uint16_t>(s % 128)});
      }
      // obstacles: boxes and poles in some cells, well clear of the ground thresholds
      if (((s * 31 + c * 17) % 23) == 0) {
        for (int k = 0; k < 4; ++k) {
          float a = a0 + (a1 - a0) * u01(rng), r = r0 + (r1 - r0) * u01(rng);
          float z = ground_z(s, r) + 0.7f + 0.8f * u01(rng);
          pts.push_back({r * std::cos(a), r * std::sin(a), z, 200, 1, static_cast<uint16_t>(s % 128)});
        }
      }
    }
  }
  std::shuffle(pts.begin(), pts.end(), rng);
  return pts;
}

// ---------------- CPU replica of the GPU algorithm ----------------
struct CP { float z; PointType type; float radius; size_t idx; };

static float fastAtan2_0_2Pi(float y, float x) { return fmodf(atan2(y, x) + 2.0f * M_PI, 2.0f * M_PI); }

static int mode_of(int cell_id, int closest, int furthest, int buf, int cont, int empty)
{
  if (closest < 0) return UNINITIALIZED;
  if (cell_id - closest >= cont) return BREAK;
  if (cell_id - furthest - empty <= buf) return CONTINUOUS;
  return DISCONTINUOUS;
}

static void seg_point(CP & p, const FilterParameters & prm, int mode, float slope, float ph, float pr)
{
  if (mode == UNINITIALIZED) {
    if (p.z > prm.detection_range_z_max || p.z < -prm.non_ground_height_threshold) { p.type = PointType::OUT_OF_RANGE; return; }
    float g = p.radius * prm.global_slope_max_ratio;
    if (p.z > g && p.z > prm.non_ground_height_threshold) { p.type = PointType::NON_GROUND; return; }
    float az = std::fabs(p.z);
    if (az < g && az < prm.non_ground_height_threshold) p.type = PointType::GROUND;
  } else if (mode == CONTINUOUS) {
    if (p.z - ph > prm.detection_range_z_max) { p.type = PointType::OUT_OF_RANGE; return; }
    float dr = p.radius - pr + prm.cell_divider_size_m, dz = p.z - ph;
    if (p.z > prm.global_slope_max_ratio * p.radius) { p.type = PointType::NON_GROUND; return; }
    if (dz > prm.local_slope_max_ratio * dr) { p.type = PointType::NON_GROUND; return; }
    float est = ph + slope * prm.cell_divider_size_m;
    if (p.z > est + prm.non_ground_height_threshold) { p.type = PointType::NON_GROUND; return; }
    if (p.z < est - prm.non_ground_height_threshold || dz < -prm.local_slope_max_ratio * dr ||
        p.z < -prm.global_slope_max_ratio * p.radius) { p.type = PointType::OUT_OF_RANGE; return; }
    p.type = PointType::GROUND;
  } else if (mode == DISCONTINUOUS) {
    if (p.z - ph > prm.detection_range_z_max) { p.type = PointType::OUT_OF_RANGE; return; }
    float dz = p.z - ph, dr = p.radius - pr;
    float g = p.radius * prm.global_slope_max_ratio, l = prm.local_slope_max_ratio * dr;
    if (p.z > g || dz > l) { p.type = PointType::NON_GROUND; return; }
    if (dz < -l || p.z < -g) { p.type = PointType::OUT_OF_RANGE; return; }
    p.type = PointType::GROUND;
  } else {
    if (p.z - ph > prm.detection_range_z_max) { p.type = PointType::OUT_OF_RANGE; return; }
    float dz = p.z - ph, dr = p.radius - pr, g = dr * prm.global_slope_max_ratio;
    p.type = (dz > g) ? PointType::NON_GROUND : ((dz < -g) ? PointType::OUT_OF_RANGE : PointType::GROUND);
  }
}

// returns label per input index (true = ground) for points that survive the box filter
static std::vector<int> cpu_reference(const std::vector<PointTypeStruct> & in, const FilterParameters & prm)
{
  std::vector<PointTypeStruct> pts;
  std::vector<size_t> orig;
  for (size_t k = 0; k < in.size(); ++k) {
    const auto & p = in[k];
    if (!(p.x < prm.min_x || p.x > prm.max_x || p.y < prm.min_y || p.y > prm.max_y || p.z < prm.min_z || p.z > prm.max_z)) {
      pts.push_back(p); orig.push_back(k);
    }
  }
  const int ns = prm.num_sectors, nc = prm.max_num_cells_per_sector;
  std::vector<std::vector<CP>> cells(prm.max_num_cells);
  for (size_t i = 0; i < pts.size(); ++i) {
    float dx = pts[i].x - prm.center_x, dy = pts[i].y - prm.center_y;
    float radius = hypotf(dx, dy), angle = fastAtan2_0_2Pi(dy, dx);
    int sid = static_cast<int>(angle * prm.inv_sector_angle_rad);
    int cid = static_cast<int>(radius / prm.cell_divider_size_m);
    cells[cid * ns + sid].push_back({pts[i].z, PointType::INIT, radius, i});
  }
  for (int s = 0; s < ns; ++s) {
    // empty cell prefix count per sector
    std::vector<int> empty(nc);
    int sum = 0;
    for (int i = 0; i < nc; ++i) { empty[i] = sum; sum += cells[i * ns + s].empty() ? 1 : 0; }
    std::vector<Cell> q(prm.gnd_cell_buffer_size);
    std::vector<int> qid(prm.gnd_cell_buffer_size);
    int closest = -1, furthest = -1, nlatest = 0, head = 0, tail = 0, empty_num = 0;
    float sx = 0, sy = 0, sxx = 0, sxy = 0, slope = 0, pr = 0, ph = 0;
    for (int i = 0; i < nc; ++i) {
      if (nlatest > 0) { furthest = qid[head]; empty_num = empty[i] - empty[furthest]; }
      auto & cell_pts = cells[i * ns + s];
      if (cell_pts.empty()) continue;
      int mode = mode_of(i, closest, furthest, prm.gnd_cell_buffer_size, prm.gnd_grid_continual_thresh, empty_num);
      double rs = 0, hs = 0; float hmin = FLT_MAX; int ng = 0;
      for (auto & p : cell_pts) {
        seg_point(p, prm, mode, slope, ph, pr);
        if (p.type == PointType::GROUND) { rs += p.radius; hs += p.z; ++ng; hmin = std::min(hmin, p.z); }
      }
      if (prm.use_recheck_ground_cluster && ng > 1 && rs / ng > prm.recheck_start_distance) {
        double mrs = 0, mhs = 0; int mng = 0;
        for (auto & p : cell_pts)
          if (p.type == PointType::GROUND && p.z > hmin + prm.non_ground_height_threshold) {
            mhs += p.z; mrs += p.radius; ++mng; p.type = PointType::NON_GROUND;
          }
        if (mng == ng) { std::fprintf(stderr, "reference: keep-last branch reached (s=%d c=%d)\n", s, i); }
        rs -= mrs; hs -= mhs; ng -= mng;
      }
      Cell cell;
      if (ng > 0) {
        cell.gnd_radius_avg = static_cast<float>(rs / ng); cell.gnd_height_avg = static_cast<float>(hs / ng);
        cell.gnd_height_min = hmin; cell.num_ground_points = ng;
      }
      if (cell.num_ground_points > 0) {
        if (nlatest >= static_cast<int>(prm.gnd_cell_buffer_size)) {
          Cell hc = q[head];
          sx -= hc.gnd_radius_avg; sy -= hc.gnd_height_avg; sxx -= hc.gnd_radius_avg * hc.gnd_radius_avg; sxy -= hc.gnd_radius_avg * hc.gnd_height_avg;
          head = (head + 1) % prm.gnd_cell_buffer_size; --nlatest;
        }
        ++nlatest;
        q[tail] = cell; qid[tail] = i;
        sx += cell.gnd_radius_avg; sy += cell.gnd_height_avg; sxx += cell.gnd_radius_avg * cell.gnd_radius_avg; sxy += cell.gnd_radius_avg * cell.gnd_height_avg;
        float denom = nlatest * sxx - sx * sx;
        if (std::fabs(denom) < 1e-6f) { slope = q[head].gnd_height_avg / q[head].gnd_radius_avg; }
        else { slope = (nlatest * sxy - sx * sy) / denom; slope = std::fmax(std::fmin(slope, prm.global_slope_max_ratio), -prm.global_slope_max_ratio); }
        tail = (tail + 1) % prm.gnd_cell_buffer_size;
        pr = cell.gnd_radius_avg; ph = cell.gnd_height_avg; closest = i;
      }
    }
  }
  // -1: removed by the box filter, so in neither output
  std::vector<int> label(in.size(), -1);
  for (auto & c : cells) for (auto & p : c) label[orig[p.idx]] = (p.type == PointType::GROUND) ? 1 : 0;
  // map back to keys
  return label;
}

using Key = std::tuple<float, float, float>;

static cuda_blackboard::CudaPointCloud2 to_device(const std::vector<PointTypeStruct> & pts)
{
  cuda_blackboard::CudaPointCloud2 c;
  c.header.frame_id = "base_link";
  c.height = 1; c.width = pts.size(); c.point_step = sizeof(PointTypeStruct); c.row_step = c.width * c.point_step;
  c.is_dense = true; c.is_bigendian = false;
  auto f = [](const char * n, uint32_t off, uint8_t t) { sensor_msgs::msg::PointField pf; pf.name = n; pf.offset = off; pf.datatype = t; pf.count = 1; return pf; };
  c.fields = {f("x", 0, 7), f("y", 4, 7), f("z", 8, 7), f("intensity", 12, 2), f("return_type", 13, 2), f("channel", 14, 4)};
  c.data = cuda_blackboard::make_unique<uint8_t[]>(pts.size() * sizeof(PointTypeStruct));
  CHECK_CUDA_ERROR(cudaMemcpy(c.data.get(), pts.data(), pts.size() * sizeof(PointTypeStruct), cudaMemcpyHostToDevice));
  return c;
}

static std::vector<PointTypeStruct> to_host(const cuda_blackboard::CudaPointCloud2 & c)
{
  std::vector<PointTypeStruct> v(c.width * c.height);
  if (!v.empty()) CHECK_CUDA_ERROR(cudaMemcpy(v.data(), c.data.get(), v.size() * sizeof(PointTypeStruct), cudaMemcpyDeviceToHost));
  return v;
}

int main(int argc, char ** argv)
{
  const int runs = argc > 1 ? std::atoi(argv[1]) : 20;
  const uint32_t seed = argc > 2 ? std::atoi(argv[2]) : 7;
  const FilterParameters prm = make_params();
  auto pts = make_scene(seed, prm);
  auto ref = cpu_reference(pts, prm);
  std::map<Key, size_t> index;
  for (size_t i = 0; i < pts.size(); ++i) index[{pts[i].x, pts[i].y, pts[i].z}] = i;
  size_t ref_ground = std::count(ref.begin(), ref.end(), 1), ref_ng = std::count(ref.begin(), ref.end(), 0);
  std::printf("scene seed=%u points=%zu sectors=%u reference ground=%zu non_ground=%zu box_filtered=%zu\n", seed, pts.size(), prm.num_sectors, ref_ground, ref_ng, pts.size() - ref_ground - ref_ng);

  CudaScanGroundSegmentationFilter filter(prm, 1000000000LL);
  auto input = to_device(pts);
  int bad_runs = 0;
  std::vector<int> first;
  for (int run = 0; run < runs; ++run) {
    cuda_blackboard::CudaPointCloud2 ground, non_ground;
    filter.classifyPointCloud(input, ground, non_ground);
    CHECK_CUDA_ERROR(hipDeviceSynchronize());
    auto g = to_host(ground), ng = to_host(non_ground);
    std::vector<int> lab(pts.size(), -1);
    for (auto & p : g) lab[index.at({p.x, p.y, p.z})] = 1;
    for (auto & p : ng) lab[index.at({p.x, p.y, p.z})] = 0;
    size_t missing = std::count(lab.begin(), lab.end(), -1), mism = 0, mism_gnd_to_ng = 0;
    missing -= std::count(ref.begin(), ref.end(), -1);
    std::set<int> bad_sectors;
    for (size_t i = 0; i < pts.size(); ++i) {
      if (lab[i] != ref[i]) {
        ++mism; if (ref[i] == 1) ++mism_gnd_to_ng;
        float a = fastAtan2_0_2Pi(pts[i].y, pts[i].x); bad_sectors.insert(static_cast<int>(a * prm.inv_sector_angle_rad));
      }
    }
    int odd = 0; for (int s : bad_sectors) odd += (s & 1);
    if (run == 0) {
      first = lab;
      uint64_t h = 1469598103934665603ull;  // FNV-1a over per-input-point labels
      for (int v : lab) { h ^= static_cast<uint8_t>(v + 1); h *= 1099511628211ull; }
      std::printf("label hash (FNV-1a, input order): %016llx\n", static_cast<unsigned long long>(h));
    }
    bool same_as_first = (lab == first);
    std::printf("run %2d: ground=%zu non_ground=%zu missing=%zu mismatches=%zu (ref ground->non_ground %zu) sectors_with_mismatch=%zu (odd %d) identical_to_run0=%d\n",
                run, g.size(), ng.size(), missing, mism, mism_gnd_to_ng, bad_sectors.size(), odd, same_as_first ? 1 : 0);
    if (mism || missing || !same_as_first) ++bad_runs;
  }
  std::printf("%s: %d of %d runs differ from the CPU reference or from run 0\n", bad_runs ? "FAIL" : "PASS", bad_runs, runs);
  return bad_runs ? 1 : 0;
}
