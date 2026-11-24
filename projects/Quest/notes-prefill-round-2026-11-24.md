# Quest prefill round 2026-11-24 (linux-gfx950): the change against `9cb8bee`

The complete, uncommitted change of the porter round of 2026-11-24 (notes.md, "Porter round 2026-11-24"), kept here because the host's disk filled before it could be built and validated in full. Apply with `git apply` on the fork at `9cb8bee`.

```diff
diff --git a/kernels/include/prefill/prefill.cuh b/kernels/include/prefill/prefill.cuh
index cc5c21e..51a0ce6 100644
--- a/kernels/include/prefill/prefill.cuh
+++ b/kernels/include/prefill/prefill.cuh
@@ -686,7 +686,11 @@ template <uint32_t group_size,
 		  RotaryMode rotary_mode,
 		  uint32_t num_frags_x,
 		  uint32_t num_frags_y,
+#if defined(USE_HIP)
+		  uint32_t kernel_num_frags_z,
+#else
 		  uint32_t num_frags_z,
+#endif
 		  uint32_t num_warps,
 		  PageStorage page_storage,
 		  QKVLayout kv_layout,
@@ -711,6 +715,12 @@ BatchPrefillWithPagedKVCacheKernel(IdType* __restrict__ request_indices,
 								   float sm_scale,
 								   const float log2_rope_rcp_scale,
 								   const float log2_rope_rcp_theta) {
+#if defined(QUEST_PREFILL_MMA)
+	constexpr uint32_t num_frags_z = matrix_core::kv_frags(
+		num_frags_x, num_frags_y, kernel_num_frags_z, matrix_core::kShape);
+#elif defined(USE_HIP)
+	constexpr uint32_t num_frags_z = kernel_num_frags_z;
+#endif
 	static_assert(sizeof(DTypeIn) == 2);
 	static_assert(sizeof(DTypeOut) == 2);
 	sm_scale *= math::log2e;
@@ -837,7 +847,57 @@ BatchPrefillWithPagedKVCacheKernel(IdType* __restrict__ request_indices,
 			 : kv_len) /
 		(16 * num_frags_z);
 
-#if defined(QUEST_PREFILL_MMA)
+#if QUEST_PREFILL_MMA == 2
+	// gfx950: the K and V tiles are copied straight into two pairs of shared memory buffers,
+	// one tile ahead, so one barrier per tile suffices. The loop is unrolled by two so that
+	// each buffer is a distinct array, which lets the compiler see that reading one buffer
+	// does not wait for the copy into the other.
+	static_assert(num_q_smem_rows == 0);
+	using mc_tile_t = matrix_core::
+		kv_tile_t<num_warps, num_frags_y, num_frags_z, page_storage, DTypeIn, IdType>;
+	constexpr uint32_t mc_tile_size = num_frags_z * 16 * channel_size_128b_in;
+	__shared__ b128_t mc_k0[mc_tile_size], mc_v0[mc_tile_size], mc_k1[mc_tile_size],
+		mc_v1[mc_tile_size];
+	mc_tile_t mc_tile;
+	matrix_core::zero_v_tile<mc_tile_t>(mc_v0);
+	matrix_core::zero_v_tile<mc_tile_t>(mc_v1);
+	matrix_core::kv_tile_pages<page_size>(mc_tile, paged_kv, page_iter_base, 0, kv_len);
+	block.sync();
+	matrix_core::kv_tile_copy<page_size>(mc_tile, paged_kv, mc_k0, mc_v0);
+	matrix_core::kv_tile_pages<page_size>(
+		mc_tile, paged_kv, page_iter_base + 16 * num_frags_z / page_size, 16 * num_frags_z, kv_len);
+	auto mc_step = [&](auto second, const uint32_t iter) {
+		b128_t* k_cur = second ? mc_k1 : mc_k0;
+		b128_t* v_cur = second ? mc_v1 : mc_v0;
+		matrix_core::kv_tile_barrier();
+		if(iter + 1 < num_iterations) {
+			matrix_core::kv_tile_copy<page_size>(
+				mc_tile, paged_kv, second ? mc_k0 : mc_k1, second ? mc_v0 : mc_v1);
+			matrix_core::kv_tile_pages<page_size>(
+				mc_tile,
+				paged_kv,
+				page_iter_base + (iter + 2) * (16 * num_frags_z / page_size),
+				(iter + 2) * 16 * num_frags_z,
+				kv_len);
+		}
+		smem_t k_cur_smem(k_cur), v_cur_smem(v_cur);
+		matrix_core::compute_qk<num_frags_x, num_frags_y, num_frags_z, DTypeIn>(
+			&qo_smem, &k_cur_smem, mc_state);
+		if(iter >= mask_iteration) {
+			matrix_core::mask_s<causal, group_size>(
+				mc_q_row_base, iter * 16 * num_frags_z, qo_len, kv_len, kv_len, mc_state);
+		}
+		matrix_core::update_mdo_states(mc_state);
+		matrix_core::compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, DTypeIn>(&v_cur_smem,
+																				  mc_state);
+	};
+	for(uint32_t iter = 0; iter < num_iterations; iter += 2) {
+		mc_step(std::false_type(), iter);
+		if(iter + 1 < num_iterations) {
+			mc_step(std::true_type(), iter + 1);
+		}
+	}
+#elif defined(QUEST_PREFILL_MMA)
 	// The same steps on the matrix instructions. Each K and V tile is loaded into registers
 	// before the product that precedes its store, so the global load latency overlaps compute.
 	matrix_core::kv_regs_t<num_warps, num_frags_y, num_frags_z> kv_regs;
@@ -1055,9 +1115,14 @@ cudaError_t BatchPrefillWithPagedKVCacheDispatched(
 		uint32_t smem_size =
 			(num_frags_x * num_warps + num_frags_z * 2) * 16 * HEAD_DIM * sizeof(DTypeIn);
 #if defined(USE_HIP)
-		if(matrix_core::q_in_registers(num_frags_x, num_frags_y, mma_shape)) {
-			smem_size = num_frags_z * 2 * 16 * HEAD_DIM * sizeof(DTypeIn);
-		}
+		smem_size =
+			mma_shape.static_kv
+				? 0
+				: ((matrix_core::q_in_registers(num_frags_x, num_frags_y, mma_shape)
+						? 0
+						: num_frags_x * num_warps) +
+				   matrix_core::kv_frags(num_frags_x, num_frags_y, num_frags_z, mma_shape) * 2) *
+					  16 * HEAD_DIM * sizeof(DTypeIn);
 #endif
 		FLASHINFER_CUDA_CALL(
 			cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
@@ -1098,7 +1163,16 @@ split_qo_indptr(IdType* qo_indptr,
 
 	const uint32_t total_q_len = qo_indptr_h[batch_size];
 	const bool avg_len_greater_than_64 = total_q_len * gqa_group_size > 64 * batch_size;
+#if defined(USE_HIP)
+	matrix_core::mma_shape_t mma_shape = {};
+	if(matrix_core::get_prefill_mma_shape(&mma_shape) != cudaSuccess) {
+		mma_shape = {};
+	}
+	const uint32_t num_frags_x =
+		avg_len_greater_than_64 ? matrix_core::max_num_frags_x(mma_shape) : 1;
+#else
 	const uint32_t num_frags_x = avg_len_greater_than_64 ? 2 : 1;
+#endif
 	const uint32_t num_rows_per_cta = num_frags_x * num_warps * 16;
 	uint32_t num_qo_tiles = 0;
 
diff --git a/kernels/include/prefill/prefill_matrix_core.cuh b/kernels/include/prefill/prefill_matrix_core.cuh
index d89d81e..2d245c3 100644
--- a/kernels/include/prefill/prefill_matrix_core.cuh
+++ b/kernels/include/prefill/prefill_matrix_core.cuh
@@ -32,16 +32,21 @@
 // Operands: Q stays in registers when it fits (q_in_registers), so shared memory holds only
 // K and V; V is stored transposed so its operand is a contiguous read; and each K and V tile
 // is loaded into registers one phase ahead of its shared memory store, since AMD has no
-// asynchronous global-to-shared copy.
+// asynchronous global-to-shared copy. gfx950 instead copies K and V from global to shared
+// memory directly, into two buffers a tile ahead (kv_tile_copy), keeps V row-major and reads
+// its operand with a transposing load (load_v_tr).
 //
-// QUEST_PREFILL_MMA selects the instruction set in a device pass: 1 for MFMA, 11 for RDNA3
-// WMMA, 12 for RDNA4 WMMA. It is left undefined for the host pass and for any other target,
-// which keep the mma.sync emulation in hip_compat/flashinfer/mma.cuh.
+// QUEST_PREFILL_MMA selects the instruction set in a device pass: 1 for MFMA 16x16x16, 2 for
+// MFMA 16x16x32 (gfx950, which also reads the V operand with a transposing LDS load), 11 for
+// RDNA3 WMMA, 12 for RDNA4 WMMA. It is left undefined for the host pass and for any other
+// target, which keep the mma.sync emulation in hip_compat/flashinfer/mma.cuh.
 
 #if !defined(QUEST_PREFILL_MMA) && defined(__HIP_DEVICE_COMPILE__)
 #if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || \
-    defined(__gfx950__) || defined(__gfx9_4_generic__)
+    defined(__gfx9_4_generic__)
 #define QUEST_PREFILL_MMA 1
+#elif defined(__gfx950__)
+#define QUEST_PREFILL_MMA 2
 #elif defined(__GFX11__)
 #define QUEST_PREFILL_MMA 11
 #elif defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
@@ -58,22 +63,27 @@ namespace flashinfer {
 namespace matrix_core {
 
 // What the device code for the current GPU uses, read by the host to size shared memory:
-// wave_size and op_elems (16-bit operand elements per lane) of the matrix instruction, or
+// wave_size, op_elems (16-bit operand elements per lane) and k of the matrix instruction, or
 // zeros where the kernel keeps the mma.sync emulation.
 struct mma_shape_t {
   uint32_t wave_size;
   uint32_t op_elems;
+  uint32_t k;
+  // Whether the K and V tiles are in static shared memory rather than the dynamic one.
+  uint32_t static_kv;
 };
 
 static __device__ __attribute__((used)) const mma_shape_t prefill_mma_shape =
 #if QUEST_PREFILL_MMA == 1
-    {64, 4};
+    {64, 4, 16, 0};
+#elif QUEST_PREFILL_MMA == 2
+    {64, 8, 32, 1};
 #elif QUEST_PREFILL_MMA == 11
-    {32, 16};
+    {32, 16, 16, 0};
 #elif QUEST_PREFILL_MMA == 12
-    {32, 8};
+    {32, 8, 16, 0};
 #else
-    {0, 0};
+    {0, 0, 0, 0};
 #endif
 
 // Whether a wavefront keeps its query tile in registers (64 of them at most) for the whole
@@ -82,7 +92,32 @@ static __device__ __attribute__((used)) const mma_shape_t prefill_mma_shape =
 __host__ __device__ constexpr bool q_in_registers(uint32_t num_frags_x, uint32_t num_frags_y,
                                                   mma_shape_t shape) {
   return shape.wave_size != 0 &&
-         num_frags_x * (shape.wave_size / 32) * num_frags_y * shape.op_elems / 2 <= 64;
+         num_frags_x * (shape.wave_size / 32) * (num_frags_y * 16 / shape.k) * shape.op_elems / 2 <=
+             64;
+}
+
+// The most 16-row query tiles a warp takes (num_frags_x). On gfx950 one tile per warp keeps a
+// wavefront within 256 registers, so two wavefronts share a SIMD, which pays more than the
+// halved reuse of each K and V tile.
+__host__ __device__ constexpr uint32_t max_num_frags_x(mma_shape_t shape) {
+  return shape.k == 32 ? 1 : 2;
+}
+
+// The number of 16-row K and V fragments per tile that the kernel instantiated with
+// num_frags_z uses. gfx950 double-buffers 32-row tiles in shared memory. A wave32 lane holds
+// twice the fp32 accumulators of a wave64 one: with the query tile in registers, more than
+// 128 registers of accumulators (S^T and O^T) and query operands at num_frags_z make the
+// compiler spill, and two fragments are used instead. (With the query tile in shared memory,
+// gfx11 at head dimension 128, two fragments spill more, not less.)
+__host__ __device__ constexpr uint32_t kv_frags(uint32_t num_frags_x, uint32_t num_frags_y,
+                                                uint32_t num_frags_z, mma_shape_t shape) {
+  const bool cap =
+      shape.static_kv ||
+      (shape.wave_size == 32 && q_in_registers(num_frags_x, num_frags_y, shape) &&
+       num_frags_x * (num_frags_y + num_frags_z) * 8 +
+               num_frags_x * num_frags_y * shape.op_elems / 2 >
+           128);
+  return cap && num_frags_z > 2 ? 2 : num_frags_z;
 }
 
 inline cudaError_t get_prefill_mma_shape(mma_shape_t* shape) {
@@ -124,6 +159,9 @@ namespace matrix_core {
 constexpr uint32_t kWaveSize = 64;
 // 16-bit operand elements per lane.
 constexpr uint32_t kOpElems = 4;
+#elif QUEST_PREFILL_MMA == 2
+constexpr uint32_t kWaveSize = 64;
+constexpr uint32_t kOpElems = 8;
 #elif QUEST_PREFILL_MMA == 11
 constexpr uint32_t kWaveSize = 32;
 constexpr uint32_t kOpElems = 16;
@@ -134,7 +172,14 @@ constexpr uint32_t kOpElems = 8;
 // fp32 accumulator elements per lane of a 16x16 tile.
 constexpr uint32_t kAccElems = 256 / kWaveSize;
 constexpr uint32_t kWarpsPerWave = kWaveSize / 32;
-constexpr mma_shape_t kShape = {kWaveSize, kOpElems};
+// k of the instruction and the number of 16-wide fragments it spans.
+#if QUEST_PREFILL_MMA == 2
+constexpr uint32_t kK = 32;
+#else
+constexpr uint32_t kK = 16;
+#endif
+constexpr uint32_t kFragsPerK = kK / 16;
+constexpr mma_shape_t kShape = {kWaveSize, kOpElems, kK, QUEST_PREFILL_MMA == 2};
 
 __device__ __forceinline__ uint32_t lane_id() {
   return (threadIdx.y * 32 + threadIdx.x) % kWaveSize;
@@ -142,7 +187,7 @@ __device__ __forceinline__ uint32_t lane_id() {
 
 // Row of D (within the 16x16 tile) held in accumulator element i.
 __device__ __forceinline__ uint32_t acc_row(uint32_t i) {
-#if QUEST_PREFILL_MMA == 1
+#if QUEST_PREFILL_MMA == 1 || QUEST_PREFILL_MMA == 2
   return 4 * (lane_id() / 16) + i;
 #elif QUEST_PREFILL_MMA == 11
   return 2 * i + lane_id() / 16;
@@ -151,10 +196,14 @@ __device__ __forceinline__ uint32_t acc_row(uint32_t i) {
 #endif
 }
 
-// k index (within 16) of operand element 0; the lane's operand elements are consecutive.
+// k index (within kK) of operand element 0; the lane's operand elements are consecutive.
+// (For 16x16x32 this holds for the head dimension; the kv dimension pairs two fragments, see
+// compute_sfm_v.)
 __device__ __forceinline__ uint32_t op_k0() {
 #if QUEST_PREFILL_MMA == 1
   return 4 * (lane_id() / 16);
+#elif QUEST_PREFILL_MMA == 2
+  return 8 * (lane_id() / 16);
 #elif QUEST_PREFILL_MMA == 11
   return 0;
 #else
@@ -202,7 +251,7 @@ __device__ __forceinline__ float from_bits(uint16_t x) {
   }
 }
 
-// c += a * b for one 16x16x16 tile.
+// c += a * b for one 16x16xkK tile.
 template <typename T>
 __device__ __forceinline__ void mma(float (&c)[kAccElems], const operand_t& a,
                                     const operand_t& b) {
@@ -215,15 +264,23 @@ __device__ __forceinline__ void mma(float (&c)[kAccElems], const operand_t& a,
     const op_v av = __builtin_bit_cast(op_v, a), bv = __builtin_bit_cast(op_v, b);
 #if QUEST_PREFILL_MMA == 1
     acc = __builtin_amdgcn_mfma_f32_16x16x16f16(av, bv, acc, 0, 0, 0);
+#elif QUEST_PREFILL_MMA == 2
+    acc = __builtin_amdgcn_mfma_f32_16x16x32_f16(av, bv, acc, 0, 0, 0);
 #elif QUEST_PREFILL_MMA == 11
     acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av, bv, acc);
 #else
     acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc);
 #endif
   } else {
+#if QUEST_PREFILL_MMA == 2
+    typedef __bf16 op_v __attribute__((ext_vector_type(kOpElems)));
+#else
     typedef short op_v __attribute__((ext_vector_type(kOpElems)));
+#endif
     const op_v av = __builtin_bit_cast(op_v, a), bv = __builtin_bit_cast(op_v, b);
-#if QUEST_PREFILL_MMA == 1
+#if QUEST_PREFILL_MMA == 2
+    acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(av, bv, acc, 0, 0, 0);
+#elif QUEST_PREFILL_MMA == 1
     acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc, 0, 0, 0);
 #elif QUEST_PREFILL_MMA == 11
     acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(av, bv, acc);
@@ -325,6 +382,22 @@ __device__ __forceinline__ void load_vt(smem_t* smem, uint32_t d, uint32_t kv0,
   }
 }
 
+#if QUEST_PREFILL_MMA == 2
+// gfx950 keeps V row-major like K and reads the V^T operand with a transposing load: in each
+// group of 16 lanes, ds_read_b64_tr_b16 gives lane i element e of the four elements read by
+// lane 4 * e + i / 4. Lane j therefore reads kv row kv0 + j / 4, head dimensions d0 + 4 * (j % 4)
+// .. + 3, and lane i receives (d0 + i, kv0 .. kv0 + 3).
+template <uint32_t channel_size_128b, typename T>
+__device__ __forceinline__ void load_v_tr(smem_t* smem, uint32_t d0, uint32_t kv0, uint32_t* dst) {
+  typedef short v4s __attribute__((ext_vector_type(4)));
+  const uint32_t j = lane_id() % 16;
+  const T* src = smem_elem<channel_size_128b, T>(smem, kv0 + j / 4, d0 + 4 * (j % 4));
+  const v4s x =
+      __builtin_amdgcn_ds_read_tr16_b64_v4i16((__attribute__((address_space(3))) v4s*)src);
+  *(uint2*)dst = __builtin_bit_cast(uint2, x);
+}
+#endif
+
 // K rows past kv_len are left as they are (their scores are masked); V rows past kv_len are
 // zero-filled, as their products are not.
 template <bool produce_v, uint32_t num_warps, uint32_t num_frags_y, uint32_t num_frags_z,
@@ -378,12 +451,165 @@ __device__ __forceinline__ void page_produce_kv(
     smem_t smem, uint32_t* smem_offset,
     paged_kv_t<page_storage, kv_layout, DType, IdType>& paged_kv, const uint32_t kv_idx_base,
     const uint32_t page_iter_base, const uint32_t kv_len, const IdType last_indptr) {
+#if QUEST_PREFILL_MMA != 2
+  // (gfx950 copies its first tiles in the main loop's prologue instead, see kv_tile_copy.)
   kv_regs_t<num_warps, num_frags_y, num_frags_z> regs;
   page_load_kv<produce_v, page_size>(regs, paged_kv, page_iter_base, last_indptr);
   page_store_kv<produce_v, num_warps, num_frags_y, num_frags_z, DType>(regs, smem, smem_offset,
                                                                        kv_idx_base, kv_len);
+#endif
+}
+
+#if QUEST_PREFILL_MMA == 2
+// gfx950 copies K and V tiles from global to shared memory directly (global_load_lds_dwordx4)
+// into two buffers, so that the copy of the next tile overlaps the compute of the current one
+// without passing through registers. One such instruction of a wavefront writes 64
+// consecutive 16-byte chunks of shared memory, lane l the l-th, so each lane loads the global
+// chunk that the permuted layout puts at its position.
+template <uint32_t num_warps, uint32_t num_frags_y, uint32_t num_frags_z,
+          PageStorage page_storage, typename DType, typename IdType>
+struct kv_tile_t {
+  static constexpr uint32_t channel_size_128b = num_frags_y * 16 / num_elems_per_128b<DType>();
+  static constexpr uint32_t num_waves = num_warps / kWarpsPerWave;
+  // 16-byte chunks of one K (or V) tile per lane.
+  static constexpr uint32_t num_chunks =
+      num_frags_z * 16 * channel_size_128b / (num_waves * kWaveSize);
+  // Rows copied by one instruction of a wavefront.
+  static constexpr uint32_t rows_per_inst = kWaveSize / channel_size_128b;
+  static_assert(kWaveSize % channel_size_128b == 0);
+  // The page of each chunk's row, read one tile ahead of its use so that the copy does not
+  // wait for it: an index into the cache, or the page's address. When a page holds whole
+  // groups of rows_per_inst rows, every lane of a wavefront reads the same page in each
+  // instruction (uniform_pages).
+  typename std::conditional<page_storage == PageStorage::kIndices, IdType, DType*>::type
+      page[num_chunks];
+  // Bit j set when chunk j belongs to a row below kv_len.
+  uint32_t valid;
+  // Whether every row of the tile is below kv_len.
+  bool full;
+
+  template <uint32_t page_size>
+  static constexpr bool uniform_pages = page_size % rows_per_inst == 0;
+  // Shared memory chunk (within the tile) written by the wavefront's j-th instruction.
+  static __device__ __forceinline__ uint32_t first_chunk(uint32_t j) {
+    return (j * num_waves + __builtin_amdgcn_readfirstlane(threadIdx.y / kWarpsPerWave)) *
+           kWaveSize;
+  }
+  static __device__ __forceinline__ uint32_t row(uint32_t j) {
+    return (first_chunk(j) + lane_id()) / channel_size_128b;
+  }
+  // Global chunk of the row that the permuted layout puts at this lane's position.
+  static __device__ __forceinline__ uint32_t col(uint32_t j) {
+    return ((first_chunk(j) + lane_id()) % channel_size_128b) ^ (row(j) % 8);
+  }
+};
+
+template <uint32_t page_size, uint32_t num_warps, uint32_t num_frags_y, uint32_t num_frags_z,
+          PageStorage page_storage, QKVLayout kv_layout, typename DType, typename IdType>
+__device__ __forceinline__ void kv_tile_pages(
+    kv_tile_t<num_warps, num_frags_y, num_frags_z, page_storage, DType, IdType>& tile,
+    paged_kv_t<page_storage, kv_layout, DType, IdType>& paged_kv, const uint32_t page_iter_base,
+    const uint32_t kv_idx_base, const uint32_t kv_len) {
+  using tile_t = kv_tile_t<num_warps, num_frags_y, num_frags_z, page_storage, DType, IdType>;
+  tile.valid = 0;
+  tile.full = kv_idx_base + num_frags_z * 16 <= kv_len;
+#pragma unroll
+  for (uint32_t j = 0; j < tile_t::num_chunks; ++j) {
+    const uint32_t row = tile_t::row(j);
+    const bool valid = kv_idx_base + row < kv_len;
+    const uint32_t page_iter = valid ? page_iter_base + row / page_size : 0;
+    if constexpr (page_storage == PageStorage::kIndices) {
+      tile.page[j] = __ldg(paged_kv.indices + page_iter);
+    } else {
+      tile.page[j] = __ldg(paged_kv.ptrs + page_iter);
+    }
+    tile.valid |= uint32_t(valid) << j;
+  }
 }
 
+template <typename T>
+__device__ __forceinline__ T* readfirstlane(T* p) {
+  const uint64_t x = reinterpret_cast<uint64_t>(p);
+  return reinterpret_cast<T*>(uint64_t(__builtin_amdgcn_readfirstlane(uint32_t(x))) |
+                              uint64_t(__builtin_amdgcn_readfirstlane(uint32_t(x >> 32))) << 32);
+}
+__device__ __forceinline__ int32_t readfirstlane(int32_t x) {
+  return __builtin_amdgcn_readfirstlane(x);
+}
+__device__ __forceinline__ uint32_t readfirstlane(uint32_t x) {
+  return __builtin_amdgcn_readfirstlane(x);
+}
+
+// With uniform, the page is read from the wavefront's first active lane, which every row
+// this instruction copies shares.
+template <uint32_t page_size, bool uniform, uint32_t num_warps, uint32_t num_frags_y,
+          uint32_t num_frags_z, PageStorage page_storage, QKVLayout kv_layout, typename DType,
+          typename IdType>
+__device__ __forceinline__ void kv_chunk_copy(
+    const kv_tile_t<num_warps, num_frags_y, num_frags_z, page_storage, DType, IdType>& tile,
+    paged_kv_t<page_storage, kv_layout, DType, IdType>& paged_kv, b128_t* k_smem,
+    b128_t* v_smem, const uint32_t j) {
+  using tile_t = kv_tile_t<num_warps, num_frags_y, num_frags_z, page_storage, DType, IdType>;
+  typedef __attribute__((address_space(3))) void lds_void;
+  const auto page_j = uniform ? readfirstlane(tile.page[j]) : tile.page[j];
+  const DType* page;
+  if constexpr (page_storage == PageStorage::kIndices) {
+    page = paged_kv.data + paged_kv.get_k_elem_offset(page_j, 0, 0, 0);
+  } else {
+    page = page_j;
+  }
+  const uint32_t offset = paged_kv.get_k_elem_offset_in_page(
+      blockIdx.z, tile_t::row(j) % page_size, tile_t::col(j) * num_elems_per_128b<DType>());
+  const DType* k = page + offset;
+  const DType* v = k + paged_kv.kv_offset_delta();
+  const uint32_t c = tile_t::first_chunk(j);
+  __builtin_amdgcn_global_load_lds((void*)k, (lds_void*)(k_smem + c), 16, 0, 0);
+  __builtin_amdgcn_global_load_lds((void*)v, (lds_void*)(v_smem + c), 16, 0, 0);
+}
+
+// Rows at or past kv_len are not copied: their K scores are masked, and their V rows hold
+// zeros (zero_v_tile) or a V row of an earlier tile, which meets a probability of zero.
+template <uint32_t page_size, uint32_t num_warps, uint32_t num_frags_y, uint32_t num_frags_z,
+          PageStorage page_storage, QKVLayout kv_layout, typename DType, typename IdType>
+__device__ __forceinline__ void kv_tile_copy(
+    const kv_tile_t<num_warps, num_frags_y, num_frags_z, page_storage, DType, IdType>& tile,
+    paged_kv_t<page_storage, kv_layout, DType, IdType>& paged_kv, b128_t* k_smem,
+    b128_t* v_smem) {
+  using tile_t = kv_tile_t<num_warps, num_frags_y, num_frags_z, page_storage, DType, IdType>;
+  if (tile.full) {
+#pragma unroll
+    for (uint32_t j = 0; j < tile_t::num_chunks; ++j) {
+      kv_chunk_copy<page_size, tile_t::template uniform_pages<page_size>>(tile, paged_kv, k_smem,
+                                                                           v_smem, j);
+    }
+  } else {
+#pragma unroll
+    for (uint32_t j = 0; j < tile_t::num_chunks; ++j) {
+      if (tile.valid & (1u << j)) {
+        kv_chunk_copy<page_size, false>(tile, paged_kv, k_smem, v_smem, j);
+      }
+    }
+  }
+}
+
+template <typename tile_t>
+__device__ __forceinline__ void zero_v_tile(b128_t* v_smem) {
+#pragma unroll
+  for (uint32_t j = 0; j < tile_t::num_chunks; ++j) {
+    *(uint4*)(v_smem + tile_t::first_chunk(j) + lane_id()) = make_uint4(0, 0, 0, 0);
+  }
+}
+
+// Wait for this thread's copies, then for the threadblock: the tile is in shared memory, and
+// every wavefront has finished reading the buffer the next copy overwrites.
+__device__ __forceinline__ void kv_tile_barrier() {
+  asm volatile("" ::: "memory");
+  __builtin_amdgcn_s_waitcnt(0x70);  // vmcnt(0) lgkmcnt(0)
+  __builtin_amdgcn_s_barrier();
+  asm volatile("" ::: "memory");
+}
+#endif
+
 /*!
  * \brief Fragment state of one wavefront: num_frags_x * kWarpsPerWave query tiles of 16 rows.
  */
@@ -391,8 +617,9 @@ template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z>
 struct state_t {
   static constexpr uint32_t num_q_tiles = num_frags_x * kWarpsPerWave;
   static constexpr bool q_in_regs = q_in_registers(num_frags_x, num_frags_y, kShape);
-  // q[qx][fy]: the B operand Q^T, scaled, when the query tile is kept in registers.
-  operand_t q[q_in_regs ? num_q_tiles : 1][q_in_regs ? num_frags_y : 1];
+  // q[qx][fy / kFragsPerK]: the B operand Q^T, scaled, when the query tile is kept in
+  // registers.
+  operand_t q[q_in_regs ? num_q_tiles : 1][q_in_regs ? num_frags_y / kFragsPerK : 1];
   // s[qx][fz]: S^T for 16 kv rows x 16 query columns.
   float s[num_q_tiles][num_frags_z][kAccElems];
   // o[qx][fy]: O^T for 16 head dimensions x 16 query columns.
@@ -439,12 +666,12 @@ __device__ __forceinline__ void compute_qk(smem_t* q_smem, smem_t* k_smem,
     }
   }
 #pragma unroll
-  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
+  for (uint32_t fy = 0; fy < num_frags_y; fy += kFragsPerK) {
     operand_t q_op[state::num_q_tiles];
 #pragma unroll
     for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
       if constexpr (state::q_in_regs) {
-        q_op[qx] = st.q[qx][fy];
+        q_op[qx] = st.q[qx][fy / kFragsPerK];
       } else {
         load_row<channel_size_128b_in, DTypeIn>(q_smem, q_row0 + qx * 16 + col, fy * 16 + k0,
                                                 q_op[qx]);
@@ -526,7 +753,7 @@ __device__ __forceinline__ void compute_sfm_v(smem_t* v_smem,
   using state = state_t<num_frags_x, num_frags_y, num_frags_z>;
   // P^T in the B layout, rounded to the input type as in the mma.sync path. The denominator
   // sums the rounded values, as the mma.sync path's rowsum does.
-  operand_t p_op[state::num_q_tiles][num_frags_z];
+  operand_t p_op[state::num_q_tiles][num_frags_z / kFragsPerK];
 #pragma unroll
   for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
 #pragma unroll
@@ -556,23 +783,33 @@ __device__ __forceinline__ void compute_sfm_v(smem_t* v_smem,
         p_op[qx][fz].r[2 * i + 1] = (g0 >> 16) | (g1 & 0xffff0000u);
       }
 #else
+      // For 16x16x32 the k elements of a lane are kv rows acc_row(0) .. + 3 of fragment fz and
+      // then of fragment fz + 1; the V^T operand below follows the same order.
 #pragma unroll
       for (uint32_t i = 0; i < kAccElems / 2; ++i) {
-        p_op[qx][fz].r[i] = uint32_t(p[2 * i]) | (uint32_t(p[2 * i + 1]) << 16);
+        p_op[qx][fz / kFragsPerK].r[(fz % kFragsPerK) * kAccElems / 2 + i] =
+            uint32_t(p[2 * i]) | (uint32_t(p[2 * i + 1]) << 16);
       }
 #endif
     }
   }
   const uint32_t col = lane_id() % 16, k0 = op_k0();
 #pragma unroll
-  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
+  for (uint32_t fz = 0; fz < num_frags_z; fz += kFragsPerK) {
 #pragma unroll
     for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
       operand_t v_op;
+#if QUEST_PREFILL_MMA == 2
+      constexpr uint32_t channel_size_128b_in = num_frags_y * 16 / num_elems_per_128b<DTypeIn>();
+      const uint32_t kv0 = 4 * (lane_id() / 16);
+      load_v_tr<channel_size_128b_in, DTypeIn>(v_smem, fy * 16, fz * 16 + kv0, &v_op.r[0]);
+      load_v_tr<channel_size_128b_in, DTypeIn>(v_smem, fy * 16, (fz + 1) * 16 + kv0, &v_op.r[2]);
+#else
       load_vt<num_frags_z>(v_smem, fy * 16 + col, fz * 16 + k0, v_op);
+#endif
 #pragma unroll
       for (uint32_t qx = 0; qx < state::num_q_tiles; ++qx) {
-        mma<DTypeIn>(st.o[qx][fy], v_op, p_op[qx][fz]);
+        mma<DTypeIn>(st.o[qx][fy], v_op, p_op[qx][fz / kFragsPerK]);
       }
     }
   }
@@ -615,7 +852,7 @@ __device__ __forceinline__ void load_q(state_t<num_frags_x, num_frags_y, num_fra
                                    qo_indptr_base + qo_idx, kv_head_idx * group_size + row % group_size,
                                    k0, qo_len, num_qo_heads);
 #pragma unroll
-    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
+    for (uint32_t fy = 0; fy < num_frags_y; fy += kFragsPerK) {
 #pragma unroll
       for (uint32_t j = 0; j < kOpElems / 2; ++j) {
         uint32_t x = 0;
@@ -624,7 +861,7 @@ __device__ __forceinline__ void load_q(state_t<num_frags_x, num_frags_y, num_fra
           x = uint32_t(to_bits<DTypeIn>(from_bits<DTypeIn>(raw & 0xffff) * sm_scale)) |
               (uint32_t(to_bits<DTypeIn>(from_bits<DTypeIn>(raw >> 16) * sm_scale)) << 16);
         }
-        st.q[qx][fy].r[j] = x;
+        st.q[qx][fy / kFragsPerK].r[j] = x;
       }
     }
   }
```
