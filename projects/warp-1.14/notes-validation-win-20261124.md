# Windows validation 2026-11-24 (windows-gfx1151), 66ba4ffb: commands and per-file counts

Scripts live in agent_space/p6ev/ (not committed): build.sh, exbuild.sh and exbuild2.sh, runfiles.sh, mk_k2.py, summarize.py.

Build: put the devel venv's Scripts directory first on PATH, then `python build_lib.py -j 8`. Wheel: `python -m build --wheel -C--build-option=-Pwindows-x86_64 --outdir dist` (from a venv holding only `build`). Install: `pip install dist/<wheel> --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple`, then `pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "torch[device-gfx1151]==2.15.0+rocm10.2.0" numpy==1.26.4 usd-core`, plus psutil and jinja2 (GLAD).

Test run, per file: from a directory outside the tree, `WARP_CACHE_PATH=<fresh dir> timeout 900 python -m warp.tests -s autodetect -p <file> -m 2`. For the two partial files add one `-k <test name>` per loaded test except `test_fill_large_cuda_0` (test_array.py) and the four `test_copy_large_stride_*` (test_copy.py); mk_k2.py `<module> <excluded substring>` loads the module and prints those flags (only its last output line; the init banner precedes it).

### Step 3: held and risk list (test files read; nothing run for this step)

`device.free_memory` here reports 112 GB, so every memory-gated test below runs unless excluded.

| file / test | what it does | size | risk and action |
|---|---|---|---|
| `test_large.py` (3 GPU tests) | `test_large_launch_large_kernel` 2*2^16*2^16 = 2^33 threads; `test_large_launch_very_large_kernel` (2^31-1)*256+1 = 5.5e11 threads (grid-stride, capped at 2^32-1 per dimension by the port); `test_large_arrays` 2^31-element int8 arrays in 2-D/3-D/4-D (about 2 GiB each) filled and read by kernels | 2 GiB live, launches of billions of threads | HIGH, held |
| `test_examples.py` | about 35 whole example simulations as subprocesses on cuda:0 (core.example_dem, fluid, mesh, sph, torch (1000 frames), wave, ..., optim, tile), 600 s each; `example_sph` was running when a device-wide launch failure broke the pool on gfx1250 | many processes, long runs | HIGH, held |
| `test_array.py::test_fill_large_cuda_0` | `wp.zeros((65536, 65537), int16)` = 2^32 + 65536 elements, 8.6 GB, `fill_` with one thread per element (more than 2^32 threads); runs because free_memory > 12 GB; added after session 18, never run on Windows | 8.6 GB | HIGH, excluded; the other 139 tests ran |
| `test_copy.py::test_copy_large_stride` x4 (cpu_cpu, cpu_cuda0, cuda0_cpu, cuda0_cuda0) | N = 500M x 2 float32 (4 GB strided source) plus a 2 GB destination, fills and copies; session 18's run (14 tests) executed these on this laptop without incident | 6 GB | HIGH, excluded; the other 10 tests ran |
| `test_tile_large_offsets.py` | one test: `wp.empty` of 2.25 GiB uint8, writes a few bytes at offsets above 2 GiB (1.9 s in session 18) | 2.25 GiB | medium, ran (1 OK, 2 s here) |
| `cuda/test_streams.py` | `test_stream_priority_timings` allocates two 1 GiB arrays before skipping on HIP; `test_stream_event_is_complete` 5 launches of 64M-thread atomic kernels | 2 GiB, 5 x 64M threads | medium, ran |
| `test_graph.py` | one test captures 100 kernels over 64M floats (256 MB) plus a pinned 256 MB copy; a `spin = 2^24` busy-loop kernel in a graph (TDR exposure on a slow GPU); a 64M-float leak test is skipped upstream | 256 MB to 1 GB | medium, ran (35 tests, 3 skipped) |
| `cuda/test_async.py` | 2373 copy/graph combinations on 1M-float arrays, 10-iteration loops; the gfx950 copy flake lives here | 4 MB per test | medium (long, concurrent streams), ran 139 s |
| `fem/test_fem_examples.py` (and `test_fem_multi_env`) | 23 FEM examples as subprocesses at small settings (resolution 10-25, 1-101 frames) | small | medium, ran 459 s |
| `cuda/test_ipc.py` | `mp.spawn` child opens an IPC handle, `process.join()` has no timeout | 1 KB | low here: `is_ipc_supported` is False, 4 skipped |
| `test_assert.py` | 11 subprocesses that abort on purpose, CPU only | none | low, ran |
| `test_diagnostics.py`, `test_logger.py` | `python -c` subprocesses | none | low, ran |
| `test_multigpu.py`, `test_peer.py` | 10-iteration copies of 1M floats between two GPUs | 4 MB | low, skipped for one GPU |
| `test_allocation_tracker.py`, `test_module_parallel_load.py` | 4 threads, small arrays; CPU module builds | small | low |
| in-kernel loops: `test_adam`/`test_sgd` (100 iterations), `test_triangle_closest_point` (1000), `tile/test_tile_shared_memory` (4096 stack-allocator iterations) | small grids | small | low, ran |
| `interop/test_jax.py`, `test_paddle.py` | packages not installed | 0 tests | none |

No other test file has a repeat loop of more than a few hundred GPU launches, an allocation above 2.25 GiB, or a multi-process stress (searched `warp/tests` for `range(`, size literals, `subprocess`, `multiprocessing`, `threading` and `free_memory`, then read each hit).


- Skips by reason (count): no suitable devices 18, conditional graph nodes not supported 16, "unreliable stdout capture" on Windows 11 (test_ipc, test_print, test_verify_fp), MathDx not built 8+1+1, a second CUDA device 7+2, ml_dtypes absent 5, mempool access 5+1+1+1, IPC unsupported 3, peer access 3+1+1+1, "NVRTC bug on Windows" 2, NVRTC two archs 2, tile tests needing 66688/67584/66816 bytes of shared memory with 65536 available 2+2+1, cuda:0 cannot access CPU memory 2, rmm 1, upstream memory-usage flakiness 1, stream priority without scheduling effect on HIP 1, "Requires CUDA" 1, NVRTC not available 1.

Per file, tests run / skipped (all OK):

test_adam.py 6/0; test_allocation_tracker.py 34/0; test_allocator.py 16/3; test_apic.py 52/0; test_apic_mesh.py 12/0; test_arithmetic.py 126/0; test_array.py 139/0; test_array_fill_capture.py 3/0; test_array_reduce.py 16/0; test_assert.py 11/0; test_async.py 2373/0; test_atomic.py 164/0; test_atomic_bitwise.py 6/0; test_atomic_cas.py 48/0; test_bf16.py 48/5; test_block_dim_dispatch.py 2/0; test_bool.py 14/0; test_builtins_resolution.py 48/0; test_bvh.py 13/0; test_capture_mode.py 2/0; test_clang_cuda.py 7/6; test_closest_point_edge_edge.py 28/0; test_codegen.py 117/0; test_codegen_instancing.py 80/0; test_coloring.py 3/0; test_compilation.py 3/0; test_compile_consts.py 33/0; test_composite_component_adjoint.py 18/0; test_conditional.py 50/0; test_conditional_captures.py 32/16; test_constant_precision.py 46/0; test_context.py 1/0; test_copy.py 10/0; test_cpu_precompiled_headers.py 3/0; test_ctypes.py 36/0; test_cuda_arch_suffix.py 10/10; test_dense.py 2/0; test_devices.py 14/0; test_diagnostics.py 14/0; test_dlpack.py 16/0; test_enum.py 14/0; test_fabricarray.py 19/0; test_factory_style_array_annotations.py 25/0; test_fast_math.py 14/1; test_fastcall.py 9/0; test_fem_examples.py 23/0; test_fem_field.py 6/0; test_fem_fp64.py 16/0; test_fem_geometry.py 22/1; test_fem_integrate.py 17/0; test_fem_linalg.py 5/0; test_fem_multi_env.py 10/1; test_fem_quadrature.py 5/0; test_fem_shape.py 4/0; test_fixedarray.py 20/0; test_fp16.py 6/0; test_func.py 35/0; test_future_annotations.py 2/0; test_generics.py 36/0; test_grad.py 41/0; test_grad_customs.py 17/0; test_grad_debug.py 16/0; test_graph.py 35/3; test_grouped_bvh.py 11/0; test_hash_grid.py 23/0; test_implicit_init.py 21/0; test_import.py 2/0; test_indexedarray.py 38/1; test_intersect.py 2/0; test_ipc.py 4/4; test_iter.py 2/0; test_jax.py 0/0; test_kernel_cache.py 5/0; test_launch.py 32/0; test_lerp.py 2/0; test_linear_solvers.py 34/0; test_logger.py 34/0; test_lvalue.py 29/0; test_map.py 45/0; test_marching_cubes.py 17/0; test_mat.py 134/0; test_mat_assign_copy.py 6/0; test_mat_basics.py 121/0; test_mat_constructors.py 54/0; test_mat_elementwise_ops.py 110/0; test_mat_linalg.py 154/0; test_mat_lite.py 6/0; test_math.py 16/0; test_mempool.py 7/2; test_mesh.py 16/0; test_mesh_query_aabb.py 15/0; test_mesh_query_point.py 9/0; test_mesh_query_ray.py 9/0; test_module_aot.py 23/3; test_module_contamination.py 2/0; test_module_hashing.py 8/0; test_module_parallel_load.py 5/0; test_modules_lite.py 4/0; test_multigpu.py 5/5; test_noise.py 14/0; test_occupancy.py 6/0; test_operators.py 20/0; test_options.py 22/0; test_overwrite.py 25/0; test_paddle.py 0/0; test_peer.py 4/2; test_pinned.py 1/0; test_print.py 10/8; test_quat.py 167/0; test_quat_assign_copy.py 6/0; test_rand.py 10/0; test_reload.py 14/0; test_rounding.py 2/0; test_runlength_encode.py 16/0; test_scalar_ops.py 44/0; test_sgd.py 14/0; test_smoothstep.py 2/0; test_snippet.py 12/0; test_sparse.py 47/0; test_spatial.py 201/0; test_spatial_assign_copy.py 6/0; test_special_values.py 74/0; test_static.py 28/0; test_streams.py 25/3; test_struct.py 46/0; test_subscript_types.py 44/0; test_tape.py 19/0; test_template_launch_bounds.py 36/0; test_texture.py 120/1; test_tile.py 126/0; test_tile_atomic_bitwise.py 3/0; test_tile_cholesky.py 33/0; test_tile_cholesky_no_mathdx.py 16/0; test_tile_empty.py 20/0; test_tile_fft.py 20/8; test_tile_fft_no_mathdx.py 16/0; test_tile_func_arg.py 27/0; test_tile_fused_ops.py 36/0; test_tile_large_offsets.py 1/0; test_tile_load.py 64/0; test_tile_mathdx.py 6/0; test_tile_matmul.py 26/0; test_tile_matmul_no_mathdx.py 22/0; test_tile_mlp.py 3/1; test_tile_reduce.py 62/2; test_tile_shared_memory.py 30/2; test_tile_solve.py 5/1; test_tile_solve_no_mathdx.py 28/0; test_tile_sort.py 2/0; test_tile_stack.py 40/0; test_tile_view.py 16/1; test_torch.py 34/0; test_transient_module.py 2/0; test_triangle_closest_point.py 2/0; test_tuple.py 16/0; test_types.py 53/0; test_unified_memory.py 27/8; test_unique_module.py 23/0; test_unpack.py 32/0; test_utils.py 62/0; test_vec.py 89/0; test_vec_assign_copy.py 8/0; test_vec_constructors.py 26/0; test_vec_lite.py 6/0; test_vec_scalar_ops.py 402/0; test_verify_fp.py 4/2; test_version.py 4/0; test_volume.py 59/0; test_volume_write.py 5/0; 

### Step 5: the held tests (2026-11-24)

Commands: `agent_space/p6ev/heldrun.sh` over `heldlist.txt` (one line per test): from the directory outside the tree, venv with the installed wheel, `ulimit -c 0`, `WARP_CACHE_PATH` emptied before each test, a `progress.log` line before each test, `timeout 900 python -m warp.tests -s autodetect -p <file> -k <exact test name> -m 2`; after each test the script checks the log for hipError/`Warp CUDA error`/device-lost lines, rc 124, and a new Display event 4101 (TDR) and would stop at the first. TdrDelay and TdrDdiDelay are 0x3c (60 s) in the registry. Pillow was added to the venv for the two examples that need it.

Wall time per test (ms, process wall including start-up and module build), all rc=0 and `Ran 1 test ... OK`, none skipped:

- test_copy_large_stride_cpu_cpu: 10505 ms
- test_copy_large_stride_cpu_cuda0: 10188 ms
- test_copy_large_stride_cuda0_cpu: 5623 ms
- test_copy_large_stride_cuda0_cuda0: 2306 ms
- test_fill_large_cuda_0: 7884 ms
- test_large_array_excessive_ndarray: 1628 ms
- test_large_array_excessive_numpy: 1594 ms
- test_large_array_excessive_zeros_cpu: 2271 ms
- test_large_array_excessive_zeros_cuda_0: 1611 ms
- test_large_launch_max_blocks_cpu: 2465 ms
- test_large_launch_max_blocks_cuda_0: 2537 ms
- test_large_arrays_cpu: 61341 ms
- test_large_arrays_cuda_0: 6140 ms
- test_large_launch_large_kernel_cuda_0: 2678 ms
- test_large_launch_very_large_kernel_cuda_0: 8284 ms
- test_core.example_custom_allocator_cuda_0: 2233 ms
- test_core.example_dem_cpu: 16219 ms
- test_core.example_dem_cuda_0: 31128 ms
- test_core.example_fft_poisson_navier_stokes_2d_cuda_0: 5329 ms
- test_core.example_fluid_cpu: 8649 ms
- test_core.example_fluid_cuda_0: 3842 ms
- test_core.example_graph_capture_cpu: 3662 ms
- test_core.example_graph_capture_cuda_0: 3372 ms
- test_core.example_marching_cubes_cuda_0: 6205 ms
- test_core.example_mesh_cpu: 3495 ms
- test_core.example_mesh_cuda_0: 4206 ms
- test_core.example_mesh_intersect_cpu: 4415 ms
- test_core.example_mesh_intersect_cuda_0: 4735 ms
- test_core.example_nvdb_cpu: 5974 ms
- test_core.example_nvdb_cuda_0: 4474 ms
- test_core.example_raycast_cpu: 3131 ms
- test_core.example_raycast_cuda_0: 3320 ms
- test_core.example_raymarch_cpu: 4382 ms
- test_core.example_sample_mesh_cpu: 3021 ms
- test_core.example_sample_mesh_cuda_0: 3481 ms
- test_core.example_sph_cpu: 6076 ms
- test_core.example_torch_cpu: 8716 ms
- test_core.example_torch_cuda_0: 9713 ms
- test_core.example_wave_cpu: 4061 ms
- test_core.example_wave_cuda_0: 3606 ms
- test_optim.example_diffray_cpu: 8445 ms
- test_optim.example_diffray_cuda_0: 51314 ms
- test_optim.example_fluid_checkpoint_cuda_0: 10621 ms
- test_optim.example_navier_stokes_perturbation_cuda_0: 6125 ms
- test_optim.example_particle_repulsion_cuda_0: 3362 ms
- test_tile.example_tile_block_cholesky_cpu: 3761 ms
- test_tile.example_tile_block_cholesky_cuda_0: 4949 ms
- test_tile.example_tile_cholesky_cpu: 3007 ms
- test_tile.example_tile_cholesky_cuda_0: 3123 ms
- test_tile.example_tile_convolution_cuda_0: 3240 ms
- test_tile.example_tile_fft_cuda_0: 3247 ms
- test_tile.example_tile_filtering_cuda_0: 3262 ms
- test_tile.example_tile_matmul_cpu: 3221 ms
- test_tile.example_tile_matmul_cuda_0: 3161 ms
- test_tile.example_tile_mcgp_cpu: 11900 ms
- test_tile.example_tile_mcgp_cuda_0: 5271 ms
- test_tile.example_tile_mlp_cuda_0: 8162 ms
- test_tile.example_tile_nbody_cpu: 2112 ms
- test_tile.example_tile_nbody_cuda_0: 3189 ms
- test_tile.example_tile_stream_compaction_cuda_0: 3043 ms
- test_core.example_sph_cuda_0: 29370 ms

### Step 6: windowed OpenGL run (2026-11-24, jeff at the laptop, same installed wheel and venv, pyglet 2.1.19)

Each item one process, `ulimit -c 0`, `timeout 300`, a `progress.log` line before it; scripts `agent_space/p6ev/gl_interop.py` and `gl_window.py`.
1. Tests skipped for lack of pyglet, OpenGL or a display in the 169-file run: **none**. No skip reason in any log names pyglet, OpenGL or a display (the 100 skips are listed above), and the suite has no GL test (`test_examples.py` has `example_render_opengl` commented out), so there was nothing to rerun. pyglet was installed only for items 2 and 3.
2. GL interop (`gl_interop.py`; hidden 64x64 pyglet window; GL vendor "ATI Technologies Inc.", renderer "AMD Radeon(TM) 8060S Graphics", GL 4.6 Compatibility Profile 26.11.2.261114): `wp.RegisteredGLBuffer` **registration succeeded on cuda:0 (zero-copy interop, no copy fallback, no warning)**, both directions. Forward: Warp kernel wrote `2*i+1` into the mapped 4096-float GL buffer, unmap, `glGetBufferSubData` read-back: exact match, max abs difference 0. Reverse: GL wrote 4096 known floats with `glBufferSubData`, a READ_ONLY registered buffer was mapped, a Warp kernel wrote `3*x` into a Warp array: exact match. RESULT PASS, rc=0. (The first attempt died on my own script, `glGetString` returns a pointer without `.decode`; fixed, not a HIP event.)
3. Windowed example: `warp.examples.fem.example_apic_fluid` driven by `gl_window.py` (the version has no `example_cloth`; this is the example that renders with the windowed `OpenGLRenderer`): 1024 x 1024 window, 68,921 particles (`--voxel-size 0.5`), 600 frames in 21.8 s, window stayed open, `get_pixels(..., split_up_tiles=False, mode="rgb")` of the last frame: shape (1024, 1024, 3), min 0.0, max 1.0, mean 0.796, std 0.195, 82% of pixels differ from the corner colour: **not blank** (`logs/gl_frame.png`: blue shaded spheres over a pale-blue gradient with a grid and axes). rc=0, no HIP error, no hang, no driver reset. The first window run (100 frames) failed only at my pixel capture (`get_pixels` with the default `split_up_tiles=True` asserts on a renderer without tiles; the simulation and window ran fine); rerun with the fixed call passed. The Python-exit `__del__`/`sys.meta_path is None` and pyglet focus-callback messages at shutdown are pyglet/renderer teardown noise, rc=0.

Verdict: every step clean at `66ba4ffb` (build, wheel, loadcheck, examples, 169 files, 61 held tests, GL). `jargon.py --port warp-1.14` is clean. The CUDA no-regression gate is not rerun here (no toolkit on this host; recorded by the Linux validators).
