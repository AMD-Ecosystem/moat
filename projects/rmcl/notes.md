# rmcl notes

## The ROCm port lives in rmagine, not here

rmcl is a thin ROS 2 layer over the rmagine ray-casting library, which it pulls as an external git dependency via `source_dependencies.yaml` rather than vendoring. The GPU compute rmcl uses IS rmagine's CUDA backend, so that is what was ported: rmagine has 21 `.cu` files to rmcl's 3.

That work was done on `AMD-Ecosystem/rmagine` between 2026-06-01 and 2026-06-05 and validated on four architectures, but `projects/rmagine/` did not exist until 2026-08-06, so it was recorded here under rmcl's name. The record has been moved to `projects/rmagine/`, which is where the notes, the plan and the validation history now live. `depends_on: [rmagine]` records the relationship the tooling can act on.

Nothing about the port changed; only where it is filed.

## What rmcl itself still needs

Its own GPU code is unported: `rmcl_ros`'s `particle_motion.cu` and `resampling.cu`, plus the MICP CUDA sensors and the ROS 2 nodes. Both `.cu` files look mechanical -- curand to hiprand, and resampling's reduction already runs the full `__syncthreads` tree, so it is wave-safe. They need ROS 2 jazzy and Embree, which were not available on the host that did the rmagine work, and they should be built through rmagine's HIP toolchain in a colcon workspace.

This needs a fresh plan: the one written in 2026-05 analysed rmcl and concluded the port target was rmagine, and it moved with the work it produced.
