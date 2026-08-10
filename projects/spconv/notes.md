# spconv notes

## Why this is back

Re-opened 2026-08-07 after FlyDSL became available.

The prior screen declined this because the kernels are generated at build time by cumm, a CUTLASS-derived Python code generator, so hipify has nothing to translate and porting spconv means porting cumm's codegen backend. That is still true, but the target has changed: FlyDSL is a Python DSL emitting AMD tensor-core kernels with layout algebra modelled on CuTe, so a cumm backend targeting FlyDSL is a much closer correspondence than CK C++ templates. Bounded by cumm's GEMM/conv templates rather than unbounded.

Upstream last pushed 2024-12-15, so check whether it is dormant before investing. The value is real -- 2.3k stars, core to 3D point-cloud perception, and no AMD spconv exists.

## The prior analysis

Do not redo it. The earlier screen's full write-up is in history:

    git show b40576d53399:projects/spconv/plan.md
    git show b40576d53399:projects/spconv/notes.md

Read it first and test only what has changed.
