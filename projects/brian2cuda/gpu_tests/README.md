# brian2cuda GPU test cases

Kept here so every validator runs the same files (the 2026-11-20 round lost its
scripts with the job directory). Each case is one process, fixed seed, and prints
one `RESULT <case> PASS|FAIL|SKIP ...` line; each docstring states what is compared
and the tolerance.

| case | what it proves |
|---|---|
| t1_lif | LIF group: quiet neurons match the closed form to 1e-12, spike counts per neuron equal Brian2's CPU runtime |
| t2_poisson_delay | Poisson input through 1 ms delays (spike-queue lock): no early input, every target's input equals the delivered source spikes |
| t3_recurrent | 200 neurons, ~12k synapses, random delays: no deadlock, mean spike count over 3 seeds within 5% of the CPU runtime |
| t4_timedarray | TimedArray 1-D and 2-D in device code equal the array values exactly; an ODE driven by it matches the CPU runtime |
| t5_poisson_lambda | `poisson()` with per-neuron lambda (0 to 500): mean and variance within 5 standard errors of lambda; scalar-lambda control |
| t6_binomial | BinomialFunction (inversion and normal approximation): means within 5 SE of n*p. Brian2 2.10.1 refuses brian2cuda's generator; the case says so and attaches the generated code by hand |
| t7_int_math | `_brian_<func>` math wrappers with integer arguments match numpy |

Run (HIP backend, one GPU, from the moat checkout, brian2cuda installed in `$PY`'s environment):

```
ulimit -c 0
USE_HIP=1 bash projects/brian2cuda/gpu_tests/run_all.sh $PY /tmp/b2c_gpu_tests
```

`cuda_codegen/compare.sh` generates six example projects with two brian2cuda trees
and diffs the generated sources (`cuda` or `hip` backend; no compile). For the CUDA
backend on a host without nvcc, put a version-only `nvcc` stub first on `PATH` and
leave `USE_HIP` unset; on the nvcc host the generated directories can then be built
with `make`. Some lines are ordered by iterating sets of objects and can move between
two runs of the same tree; the script reports a pure reordering separately.
