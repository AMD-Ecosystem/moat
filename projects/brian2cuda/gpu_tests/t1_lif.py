"""LIF group, 100 neurons, 20 ms: GPU against the closed form and Brian2's CPU runtime.

Neurons with drive I < 1 never spike, so v(t) = I * (1 - exp(-t/tau)) exactly.
Spike counts per neuron must equal the CPU runtime's (same exact integration).
"""
import sys
import numpy as np
from _common import gpu_setup, ref_setup, run_ref, result, builddir_arg, backend

REF = '--ref' in sys.argv
if REF:
    ref_setup()
else:
    gpu_setup(builddir_arg('t1_lif'))

from brian2 import NeuronGroup, SpikeMonitor, StateMonitor, run, ms

tau = 10 * ms
G = NeuronGroup(100, '''dv/dt = (I - v)/tau : 1
                        I : 1 (constant)''',
                threshold='v > 1', reset='v = 0', method='exact')
G.I = 'i * 0.025'          # neurons 0..40 have I <= 1 and never spike
M = StateMonitor(G, 'v', record=True)
S = SpikeMonitor(G)
run(20 * ms)

counts = np.bincount(np.asarray(S.i), minlength=100)
if REF:
    print(' '.join(map(str, counts)))
    sys.exit(0)

ref_counts = np.array(run_ref(__file__).split(), dtype=int)
quiet = np.arange(100)[np.arange(100) * 0.025 < 1.0]
t = np.asarray(M.t / ms)
expected = (quiet[:, None] * 0.025) * (1 - np.exp(-t[None, :] / 10.0))
err = np.max(np.abs(np.asarray(M.v)[quiet] - expected))
ok = err < 1e-12 and np.array_equal(counts, ref_counts) and counts.sum() > 0
result('t1_lif', ok, f'backend={backend()} spikes={counts.sum()} cpu_spikes={ref_counts.sum()} '
       f'per_neuron_equal={np.array_equal(counts, ref_counts)} quiet_max_err={err:.2e}')
