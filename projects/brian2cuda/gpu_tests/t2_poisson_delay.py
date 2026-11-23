"""Poisson input into 50 targets through 1 ms synaptic delays (spike-queue path).

Checks: no target receives input before the delay has elapsed, every target is
driven, and each target's received input equals the number of source spikes on
its synapses (counted on the host from the SpikeMonitor and connectivity).
"""
import sys
import numpy as np
from _common import gpu_setup, result, builddir_arg, backend

gpu_setup(builddir_arg('t2_poisson_delay'))

from brian2 import (PoissonGroup, NeuronGroup, Synapses, SpikeMonitor,
                    StateMonitor, run, ms, Hz)

P = PoissonGroup(100, rates=200 * Hz)
T = NeuronGroup(50, 'n_in : 1')               # counts arriving spikes, never resets
S = Synapses(P, T, on_pre='n_in += 1', delay=1 * ms)
S.connect(p=0.2)
MP = SpikeMonitor(P)
MT = StateMonitor(T, 'n_in', record=True)
run(20 * ms)

src_i = np.asarray(S.i)
tgt_j = np.asarray(S.j)
sp_i = np.asarray(MP.i)
sp_t = np.asarray(MP.t / ms)
n_in = np.asarray(MT.n_in)
t = np.asarray(MT.t / ms)

# a spike in step k is delivered in step k + 10; the run has steps 0..199
delivered = sp_t <= 18.9 + 1e-6
per_source = np.bincount(sp_i[delivered], minlength=100)
expected = np.bincount(tgt_j, weights=per_source[src_i], minlength=50)
early = n_in[:, t < 1.0 - 1e-9].max()
final = np.asarray(T.n_in[:])
ok = (len(src_i) > 0 and early == 0 and np.all(final > 0)
      and np.array_equal(final, expected))
result('t2_poisson_delay', ok, f'backend={backend()} synapses={len(src_i)} source_spikes={len(sp_i)} '
       f'early_input={early:.0f} targets_driven={int(np.sum(final > 0))}/50 '
       f'input_total={final.sum():.0f} expected_total={expected.sum():.0f}')
