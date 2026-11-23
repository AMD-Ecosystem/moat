"""C99 math functions called with an integer argument in device code.

The generated _brian_<func> template converts integral arguments to the
default float type in the device branch. y = sin(i) + exp(-i) + sqrt(i) +
log1p(i) with integer i, set from device code, must match numpy to 1e-12.
"""
import sys
import numpy as np
from _common import gpu_setup, result, builddir_arg, backend

gpu_setup(builddir_arg('t7_int_math'))

from brian2 import NeuronGroup, run, ms

N = 1000
G = NeuronGroup(N, '''y : 1
                      k : integer (constant)''')
G.k = 'i % 50'
G.run_regularly('y = sin(k) + exp(-k) + sqrt(k) + log1p(k) + floor(k/3)', dt=0.1 * ms)
run(0.2 * ms)

k = np.arange(N) % 50
exp_y = np.sin(k) + np.exp(-k) + np.sqrt(k) + np.log1p(k) + k // 3
err = np.max(np.abs(np.asarray(G.y[:]) - exp_y))
result('t7_int_math', err < 1e-12, f'backend={backend()} max_err={err:.2e}')
