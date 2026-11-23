"""TimedArray (1-D and 2-D) evaluated in device code.

x = ta1(t) and y = ta2(t, i) are recorded through StateMonitor and must equal
the array values exactly; v, an ODE driven by ta1(t), must match Brian2's CPU
runtime to 1e-12.
"""
import sys
import numpy as np
from _common import gpu_setup, ref_setup, run_ref, result, builddir_arg, backend

REF = '--ref' in sys.argv
if REF:
    ref_setup()
else:
    gpu_setup(builddir_arg('t4_timedarray'))

from brian2 import TimedArray, NeuronGroup, StateMonitor, run, ms

N, STEPS = 20, 200                       # dt = 0.1 ms, 20 ms
vals1 = np.sin(np.arange(40) * 0.37) + 1.5                          # 0.5 ms grid
vals2 = np.cos(np.arange(80)[:, None] * 0.11 + np.arange(N)[None, :] * 0.7)  # 0.25 ms grid
ta1 = TimedArray(vals1, dt=0.5 * ms)
ta2 = TimedArray(vals2, dt=0.25 * ms)
G = NeuronGroup(N, '''dv/dt = (ta1(t) - v)/(5*ms) : 1
                      x = ta1(t) : 1
                      y = ta2(t, i) : 1''', method='euler')
M = StateMonitor(G, ['v', 'x', 'y'], record=True)
run(STEPS * 0.1 * ms)

v = np.asarray(M.v)
if REF:
    np.savetxt(sys.stdout, v[:, -1:].T, fmt='%.17g')
    sys.exit(0)

t = np.round(np.asarray(M.t / ms) * 10).astype(int)          # step index
x_exp = vals1[np.minimum(t // 5, len(vals1) - 1)]
y_exp = vals2[np.minimum(t // 2.5, len(vals2) - 1).astype(int)].T
x_err = np.max(np.abs(np.asarray(M.x) - x_exp[None, :]))
y_err = np.max(np.abs(np.asarray(M.y) - y_exp))
v_ref = np.array(run_ref(__file__).split(), dtype=float)
v_err = np.max(np.abs(v[:, -1] - v_ref))
ok = x_err == 0 and y_err == 0 and v_err < 1e-12 and np.ptp(v[:, -1]) == 0
result('t4_timedarray', ok, f'backend={backend()} x_max_err={x_err:.2e} y_max_err={y_err:.2e} '
       f'v_end={v[0, -1]:.15f} v_max_err_vs_cpu={v_err:.2e}')
