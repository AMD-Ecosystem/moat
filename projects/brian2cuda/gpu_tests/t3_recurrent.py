"""200-neuron recurrent network, ~12k synapses with random delays, 100 ms.

Stresses concurrent spike-queue pushes. GPU and CPU runtime draw different
random streams, so the comparison is statistical: the GPU spike count, averaged
over 3 seeds, must be within 5% of the CPU runtime's mean over the same seeds.
"""
import sys
import numpy as np
import _common
from _common import result, builddir_arg, backend

REF = '--ref' in sys.argv
seed_off = int(sys.argv[-1]) if REF else None


def build_and_run(offset):
    from brian2 import NeuronGroup, Synapses, SpikeMonitor, run, ms, seed
    seed(_common.SEED + offset)
    G = NeuronGroup(200, '''dv/dt = (I - v)/(10*ms) : 1
                            I : 1 (constant)''',
                    threshold='v > 1', reset='v = 0', refractory=2 * ms,
                    method='exact')
    G.I = '0.9 + 0.3*rand()'
    G.v = 'rand()'
    S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
    S.connect(p=0.3, condition='i != j')
    S.w = '0.02*rand() - 0.006'
    S.delay = '0.5*ms + 4*ms*rand()'
    M = SpikeMonitor(G)
    run(100 * ms)
    return len(S.i[:]), M.num_spikes


if REF:
    from brian2 import prefs
    prefs.codegen.target = 'numpy'
    syn, spikes = build_and_run(seed_off)
    print(syn, spikes)
    sys.exit(0)

if len(sys.argv) > 2 and sys.argv[2] == '--one':
    off = int(sys.argv[3])
    import brian2cuda  # noqa: F401
    from brian2 import set_device
    set_device('cuda_standalone', directory=builddir_arg(f't3_recurrent_{off}'),
               build_on_run=True)
    syn, spikes = build_and_run(off)
    print(syn, spikes)
    sys.exit(0)

import subprocess
gpu, cpu, syns = [], [], []
for off in range(3):
    out = subprocess.run([sys.executable, __file__, sys.argv[1] if len(sys.argv) > 1 else '/tmp/b2c_gpu_tests',
                          '--one', str(off)], check=True, capture_output=True, text=True).stdout.split()
    syns.append(int(out[-2])); gpu.append(int(out[-1]))
    cpu.append(int(_common.run_ref(__file__, off).split()[-1]))
rel = abs(np.mean(gpu) - np.mean(cpu)) / np.mean(cpu)
ok = min(syns) > 10000 and min(gpu) > 0 and rel < 0.05
result('t3_recurrent', ok, f'backend={backend()} synapses={syns} gpu_spikes={gpu} cpu_spikes={cpu} '
       f'mean_rel_diff={rel:.3f}')
