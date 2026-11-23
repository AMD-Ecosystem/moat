"""Generate a cuda_standalone project for one case (no compile, no run).

usage: gen_case.py <case> <outdir> [cuda|hip]

`cuda` asserts the CUDA backend is selected (GPU detection off, compute
capability 7.5, as upstream's compile-only test does), so it runs on a host
without an NVIDIA GPU. Used to diff generated sources between two trees.
"""
import sys

case, outdir = sys.argv[1:3]
want = sys.argv[3] if len(sys.argv) > 3 else 'cuda'

import numpy as np
import brian2cuda  # noqa: F401
from brian2 import *
try:
    from brian2cuda.utils.hip_backend import is_hip_backend
except ImportError:  # tree without the HIP backend
    def is_hip_backend():
        return False

set_device('cuda_standalone', build_on_run=False)
prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0
prefs.devices.cuda_standalone.cuda_backend.compute_capability = 7.5
seed(20261123)
assert is_hip_backend() == (want == 'hip'), f'expected the {want} backend'
print('brian2cuda from', brian2cuda.__file__, 'backend', want)

if case == 'smoke':
    G = NeuronGroup(100, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    S = Synapses(G, G, on_pre='v += 0.1')
    S.connect(j='i')
    run(10*ms)

elif case == 'cuba':
    taum, taue, taui = 20*ms, 5*ms, 10*ms
    eqs = '''dv/dt = (ge+gi-(v+49*mV))/taum : volt (unless refractory)
             dge/dt = -ge/taue : volt
             dgi/dt = -gi/taui : volt'''
    P = NeuronGroup(1000, eqs, threshold='v>-50*mV', reset='v=-60*mV',
                    refractory=5*ms, method='exact')
    P.v = '-60*mV + rand()*10*mV'
    Ce = Synapses(P, P, 'w : volt', on_pre='ge += w')
    Ci = Synapses(P, P, on_pre='gi -= 9*mV')
    Ce.connect('i<800', p=0.02)
    Ci.connect('i>=800', p=0.02)
    Ce.w = '1.62*mV*rand()'
    Ce.delay = 'rand()*3*ms'
    Ci.delay = 1*ms
    Pin = PoissonGroup(100, rates=20*Hz)
    Sin = Synapses(Pin, P, on_pre='v += 0.5*mV')
    Sin.connect(j='k for k in range(i*10, (i+1)*10)')
    M1 = SpikeMonitor(P)
    M2 = StateMonitor(P, 'v', record=[0, 1, 2])
    M3 = PopulationRateMonitor(P)
    M4 = EventMonitor(P, 'spike', variables=['v'])
    run(20*ms)

elif case == 'stdp':
    taupre = taupost = 20*ms
    G = NeuronGroup(200, '''dv/dt = (I - v)/(10*ms) + 0.1*xi/sqrt(ms) : 1
                            I : 1''', threshold='v>1', reset='v=0', method='euler')
    H = NeuronGroup(50, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0')
    S = Synapses(G, H, '''w : 1
                          dapre/dt = -apre/taupre : 1 (event-driven)
                          dapost/dt = -apost/taupost : 1 (event-driven)''',
                 on_pre='''v_post += w
                           apre += 0.01
                           w = clip(w + apost, 0, 1)''',
                 on_post='''apost -= 0.0105
                            w = clip(w + apre, 0, 1)''')
    S.connect(p=0.3)
    S.w = 'rand()*0.5'
    G.I = '1.1 + 0.1*randn()'
    M1 = SpikeMonitor(H)
    run(20*ms)

elif case == 'timed_poisson':
    # TimedArray 1-D and 2-D, poisson() with per-neuron and scalar lambda,
    # integer-argument math functions
    ta = TimedArray(np.linspace(0, 2, 50), dt=1*ms)
    ta2 = TimedArray(np.ones((20, 100)), dt=1*ms)
    G = NeuronGroup(100, '''dv/dt = (ta(t) + ta2(t, i) - v)/(10*ms) : 1
                            lam : 1
                            k : integer
                            y : 1
                            n : integer
                            m : integer''', threshold='v>1', reset='v=0')
    G.lam = 'i*0.1'
    G.k = 'i % 7'
    G.run_regularly('''n = poisson(lam)
                       m = poisson(2.0)
                       y = sin(k) + exp(-k)''', dt=1*ms)
    M1 = StateMonitor(G, 'n', record=[0, 50])
    run(20*ms)

elif case == 'binomial':
    # brian2 releases that unpack two values refuse brian2cuda's 3-tuple;
    # attach the generator's code and headers by hand (as gpu_tests/t6 does)
    gen = BinomialFunction.implementations['cuda']
    kwds = {}

    def two_tuple(**kw):
        code, deps, ck = gen(**kw)
        kwds[kw['name']] = (code, deps, ck)
        return code, deps

    BinomialFunction.implementations['cuda'] = two_tuple
    funcs = []
    for n, p in [(10, 0.2), (10000, 0.4)]:
        f = BinomialFunction(n, p)
        code, deps, ck = kwds[f.name]
        f.implementations.add_implementation('cuda', code, dependencies=deps,
                                             name=f.name, compiler_kwds=ck)
        funcs.append(f)
    b_inv, b_norm = funcs
    G = NeuronGroup(10, '''s1 : 1
                           s2 : 1''')
    G.run_regularly('''s1 += b_inv()
                       s2 += b_norm()''')
    run(1*ms)

elif case == 'spatial':
    morpho = Soma(30*um)
    morpho.axon = Cylinder(diameter=1*um, length=300*um, n=50)
    gL = 1e-4*siemens/cm**2
    EL = -70*mV
    eqs = '''Im = gL * (EL - v) : amp/meter**2
             I : amp (point current)'''
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1*uF/cm**2, Ri=100*ohm*cm)
    neuron.v = EL
    neuron.I[0] = 0.2*nA
    M1 = StateMonitor(neuron, 'v', record=True)
    run(5*ms)

else:
    raise SystemExit('unknown case ' + case)

device.build(directory=outdir, compile=False, run=False)
print('GENERATED', outdir)
