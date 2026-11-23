"""BinomialFunction in device code: inversion (n=10, p=0.2) and normal
approximation (n=10000, p=0.4) variants.

brian2cuda's binomial code generator returns (code, dependencies,
compiler_kwds); Brian2 releases that unpack two values (2.10.1 among them)
refuse it in BinomialFunction.__init__ with "too many values to unpack". The
case reports that, then builds the same functions through a test-only adapter
(the generator's code and headers attached by hand) so the generated device
code is still compiled and run. With 20000 neurons x 100 draws, each sample
mean must be within 5 standard errors of n*p.
"""
import sys
import numpy as np
from _common import gpu_setup, result, builddir_arg, backend

gpu_setup(builddir_arg('t6_binomial'))

from brian2 import NeuronGroup, BinomialFunction, run, ms

note = 'brian2 accepts the generator'
try:
    BinomialFunction(10, 0.2)
except ValueError as exc:
    if 'unpack' not in str(exc):
        raise
    note = f'brian2 refuses the generator ({exc}); adapter used'
    gen = BinomialFunction.implementations['cuda']
    kwds = {}

    def two_tuple(**kw):
        code, deps, compiler_kwds = gen(**kw)
        kwds[kw['name']] = (code, deps, compiler_kwds)
        return code, deps

    BinomialFunction.implementations['cuda'] = two_tuple


def binomial(n, p):
    f = BinomialFunction(n, p)
    if f.name in globals().get('kwds', {}):
        code, deps, compiler_kwds = kwds[f.name]
        f.implementations.add_implementation('cuda', code,
                                 dependencies=deps, name=f.name,
                                 compiler_kwds=compiler_kwds)
    return f


N, DRAWS = 20000, 100
b_inv = binomial(10, 0.2)
b_norm = binomial(10000, 0.4)
G = NeuronGroup(N, '''s1 : 1
                      s2 : 1''')
G.run_regularly('''s1 += b_inv()
                   s2 += b_norm()''', dt=0.1 * ms)
run(DRAWS * 0.1 * ms)

m1 = np.asarray(G.s1[:]).sum() / (N * DRAWS)
m2 = np.asarray(G.s2[:]).sum() / (N * DRAWS)
z1 = abs(m1 - 2.0) / np.sqrt(10 * 0.2 * 0.8 / (N * DRAWS))
z2 = abs(m2 - 4000.0) / np.sqrt(10000 * 0.4 * 0.6 / (N * DRAWS))
result('t6_binomial', z1 < 5 and z2 < 5,
       f'backend={backend()} inv_mean={m1:.4f} (2) z={z1:.2f} norm_mean={m2:.3f} (4000) z={z2:.2f} [{note}]')
