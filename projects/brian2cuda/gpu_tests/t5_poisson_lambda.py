"""poisson() with a per-neuron lambda (device-side cuRAND/hipRAND path).

12 lambda classes x 1000 neurons, 200 draws each (200k samples per class).
Per class, the sample mean must be within 5 standard errors of lambda
(SE = sqrt(lambda / 200000)) and the sample variance within 5 standard errors of lambda
(relative SE of a Poisson sample variance = sqrt((1/lambda + 2) / 200000)).
lambda = 0 must give only zeros. A scalar-lambda poisson (host RNG path) runs
in the same network as a control.
"""
import sys
import numpy as np
from _common import gpu_setup, result, builddir_arg, backend

gpu_setup(builddir_arg('t5_poisson_lambda'))

from brian2 import NeuronGroup, run, ms

LAMS = np.array([0.0, 0.05, 0.5, 1.0, 2.5, 5.0, 9.5, 20.0, 50.0, 63.0, 100.0, 500.0])
PER, DRAWS = 1000, 200
G = NeuronGroup(len(LAMS) * PER, '''lam : 1 (constant)
                                    s1 : 1
                                    s2 : 1
                                    c1 : 1''')
G.lam = np.repeat(LAMS, PER)
G.run_regularly('''n = poisson(lam)
                   s1 += n
                   s2 += n*n
                   c1 += poisson(3.0)''', dt=0.1 * ms)
run(DRAWS * 0.1 * ms)

s1 = np.asarray(G.s1[:]).reshape(len(LAMS), PER)
s2 = np.asarray(G.s2[:]).reshape(len(LAMS), PER)
c1 = np.asarray(G.c1[:])
n = PER * DRAWS
mean = s1.sum(1) / n
var = s2.sum(1) / n - mean ** 2
se = np.sqrt(np.maximum(LAMS, 1e-12) / n)
z = np.abs(mean - LAMS) / se
rows = []
ok = True
for lam, m, vv, zz in zip(LAMS, mean, var, z):
    if lam == 0:
        good = m == 0 and vv == 0
    else:
        good = zz < 5 and abs(vv / lam - 1) < 5 * np.sqrt((1 / lam + 2) / n)
    ok &= bool(good)
    rows.append(f'{lam:g}:{m:.4f}/{vv:.4f}{"" if good else "!"}')
c_mean = c1.sum() / (len(c1) * DRAWS)
ok &= abs(c_mean - 3.0) < 5 * np.sqrt(3.0 / (len(c1) * DRAWS))
result('t5_poisson_lambda', ok, f'backend={backend()} lam:mean/var ' + ' '.join(rows) +
       f' max_z={z[1:].max():.2f} scalar_lambda3_mean={c_mean:.4f}')
