"""Shared setup for the brian2cuda GPU test cases.

Each case is its own script so every simulation gets a fresh process (Brian2's
device state is global). A case takes the build directory as its first argument,
prints one `RESULT <case> PASS|FAIL ...` line and exits 0 only on PASS.
Cases that need a reference run Brian2's CPU runtime (numpy target) in a child
process of the same script with `--ref`.
"""
import os
import subprocess
import sys

SEED = 20261123


def gpu_setup(builddir):
    """Select cuda_standalone (the HIP backend when USE_HIP=1 or hipcc-only)."""
    import brian2cuda  # noqa: F401
    from brian2 import set_device, seed
    set_device('cuda_standalone', directory=builddir, build_on_run=True)
    seed(SEED)


def ref_setup():
    from brian2 import prefs, seed
    prefs.codegen.target = 'numpy'
    seed(SEED)


def run_ref(script, *args):
    """Run `script --ref args...` in a child process, return its stdout lines."""
    out = subprocess.run([sys.executable, script, '--ref', *map(str, args)],
                         check=True, capture_output=True, text=True).stdout
    return out


def result(case, ok, msg):
    print(f'RESULT {case} {"PASS" if ok else "FAIL"} {msg}', flush=True)
    sys.exit(0 if ok else 1)


def backend():
    from brian2cuda.utils.hip_backend import is_hip_backend
    return 'hip' if is_hip_backend() else 'cuda'


def builddir_arg(name):
    base = sys.argv[1] if len(sys.argv) > 1 else '/tmp/b2c_gpu_tests'
    d = os.path.join(base, name)
    os.makedirs(base, exist_ok=True)
    return d
