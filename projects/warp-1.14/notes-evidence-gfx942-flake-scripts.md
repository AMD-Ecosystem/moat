# warp-1.14 evidence session (2026-11-20, linux-gfx942): test_async copy flake scripts

All under the job worktree's `agent_space/`; `w` is the fork clone at `7059a340`, `flake/copy_flake` is session 21's `copy_flake.hip` (in `notes-session21-scripts.md`) built with `hipcc --offload-arch=gfx942 -O2` from the job venv's ROCm 10.2 SDK.

## `flake_b.sh`

```
#!/bin/bash
# flake_b.sh from session 21, adapted: tree $1, log dir $2, $3 copy_flake load processes (4 streams each), $4 runs.
# The subset runs on the dispatcher's pin (HIP_VISIBLE_DEVICES, kept); the load runs on GPU $5 (default: the same pin).
T=$1; L=$2; NL=$3; N=$4; LG=${5:-$HIP_VISIBLE_DEVICES}; mkdir -p $L
ulimit -c 0
export FLAKE_DUMP=$(dirname $(readlink -f $0))/logs/dumps
A=$(dirname $(readlink -f $0))
pids=()
for k in $(seq $NL); do HIP_VISIBLE_DEVICES=$LG $A/flake/copy_flake load 7200 4 > /dev/null 2>&1 & pids+=($!); done
sleep 2
amd-smi process -g $HIP_VISIBLE_DEVICES > $L/smi_subset_gpu.txt 2>&1; [ "$LG" != "$HIP_VISIBLE_DEVICES" ] && amd-smi process -g $LG > $L/smi_load_gpu.txt 2>&1
f=0
cd $T
for i in $(seq $N); do
  WARP_CACHE_PATH=$L/cache $FLEET_VENV/bin/python -m warp.tests -s autodetect -p test_async.py -k s2f_d2d -k s2c_d2d > $L/run_$i.log 2>&1
  if ! grep -qE "^OK" $L/run_$i.log; then f=$((f+1)); echo "run $i: $(grep -E '^(OK|FAILED)' $L/run_$i.log) $(grep -E '\.\.\. (FAIL|ERROR)' $L/run_$i.log | sed -E 's/ \(warp.*//' | tr '\n' ' ') $(grep -oE 'Mismatched elements: [0-9]+' $L/run_$i.log | tr '\n' ' ')"; grep -hE '^FLAKE' $L/run_$i.log; fi
  [ $((i % 25)) -eq 0 ] && echo "progress $i/$N failed=$f $(date +%T)" && amd-smi process -g $HIP_VISIBLE_DEVICES >> $L/smi_during.txt 2>&1
done
for p in "${pids[@]}"; do kill $p 2>/dev/null; done
wait 2>/dev/null
echo "$f of $N runs failed with $NL load processes on GPU $LG (subset on GPU $HIP_VISIBLE_DEVICES)"
```

## `flake_c.sh`

```
#!/bin/bash
# flake_a.sh from session 21, adapted: the full default suite runs on GPU $4 (default: the pin) in the background
# and is restarted when it finishes, while the test_async subset runs $3 times on the pin. Tree $1, log dir $2.
T=$1; L=$2; N=$3; SG=${4:-$HIP_VISIBLE_DEVICES}; mkdir -p $L
ulimit -c 0
export FLAKE_DUMP=$(dirname $(readlink -f $0))/logs/dumps
cd $T
s=0; f=0
start_suite() { s=$((s+1)); HIP_VISIBLE_DEVICES=$SG WARP_CACHE_PATH=$L/cache_suite $FLEET_VENV/bin/python -m warp.tests -s autodetect > $L/suite_$s.log 2>&1 & S=$!; echo "suite $s started $(date +%T)"; }
start_suite
sleep 20
for i in $(seq $N); do
  if ! kill -0 $S 2>/dev/null; then echo "suite $s: $(grep -E '^(Ran|OK|FAILED)' $L/suite_$s.log | tr '\n' ' ')"; start_suite; sleep 20; fi
  WARP_CACHE_PATH=$L/cache_subset $FLEET_VENV/bin/python -m warp.tests -s autodetect -p test_async.py -k s2f_d2d -k s2c_d2d > $L/subset_$i.log 2>&1
  if ! grep -qE "^OK" $L/subset_$i.log; then f=$((f+1)); echo "run $i $(date +%T): $(grep -E '^(OK|FAILED)' $L/subset_$i.log) $(grep -E '\.\.\. (FAIL|ERROR)' $L/subset_$i.log | sed -E 's/ \(warp.*//' | tr '\n' ' ') $(grep -oE 'Mismatched elements: [0-9]+' $L/subset_$i.log | tr '\n' ' ')"; grep -hE '^FLAKE' $L/subset_$i.log; fi
  [ $((i % 10)) -eq 0 ] && echo "progress $i/$N failed=$f $(date +%T)" && amd-smi process -g $HIP_VISIBLE_DEVICES >> $L/smi_during.txt 2>&1
done
kill -- $S 2>/dev/null; pkill -P $S 2>/dev/null; sleep 5
echo "suite $s (killed or finished): $(grep -E '^(Ran|OK|FAILED)' $L/suite_$s.log | tr '\n' ' ')"
echo "$f of $N subset runs failed beside the full suite on GPU $SG (subset on GPU $HIP_VISIBLE_DEVICES)"
```

## `cf_load.sh`

```
#!/bin/bash
# copy_flake check $1 iterations [$2 = sync|pool] on the pin with 8 copy_flake load processes (4 streams each) on GPU $3 (default: the pin).
ulimit -c 0
A=$(dirname $(readlink -f $0)); LG=${3:-$HIP_VISIBLE_DEVICES}
pids=(); for k in $(seq 8); do HIP_VISIBLE_DEVICES=$LG $A/flake/copy_flake load 7200 4 > /dev/null 2>&1 & pids+=($!); done
sleep 2
$A/flake/copy_flake check $1 $2
for p in "${pids[@]}"; do kill $p; done; wait 2>/dev/null
```

## `analyze_dump.py`

```
# Per failing test: each read (dst reads x1, x2, x3 and the source read-back ref) against the expected values
# arange(offset, offset + n) (offset recovered as the most common value of read - index), grouped by
# 256-thread block and block index mod 8.
import sys, numpy as np
for f in sys.argv[1:]:
    d = np.load(f); n = len(d["ref"]); nb = (n + 255) // 256; idx = np.arange(n, dtype=np.float64)
    v, c = np.unique(np.concatenate([d[k] - idx for k in ("x1", "ref")]), return_counts=True)
    off = v[c.argmax()]; exp = (idx + off).astype(np.float32)
    print(f.rsplit("/", 1)[-1], f"offset {off:.0f}")
    for tag in [k for k in ("ref", "ref2", "x1", "x2", "x3") if k in d]:
        x = d[tag]; w = np.nonzero(x != exp)[0]
        if not len(w):
            print(f"  {tag}: correct"); continue
        b, cnt = np.unique(w // 256, return_counts=True)
        full = np.minimum(256, n - b * 256)
        print(f"  {tag}: {len(w)} wrong ({int((x[w] == 0).sum())} zero), {len(b)} blocks, {int((cnt == full).sum())} whole")
        for r in range(8):
            sel = b % 8 == r
            if not sel.any(): continue
            br, cr, fr = b[sel], cnt[sel], full[sel]
            allr = np.arange(r, nb, 8); span = allr[(allr >= br.min()) & (allr <= br.max())]
            print(f"    mod8={r}: {sel.sum()} blocks ({int((cr == fr).sum())} whole) of {len(allr)} in class, blocks {br.min()}..{br.max()}, "
                  f"{'every class block in that span' if len(span) == len(br) else f'{len(span) - len(br)} class blocks in span correct'}")
```

## `summarize_dump.py`

```
# One line per failing test dump: for each read, the wrong count and, per mod-8 class, wrong blocks (whole) / class size.
import sys, numpy as np
for f in sys.argv[1:]:
    d = np.load(f); n = len(d["ref"]); nb = (n + 255) // 256; idx = np.arange(n, dtype=np.float64)
    v, c = np.unique(np.concatenate([d[k] - idx for k in ("x1", "ref")]), return_counts=True)
    exp = (idx + v[c.argmax()]).astype(np.float32)
    parts = []
    for tag in [k for k in ("ref", "ref2", "x1", "x2", "x3") if k in d]:
        x = d[tag]; w = np.nonzero(x != exp)[0]
        if not len(w):
            parts.append(f"{tag}=ok"); continue
        b, cnt = np.unique(w // 256, return_counts=True); full = np.minimum(256, n - b * 256)
        cls = []
        for r in range(8):
            sel = b % 8 == r
            if sel.any():
                cls.append(f"{r}:{sel.sum()}({int((cnt[sel] == full[sel]).sum())})/{len(range(r, nb, 8))}")
        parts.append(f"{tag}={len(w)}[{int((x[w]==0).sum())}z] " + " ".join(cls))
    t = f.rsplit("/", 1)[-1].split("_", 3)[3].replace(".npz", "")
    print(f"{f.rsplit('/',1)[-1].split('_')[2]} {t}: " + " | ".join(parts))
```

## `table.py`

```
# Per failing test in run logs: run, test, mismatch count, and the dump's verdict (which read, persistent mod-8 class, blocks).
import sys, re, glob, os, numpy as np
logdir, pat = sys.argv[1], sys.argv[2]
dumps = [f for d in sys.argv[3:] for f in glob.glob(d + "/flake_*.npz")]
used = set()
def verdict(f):
    d = np.load(f); n = len(d["ref"]); nb = (n + 255) // 256; idx = np.arange(n, dtype=np.float64)
    v, c = np.unique(np.concatenate([d[k] - idx for k in ("x1", "ref")]), return_counts=True)
    exp = (idx + v[c.argmax()]).astype(np.float32)
    def cls(x):
        w = np.nonzero(x != exp)[0]
        if not len(w): return None
        b, cnt = np.unique(w // 256, return_counts=True)
        return {r: (b[b % 8 == r], cnt[b % 8 == r]) for r in range(8) if (b % 8 == r).any()}, len(w)
    out = []
    if cls(d["ref"]):
        cl, nw = cls(d["ref"]); out.append(f"source read-back wrong ({nw}); re-read {'ok' if cls(d['ref2']) is None else 'WRONG'} " if "ref2" in d else f"source read-back wrong ({nw}) ")
        main = max(cl, key=lambda r: (cl[r][1] == 256).sum())
    else:
        c1, n1 = cls(d["x1"]); c3 = cls(d["x3"])
        if c3 is None: return "dst wrong in first read only"
        cl, n3 = c3; out.append(f"dst {n1} -> {n3} after re-read/sync; ")
        extra = sorted(set(c1) - set(cl))
        if extra: out.append(f"first read also class {extra} (cleared on re-read); ")
        main = max(cl, key=lambda r: len(cl[r][0]))
    b, cnt = cl[main]; full = np.minimum(256, n - b * 256)
    allr = np.arange(main, nb, 8); span = allr[(allr >= b.min()) & (allr <= b.max())]
    out.append(f"class {main} mod 8: {len(b)}/{len(allr)} blocks, {int((cnt == full).sum())} whole, blocks {b.min()}..{b.max()}{'' if len(span) == len(b) else ' (gaps)'}"
               + ("" if len(cl) == 1 else f"; other classes partial: {sorted(r for r in cl if r != main)}"))
    return "".join(out)
for lf in sorted(glob.glob(f"{logdir}/{pat}"), key=lambda p: int(re.findall(r"(\d+)\.log", p)[0])):
    s = open(lf).read()
    if re.search(r"^OK", s, re.M): continue
    run = re.findall(r"(\d+)\.log", lf)[0]
    for m in re.finditer(r"^={70}\n(test_\S+) \(([^)]+)\)\n.*?Mismatched elements: (\d+)", s, re.M | re.S):
        t, mm = m.group(1), int(m.group(3))
        cand = [f for f in dumps if f.endswith(t + ".npz") and f not in used]
        hit = None
        for f in sorted(cand, key=os.path.getmtime):
            d = np.load(f)
            if int((d["x1"] != d["ref"]).sum()) == mm: hit = f; break
        if hit: used.add(hit)
        print(f"| {run} | `{t[10:]}` | {mm} | {verdict(hit) if hit else 'no dump (before the dump was added)'} |")
```

## Scratch diagnostic in `w` (never committed): `git -C w diff`

Session 21's diagnostic plus, on a mismatch, an `np.savez_compressed` of the reads (`x1`, `x2`, `x3` of `dst`, the reference `ref` = `src.numpy()`, and `ref2` = a second `src.numpy()`; `ref2` was added after the first 48 load runs and the dump after the first 15) and a per-read block summary (`FLAKEBLK`).

```diff
diff --git a/warp/tests/cuda/test_async.py b/warp/tests/cuda/test_async.py
index 196424ae..5ac5a4d3 100644
--- a/warp/tests/cuda/test_async.py
+++ b/warp/tests/cuda/test_async.py
@@ -431,7 +431,31 @@ def copy_template(test, src_ctor, dst_ctor, src_device, dst_device, n, params: C
             if stream is not None:
                 wp.synchronize_stream(stream)
 
-            assert_np_equal(dst.numpy(), src.numpy())
+            x1 = dst.numpy()
+            ref = src.numpy()
+            w1 = np.nonzero(x1 != ref)[0]
+            if len(w1):
+                # SCRATCH DIAGNOSTIC (not for commit): where is the stale data?
+                x2 = dst.numpy()
+                wp.synchronize()
+                x3 = dst.numpy()
+                w2 = np.nonzero(x2 != ref)[0]
+                w3 = np.nonzero(x3 != ref)[0]
+                print(f"FLAKEDIAG {test.id()}: first read {len(w1)} wrong [{w1[0]}, {w1[-1]}] blocks [{w1[0] // 256}, {w1[-1] // 256}] "
+                      f"values {x1[w1[0]]}..{x1[w1[-1]]}; second read {len(w2)} wrong; after wp.synchronize {len(w3)} wrong", flush=True)
+                # SCRATCH (gfx942 evidence session): keep the reads, then the block pattern of each read
+                import os, time
+                np.savez_compressed(os.path.join(os.environ.get("FLAKE_DUMP", "/tmp"), f"flake_{os.getpid()}_{time.time():.0f}_{test.id().rsplit('.', 1)[-1]}.npz"), x1=x1, x2=x2, x3=x3, ref=ref, ref2=src.numpy())
+                for tag, w in (("read1", w1), ("read2", w2), ("read3", w3)):
+                    if not len(w):
+                        continue
+                    b, c = np.unique(w // 256, return_counts=True)
+                    nb = (len(ref) + 255) // 256
+                    full = np.array([min(256, len(ref) - k * 256) for k in b])
+                    mods = {m: sorted(set((b % m).tolist())) for m in (2, 4, 8, 16, 38, 304)}
+                    print(f"FLAKEBLK {test.id()} {tag}: n={len(ref)} grid={nb} wrong_blocks={len(b)} whole_blocks={int((c == full).sum())} "
+                          f"partial_blocks={int((c != full).sum())} residues={ {m: (v if len(v) <= 8 else f'{len(v)} residues') for m, v in mods.items()} }", flush=True)
+            assert_np_equal(x1, ref)
 
             if params.with_grad:
                 assert_np_equal(dst.grad.numpy(), src.grad.numpy())
```
