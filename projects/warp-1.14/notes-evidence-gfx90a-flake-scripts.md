# warp-1.14 evidence session (2026-11-20, linux-gfx90a): test_async flake scripts

Job directory `<job>` = the fleet job directory; the fork clone is `<job>/warp` at `7059a340`, scripts in `<job>/flk`. `copy_flake.hip` is session 21's file unchanged (`notes-session21-scripts.md`), built with `$(python -m rocm_sdk path --root)/bin/hipcc -O2 --offload-arch=gfx90a copy_flake.hip -o copy_flake`.

## `flake_b.sh` (session 21's, with GPU 0 pinned, an optional load GPU and `amd-smi process` snapshots)

```
#!/bin/bash
# Subset (test_async s2f_d2d / s2c_d2d) run N times in tree $1 on GPU 0 (dispatcher pin), with $3 background
# copy_flake load processes (4 streams each) on GPU $5 (default 0 = same device) or none. Log dir $2.
T=$1; L=$2; NL=$3; N=$4; LG=${5:-0}; mkdir -p $L
ulimit -c 0
J=<job>
export LD_LIBRARY_PATH=$($FLEET_VENV/bin/python -m rocm_sdk path --root)/lib
pids=()
for k in $(seq $NL); do HIP_VISIBLE_DEVICES=$LG $J/flk/copy_flake load 7200 4 > /dev/null 2>&1 & pids+=($!); done
export HIP_VISIBLE_DEVICES=0
sleep 3; amd-smi process > $L/smi_during_start.txt 2>&1
f=0
cd $T
for i in $(seq $N); do
  WARP_CACHE_PATH=$L/cache $FLEET_VENV/bin/python -m warp.tests -s autodetect -p test_async.py -k s2f_d2d -k s2c_d2d > $L/run_$i.log 2>&1
  [ $i -eq $((N/2)) ] && amd-smi process > $L/smi_during_mid.txt 2>&1
  if ! grep -qE "^OK" $L/run_$i.log; then f=$((f+1)); echo "run $i: $(grep -E '^(OK|FAILED)' $L/run_$i.log) $(grep -E '\.\.\. (FAIL|ERROR)' $L/run_$i.log | sed -E 's/ \(warp.*//' | tr '\n' ' ') $(grep -oE 'Mismatched elements: [0-9]+' $L/run_$i.log | tr '\n' ' ')"; fi
done
alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=$((alive+1)); kill $p 2>/dev/null; done
echo "load processes alive at end: $alive of $NL"
echo "$f of $N runs failed with $NL load processes on GPU $LG"
```

## `flake_a.sh` (session 21's, run for a fixed N subset runs; the suite restarts whenever it ends)

```
#!/bin/bash
# The full default suite runs on GPU 0 in the background (restarted whenever it ends) while the subset runs N times
# beside it. Tree $1, log dir $2.
T=$1; L=$2; N=$3; mkdir -p $L
ulimit -c 0
export HIP_VISIBLE_DEVICES=0
cd $T
s=1; WARP_CACHE_PATH=$L/cache_suite $FLEET_VENV/bin/python -m warp.tests -s autodetect > $L/suite_$s.log 2>&1 & S=$!
sleep 60; amd-smi process > $L/smi_during_start.txt 2>&1
f=0
for i in $(seq $N); do
  if ! kill -0 $S 2>/dev/null; then echo "suite $s: $(grep -E '^(Ran|OK|FAILED)' $L/suite_$s.log | tr '\n' ' ')"; s=$((s+1)); WARP_CACHE_PATH=$L/cache_suite $FLEET_VENV/bin/python -m warp.tests -s autodetect > $L/suite_$s.log 2>&1 & S=$!; sleep 20; fi
  WARP_CACHE_PATH=$L/cache_subset $FLEET_VENV/bin/python -m warp.tests -s autodetect -p test_async.py -k s2f_d2d -k s2c_d2d > $L/run_$i.log 2>&1
  [ $i -eq $((N/2)) ] && amd-smi process > $L/smi_during_mid.txt 2>&1
  if ! grep -qE "^OK" $L/run_$i.log; then f=$((f+1)); echo "run $i $(date +%T): $(grep -E '^(OK|FAILED)' $L/run_$i.log) $(grep -E '\.\.\. (FAIL|ERROR)' $L/run_$i.log | sed -E 's/ \(warp.*//' | tr '\n' ' ') $(grep -oE 'Mismatched elements: [0-9]+' $L/run_$i.log | tr '\n' ' ')"; fi
done
echo "$f of $N subset runs failed beside the suite ($s suite starts)"
echo "suite $s at stop: still running? $(kill -0 $S 2>/dev/null && echo yes || echo no)"; kill $S 2>/dev/null; pkill -P $S 2>/dev/null
grep -hE '^(Ran|OK|FAILED)' $L/suite_*.log
```

## Scratch diagnostic in `<job>/warp` (never committed): `git diff`

```diff
diff --git a/warp/tests/cuda/test_async.py b/warp/tests/cuda/test_async.py
index 196424ae..1651ee01 100644
--- a/warp/tests/cuda/test_async.py
+++ b/warp/tests/cuda/test_async.py
@@ -431,7 +431,26 @@ def copy_template(test, src_ctor, dst_ctor, src_device, dst_device, n, params: C
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
+                blk = np.unique(w1 // 256)
+                nb = (len(ref) + 255) // 256
+                whole = all(np.all(x1[b * 256:min((b + 1) * 256, len(ref))] != ref[b * 256:min((b + 1) * 256, len(ref))]) for b in blk)
+                res = {m: sorted(set((blk % m).tolist())) for m in (2, 3, 4, 5, 6, 7, 8, 16)}
+                const = [m for m, r in res.items() if len(r) == 1]
+                print(f"FLAKEBLK {test.id()}: n={len(ref)} grid_blocks={nb} wrong_blocks={len(blk)} whole_blocks={whole} "
+                      f"constant_residue_moduli={const} residues={ {m: r[:8] for m, r in res.items()} } first_blocks={blk[:12].tolist()}", flush=True)
+            assert_np_equal(x1, ref)
 
             if params.with_grad:
                 assert_np_equal(dst.grad.numpy(), src.grad.numpy())
```

## `cf2_run.sh` (copy_flake2 session; `copy_flake2.hip` is in notes.md, "Evidence session 2 (2026-11-20, linux-gfx942)")

```
#!/bin/bash
# copy_flake2 on GPU 0 (dispatcher pin) with $2 copy_flake load processes (4 streams each) on the same GPU.
# Usage: cf2_run.sh <label> <nload> <iterations> [copy_flake2 options]; log <job>/flk2/<label>.log.
# Load processes are started here and stopped by their recorded PIDs only.
L=$1; NL=$2; N=$3; shift 3
J=<job>
ulimit -c 0
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$($FLEET_VENV/bin/python -m rocm_sdk path --root)/lib
pids=()
for k in $(seq $NL); do $J/flk/copy_flake load 7200 4 > /dev/null 2>&1 & pids+=($!); done
echo "load pids: ${pids[*]}" > $J/flk2/$L.log
sleep 3
amd-smi process -g 0 > $J/flk2/${L}_smi_start.txt 2>&1
cd /tmp
t0=$(date +%s)
$J/flk2/copy_flake2 $N "$@" >> $J/flk2/$L.log 2>&1 &
C=$!
sleep 60; amd-smi process -g 0 > $J/flk2/${L}_smi_during.txt 2>&1
wait $C; rc=$?
alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=$((alive+1)); done
for p in "${pids[@]}"; do kill $p 2>/dev/null; done
for p in "${pids[@]}"; do wait $p 2>/dev/null; done
echo "rc=$rc seconds=$(( $(date +%s) - t0 )) load alive at end: $alive of $NL" >> $J/flk2/$L.log
tail -2 $J/flk2/$L.log
```
