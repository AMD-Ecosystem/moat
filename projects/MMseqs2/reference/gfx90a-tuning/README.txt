gfx90a tuned-vs-generic timing for the curated moat-gpu-coverage history (2026-11-13, MI250X GCD, HIP_VISIBLE_DEVICES=1).
generic = build of 5ee63b2d (fallback + ClampToInvalid, no gfx90a table); tuned = build of 94e87c29 (tree identical to af0688d8, the branch tip).
build.sh <src> <build>: the configure/build used for both. bench.sh <q1k|t8> <variant>...: one timed ungappedprefilter run per variant, with createtsv row count and sorted md5.
q1k = first 1000 entries of util/regression/data/query.fasta; t = makepaddedseqdb of targetannotation.fasta.
q200 = first 200 entries; t8 = targetannotation.fasta replicated 8 times, header ids suffixed _r0.._r7, then makepaddedseqdb.
Host load (1-min) is in each log line; other jobs were compiling during several runs (load up to 127).
kernel-stats/: rocprofv3 --kernel-trace --stats of one q200-vs-t8 run per build.
