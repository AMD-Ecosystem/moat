#!/usr/bin/env python3
"""Generate make_aligner{,_endpos}_amd_gfx1201.cu from tuning_gapless CSV output.

Selection rule (same as the gfx90a/gfx950 rounds):
- per tile size, the config with max maxgcups (numTiles > 0), over upstream's AMD spans
- single tile: drop t if some kept larger T has g_T * t / T > g_t
- multi tile: tiles >= 256 within 10% of best multi-tile maxgcups
Usage: gen_tuned.py <rawdir> <aligners_dir> <outdir>
"""
import csv, re, sys, pathlib

raw, aldir, out = map(pathlib.Path, sys.argv[1:4])


def load(*names):
    rows = []
    for n in names:
        for line in (raw / n).read_text().splitlines():
            p = line.strip().split(',')
            if len(p) != 13 or p[0] == 'gpuArch':
                continue
            r = dict(block=int(p[2]), tile=int(p[3]), group=int(p[4]), items=int(p[5]),
                     maxg=float(p[8]), dt=p[9], ap=p[10], kind=p[11], ntiles=int(p[12]))
            if r['ntiles'] > 0:
                rows.append(r)
    return rows


def best_per_tile(rows, lo, hi):
    best = {}
    for r in rows:
        if lo <= r['tile'] <= hi and (r['tile'] not in best or r['maxg'] > best[r['tile']]['maxg']):
            best[r['tile']] = r
    return [best[t] for t in sorted(best)]


def single(rows, lo, hi):
    b = best_per_tile(rows, lo, hi)
    kept = []
    for r in reversed(b):
        if not any(k['maxg'] * r['tile'] / k['tile'] > r['maxg'] for k in kept):
            kept.append(r)
    return list(reversed(kept))


def multi(rows, lo, hi):
    b = best_per_tile(rows, lo, hi)
    top = max(r['maxg'] for r in b)
    return [r for r in b if r['maxg'] >= 0.9 * top], top


def tuple_text(sel, indent='                '):
    return ',\n'.join(f"{indent}GaplessConfig<21, Approach::{r['ap']}, Datatype::{r['dt']}, {r['block']}, {r['group']}, {r['items']}>"
                      for r in sel)


def hints_text(sel):
    v = [str(int(r['maxg'])) for r in sel]
    lines = [','.join(v[i:i + 10]) for i in range(0, len(v), 10)]
    return (',\n' + ' ' * 20).join(lines)


def fill(template, lists):
    text = template.replace('gfx950', 'gfx1201')
    tuples = iter(lists)
    hints = iter(lists)

    def tsub(m):
        if 'GaplessConfig' not in m.group(0):
            return m.group(0)
        return m.group(1) + tuple_text(next(tuples)) + '\n' + m.group(3)

    text = re.sub(r'(using type = std::tuple<\n)(.*?)\n(\s*>;)', tsub, text, flags=re.S)
    text = re.sub(r'(constexpr auto result = std::array\{)(.*?)(\};)',
                  lambda m: m.group(1) + hints_text(next(hints)) + m.group(3), text, flags=re.S)
    return text


report = []


def rep(name, sel, extra=''):
    report.append(f"{name}: {len(sel)} configs, tiles {sel[0]['tile']}..{sel[-1]['tile']}, "
                  f"GCUPS {int(min(r['maxg'] for r in sel))}-{int(max(r['maxg'] for r in sel))} {extra}")
    for r in sel:
        report.append(f"   {r['tile']:5d} {r['dt']:6s} {r['ap']:16s} {r['block']:4d} {r['group']:3d} {r['items']:3d} {r['maxg']:.0f}")


h1 = single(load('half2_t1.csv'), 16, 1536); rep('half2 single', h1)
h4, t = multi(load('half2_t4.csv'), 256, 1536); rep('half2 multi', h4, f'best {t:.0f}')
f1 = single(load('f32_t1.csv'), 8, 768); rep('fp32 single', f1)
f4, t = multi(load('f32_t4.csv'), 256, 768); rep('fp32 multi', f4, f'best {t:.0f}')
e1 = single(load('endpos_t1.csv'), 8, 768); rep('endpos single', e1)
e4, t = multi(load('endpos_t4_a.csv', 'endpos_t4_b.csv'), 256, 768); rep('endpos multi', e4, f'best {t:.0f}')

out.mkdir(parents=True, exist_ok=True)
(out / 'make_aligner_amd_gfx1201.cu').write_text(fill((aldir / 'make_aligner_amd_gfx950.cu').read_text(), [h1, h4, f1, f4]))
(out / 'make_aligner_endpos_amd_gfx1201.cu').write_text(fill((aldir / 'make_aligner_endpos_amd_gfx950.cu').read_text(), [e1, e4]))
print('\n'.join(report))
