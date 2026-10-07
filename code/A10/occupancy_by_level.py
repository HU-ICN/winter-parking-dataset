#!/usr/bin/env python3
"""Per-snow-level breakdown of the sealed occupancy-120 evaluation (descriptive; frames grouped by the day's
snow label in snow_day_index.csv). Reads the sealed package only; writes next to it, never into it."""
import csv, hashlib, json, sys, time
from pathlib import Path
PKG = Path(sys.argv[1] if len(sys.argv) > 1 else 'experiments/A10/results/2026-09-09T215146+0900')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
lv = {r['date']: r['snow_label'] for r in csv.DictReader(open('snow_day_index.csv'))}
o = json.loads((PKG / 'occupancy_results.json').read_text()); out = {}
for k in ('golden_40', 'independent_80', 'all_120'):
    agg = {}
    for fr, c in o['results'][k]['per_frame'].items():
        a = agg.setdefault(lv[fr[6:14]], dict(frames=0, tp=0, fp=0, fn=0, tn=0))
        a['frames'] += 1
        for m in ('tp', 'fp', 'fn', 'tn'):
            a[m] += c[m]
    for L, a in agg.items():
        n = a['tp'] + a['fp'] + a['fn'] + a['tn']; d = 2 * a['tp'] + a['fp'] + a['fn']
        a['accuracy'] = a['tp'] + a['tn'] and (a['tp'] + a['tn']) / n; a['occupied_f1'] = (2 * a['tp'] / d) if d else None
    out[k] = dict(sorted(agg.items()))
res = dict(source_package=str(PKG), source_sha256=sha(PKG / 'occupancy_results.json'), snow_index_sha256=sha('snow_day_index.csv'), script_sha256=sha(__file__), created_at=time.strftime('%Y-%m-%dT%H%M%S%z'), by_level=out)
dst = PKG.parent / f'occupancy_by_level_{PKG.name}.json'; dst.write_text(json.dumps(res, indent=1)); print(json.dumps(out, indent=None)); print('->', dst)
