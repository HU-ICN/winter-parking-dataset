#!/usr/bin/env python3
"""E24 post-hoc descriptive breakdown (registered after the sealed run): per-snow-level recall / precision of the
E24 models on FIXED-AUDITED-TEST at the deployment threshold (conf >= 0.25, greedy IoU >= 0.5 matching, highest IoU
first) from the archived predictions.json of the sealed evaluation package. Inputs are digest-bound to the package's
OUTPUT_SHA256SUMS. Descriptive only; not part of the pre-registered reading."""
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
PKG = Path('experiments/E24/eval_results/2026-09-28T070250+0800'); ARCH = Path.home() / 'snow_review_archive/E24_eval_2026-09-28T070250+0800'
W, H = 2304, 1536; CONF = 0.25; THR = 0.5
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sums = {l.split('  ', 1)[1]: l.split('  ', 1)[0] for l in (PKG / 'OUTPUT_SHA256SUMS').read_text().splitlines()}
lv = {r['date']: r['snow_label'] for r in csv.DictReader(open('snow_day_index.csv'))}
lab_dir = PKG / 'datasets/FIXED-AUDITED-TEST/labels/val'
labels = {}
for p in sorted(lab_dir.glob('*.txt')):
    assert sha(p) == sums[f'datasets/FIXED-AUDITED-TEST/labels/val/{p.name}'], p.name
    b = []
    for l in p.read_text().splitlines():
        c, x, y, w, h = map(float, l.split()[:5]); b.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
    labels[p.stem] = np.array(b).reshape(-1, 4)
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1]); ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None); aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-9, None)
def match(pred, gt):
    M = iou_matrix(pred, gt); ua = set(); ub = set(); m = 0
    for v, i, j in sorted(((M[i, j], i, j) for i in range(len(pred)) for j in range(len(gt)) if M[i, j] >= THR), reverse=True):
        if i in ua or j in ub: continue
        ua.add(i); ub.add(j); m += 1
    return m
days = {}
for stem in labels: days.setdefault(lv[stem[6:14]], set()).add(stem[6:14])
out = dict(created_at=time.strftime('%Y-%m-%dT%H%M%S%z'), package=str(PKG), output_sums_sha256=sha(PKG / 'OUTPUT_SHA256SUMS'), snow_day_index_sha256=sha('snow_day_index.csv'), script_sha256=sha(__file__), test_days_per_level={k: len(v) for k, v in sorted(days.items())}, test_frames_per_level={k: sum(1 for st in labels if lv[st[6:14]] == k) for k in sorted(days)}, rule=dict(conf=CONF, iou=THR, match='greedy highest IoU first'), note='post-hoc, descriptive; not part of the pre-registered reading; one test day each at S0 and S3', models={})
for tag in ['E24_pucpr_car_s0', 'E24_pucpr_car_s1', 'E24_pucpr_car_s2', 'E24_pucpr_stall_s0', 'E24_pucpr_stall_s1', 'E24_pucpr_stall_s2']:
    pj = ARCH / 'val' / f'{tag}__FIXED-AUDITED-TEST' / 'predictions.json'
    assert sha(pj) == sums[f'val/{tag}__FIXED-AUDITED-TEST/predictions.json'], tag
    preds = {}
    for x in json.loads(pj.read_text()):
        if x['score'] >= CONF:
            bx = x['bbox']; preds.setdefault(x['image_id'], []).append([bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]])
    agg = {}
    for stem, gt in labels.items():
        L = lv[stem[6:14]]; a = agg.setdefault(L, dict(frames=0, gt=0, pred=0, matched=0)); pr = np.array(preds.get(stem, [])).reshape(-1, 4)
        a['frames'] += 1; a['gt'] += len(gt); a['pred'] += len(pr); a['matched'] += match(pr, gt)
    tot = dict(frames=sum(a['frames'] for a in agg.values()), gt=sum(a['gt'] for a in agg.values()), pred=sum(a['pred'] for a in agg.values()), matched=sum(a['matched'] for a in agg.values()))
    for a in list(agg.values()) + [tot]:
        a['recall'] = a['matched'] / a['gt'] if a['gt'] else None; a['precision'] = a['matched'] / a['pred'] if a['pred'] else None
    out['models'][tag] = dict(by_level=dict(sorted(agg.items())), all=tot)
Path('experiments/E24/eval_results/per_level_recall_2026-09-28T070250+0800.json').write_text(json.dumps(out, indent=1))
for tag, d in out['models'].items():
    print(tag, ' | '.join(f"{L}: R {a['recall']:.3f} P {a['precision'] if a['precision'] is None else round(a['precision'],3)} (gt {a['gt']}, {a['frames']} fr)" for L, a in d['by_level'].items()), '| all R %.3f P %.3f' % (d['all']['recall'], d['all']['precision']))
