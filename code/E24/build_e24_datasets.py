#!/usr/bin/env python3
"""E24 step 2: build the two training conditions on IDENTICAL PUCPR frames (4090).

Inputs: E24_pucpr_plus_match.json (125 exact pixel matches), PUCPR+ devkit (car boxes 'x1 y1 x2 y2 cls', 1280x720),
PKLot PUCPR XML (stall rotatedRect). Outputs under --out:
  pucpr_car/   images/{train,val} (symlinks to the PKLot source frames), labels/{train,val}: PUCPR+ car boxes -> YOLO
  pucpr_stall/ same images; labels: occupied stalls -> axis-aligned bounding box of the XML rotatedRect (the E22 rule:
               centre = rotatedRect centre; w' = |w cos a| + |h sin a|, h' = |w sin a| + |h cos a|), class 0
Splits follow PUCPR+ ImageSets (train 100 -> train, test 25 -> val, used for early stopping and in-domain scoring).
Verification: for every frame that also exists in the E22 cross_pklot_hbb mirror, the stall labels produced here are
compared with the E22 file (exact text equality after normalisation); counts and any mismatch are written to the
meta file. E24_manifests.meta.json records per-file SHA256 of every label, the image digests, the rule text and the
split lists. Refuses to overwrite an existing output directory.
"""
import argparse, hashlib, json, math, os, re, time
from pathlib import Path

W, H = 1280, 720


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def car_lines(txt):
    out = []
    for l in Path(txt).read_text().splitlines():
        if not l.strip():
            continue
        x1, y1, x2, y2 = [float(v) for v in l.split()[:4]]
        x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2)); y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append('0 %.6f %.6f %.6f %.6f' % ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    return out


def stall_lines(xml):
    t = Path(xml).read_text(); out = []
    for sid, occ, body in re.findall(r'<space id="(\d+)" occupied="(\d)">(.*?)</space>', t, re.S):
        if occ != '1':
            continue
        m = re.search(r'<center x="(\d+)" y="(\d+)" />\s*<size w="(\d+)" h="(\d+)" />\s*<angle d="(-?\d+)" />', body)
        if not m:
            continue
        cx, cy, w, h, a = map(int, m.groups()); r = math.radians(a)
        bw = abs(w * math.cos(r)) + abs(h * math.sin(r)); bh = abs(w * math.sin(r)) + abs(h * math.cos(r))
        out.append('0 %.6f %.6f %.6f %.6f' % (cx / W, cy / H, bw / W, bh / H))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--match', type=Path, default=Path('/mnt/ssd/datasets/pucpr_plus/E24_pucpr_plus_match.json'))
    ap.add_argument('--plus', type=Path, default=Path('/mnt/ssd/datasets/pucpr_plus/PUCPR+_devkit/data'))
    ap.add_argument('--e22-labels', type=Path, default=Path('/mnt/ssd/datasets/custom_dataset/cross_pklot_hbb/labels'))
    ap.add_argument('--out', type=Path, default=Path('/mnt/ssd/snow_review_runs/E24/data'))
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; refusing to overwrite')
    m = json.loads(a.match.read_text()); assert m['unmatched'] == 0 and m['exact'] == 125
    splits = {s: set(l.strip() for l in (a.plus / 'ImageSets' / f'{s}.txt').read_text().splitlines() if l.strip()) for s in ('train', 'test')}
    e22 = {p.name: p for s in ('train', 'val') for p in (a.e22_labels / s).glob('PUCPR_*.txt')} if a.e22_labels.exists() else {}
    meta = dict(created_at=time.strftime('%Y-%m-%dT%H%M%S%z'), match_sha256=sha(a.match), rule_car='PUCPR+ x1 y1 x2 y2 -> clipped, class 0, normalised xywh',
                rule_stall='occupied XML spaces -> axis-aligned bbox of rotatedRect (E22 rule), class 0', conditions={}, e22_verification=dict(compared=0, equal=0, mismatches=[]))
    for cond in ('pucpr_car', 'pucpr_stall'):
        files = {}; counts = dict(train=0, val=0); boxes = dict(train=0, val=0)
        for r in m['rows']:
            stem = r['plus'][:-4]; split = 'train' if stem in splits['train'] else 'val'; src = Path(r['source'])
            name = 'PUCPR_' + src.parent.parent.name + '_' + src.stem  # PUCPR_<weather>_<timestamp>, as in E22
            (a.out / cond / 'images' / split).mkdir(parents=True, exist_ok=True); (a.out / cond / 'labels' / split).mkdir(parents=True, exist_ok=True)
            os.symlink(src, a.out / cond / 'images' / split / (name + '.jpg'))
            lines = car_lines(a.plus / 'Annotations' / (stem + '.txt')) if cond == 'pucpr_car' else stall_lines(src.with_suffix('.xml'))
            lp = a.out / cond / 'labels' / split / (name + '.txt'); lp.write_text('\n'.join(lines) + ('\n' if lines else ''))
            files[name] = dict(split=split, source=str(src), plus=r['plus'], image_pixel_sha256=r['plus_pixel_sha256'], label_sha256=sha(lp), n_boxes=len(lines))
            counts[split] += 1; boxes[split] += len(lines)
            if cond == 'pucpr_stall' and (name + '.txt') in e22:
                mine = [l.split() for l in lines]; theirs = [l.split() for l in e22[name + '.txt'].read_text().splitlines() if l.strip()]
                meta['e22_verification']['compared'] += 1
                if mine == theirs:
                    meta['e22_verification']['equal'] += 1
                else:
                    meta['e22_verification']['mismatches'].append(dict(name=name, mine=len(mine), e22=len(theirs), first_diff=next(((i, x, y) for i, (x, y) in enumerate(zip(mine, theirs)) if x != y), None)))
        (a.out / cond / 'data.yaml').write_text(f'path: {a.out / cond}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n')
        for s in ('train', 'val'):
            (a.out / cond / f'{s}_list.txt').write_text('\n'.join(sorted(n for n, v in files.items() if v['split'] == s)) + '\n')
        meta['conditions'][cond] = dict(n=counts, boxes=boxes, files=files, train_list_sha256=sha(a.out / cond / 'train_list.txt'), val_list_sha256=sha(a.out / cond / 'val_list.txt'))
    (a.out / 'E24_manifests.meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({c: {k: v for k, v in d.items() if k != 'files'} for c, d in meta['conditions'].items()}, indent=1)); print('e22_verification:', {k: (v if k != 'mismatches' else v[:3]) for k, v in meta['e22_verification'].items()})


if __name__ == '__main__':
    main()
