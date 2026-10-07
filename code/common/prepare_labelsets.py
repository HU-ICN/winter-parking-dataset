#!/usr/bin/env python3
"""Build YOLO-format label directories for the registered evaluation label sets.

  FIXED-AUDITED-TEST   : the 723 test labels as they are (source dir given).
  FIXED-AUDITED-331/88 : subsets of the test labels selected by the fixed manifests.
  INDEPENDENT-BLIND-*  : converted from SEALED annotation outputs (X-AnyLabeling-style JSON,
                         rectangle shapes, original-image pixel coordinates) to YOLO txt
                         (class 0, normalized cx cy w h), one file per frame, empty file for
                         empty frames. The seal list must exist and every source file must match it.
Each built set gets LABELSET_MANIFEST.json with per-file SHA256 and an aggregate hash.
No comparison between label sets is performed here.
"""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def write_manifest(out: Path, label_set_id: str, extra: dict):
    files = sorted(p.name for p in out.glob('*.txt'))
    digests = {n: sha(out / n) for n in files}
    agg = hashlib.sha256(''.join(digests[n] for n in files).encode()).hexdigest()
    man = dict(label_set_id=label_set_id, n_files=len(files), n_boxes=sum(sum(1 for l in (out / n).read_text().splitlines() if l.strip()) for n in files),
               aggregate_sha256=agg, files=digests, **extra)
    (out / 'LABELSET_MANIFEST.json').write_text(json.dumps(man, indent=1))
    print(label_set_id, 'files', man['n_files'], 'boxes', man['n_boxes'], 'aggregate', agg[:16])


def build_fixed_subset(test_labels: Path, manifest: Path, out: Path, label_set_id: str):
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(manifest.open(newline='', encoding='utf-8')))
    for r in rows:
        stem = Path(r['image_name']).stem; src = test_labels / f'{stem}.txt'
        if not src.exists():
            sys.exit(f'{label_set_id}: missing test label {src}')
        (out / f'{stem}.txt').write_bytes(src.read_bytes())
    write_manifest(out, label_set_id, dict(source=str(test_labels), manifest=str(manifest), manifest_sha256=sha(manifest)))


def build_blind(outputs: Path, manifest: Path, seal_list: Path, out: Path, label_set_id: str, w: int = 2304, h: int = 1536):
    seal = {l.split()[1]: l.split()[0] for l in seal_list.read_text().splitlines() if l.strip()}
    out.mkdir(parents=True, exist_ok=True); rows = list(csv.DictReader(manifest.open(newline='', encoding='utf-8'))); n_boxes = 0; buried = 0; flags = {}
    for r in rows:
        src = outputs / f"{r['anonymous_id']}.json"
        if src.name not in seal or sha(src) != seal[src.name]:
            sys.exit(f'{label_set_id}: {src.name} is not sealed or differs from the seal list')
        d = json.load(src.open(encoding='utf-8'))
        if d.get('imagePath', d.get('image_name')) != r['image_name']:
            sys.exit(f'{label_set_id}: {src.name} image mismatch')
        if not d['revision_annotation'].get('complete'):
            sys.exit(f'{label_set_id}: {src.name} not complete')
        lines = []
        for s in d.get('shapes', []):
            if s.get('shape_type', 'rectangle') != 'rectangle':
                continue
            xs = [p[0] for p in s['points']]; ys = [p[1] for p in s['points']]
            x1, x2 = max(0.0, min(xs)), min(float(w), max(xs)); y1, y2 = max(0.0, min(ys)), min(float(h), max(ys))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            if (s.get('flags') or {}).get('partially_buried'):
                buried += 1; flags.setdefault(Path(r['image_name']).stem, []).append(len(lines) - 1)
        n_boxes += len(lines)
        (out / f"{Path(r['image_name']).stem}.txt").write_text('\n'.join(lines) + ('\n' if lines else ''))
    (out / 'buried_flags.json').write_text(json.dumps(dict(note='line indices (0-based) of partially_buried boxes per frame stem; not part of the label aggregate hash', frames=flags), indent=1))
    write_manifest(out, label_set_id, dict(source_outputs=str(outputs), seal_list=str(seal_list), seal_list_sha256=sha(seal_list), manifest=str(manifest), manifest_sha256=sha(manifest), partially_buried_boxes=buried, buried_flags_sha256=sha(out / 'buried_flags.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-labels', type=Path, required=True, help='dir with the 723 FIXED-AUDITED-TEST YOLO labels')
    ap.add_argument('--out-root', type=Path, default=Path('experiments/labelsets'))
    ap.add_argument('--sets', nargs='+', default=['FIXED-AUDITED-331', 'FIXED-AUDITED-88', 'INDEPENDENT-BLIND-331'])
    a = ap.parse_args()
    M = Path('annotation/manifests'); O = Path('annotation/outputs'); S = Path('annotation/seals')
    for s in a.sets:
        out = a.out_root / s
        if s == 'FIXED-AUDITED-TEST':
            out.mkdir(parents=True, exist_ok=True)
            srcs = sorted(a.test_labels.glob('*.txt'))
            for src in srcs: (out / src.name).write_bytes(src.read_bytes())
            write_manifest(out, s, dict(source=str(a.test_labels), n_source_files=len(srcs)))
        elif s == 'FIXED-AUDITED-419':  # union of the two fixed subsets, file copies, manifest binds both sources
            out.mkdir(parents=True, exist_ok=True); srcs = {}
            for src_name in ('FIXED-AUDITED-331', 'FIXED-AUDITED-88'):
                sd = a.out_root / src_name; sm = json.loads((sd / 'LABELSET_MANIFEST.json').read_text()); srcs[str(sd)] = sm['aggregate_sha256']
                for fn in sm['files']:
                    if (out / fn).exists(): sys.exit(f'{s}: duplicate frame {fn} across sources')
                    (out / fn).write_bytes((sd / fn).read_bytes())
            write_manifest(out, s, dict(sources=srcs))
        elif s == 'FIXED-AUDITED-331': build_fixed_subset(a.test_labels, M / 'fixed_audited_331.csv', out, s)
        elif s == 'FIXED-AUDITED-88': build_fixed_subset(a.test_labels, M / 'fixed_audited_s0s3_88.csv', out, s)
        elif s == 'INDEPENDENT-BLIND-331': build_blind(O / 'detection', M / 'independent_blind_331.csv', S / 'INDEPENDENT-BLIND-331.sha256', out, s)
        elif s == 'INDEPENDENT-BLIND-S0S3-88': build_blind(O / 'detection_s0s3_88', M / 'independent_blind_s0s3_88.csv', S / 'INDEPENDENT-BLIND-S0S3-88.sha256', out, s)
        else: sys.exit(f'unknown set {s}')


if __name__ == '__main__':
    main()
