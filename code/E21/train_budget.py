#!/usr/bin/env python3
"""E21: label efficiency in DAYS, three paired replicates (FREEZE_SPEC_E17_E21_E22.md, section E21).

--budget dNN --seed s : train on ALL train frames of the days listed in the frozen manifest
  experiments/E21/manifests/dNN_ss.txt (day selection rule frozen in E21_manifests.meta.json; s=0 reproduces
  Table 5 / E14). Data come only from the A07 mirror of snowpark_yolo2 (images/train, labels/train, fixed val).
Identity assertions BEFORE training: manifest raw-file sha256 == the value in E21_manifests.meta.json; every
listed frame exists in the mirror; init weights sha256 == --expected-init-sha256 (yolo11s COCO, 85a76fe8...).
Hyperparameters identical to E17 / E22 / Table 5: yolo11s, imgsz 1280, epochs 60, patience 15, batch 16,
seed s, amp, deterministic; early stopping on the fixed snowpark_yolo2 val split (626). Full budget (73 days)
= E17 stage c replicates (same identity check), not retrained here.
--check-only runs the assertions and writes nothing but a check line.
Evidence: run_manifest.json (manifest sha, labels aggregate, init sha, args, environment, best.pt sha).
"""
import argparse, hashlib, json, os, platform, shutil, sys, time
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', required=True, help='d01 | d02 | d04 | d08 | d16 | d32'); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--mirror', default='/mnt/ssd/snow_review_runs/mirror/snowpark_yolo2'); ap.add_argument('--manifest-dir', default='/mnt/ssd/snow_review_runs/E21/manifests')
    ap.add_argument('--init', default='/mnt/ssd/snow_review_runs/weights/yolo11s.pt'); ap.add_argument('--expected-init-sha256', default='85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5')
    ap.add_argument('--out-root', default='/mnt/ssd/snow_review_runs/E21'); ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    name = f'{a.budget}_s{a.seed}'; mdir = Path(a.manifest_dir); mpath = mdir / f'{name}.txt'; meta = json.loads((mdir / 'E21_manifests.meta.json').read_text())
    if name not in meta['budgets']:
        sys.exit(f'{name} is not a frozen budget')
    b = meta['budgets'][name]
    if not mpath.exists() or sha(mpath) != b['manifest_sha256']:
        sys.exit(f'{name}: manifest missing or sha differs from E21_manifests.meta.json')
    frames = [l.strip() for l in mpath.read_text().splitlines() if l.strip()]
    if len(frames) != b['frames'] or len(set(frames)) != len(frames):
        sys.exit(f'{name}: manifest has {len(frames)} frames, meta says {b["frames"]}')
    days = sorted({f[6:14] for f in frames})
    if days != sorted(b['days']):
        sys.exit(f'{name}: frame dates {days} differ from the frozen days {sorted(b["days"])}')
    mirror = Path(a.mirror)
    missing = [f for f in frames if not (mirror / 'images' / 'train' / f).exists() or not (mirror / 'labels' / 'train' / (f[:-4] + '.txt')).exists()]
    if missing:
        sys.exit(f'{name}: {len(missing)} frames missing in the mirror')
    init_sha = sha(a.init)
    if init_sha != a.expected_init_sha256:
        sys.exit(f'init weights sha {init_sha[:12]} != expected {a.expected_init_sha256[:12]}')
    labels_sha = hashlib.sha256(''.join(sha(mirror / 'labels' / 'train' / (f[:-4] + '.txt')) for f in frames).encode()).hexdigest()
    if a.check_only:
        print(f'CHECK OK {name} days={len(days)} frames={len(frames)} manifest={b["manifest_sha256"][:12]} labels={labels_sha[:12]} init={init_sha[:12]}'); return
    out = Path(a.out_root) / name
    if out.exists():
        sys.exit(f'{out} exists; refusing to overwrite')
    import torch, ultralytics
    from ultralytics import YOLO
    ts = time.strftime('%Y-%m-%dT%H%M%S%z'); out.mkdir(parents=True); ds = out / 'dataset'; (ds / 'images' / 'train').mkdir(parents=True); (ds / 'labels' / 'train').mkdir(parents=True)
    for f in frames:
        os.symlink(os.path.realpath(mirror / 'images' / 'train' / f), ds / 'images' / 'train' / f); shutil.copy2(mirror / 'labels' / 'train' / (f[:-4] + '.txt'), ds / 'labels' / 'train' / (f[:-4] + '.txt'))
    (ds / 'data.yaml').write_text(f"path: {ds}\ntrain: images/train\nval: {mirror}/images/val\nnames:\n  0: car\n")
    manifest = dict(experiment='E21', budget=a.budget, n_days=len(days), days=days, replicate_seed=a.seed, started_at=ts, mirror=str(mirror), manifest_file=str(mpath), manifest_sha256=b['manifest_sha256'],
                    n_train=len(frames), train_labels_sha256=labels_sha, init_weights=a.init, init_sha256=init_sha,
                    args=dict(imgsz=1280, epochs=60, patience=15, batch=16, seed=a.seed, amp=True, deterministic=True, model='yolo11s'),
                    environment=f"host={platform.node()} python={sys.version.split()[0]} torch={torch.__version__} ultralytics={ultralytics.__version__} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=1))
    YOLO(a.init).train(data=str(ds / 'data.yaml'), imgsz=1280, epochs=60, patience=15, batch=16, seed=a.seed, amp=True, deterministic=True, project=str(out), name='train', exist_ok=True, verbose=False, plots=False)
    best = out / 'train' / 'weights' / 'best.pt'; manifest['best_sha256'] = sha(best); manifest['finished_at'] = time.strftime('%Y-%m-%dT%H%M%S%z')
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=1)); print('DONE', name, manifest['best_sha256'])


if __name__ == '__main__':
    main()
