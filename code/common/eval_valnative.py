#!/usr/bin/env python3
"""Uniform val-native evaluation of a single-class ('car') detector on a registered label set.

Frozen protocol (all experiments E17/E21/E22/X02, and the paper's revised tables):
  ultralytics model.val(), imgsz=1280, conf=0.001, iou=0.7 (NMS), max_det=300, batch=16, rect val.
  Metrics: mAP@0.5, mAP@[.5:.95], precision, recall at max-F1 (ultralytics defaults), n_images, n_gt.
Label set = a directory of YOLO txt files (class 0). Images are COPIED from --images into an
isolated dataset dir (historical dirs contain dangling links and stale caches).
Evidence: run_manifest.json (weights sha256, labels aggregate sha256, image list sha256, args,
environment), ultralytics predictions.json (all classes, pre-filter -- do not re-score it without
the coordinate caveat recorded in EXPERIMENT_LOG E18), results.
"""
import argparse, hashlib, json, os, platform, shutil, sys, time
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True); ap.add_argument('--labels', required=True, help='dir of YOLO txt labels defining the label set')
    ap.add_argument('--label-set-id', required=True); ap.add_argument('--images', default='/mnt/ssd/datasets/custom_dataset/image3')
    ap.add_argument('--out-root', default='/mnt/ssd/snow_review_runs/eval'); ap.add_argument('--tag', default='')
    ap.add_argument('--imgsz', type=int, default=1280); ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    import torch, ultralytics
    from ultralytics import YOLO
    ts = time.strftime('%Y-%m-%dT%H%M%S%z'); out = Path(a.out_root) / f'{a.label_set_id}_{a.tag or Path(a.weights).parent.parent.name}_{ts}'; out.mkdir(parents=True)
    ds = out / 'dataset'; (ds / 'images' / 'val').mkdir(parents=True); (ds / 'labels' / 'val').mkdir(parents=True)
    lbls = sorted(Path(a.labels).glob('*.txt')); names = []
    for l in lbls:
        img = Path(a.images) / (l.stem + '.png')
        if not img.exists(): raise SystemExit(f'image missing for {l.stem}')
        shutil.copy2(img, ds / 'images' / 'val' / img.name); shutil.copy2(l, ds / 'labels' / 'val' / l.name); names.append(img.name)
    (ds / 'data.yaml').write_text(f"path: {ds}\ntrain: images/val\nval: images/val\nnames:\n  0: car\n")
    manifest = dict(label_set_id=a.label_set_id, n_images=len(names), images_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
                    labels_sha256=hashlib.sha256(''.join(sha(l) for l in lbls).encode()).hexdigest(), weights=str(a.weights), weights_sha256=sha(a.weights),
                    args=dict(imgsz=a.imgsz, batch=a.batch, conf=0.001, iou=0.7, max_det=300, rect=True), started_at=ts,
                    environment=f"host={platform.node()} python={sys.version.split()[0]} torch={torch.__version__} ultralytics={ultralytics.__version__} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    r = YOLO(a.weights).val(data=str(ds / 'data.yaml'), split='val', imgsz=a.imgsz, batch=a.batch, conf=0.001, iou=0.7, max_det=300, save_json=True, plots=False, project=str(out), name='val', exist_ok=True, verbose=False)
    manifest['results'] = dict(map50=float(r.box.map50), map5095=float(r.box.map), precision=float(r.box.mp), recall=float(r.box.mr))
    manifest['finished_at'] = time.strftime('%Y-%m-%dT%H%M%S%z')
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=1)); (out / 'command.sh').write_text('#!/bin/sh\n' + ' '.join([sys.executable] + sys.argv) + '\n')
    print('RESULT', a.label_set_id, Path(a.weights).parent.parent.name, json.dumps(manifest['results'])); print('OUT', out)

if __name__ == '__main__': main()
