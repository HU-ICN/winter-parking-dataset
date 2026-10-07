#!/usr/bin/env python3
"""E18: zero-shot COCO-pretrained YOLO11 n/s/m on the held-out test split (no training).

Two frozen class mappings, both scored as the single dataset class 'car' via single_cls:
  car_only : COCO class 2 (car)
  vehicle  : COCO classes 2, 5, 7 (car, bus, truck)
Evidence package per run: run_manifest.json, environment.txt, command.sh, data yaml,
weights sha256, raw ultralytics results (JSON/CSV), per-image predictions, validation.json.
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time, shutil
from pathlib import Path

def sha(p): 
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', default='/mnt/ssd/datasets/custom_dataset/snowpark_yolo2')
    ap.add_argument('--weights-dir', default='/mnt/ssd/snow_review_runs/weights')
    ap.add_argument('--out-root', default='/mnt/ssd/snow_review_runs')
    ap.add_argument('--models', nargs='+', default=['yolo11n', 'yolo11s', 'yolo11m'])
    ap.add_argument('--imgsz', type=int, default=1280); ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--label-set-id', default='FIXED-AUDITED-TEST')
    a = ap.parse_args()
    import torch, ultralytics
    from ultralytics import YOLO
    ts = time.strftime('%Y-%m-%dT%H%M%S%z'); out = Path(a.out_root) / f'E18_{ts}'; out.mkdir(parents=True)
    root = Path(a.dataset_root); test_imgs = sorted((root / 'images' / 'test').glob('*.png')); test_lbls = sorted((root / 'labels' / 'test').glob('*.txt'))
    yaml = out / 'e18_data.yaml'
    yaml.write_text(f"path: {root}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n  0: car\n")
    env = f"host={platform.node()} os={platform.platform()} python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda} ultralytics={ultralytics.__version__} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}\n"
    (out / 'environment.txt').write_text(env)
    (out / 'command.sh').write_text('#!/bin/sh\n' + ' '.join([sys.executable] + sys.argv) + '\n')
    manifest = dict(experiment='E18', label_set_id=a.label_set_id, split='test', n_images=len(test_imgs), n_label_files=len(test_lbls),
                    imgsz=a.imgsz, batch=a.batch, conf=0.001, iou=0.7, single_cls=True, started_at=ts, host=platform.node(),
                    mappings={'car_only': [2], 'vehicle': [2, 5, 7]}, weights={}, runs={})
    for m in a.models:
        w = Path(a.weights_dir) / f'{m}.pt'; manifest['weights'][m] = dict(path=str(w), sha256=sha(w))
        for name, classes in manifest['mappings'].items():
            rd = out / f'{m}_{name}'; model = YOLO(str(w))
            r = model.val(data=str(yaml), split='test', imgsz=a.imgsz, batch=a.batch, conf=0.001, iou=0.7, classes=classes, single_cls=True,
                          save_json=True, plots=False, project=str(out), name=f'{m}_{name}', exist_ok=True, verbose=False)
            res = dict(map50=float(r.box.map50), map5095=float(r.box.map), precision=float(r.box.mp), recall=float(r.box.mr), classes=classes)
            manifest['runs'][f'{m}_{name}'] = res; print(m, name, json.dumps(res))
            # per-image predictions (YOLO txt with conf) for later re-scoring on independent label sets
            pd = rd / 'predictions_txt'; pd.mkdir(exist_ok=True)
            for img in test_imgs:
                pr = model.predict(str(img), imgsz=a.imgsz, conf=0.001, iou=0.7, classes=classes, verbose=False)[0]
                with open(pd / (img.stem + '.txt'), 'w') as fh:
                    for b in pr.boxes:
                        x, y, bw, bh = b.xywhn[0].tolist(); fh.write(f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f} {float(b.conf[0]):.5f}\n")
    manifest['finished_at'] = time.strftime('%Y-%m-%dT%H%M%S%z')
    manifest['validation'] = dict(images_eq_labels=len(test_imgs) == len(test_lbls), n_images=len(test_imgs))
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=1)); print('OUT', out)

if __name__ == '__main__': main()
