#!/usr/bin/env python3
"""E24 step 3 (v2): train one condition (pucpr_car | pucpr_stall) with one seed; identical protocol to E22 / E21.

Identity gate BEFORE training (verify_condition), on the data that YOLO will actually read:
  * E24_manifests.meta.json digest == --expected-meta-sha256 (pre-registered);
  * seed in the frozen replicate set {0, 1, 2};
  * labels/{train,val}: file set == frozen set, every label digest == frozen digest;
  * images/{train,val}: file set == frozen set (no extra file), every image decodes to the frozen pixel SHA256;
  * data.yaml text == the canonical text for this condition (path, train/val dirs, single class 'car');
  * init weights sha256 == expected (full digest).
run_manifest.json binds the script digest, the meta digest, the environment and the effective training arguments.
--self-test builds a temporary fixture and proves that a wrong image, a wrong data.yaml, an extra image and a
seed outside {0,1,2} are each refused (writes nothing to the formal directories).
"""
import argparse, hashlib, json, os, platform, shutil, sys, tempfile, time
from pathlib import Path

SEEDS = (0, 1, 2); CONDITIONS = ('pucpr_car', 'pucpr_stall')
ARGS = dict(imgsz=1280, epochs=60, patience=15, batch=16, amp=True, deterministic=True, model='yolo11s')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def pix_sha(p):
    import numpy as np
    from PIL import Image
    return hashlib.sha256(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest()


def canonical_yaml(d):
    return f'path: {d}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n'


def verify_condition(root, condition, seed, expected_meta_sha):
    """Raise SystemExit on the first violation; return the frozen condition meta."""
    if condition not in CONDITIONS:
        raise SystemExit(f'unknown condition {condition}')
    if seed not in SEEDS:
        raise SystemExit(f'seed {seed} is not in the frozen replicate set {SEEDS}')
    meta_p = root / 'E24_manifests.meta.json'
    if not meta_p.exists() or sha(meta_p) != expected_meta_sha:
        raise SystemExit('E24_manifests.meta.json missing or differs from the pre-registered digest')
    meta = json.loads(meta_p.read_text())['conditions'][condition]; d = root / condition
    yaml = d / 'data.yaml'
    if not yaml.exists() or yaml.read_text() != canonical_yaml(d):
        raise SystemExit(f'{condition}: data.yaml missing or differs from the canonical text')
    for s in ('train', 'val'):
        frozen = sorted(n for n, v in meta['files'].items() if v['split'] == s)
        labels = sorted(p.stem for p in (d / 'labels' / s).iterdir()); images = sorted(p.stem for p in (d / 'images' / s).iterdir())
        if labels != frozen or len(frozen) != meta['n'][s]:
            raise SystemExit(f'{condition}/{s}: label file set differs from the frozen meta')
        if images != frozen:
            raise SystemExit(f'{condition}/{s}: image file set differs from the frozen meta (extra or missing images)')
        for n in frozen:
            if sha(d / 'labels' / s / (n + '.txt')) != meta['files'][n]['label_sha256']:
                raise SystemExit(f'{condition}/{s}: label digest differs for a frame')
            img = d / 'images' / s / (n + '.jpg')
            if not img.exists() or pix_sha(img) != meta['files'][n]['image_pixel_sha256']:
                raise SystemExit(f'{condition}/{s}: image pixel digest differs for a frame')
    return meta


def self_test(root, condition, expected_meta_sha):
    """Copy the condition into a temp root, verify it passes, then prove four corruptions are refused."""
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(prefix='e24_selftest_'))
    try:
        shutil.copy2(root / 'E24_manifests.meta.json', tmp / 'E24_manifests.meta.json')
        def fresh():
            d = tmp / condition
            if d.exists():
                shutil.rmtree(d)
            for s in ('train', 'val'):
                (d / 'images' / s).mkdir(parents=True); (d / 'labels' / s).mkdir(parents=True)
                for p in (root / condition / 'images' / s).iterdir():
                    os.symlink(os.path.realpath(p), d / 'images' / s / p.name)
                for p in (root / condition / 'labels' / s).iterdir():
                    shutil.copy2(p, d / 'labels' / s / p.name)
            (d / 'data.yaml').write_text(canonical_yaml(d)); return d
        d = fresh(); verify_condition(tmp, condition, 0, expected_meta_sha); print('self-test: clean fixture passes')
        cases = []
        def expect_refusal(label, mutate, seed=0):
            dd = fresh(); mutate(dd)
            try:
                verify_condition(tmp, condition, seed, expected_meta_sha); raise AssertionError(f'{label}: NOT refused')
            except SystemExit as e:
                cases.append((label, str(e)[:80]))
        def wrong_image(dd):
            p = dd / 'images' / 'train' / sorted(os.listdir(dd / 'images' / 'train'))[0]; p.unlink(); Image.new('RGB', (1280, 720), (7, 7, 7)).save(p, quality=95)
        expect_refusal('wrong image', wrong_image)
        expect_refusal('wrong data.yaml', lambda dd: (dd / 'data.yaml').write_text(canonical_yaml(dd).replace('0: car', '0: car\n  1: truck')))
        expect_refusal('extra image', lambda dd: shutil.copy2(os.path.realpath(dd / 'images' / 'train' / sorted(os.listdir(dd / 'images' / 'train'))[0]), dd / 'images' / 'train' / 'EXTRA_frame.jpg'))
        expect_refusal('seed 99', lambda dd: None, seed=99)
        expect_refusal('wrong label', lambda dd: (dd / 'labels' / 'train' / sorted(os.listdir(dd / 'labels' / 'train'))[0]).write_text('0 0.5 0.5 0.1 0.1\n'))
        for c in cases:
            print('self-test refused:', c)
        print('SELF-TEST PASSED', len(cases), 'corruptions refused')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', choices=CONDITIONS, required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--data-root', type=Path, default=Path('/mnt/ssd/snow_review_runs/E24/data')); ap.add_argument('--out-root', type=Path, default=Path('/mnt/ssd/snow_review_runs/E24'))
    ap.add_argument('--init', type=Path, default=Path('/mnt/ssd/snow_review_runs/weights/yolo11s.pt')); ap.add_argument('--expected-init-sha256', default='85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5')
    ap.add_argument('--expected-meta-sha256', required=True); ap.add_argument('--check-only', action='store_true'); ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test(a.data_root, a.condition, a.expected_meta_sha256); return
    meta = verify_condition(a.data_root, a.condition, a.seed, a.expected_meta_sha256)
    init_sha = sha(a.init)
    if init_sha != a.expected_init_sha256:
        raise SystemExit('init weights digest differs from the expected full digest')
    if a.check_only:
        print('CHECK OK', a.condition, 'seed', a.seed, meta['n'], 'images verified by pixel digest, data.yaml canonical, init', init_sha[:12]); return
    out = a.out_root / f'{a.condition}_s{a.seed}'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    import torch, ultralytics
    from ultralytics import YOLO
    out.mkdir(parents=True); d = a.data_root / a.condition
    man = dict(experiment='E24', script_sha256=sha(__file__), condition=a.condition, training_seed=a.seed, data=str(d), data_yaml_sha256=sha(d / 'data.yaml'), meta_sha256=a.expected_meta_sha256,
               n=meta['n'], boxes=meta['boxes'], train_list_sha256=meta['train_list_sha256'], val_list_sha256=meta['val_list_sha256'], init_weights=str(a.init), init_sha256=init_sha,
               args=dict(ARGS, seed=a.seed), started_at=time.strftime('%Y-%m-%dT%H%M%S%z'),
               environment=dict(host=platform.node(), python=sys.version.split()[0], torch=torch.__version__, ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))
    (out / 'run_manifest.json').write_text(json.dumps(man, indent=1))
    YOLO(str(a.init)).train(data=str(d / 'data.yaml'), imgsz=ARGS['imgsz'], epochs=ARGS['epochs'], patience=ARGS['patience'], batch=ARGS['batch'], seed=a.seed, amp=ARGS['amp'], deterministic=ARGS['deterministic'],
                             project=str(out), name='train', exist_ok=True, verbose=False, plots=False)
    best = out / 'train' / 'weights' / 'best.pt'; man['best_sha256'] = sha(best); man['finished_at'] = time.strftime('%Y-%m-%dT%H%M%S%z')
    (out / 'run_manifest.json').write_text(json.dumps(man, indent=1)); print('DONE', a.condition, a.seed, man['best_sha256'])


if __name__ == '__main__':
    main()
