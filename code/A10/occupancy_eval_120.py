#!/usr/bin/env python3
"""Independent per-stall occupancy evaluation on 120 frames (40 golden + 80 sealed independent frames).
Version 3 (2026-09-07; auditor items after 35f64a1: the freeze is made on the prediction host with all 120 image digests
compared with the sealed payloads' image_sha256, the weights verified and the prediction environment recorded and compared
before --predict; full-SHA256 equality for the golden files and slots; exclusive box-count field). Runs only after review. Covers the frozen FINAL model only
(this is not the E21 per-budget occupancy curve).

Rules are the FROZEN A10 evaluator (experiments/A10/occupancy_evaluator.py: centroid rule, R = 90 px, offsets applied
to the stall polygons, one box marks at most one stall, unresolved stalls excluded from the denominator, empty
predictions scored). Nothing in the rule is changed; sealed labels are never edited; the annotator's straddling
convention is NOT excluded (main results keep every sample).

Two prediction sources, kept apart (auditor item 2):
  * HISTORICAL occ_pred.json (40 golden frames, digest 9b94d53e...): used ONLY for the A10 golden regression test
    (must reproduce accuracy 0.969512 / F1 0.93141).
  * UNIFIED predictions: the frozen final model run TODAY on all 120 frames with one environment, one argument set
    and one coordinate precision (predict imgsz 1280, conf 0.25, NMS IoU 0.7, max_det 300, class 0). The main results
    40 / 80 / 120 use these; the 40-frame unified result is also reported next to the golden number as a
    reproducibility statement (differences are reported, not asserted).

Modes
  --freeze    (CPU) verifies and freezes every input BEFORE any prediction: golden four files (digests), the 80-frame
              seal chain (seal list == meta.files, aggregate, every file, meta bound to a verified receipt), the manifest,
              every payload (exactly 82 legal states, complete, image name), the 120 images (content digests, size),
              offsets, code (this script, the frozen evaluator), environment; writes OCC120_FREEZE.json.
  --predict   (4090) requires the freeze to match; writes unified predictions for the 120 frames into a NEW run
              directory (experiments/A10/predictions/<ts>/unified_pred_120.json + meta with image digests, weights
              digest, arguments, environment); never overwrites a fixed prediction file.
  --evaluate  (CPU) requires the freeze to match and the unified predictions to be bound (weights digest, arguments,
              image digests); golden regression with the historical predictions; main results 40 / 80 / 120 with the
              unified predictions; frame-cluster bootstrap 95% CIs (2,000, seed 0) for accuracy / occupied precision /
              recall / F1 with F1 = 2TP / (2TP + FP + FN), undefined ONLY when that denominator is 0 (a replicate with an
              undefined metric is dropped and counted); results sealed in experiments/A10/results/<ts>/.
"""
import argparse, csv, hashlib, json, os, platform, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occupancy_evaluator import evaluate, occupied_set, validate_gt  # noqa: E402  frozen rules

GOLDEN = dict(gt='occupancy_gt.json', offsets='frame_offsets.json', pred='occ_pred.json', slots='slots.json')
GOLDEN_SHA = {'gt': 'e86bede9e6114d306cd8cf1bd93f426abcd0bbb16ae4700a4b1ab7f7b1884005', 'offsets': '3865529ed531fa0b39274a22b4daa9994e7e832c4382885fe8bea7280eae875e', 'pred': '9b94d53e82956afd1f584771c86cc0e957f478ea9217ee4722df55084be966d4', 'slots': 'acc8361fb4a5f7bb0a5c1ccb4df811794e5dc85b4f493d4bfa99a1f6cdbfbb4f'}  # full SHA256 (v3)
GOLDEN_EXPECTED = dict(accuracy=0.969512, f1=0.93141)
FINAL_WEIGHTS = dict(path='/mnt/ssd/datasets/custom_dataset/snowpark_yolo2/runs/train/weights/best.pt', sha256='ce9e9c772d7738e8b15b63b848ca18ce6e459d60f7aebb62666d6aa5ba3231bc')
PRED_ARGS = dict(imgsz=1280, conf=0.25, iou=0.7, max_det=300, classes=[0])
RULE, RADIUS, BOOT, SEED = 'centroid', 90.0, 2000, 0
SEAL = 'annotation/seals/INDEPENDENT-OCC-120.sha256'; META = 'annotation/seals/INDEPENDENT-OCC-120.meta.json'; MANIFEST = 'experiments/E21/occupancy/independent_occ_add80.csv'; OUTDIR = 'annotation/outputs/occupancy_add80'
FREEZE_FILE = Path('experiments/A10/OCC120_FREEZE.json'); STATES = {'empty', 'occupied', 'unresolved'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def bind_golden():
    b = {}
    for k, f in GOLDEN.items():
        d = sha(f)
        if d != GOLDEN_SHA[k]:
            raise SystemExit(f'golden input {f} digest {d[:16]} != frozen {GOLDEN_SHA[k][:16]}')
        b[k] = d
    gt = json.load(open(GOLDEN['gt'])); off = json.load(open(GOLDEN['offsets'])); pred = json.load(open(GOLDEN['pred'])); slots = json.load(open(GOLDEN['slots']))['slots']
    if not (sorted(gt) == sorted(off) == sorted(pred)) or len(gt) != 40:
        raise SystemExit('golden 40: frame keys of gt/offsets/pred differ or count != 40')
    ids = [s['slot_id'] for s in slots]
    if len(ids) != 82 or len(set(ids)) != 82:
        raise SystemExit('slots.json must define 82 unique stalls')
    return b, gt, off, pred, slots, ids


def bind_sealed_80(ids):
    seal = {l.split()[1]: l.split()[0] for l in open(SEAL).read().splitlines() if l.strip()}; meta = json.load(open(META))
    if seal != meta['files'] or hashlib.sha256(''.join(seal[n] for n in sorted(seal)).encode()).hexdigest() != meta['seal_sha256_of_file_digests']:
        raise SystemExit('80-frame seal list / meta / aggregate do not agree')
    if not meta.get('offmachine_snapshot') or 'PENDING' in str(meta.get('offmachine_snapshot')) or not Path(meta.get('offmachine_receipt', '')).exists():
        raise SystemExit('80-frame seal is not bound to a verified off-machine receipt')
    rc = json.load(open(meta['offmachine_receipt']))
    if rc.get('status') != 'verified' or not rc.get('restore_verified'):
        raise SystemExit('bound receipt is not verified')
    if meta['bound_hashes']['manifest'] != sha(MANIFEST):
        raise SystemExit('seal meta does not bind this manifest')
    rows = list(csv.DictReader(open(MANIFEST, newline='', encoding='utf-8')))
    if len(rows) != 80 or len(seal) != 80 or sorted(p.name for p in Path(OUTDIR).glob('*.json')) != sorted(seal):
        raise SystemExit('expected exactly 80 sealed outputs matching 80 manifest rows')
    gt, off, payload_sha, payload_image_sha, versions = {}, {}, {}, {}, set(); sid = {str(i) for i in ids}
    for r in rows:
        name = f"{r['anonymous_id']}.json"; p = Path(OUTDIR) / name; d = sha(p)
        if d != seal[name]:
            raise SystemExit(f'{name}: differs from the seal')
        j = json.load(open(p)); m = j['revision_annotation']
        if not m.get('complete') or j.get('image_name') != r['image_name'] or j.get('anonymous_id') != r['anonymous_id']:
            raise SystemExit(f'{name}: incomplete or identity mismatch')
        st = j['states']
        if set(st) != sid or len(st) != 82 or not set(st.values()) <= STATES:
            raise SystemExit(f'{name}: states must cover exactly the 82 stalls with legal values')
        if not (isinstance(j.get('offset'), list) and len(j['offset']) == 2):
            raise SystemExit(f'{name}: offset missing')
        if m.get('slots_sha256') != GOLDEN_SHA['slots']:
            raise SystemExit(f'{name}: payload bound to another slots.json')
        if not (isinstance(m.get('image_sha256'), str) and len(m['image_sha256']) == 64):
            raise SystemExit(f'{name}: payload carries no full image digest')
        stem = Path(r['image_name']).stem; versions.add(m.get('tool_version')); payload_sha[stem] = d; payload_image_sha[stem] = m['image_sha256']
        gt[stem] = dict(occupied=[int(s) for s, v in st.items() if v == 'occupied'], unresolved=[int(s) for s, v in st.items() if v == 'unresolved']); off[stem] = [float(x) for x in j['offset']]
    b = dict(seal_list_sha256=sha(SEAL), seal_meta_sha256=sha(META), receipt=meta['offmachine_receipt'], receipt_sha256=sha(meta['offmachine_receipt']), manifest_sha256=sha(MANIFEST), n=80, tool_versions=sorted(versions), payload_sha256=payload_sha, payload_image_sha256=payload_image_sha)
    return b, gt, off


def image_digests(images, stems):
    from PIL import Image
    out = {}
    for st in stems:
        img = images / f'{st}.png'
        if not img.exists():
            raise SystemExit(f'image missing for {st}')
        with Image.open(img) as im:
            if im.size != (2304, 1536):
                raise SystemExit(f'{st}: image size {im.size}')
        out[st] = sha(img)
    return out


def prediction_environment():
    import torch, ultralytics
    return dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, ultralytics=ultralytics.__version__, cuda=getattr(torch.version, 'cuda', None), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')


def freeze_items(a, with_prediction_host=True):
    """v3: the freeze is made on the PREDICTION host: all 120 images are read (missing image -> abort), the 80 sealed
    payloads' image_sha256 must equal the actual image digests, the final weights are hashed and compared, and the
    prediction environment (torch / ultralytics / CUDA / GPU) is recorded. CPU evaluation may run elsewhere; it
    re-verifies everything except the images/weights/prediction environment, which are re-verified through the
    prediction meta instead."""
    gb, gt40, off40, pred40, slots, ids = bind_golden(); sb, gt80, off80 = bind_sealed_80(ids)
    if set(gt40) & set(gt80) or len(set(gt40) | set(gt80)) != 120:
        raise SystemExit('golden 40 and sealed 80 are not disjoint or do not total 120 frames')
    items = dict(version=3, golden=gb, sealed_80=sb, script=sha(__file__), evaluator=sha('experiments/A10/occupancy_evaluator.py'), weights=FINAL_WEIGHTS, pred_args=PRED_ARGS, rule=dict(rule=RULE, radius=RADIUS), bootstrap=dict(replicates=BOOT, seed=SEED),
                 offsets_80_sha256=hashlib.sha256(json.dumps(off80, sort_keys=True).encode()).hexdigest(), gt_80_sha256=hashlib.sha256(json.dumps(gt80, sort_keys=True).encode()).hexdigest(), frames=dict(golden=sorted(gt40), sealed=sorted(gt80)),
                 cpu_environment=dict(host=platform.node(), python=sys.version.split()[0], numpy=np.__version__))
    if with_prediction_host:
        imgs = image_digests(a.images, sorted(set(gt40) | set(gt80)))  # aborts on any missing image
        bad = [st for st, d in sb['payload_image_sha256'].items() if imgs.get(st) != d]
        if bad:
            raise SystemExit(f'{len(bad)} sealed payloads carry an image digest that differs from the image on disk')
        if not Path(FINAL_WEIGHTS['path']).exists() or sha(FINAL_WEIGHTS['path']) != FINAL_WEIGHTS['sha256']:
            raise SystemExit('final model weights missing or differ from the frozen digest')
        items['images'] = imgs; items['weights_verified'] = True; items['prediction_environment'] = prediction_environment()
    return items, (gt40, off40, pred40, slots, ids, gt80, off80)


def check_freeze(a, prediction_host):
    """On the prediction host every frozen item (incl. 120 image digests, weights, prediction environment) must match.
    On a CPU evaluation host the images/weights/prediction environment are compared through the prediction meta."""
    if not FREEZE_FILE.exists():
        raise SystemExit('OCC120_FREEZE.json missing: run --freeze on the prediction host first (after the auditor approved the script)')
    frozen = json.loads(FREEZE_FILE.read_text())['items']
    if not (isinstance(frozen.get('images'), dict) and len(frozen['images']) == 120 and frozen.get('weights_verified') is True and isinstance(frozen.get('prediction_environment'), dict)):
        raise SystemExit('freeze is incomplete: it must carry 120 image digests, a verified weights flag and the prediction environment')
    now, data = freeze_items(a, with_prediction_host=prediction_host)
    keys = [k for k in frozen if k not in ('cpu_environment', 'images', 'weights_verified', 'prediction_environment')]
    diff = [k for k in keys if frozen.get(k) != now.get(k)]
    if diff:
        raise SystemExit(f'frozen inputs changed: {diff}')
    if prediction_host:
        for k in ('images', 'prediction_environment'):
            if frozen[k] != now[k]:
                raise SystemExit(f'{k} differ from the freeze (prediction host)')
    return frozen, data


def per_frame_counts(pred, gt, slots, offsets):
    ids = [s['slot_id'] for s in slots]; out = {}
    for f in sorted(gt):
        o = occupied_set(pred[f], slots, offsets.get(f, [0, 0]), RULE, RADIUS); g, unres = validate_gt(gt[f], ids, f); tp = fp = fn = tn = 0
        for sid in ids:
            if sid in unres:
                continue
            p, t = sid in o, sid in g; tp += p and t; fp += p and not t; fn += (not p) and t; tn += (not p) and (not t)
        out[f] = dict(tp=tp, fp=fp, fn=fn, tn=tn)
    return out


def metrics_from(counts):
    tp = sum(c['tp'] for c in counts); fp = sum(c['fp'] for c in counts); fn = sum(c['fn'] for c in counts); tn = sum(c['tn'] for c in counts); n = tp + fp + fn + tn
    return dict(accuracy=(tp + tn) / n if n else None, precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
                f1=(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None, tp=tp, fp=fp, fn=fn, tn=tn)


def bootstrap(pf, rng):
    frames = sorted(pf); n = len(frames); stats = {k: [] for k in ('accuracy', 'precision', 'recall', 'f1')}; dropped = {k: 0 for k in stats}
    for _ in range(BOOT):
        idx = rng.randint(0, n, size=n); m = metrics_from([pf[frames[i]] for i in idx])
        for k in stats:
            if m[k] is None:
                dropped[k] += 1
            else:
                stats[k].append(m[k])
    return {k: dict(ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 0.95 * BOOT else None, valid_replicates=len(v), undefined_replicates=dropped[k]) for k, v in stats.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--freeze', action='store_true'); ap.add_argument('--predict', action='store_true'); ap.add_argument('--evaluate', action='store_true')
    ap.add_argument('--images', type=Path, default=Path('/mnt/ssd/datasets/custom_dataset/image3')); ap.add_argument('--pred-dir', type=Path, help='experiments/A10/predictions/<ts> holding unified_pred_120.json (for --evaluate)')
    a = ap.parse_args(); os.chdir(Path(__file__).resolve().parents[2])
    if a.freeze:
        items, _ = freeze_items(a, with_prediction_host=True); FREEZE_FILE.write_text(json.dumps(dict(frozen_at=time.strftime('%Y-%m-%dT%H%M%S%z'), host=platform.node(), items=items), indent=1)); print('FROZEN', FREEZE_FILE); return
    frozen, (gt40, off40, pred40, slots, ids, gt80, off80) = check_freeze(a, prediction_host=a.predict)
    if a.predict:
        import torch, ultralytics
        from ultralytics import YOLO
        model = YOLO(FINAL_WEIGHTS['path']); stems = sorted(set(gt40) | set(gt80)); preds = {}
        for st in stems:
            r = model.predict(source=str(a.images / f'{st}.png'), verbose=False, **PRED_ARGS)[0]; preds[st] = [[round(float(x), 2) for x in b] for b in r.boxes.xyxy.cpu().numpy()]
        run = Path('experiments/A10/predictions') / time.strftime('%Y-%m-%dT%H%M%S%z'); run.mkdir(parents=True, exist_ok=False)
        (run / 'unified_pred_120.json').write_text(json.dumps(preds, indent=0))
        meta = dict(weights=FINAL_WEIGHTS, weights_sha256_at_prediction=sha(FINAL_WEIGHTS['path']), pred_args=PRED_ARGS, n_frames=len(preds), image_sha256=frozen['images'], freeze_sha256=sha(FREEZE_FILE), pred_sha256=sha(run / 'unified_pred_120.json'),
                    prediction_environment=prediction_environment(), host=platform.node(), created_at=time.strftime('%Y-%m-%dT%H%M%S%z'))
        (run / 'unified_pred_120.meta.json').write_text(json.dumps(meta, indent=1)); print('PREDICTED', len(preds), 'frames ->', run); return
    if not a.evaluate:
        raise SystemExit('choose --freeze, --predict or --evaluate')
    if not a.pred_dir:
        raise SystemExit('--evaluate needs --pred-dir')
    meta = json.loads((a.pred_dir / 'unified_pred_120.meta.json').read_text()); pf_ = a.pred_dir / 'unified_pred_120.json'
    if meta['weights'] != FINAL_WEIGHTS or meta.get('weights_sha256_at_prediction') != FINAL_WEIGHTS['sha256'] or meta['pred_args'] != PRED_ARGS or meta['pred_sha256'] != sha(pf_) or meta['freeze_sha256'] != sha(FREEZE_FILE):
        raise SystemExit('unified predictions are not bound to the frozen model / arguments / freeze')
    if meta['image_sha256'] != frozen['images'] or meta.get('prediction_environment') != frozen['prediction_environment']:
        raise SystemExit('unified predictions were made on images or in an environment that differ from the freeze')
    upred = json.load(open(pf_))
    if sorted(upred) != sorted(set(gt40) | set(gt80)):
        raise SystemExit('unified predictions do not cover exactly the 120 frames')
    ts = time.strftime('%Y-%m-%dT%H%M%S%z'); run_dir = Path('experiments/A10/results') / ts; run_dir.mkdir(parents=True, exist_ok=False); rng = np.random.RandomState(SEED)
    # golden regression with the HISTORICAL predictions only
    g = evaluate(pred40, gt40, slots, off40, RULE, RADIUS)
    if abs(g['accuracy'] - GOLDEN_EXPECTED['accuracy']) > 5e-7 or abs(g['f1'] - GOLDEN_EXPECTED['f1']) > 5e-6:
        raise SystemExit(f"golden regression failed: accuracy {g['accuracy']:.6f} f1 {g['f1']:.5f}")
    res = {}
    for name, gt, off in (('golden_40', gt40, off40), ('independent_80', gt80, off80), ('all_120', {**gt40, **gt80}, {**off40, **off80})):
        pred = {f: upred[f] for f in gt}; frozen_eval = evaluate(pred, gt, slots, off, RULE, RADIUS); pf = per_frame_counts(pred, gt, slots, off); chk = metrics_from(list(pf.values()))
        if abs(chk['accuracy'] - frozen_eval['accuracy']) > 1e-12 or (chk['tp'], chk['fp'], chk['fn'], chk['tn']) != (frozen_eval['tp'], frozen_eval['fp'], frozen_eval['fn'], frozen_eval['tn']):
            raise SystemExit(f'{name}: per-frame bookkeeping does not reproduce the frozen evaluator totals')
        res[name] = dict(frozen_evaluator=frozen_eval, point=chk, bootstrap=bootstrap(pf, rng), per_frame=pf)
    repro = dict(note='golden 40: historical predictions (regression test) vs unified predictions today; reported, not asserted',
                 historical=dict(accuracy=g['accuracy'], f1=g['f1']), unified=dict(accuracy=res['golden_40']['frozen_evaluator']['accuracy'], f1=res['golden_40']['frozen_evaluator']['f1']),
                 frames_with_identical_box_sets=sum(1 for f in gt40 if sorted(map(tuple, pred40[f])) == sorted(map(tuple, upred[f]))), frames_with_equal_box_count_but_different_boxes=sum(1 for f in gt40 if len(pred40[f]) == len(upred[f]) and sorted(map(tuple, pred40[f])) != sorted(map(tuple, upred[f]))))
    out = dict(version=3, script_sha256=sha(__file__), evaluator_sha256=sha('experiments/A10/occupancy_evaluator.py'), freeze=frozen, freeze_sha256=sha(FREEZE_FILE), predictions=dict(dir=str(a.pred_dir), meta=meta), rule=dict(rule=RULE, radius_px=RADIUS),
               bootstrap=dict(replicates=BOOT, seed=SEED, scheme='frames resampled with replacement; F1 = 2TP/(2TP+FP+FN), undefined only when that denominator is 0; undefined replicates dropped and counted'),
               golden_regression=dict(historical_predictions=g, expected=GOLDEN_EXPECTED, passed=True), golden_reproducibility=repro, results=res,
               scope='frozen final model only; not the E21 per-budget occupancy curve',
               notes=['the annotator marked both stalls spanned by a straddling vehicle as occupied; the frozen rule maps one box to one stall; no sample is excluded (auditor ruling 2026-09-07)',
                      'main results use the unified predictions for all 120 frames; the historical 40-frame predictions serve only the golden regression test'],
               created_at=ts, environment=dict(host=platform.node(), python=sys.version.split()[0], numpy=np.__version__))
    (run_dir / 'occupancy_results.json').write_text(json.dumps(out, indent=1))
    files = sorted(p for p in run_dir.rglob('*') if p.is_file() and p.name != 'OUTPUT_SHA256SUMS'); sums = ''.join(f'{sha(p)}  {p.relative_to(run_dir)}\n' for p in files); (run_dir / 'OUTPUT_SHA256SUMS').write_text(sums)
    print('DONE', run_dir, hashlib.sha256(sums.encode()).hexdigest()[:16])


if __name__ == '__main__':
    main()
