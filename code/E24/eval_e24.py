#!/usr/bin/env python3
"""E24 step 4 (v4.2): evaluate the six E24 models (2 conditions x 3 seeds) on four sets each (24 jobs):
  in_domain (the condition's 25-frame source-domain validation split, its own labels), FIXED-AUDITED-TEST (723),
  FIXED-AUDITED-419 (the 419 blind frames with their audited labels taken file by file from the 723 set) and
  INDEPENDENT-BLIND-419.

Gates (shared by --dry-run and the real run; every violation aborts before any evaluation):
  * the freeze file exists, its field set is EXACTLY the required set (a missing or extra field aborts) and every
    field is recomputed and compared: meta digest, LABELSET_MANIFEST digests of FIXED-AUDITED-TEST and
    INDEPENDENT-BLIND-419, the 723 test-image digests (images manifest), image size, VAL_ARGS, the collapse and
    diagnostic thresholds, this script's digest, the unblind_eval.py import digest, the training script digest, the
    plan digest, the model registry digest, the environment;
  * the two snowy-lot label sets load with their per-file digests; 419 blind stems ⊂ 723 audited stems;
  * E24_MODELS.json (digest bound in the freeze) lists exactly 2 conditions x 3 seeds {0,1,2}, unique tags; per model
    the weights digest equals the recomputed sha256 and best_sha256 of the bound run_manifest.json, whose recomputed
    digest equals the registered run_manifest_sha256, whose script_sha256 equals the frozen training-script digest and
    whose effective args equal the frozen protocol (imgsz 1280, epochs 60, patience 15, batch 16, seed);
  * the source-domain data of each condition re-verified as at training (train_e24.verify_condition).
Per job: mAP@0.5, mAP@[.5:.95], precision, recall, n_images; predictions.json yields the number of boxes at
conf >= 0.001 / >= 0.25 and the number of frames with any box. Ultralytics writes no predictions.json when a run
produced no prediction at all. Evidence is taken from the ORIGINAL validation run itself through the on_val_end
callback (validator.seen, the dataset's image list, len(validator.jdict), args.save_json): the file is replaced by an
empty predictions.json plus a ZERO_PREDICTIONS_VERIFIED.json record only when the run processed exactly the expected
images, save_json was on and the run's jdict is empty; a missing file with a non-empty jdict, or an incomplete run,
aborts. verify_zero_predictions is called unconditionally after every val(): it first requires complete evidence
(callback fired, jdict a real list, dataset image list present with exactly n_img entries, seen == n_img, save_json
on; absent or None fields are never treated as zero), then either cross-checks an existing file's length against
len(jdict) or, for a missing file, accepts only len(jdict) == 0 and writes '[]' with a provenance record. No second
inference is ever run. Library output is
captured to library_output.log with stdout/stderr flushed before and after the redirection; the duplicated
descriptors are closed in the finally block. Summary S: mean +- SD
(ddof=1) over seeds for mAP@0.5, mAP@[.5:.95], precision, recall; per-set flags: all seeds mAP@0.5 < 0.10 (collapse)
and, for in_domain, all seeds >= 0.80 (source-domain diagnostic gate). Library output and metrics go to files in the
run directory; the console reports job progress only. OUTPUT_SHA256SUMS over the run directory.
--freeze-now refuses to overwrite an existing freeze file.
"""
import argparse, hashlib, json, os, platform, shutil, statistics as st, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'E17')); sys.path.insert(0, str(HERE))
from unblind_eval import sha, load_labelset, build_dataset, environment_versions, VAL_ARGS  # noqa: E402
import unblind_eval  # noqa: E402
from train_e24 import verify_condition, CONDITIONS, SEEDS, ARGS as TRAIN_ARGS  # noqa: E402

SNOW_SETS = ['FIXED-AUDITED-TEST', 'FIXED-AUDITED-419', 'INDEPENDENT-BLIND-419']; COLLAPSE = 0.10; DIAG = 0.80; W, H = 2304, 1536
REQUIRED_FREEZE_KEYS = {'e24_meta_sha256', 'labelset_manifest:FIXED-AUDITED-TEST', 'labelset_manifest:INDEPENDENT-BLIND-419', 'images_manifest', 'n_images', 'image_size', 'val_args', 'collapse_threshold', 'diagnostic_threshold', 'eval_script_sha256', 'unblind_eval_sha256', 'train_script_sha256', 'plan_sha256', 'models_sha256', 'environment'}


def images_manifest(labelsets_root, images):
    from PIL import Image
    stems = sorted(p.stem for p in (labelsets_root / 'FIXED-AUDITED-TEST').glob('*.txt')); lines = []
    for s_ in stems:
        img = images / f'{s_}.png'
        if not img.exists():
            raise SystemExit('test image missing')
        with Image.open(img) as im:
            if im.size != (W, H):
                raise SystemExit('test image size differs')
        lines.append(f'{sha(img)}  {img.name}')
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest(), len(stems)


def freeze_items(a):
    im, n = images_manifest(a.labelsets_root, a.images)
    return dict(e24_meta_sha256=sha(a.data_root / 'E24_manifests.meta.json'), **{f'labelset_manifest:{s}': sha(a.labelsets_root / s / 'LABELSET_MANIFEST.json') for s in ('FIXED-AUDITED-TEST', 'INDEPENDENT-BLIND-419')},
                images_manifest=im, n_images=n, image_size=[W, H], val_args=VAL_ARGS, collapse_threshold=COLLAPSE, diagnostic_threshold=DIAG,
                eval_script_sha256=sha(__file__), unblind_eval_sha256=sha(unblind_eval.__file__), train_script_sha256=sha(HERE / 'train_e24.py'), plan_sha256=sha(HERE / 'E24_PLAN.md'), models_sha256=sha(a.models), environment=environment_versions())


def derive_fixed_419(sets, run):
    """FIXED-AUDITED-419: the blind 419 stems with their audited labels copied file by file from the 723 set."""
    blind = sets['INDEPENDENT-BLIND-419']; fixed = sets['FIXED-AUDITED-TEST']
    if not blind['stems'] <= fixed['stems']:
        raise SystemExit('blind 419 stems are not a subset of the audited 723 stems')
    dst = run / 'labelsets' / 'FIXED-AUDITED-419'; dst.mkdir(parents=True, exist_ok=True); files = {}
    for s_ in sorted(blind['stems']):
        shutil.copy2(fixed['dir'] / f'{s_}.txt', dst / f'{s_}.txt'); files[f'{s_}.txt'] = fixed['man']['files'][f'{s_}.txt']
    man = dict(label_set_id='FIXED-AUDITED-419', n_files=len(files), aggregate_sha256=hashlib.sha256(''.join(files[n] for n in sorted(files)).encode()).hexdigest(), files=files, derived_from='FIXED-AUDITED-TEST', derived_from_aggregate=fixed['man']['aggregate_sha256'], frames_from='INDEPENDENT-BLIND-419')
    (dst / 'LABELSET_MANIFEST.json').write_text(json.dumps(man, indent=1))
    return dict(dir=dst, man=man, stems=set(blind['stems']), flags=None)


def gate(a):
    """All input checks; returns (freeze items, registry, sets). Used by --dry-run and the real run alike."""
    fz = json.loads(a.freeze.read_text())['items']
    if set(fz) != REQUIRED_FREEZE_KEYS:
        raise SystemExit(f'freeze field set differs from the required set: missing {sorted(REQUIRED_FREEZE_KEYS - set(fz))}, extra {sorted(set(fz) - REQUIRED_FREEZE_KEYS)}')
    now = freeze_items(a)
    for k in sorted(REQUIRED_FREEZE_KEYS):
        if now.get(k) != fz[k]:
            raise SystemExit(f'freeze field differs: {k}')
    reg = json.loads(a.models.read_text()); models = reg['models']
    grid = sorted((m['condition'], m['seed']) for m in models)
    if grid != sorted((c, s_) for c in CONDITIONS for s_ in SEEDS) or len({m['tag'] for m in models}) != 6:
        raise SystemExit('E24_MODELS.json is not exactly the 2 x 3 grid with unique tags')
    for m in models:
        rmp = Path(m['run_manifest'])
        if sha(rmp) != m.get('run_manifest_sha256'):
            raise SystemExit(f"{m['tag']}: run_manifest digest differs from the registry")
        rm = json.loads(rmp.read_text())
        if sha(m['path']) != m['sha256'] or rm.get('best_sha256') != m['sha256'] or rm['condition'] != m['condition'] or rm['training_seed'] != m['seed'] or rm['meta_sha256'] != fz['e24_meta_sha256']:
            raise SystemExit(f"{m['tag']}: weights or run manifest not bound")
        if rm.get('script_sha256') != fz['train_script_sha256'] or rm.get('args') != dict(TRAIN_ARGS, seed=m['seed']):
            raise SystemExit(f"{m['tag']}: training script or effective arguments differ from the frozen protocol")
    for c in CONDITIONS:
        verify_condition(a.data_root, c, 0, fz['e24_meta_sha256'])
    sets = {n: load_labelset(a.labelsets_root, n) for n in ('FIXED-AUDITED-TEST', 'INDEPENDENT-BLIND-419')}
    if not sets['INDEPENDENT-BLIND-419']['stems'] <= sets['FIXED-AUDITED-TEST']['stems']:
        raise SystemExit('419 not a subset of 723')
    return fz, reg, sets


def capture_validator(v):
    """on_val_end callback payload: evidence from the ORIGINAL validation run (no second inference). Nothing is
    defaulted: absent or None attributes are recorded as absent and rejected by verify_zero_predictions."""
    missing = object()
    try:
        im_files = sorted(str(f) for f in v.dataloader.dataset.im_files)
    except Exception:
        im_files = None
    jdict = getattr(v, 'jdict', missing)
    return dict(fired=True, seen=getattr(v, 'seen', None), jdict_is_list=isinstance(jdict, list), n_jdict=(len(jdict) if isinstance(jdict, list) else None),
                save_json=getattr(getattr(v, 'args', None), 'save_json', None), im_files=im_files)


def verify_zero_predictions(cap, n_img, pj, key):
    """Ultralytics omits predictions.json when the run's jdict is empty. Accept ONLY on evidence of the original run:
    it processed exactly n_img images (seen and the dataset image list), save_json was on and jdict is empty; then
    write an empty predictions.json and a provenance record. A non-empty jdict with a missing file, or an incomplete
    run, is an output fault -> abort. Called unconditionally after every val(): with an existing file it only
    cross-checks the file length against len(jdict). Pure bookkeeping: no library call, no console output. Evidence
    checks are inline so that the function is self-contained."""
    if not cap or not cap.get('fired'):
        raise SystemExit(f'{key}: no on_val_end evidence from the validation run -> output fault')
    if not isinstance(cap.get('im_files'), list) or len(cap['im_files']) != n_img:
        raise SystemExit(f"{key}: dataset image list missing or not {n_img} images -> output fault")
    if not isinstance(cap.get('seen'), int) or cap['seen'] != n_img:
        raise SystemExit(f"{key}: validator processed {cap.get('seen')} images, expected {n_img} -> output fault")
    if not cap.get('jdict_is_list') or not isinstance(cap.get('n_jdict'), int):
        raise SystemExit(f'{key}: validator jdict missing or not a list -> output fault')
    if cap.get('save_json') is not True:
        raise SystemExit(f'{key}: save_json was not on in the validation run -> output fault')
    if pj.exists():
        if len(json.loads(pj.read_text())) != cap['n_jdict']:
            raise SystemExit(f"{key}: predictions.json length differs from the run's jdict ({cap['n_jdict']}) -> output fault")
        return
    if cap['n_jdict'] != 0:
        raise SystemExit(f"{key}: predictions.json missing although the run produced {cap['n_jdict']} predictions -> output fault")
    pj.parent.mkdir(parents=True, exist_ok=True); pj.write_text('[]')
    (pj.parent / 'ZERO_PREDICTIONS_VERIFIED.json').write_text(json.dumps(dict(key=key, images_processed=cap['seen'], n_images_in_dataset=len(cap['im_files']), jdict_len=0, save_json=True, method='evidence of the original validation run via on_val_end (validator.seen, dataset image list, len(jdict)); no second inference', verified_at=time.strftime('%Y-%m-%dT%H%M%S%z')), indent=1))


def count_predictions(pred_json, n_images):
    p = json.loads(Path(pred_json).read_text()); per = {}
    for x in p:
        per.setdefault(x['image_id'], [0, 0]); per[x['image_id']][0] += 1; per[x['image_id']][1] += (x['score'] >= 0.25)
    return dict(boxes_conf001=sum(v[0] for v in per.values()), boxes_conf025=sum(v[1] for v in per.values()), frames_with_box_conf001=len(per), frames_with_box_conf025=sum(1 for v in per.values() if v[1]), n_images=n_images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=Path, default=HERE / 'E24_MODELS.json'); ap.add_argument('--freeze', type=Path, default=HERE / 'E24_FREEZE.json')
    ap.add_argument('--data-root', type=Path, default=Path('/mnt/ssd/snow_review_runs/E24/data')); ap.add_argument('--labelsets-root', type=Path, required=True)
    ap.add_argument('--images', type=Path, default=Path('/mnt/ssd/datasets/custom_dataset/image3')); ap.add_argument('--out-root', type=Path, default=Path('/mnt/ssd/snow_review_runs/E24/eval'))
    ap.add_argument('--freeze-now', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.freeze_now:
        if a.freeze.exists():
            raise SystemExit(f'{a.freeze} exists; a changed freeze needs a new version, not an overwrite')
        a.freeze.write_text(json.dumps(dict(frozen_at=time.strftime('%Y-%m-%dT%H%M%S%z'), host=platform.node(), items=freeze_items(a)), indent=1)); print('FROZEN', a.freeze); return
    fz, reg, sets = gate(a)
    run = a.out_root / time.strftime('%Y-%m-%dT%H%M%S%z'); run.mkdir(parents=True)
    jobs = [(m, s_) for m in reg['models'] for s_ in ['in_domain'] + SNOW_SETS]
    man = dict(experiment='E24', script_sha256=sha(__file__), freeze_sha256=sha(a.freeze), models_sha256=sha(a.models), n_jobs=len(jobs), val_args=VAL_ARGS, started_at=time.strftime('%Y-%m-%dT%H%M%S%z'), environment=environment_versions())
    (run / 'run_manifest.json').write_text(json.dumps(man, indent=1))
    if a.dry_run:
        print('DRY-RUN OK: all gates passed;', len(jobs), 'jobs ->', run); return
    from ultralytics import YOLO
    sets['FIXED-AUDITED-419'] = derive_fixed_419(sets, run)
    ds = {n: build_dataset(sets[n], a.images, run, n, coco=False) for n in SNOW_SETS}
    log = (run / 'library_output.log').open('a'); A = {}
    for i, (m, s_) in enumerate(jobs, 1):
        key = f"{m['tag']}|{s_}"
        if s_ == 'in_domain':
            yaml = a.data_root / m['condition'] / 'data.yaml'; n_img = 25
        else:
            d, stems = ds[s_]; yaml = d / 'data.yaml'; yaml.write_text(f'path: {d}\ntrain: images/val\nval: images/val\nnames:\n  0: car\n'); n_img = len(stems)
        old = (os.dup(1), os.dup(2)); cap = {}
        try:
            sys.stdout.flush(); sys.stderr.flush(); log.flush(); os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            model = YOLO(m['path']); model.add_callback('on_val_end', lambda v: cap.update(capture_validator(v)))
            r = model.val(data=str(yaml), split='val', save_json=True, plots=False, project=str(run / 'val'), name=key.replace('|', '__'), exist_ok=True, verbose=False, **VAL_ARGS)
        finally:
            sys.stdout.flush(); sys.stderr.flush(); log.flush(); os.dup2(old[0], 1); os.dup2(old[1], 2); os.close(old[0]); os.close(old[1])
        pj = run / 'val' / key.replace('|', '__') / 'predictions.json'
        verify_zero_predictions(cap, n_img, pj, key)  # unconditional: complete evidence required; existing file cross-checked, missing file accepted only as verified zero
        A[key] = dict(condition=m['condition'], seed=m['seed'], set=s_, map50=float(r.box.map50), map5095=float(r.box.map), precision=float(r.box.mp), recall=float(r.box.mr), n_images=n_img, validator=dict(seen=cap['seen'], n_jdict=cap['n_jdict'], n_dataset_images=len(cap['im_files'])), predictions=count_predictions(pj, n_img))
        (run / 'results_A.json').write_text(json.dumps(A, indent=1)); print(f'job {i}/{len(jobs)} done: {key}')
    S = {}
    for cond in CONDITIONS:
        S[cond] = {}
        for s_ in ['in_domain'] + SNOW_SETS:
            vals = {k: [A[f"{m['tag']}|{s_}"][k] for m in reg['models'] if m['condition'] == cond] for k in ('map50', 'map5095', 'precision', 'recall')}
            S[cond][s_] = {k: dict(values=v, mean=st.mean(v), sd_ddof1=st.stdev(v), n=len(v)) for k, v in vals.items()}
            S[cond][s_]['boxes_conf025_per_seed'] = [A[f"{m['tag']}|{s_}"]['predictions']['boxes_conf025'] for m in reg['models'] if m['condition'] == cond]
            if s_ == 'in_domain':
                S[cond][s_]['diagnostic_gate_all_seeds_map50_ge_0_80'] = all(v >= DIAG for v in vals['map50'])
            else:
                S[cond][s_]['collapse_all_seeds_map50_lt_0_10'] = all(v < COLLAPSE for v in vals['map50'])
    (run / 'results_S.json').write_text(json.dumps(S, indent=1)); man['finished_at'] = time.strftime('%Y-%m-%dT%H%M%S%z'); (run / 'run_manifest.json').write_text(json.dumps(man, indent=1)); log.close()
    files = sorted(p for p in run.rglob('*') if p.is_file() and not p.is_symlink() and p.name != 'OUTPUT_SHA256SUMS')
    (run / 'OUTPUT_SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(run)}\n' for p in files)); print('DONE', run, len(files) + 1, 'files')


if __name__ == '__main__':
    main()
