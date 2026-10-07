#!/usr/bin/env python3
"""E21 / E22 models on the independent (blind) and fixed label sets. Version 2 (2026-09-07; auditor items after 4ca030f: all 24 reference items must exist and be compared, model grid and
label-set cardinalities verified at --freeze and at run, reference models evaluated first; 30 x 9 = 270 val jobs), written
under the auditor's boundaries of REVIEW_LOG stage 62; it runs only after the auditor's review.

Same mechanics as experiments/E17/unblind_eval.py (v3): --freeze / --dry-run / run, quiet library output, every
output written before anyone reads it, recursive sealing of the run directory. Independent registry
(BUDGET_BLIND_MODELS.json: 18 E21 + 9 E22 + 3 E17 stage-c reference models); the E17/E18 registry, freeze and
result package are not modified.

Label sets (all from experiments/labelsets, per-file digests and set relations verified):
  FIXED-AUDITED-TEST (723), FIXED-AUDITED-331 / INDEPENDENT-BLIND-331 (same 331 frames),
  FIXED-AUDITED-88 / INDEPENDENT-BLIND-S0S3-88 (same 88 frames), FIXED-AUDITED-419 / INDEPENDENT-BLIND-419
  (same 419 frames = 331 + 88), plus the Amendment-1 87-frame pair derived in the run directory.
Part A  val-native metrics (imgsz 1280, conf 0.001, NMS IoU 0.7, max_det 300, batch 16), every model x every set.
Part S  descriptive summaries, written as plain numbers (NO confidence intervals for cross-label differences):
  E21: per budget and label set, seed mean and sample SD (ddof=1) of mAP@[.5:.95] and mAP@0.5; the full-data
       reference on the SAME label set = mean of the three stage-c replicates (never the fixed-label .8958 for a
       blind set); the first budget whose seed mean is within 0.02 of that same-set reference (descriptive
       counterpart of the E21 fixed-label criterion, which stays decided on FIXED-AUDITED-TEST); per-budget SD
       reported (the frozen 0.03 SD rule was pre-registered for FIXED-AUDITED-TEST only and is reported here as a
       supplementary flag, not a stop).
  E22: per configuration and label set, seed mean and sample SD; the frozen E22 rule (any configuration within
       0.10 of the same-site full-data mean) was decided on FIXED-AUDITED-TEST; here reported descriptively per set.
  Cross-label point differences (FIXED minus BLIND on identical frames) with the seed spread, labelled as point
  differences; 87-frame sensitivity alongside the 88-frame values, never replacing them.
Part X  reproducibility cross-check: the three stage-c models re-evaluated here on the six sets of the sealed
  unblinding package must equal the sealed values (|diff| <= 1e-9); the run aborts otherwise.
"""
import argparse, contextlib, csv, hashlib, json, logging, os, platform, shutil, statistics, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'E17'))
from unblind_eval import sha, say, n_lines, load_labelset, derive_excluded, build_dataset, environment_versions, VAL_ARGS  # noqa: E402

HERE = Path(__file__).resolve().parent
FREEZE_FILE = HERE / 'BUDGET_BLIND_FREEZE.json'
BASE_SETS = ['FIXED-AUDITED-TEST', 'FIXED-AUDITED-331', 'INDEPENDENT-BLIND-331', 'FIXED-AUDITED-88', 'INDEPENDENT-BLIND-S0S3-88', 'FIXED-AUDITED-419', 'INDEPENDENT-BLIND-419']
PAIRS = [('FIXED-AUDITED-331', 'INDEPENDENT-BLIND-331'), ('FIXED-AUDITED-88', 'INDEPENDENT-BLIND-S0S3-88'), ('FIXED-AUDITED-419', 'INDEPENDENT-BLIND-419')]
BUDGETS = ['d01', 'd02', 'd04', 'd08', 'd16', 'd32']; CONFIGS = ['cross_pklot_hbb', 'cross_syn_hbb', 'cross_mixed_hbb']
WITHIN = 0.02; SD_FLAG = 0.03; E22_WITHIN = 0.10


def check_relations(sets):
    S = {k: v['stems'] for k, v in sets.items()}; F = {k: v['man']['files'] for k, v in sets.items()}
    req = [S['FIXED-AUDITED-331'] <= S['FIXED-AUDITED-TEST'], S['FIXED-AUDITED-88'] <= S['FIXED-AUDITED-TEST'], not (S['FIXED-AUDITED-331'] & S['FIXED-AUDITED-88']),
           S['FIXED-AUDITED-331'] == S['INDEPENDENT-BLIND-331'], S['FIXED-AUDITED-88'] == S['INDEPENDENT-BLIND-S0S3-88'],
           S['FIXED-AUDITED-419'] == S['FIXED-AUDITED-331'] | S['FIXED-AUDITED-88'], S['INDEPENDENT-BLIND-419'] == S['INDEPENDENT-BLIND-331'] | S['INDEPENDENT-BLIND-S0S3-88'], S['FIXED-AUDITED-419'] == S['INDEPENDENT-BLIND-419'],
           all(F['FIXED-AUDITED-419'][n] == F[src][n] for src in ('FIXED-AUDITED-331', 'FIXED-AUDITED-88') for n in F[src]),
           all(F['INDEPENDENT-BLIND-419'][n] == F[src][n] for src in ('INDEPENDENT-BLIND-331', 'INDEPENDENT-BLIND-S0S3-88') for n in F[src]),
           all(F['FIXED-AUDITED-331'][n] == F['FIXED-AUDITED-TEST'][n] for n in F['FIXED-AUDITED-331']), all(F['FIXED-AUDITED-88'][n] == F['FIXED-AUDITED-TEST'][n] for n in F['FIXED-AUDITED-88']),
           len(S['FIXED-AUDITED-TEST']) == 723, len(S['FIXED-AUDITED-419']) == 419]
    if not all(req):
        raise SystemExit(f'label-set relations violated: {req}')


def freeze_inputs(a, reg, W, H):
    from PIL import Image
    items = {'script': sha(Path(__file__).resolve()), 'unblind_eval_import': sha(Path(__file__).resolve().parents[1] / 'E17' / 'unblind_eval.py'), 'model_registry': sha(a.models)}
    for n in BASE_SETS:
        d = a.labelsets_root / n; items[f'labelset_manifest:{n}'] = sha(d / 'LABELSET_MANIFEST.json')
        if (d / 'buried_flags.json').exists():
            items[f'buried_flags:{n}'] = sha(d / 'buried_flags.json')
    em = Path(reg['amendment1']['exclude_manifest']); items['exclude_manifest'] = sha(em)
    if items['exclude_manifest'] != reg['amendment1']['exclude_manifest_sha256']:
        raise SystemExit('registry exclude_manifest_sha256 does not match the manifest file')
    rp = reg['reference_package']
    if sha(Path(rp['path']) / 'results_A.json') != rp['results_A_sha256'] or sha(Path(rp['path']) / 'OUTPUT_SHA256SUMS') != rp['output_sums_sha256']:
        raise SystemExit('reference package (sealed unblinding results) differs from the registry digests')
    items['reference_results_A'] = rp['results_A_sha256']; items['reference_output_sums'] = rp['output_sums_sha256']
    items['weights'] = {m['tag']: sha(m['path']) for m in reg['models']}
    for m in reg['models']:
        if items['weights'][m['tag']] != m['sha256']:
            raise SystemExit(f"{m['tag']}: weights sha differs from the registry")
    stems = sorted(p.stem for p in (a.labelsets_root / 'FIXED-AUDITED-TEST').glob('*.txt')); img_lines = []
    for st in stems:
        img = a.images / f'{st}.png'
        if not img.exists():
            raise SystemExit('image missing for a test frame')
        with Image.open(img) as im:
            if im.size != (W, H):
                raise SystemExit(f'image size {im.size} != --image-size {(W, H)}')
        img_lines.append(f'{sha(img)}  {img.name}')
    items['images_dir'] = str(a.images.resolve()); items['images_manifest'] = hashlib.sha256('\n'.join(img_lines).encode()).hexdigest(); items['n_images'] = len(stems); items['image_size'] = [W, H]
    items['parameters'] = dict(val=VAL_ARGS, within=WITHIN, sd_flag=SD_FLAG, e22_within=E22_WITHIN); items['environment'] = environment_versions()
    return items


def check_grid_and_sets(reg, labelsets_root):
    """Model grid must be complete (18 E21 = 6 budgets x 3 seeds, 9 E22 = 3 configs x 3 seeds, 3 E17 stage-c reference;
    30 unique tags) and the seven base label sets must exist with their exact cardinalities."""
    tags = [m['tag'] for m in reg['models']]
    if len(tags) != len(set(tags)) or len(tags) != 30:
        raise SystemExit(f'model registry must hold 30 unique tags, has {len(tags)} ({len(set(tags))} unique)')
    want = {f'E21_{b}_s{s}' for b in BUDGETS for s in (0, 1, 2)} | {f'E22_{c}_s{s}' for c in CONFIGS for s in (0, 1, 2)} | {f'E17_c_s{s}' for s in (0, 1, 2)}
    if set(tags) != want:
        raise SystemExit(f'model grid incomplete or unexpected: missing {sorted(want - set(tags))}, extra {sorted(set(tags) - want)}')
    card = {'FIXED-AUDITED-TEST': 723, 'FIXED-AUDITED-331': 331, 'INDEPENDENT-BLIND-331': 331, 'FIXED-AUDITED-88': 88, 'INDEPENDENT-BLIND-S0S3-88': 88, 'FIXED-AUDITED-419': 419, 'INDEPENDENT-BLIND-419': 419}
    sets = {n: load_labelset(labelsets_root, n) for n in BASE_SETS}
    for n, k in card.items():
        if len(sets[n]['stems']) != k:
            raise SystemExit(f'{n}: expected {k} frames, found {len(sets[n]["stems"])}')
    check_relations(sets)
    return sets


def verify_reference(reg, A=None):
    """The 24 stage-c reference values (3 models x 8 sets of the sealed unblinding package) are extracted from the
    verified package itself, compared with the registry copy, and - when A is given - all 24 must be present in A and
    equal within 1e-9; a missing item aborts (no silent skip)."""
    rp = reg['reference_package']; pkg = Path(rp['path'])
    if sha(pkg / 'results_A.json') != rp['results_A_sha256'] or sha(pkg / 'OUTPUT_SHA256SUMS') != rp['output_sums_sha256']:
        raise SystemExit('reference package digests differ from the registry')
    sealed = {k: v for k, v in json.loads((pkg / 'results_A.json').read_text()).items() if k.startswith('E17_c_')}
    if len(sealed) != 24 or sealed != rp['stage_c_values']:
        raise SystemExit(f'reference values: expected exactly 24 sealed stage-c items equal to the registry copy, found {len(sealed)}')
    if A is None:
        return sealed
    X = {}
    for k, v in sealed.items():
        tag, _, n = k.split('|'); here = A.get(f'{tag}|{n}')
        if here is None:
            raise SystemExit(f'reference job missing in this run: {tag} on {n}')
        X[k] = dict(sealed=v['map5095'], here=here['map5095'], abs_diff=abs(v['map5095'] - here['map5095']))
    if len(X) != 24:
        raise SystemExit('reference cross-check incomplete')
    bad = [k for k, x in X.items() if x['abs_diff'] > 1e-9]
    if bad:
        raise SystemExit(f'reproducibility cross-check failed for {bad}')
    return X


def mean_sd(v):
    return dict(values=v, mean=float(statistics.mean(v)), sd_ddof1=float(statistics.stdev(v)) if len(v) > 1 else None, n=len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=Path, default=HERE / 'BUDGET_BLIND_MODELS.json'); ap.add_argument('--labelsets-root', type=Path, required=True)
    ap.add_argument('--images', type=Path, default=Path('/mnt/ssd/datasets/custom_dataset/image3')); ap.add_argument('--out-root', type=Path, default=Path('/mnt/ssd/snow_review_runs/BUDGET_BLIND'))
    ap.add_argument('--image-size', nargs=2, type=int, default=[2304, 1536]); ap.add_argument('--freeze', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(); W, H = a.image_size; reg = json.loads(a.models.read_text())
    for m in reg['models']:
        if not (isinstance(m.get('sha256'), str) and len(m['sha256']) == 64):
            raise SystemExit(f"{m['tag']}: registry must carry the full sha256")
    if a.freeze:
        check_grid_and_sets(reg, a.labelsets_root); verify_reference(reg)
        items = freeze_inputs(a, reg, W, H); FREEZE_FILE.write_text(json.dumps(dict(frozen_at=time.strftime('%Y-%m-%dT%H%M%S%z'), host=platform.node(), items=items), indent=1)); say('FROZEN', FREEZE_FILE); return
    if not FREEZE_FILE.exists():
        raise SystemExit('BUDGET_BLIND_FREEZE.json missing: run --freeze first (after the auditor approved the script)')
    frozen = json.loads(FREEZE_FILE.read_text())['items']; now = freeze_inputs(a, reg, W, H)
    diff = [k for k in set(frozen) | set(now) if frozen.get(k) != now.get(k)]
    if diff:
        raise SystemExit(f'frozen inputs changed since --freeze: {sorted(diff)}')
    sets = check_grid_and_sets(reg, a.labelsets_root); verify_reference(reg)
    am1 = reg['amendment1']; rows = {r['anonymous_id']: r for r in csv.DictReader(Path(am1['exclude_manifest']).open(newline='', encoding='utf-8'))}
    stems_excl = {Path(rows[i]['image_name']).stem for i in am1['exclude_ids']}
    ts = time.strftime('%Y-%m-%dT%H%M%S%z'); run_dir = a.out_root / ts; run_dir.mkdir(parents=True, exist_ok=False)
    derived = {}
    for n in am1['exclude_sets']:
        if not stems_excl <= sets[n]['stems']:
            raise SystemExit(f'excluded frame not in {n}')
        d = derive_excluded(sets[n], n, stems_excl, run_dir); derived[d['man']['label_set_id']] = d
    ds_ = [d['stems'] for d in derived.values()]
    if len(derived) != 2 or any(len(s) != am1['expected_n_after'] for s in ds_) or ds_[0] != ds_[1]:
        raise SystemExit('Amendment 1 derived sets are not both of the expected size with identical frames')
    sets.update(derived); pairs = list(PAIRS) + [(f'{f}-EXCL{len(stems_excl)}', f'{b}-EXCL{len(stems_excl)}') for f, b in PAIRS[1:2]]
    models = sorted(reg['models'], key=lambda m: (0 if m['tag'].startswith('E17_c_') else 1, m['tag']))  # reference models first
    jobs = [(m['tag'], n) for m in models for n in sets]  # 30 models x 9 sets = 270 val-native jobs
    manifest = dict(script_version=1, registry=str(a.models), freeze=frozen, amendment1=dict(exclude_ids=am1['exclude_ids'], derived_sets=sorted(derived)), models=models,
                    labelsets={n: dict(n_files=v['man']['n_files'], n_boxes=v['man']['n_boxes'], aggregate_sha256=v['man']['aggregate_sha256']) for n, v in sets.items()},
                    val_args=VAL_ARGS, descriptive_rules=dict(within=WITHIN, sd_flag=SD_FLAG, e22_within=E22_WITHIN, note='descriptive only; the pre-registered decisions were taken on FIXED-AUDITED-TEST; no confidence intervals are claimed for cross-label differences'),
                    started_at=ts, n_jobs=len(jobs), args={k: str(v) for k, v in vars(a).items()}, environment=f'host={platform.node()} python={sys.version.split()[0]}')
    if a.dry_run:
        (run_dir / 'dryrun.json').write_text(json.dumps(dict(manifest, jobs=jobs), indent=1)); say('DRY-RUN OK', run_dir, 'jobs', len(jobs)); return
    lib_log = (run_dir / 'library_output.log').open('a')
    with contextlib.redirect_stdout(lib_log), contextlib.redirect_stderr(lib_log):
        import torch, ultralytics
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER
        for h in list(LOGGER.handlers):
            LOGGER.removeHandler(h)
        LOGGER.addHandler(logging.StreamHandler(lib_log)); LOGGER.propagate = False
        manifest['environment'] += f' torch={torch.__version__} ultralytics={ultralytics.__version__} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}'
        (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=1))
        # ---- Part A
        A = {}; datasets = {n: build_dataset(ls, a.images, run_dir, n, coco=False) for n, ls in sets.items()}; X = None
        for m in models:
            for n in sets:
                ds, stems = datasets[n]; yaml = ds / 'data.yaml'; yaml.write_text(f'path: {ds}\ntrain: images/val\nval: images/val\nnames:\n  0: car\n')
                key = f"{m['tag']}|{n}"; say('A', key)
                r = YOLO(m['path']).val(data=str(yaml), split='val', save_json=True, plots=False, project=str(run_dir / 'val'), name=key.replace('|', '__'), exist_ok=True, verbose=False, **VAL_ARGS)
                A[key] = dict(map50=float(r.box.map50), map5095=float(r.box.map), precision=float(r.box.mp), recall=float(r.box.mr), n_images=len(stems))
                (run_dir / 'results_A.json').write_text(json.dumps(A, indent=1))
            if X is None and all(f'E17_c_s{s}|{n}' in A for s in (0, 1, 2) for n in sets):
                # ---- Part X: reproducibility cross-check right after the three reference models (all 24 sealed items required)
                X = verify_reference(reg, A); (run_dir / 'results_X.json').write_text(json.dumps(X, indent=1)); say('X reference cross-check passed (24/24)')
        if X is None:
            raise SystemExit('reference cross-check never ran')
        # ---- Part S: descriptive summaries (numbers only, no CI claims)
        def g(tag, n, k='map5095'):
            return A[f'{tag}|{n}'][k]
        S = dict(E21={}, E22={}, cross_label_point_differences={}, reference_full_data={})
        for n in sets:
            S['reference_full_data'][n] = {k: mean_sd([g(f'E17_c_s{s}', n, k) for s in (0, 1, 2)]) for k in ('map5095', 'map50')}
        for n in sets:
            per = {}
            for b in BUDGETS:
                per[b] = {k: mean_sd([g(f'E21_{b}_s{s}', n, k) for s in (0, 1, 2)]) for k in ('map5095', 'map50')}
                per[b]['sd_flag_0_03'] = per[b]['map5095']['sd_ddof1'] > SD_FLAG
            ref_mean = S['reference_full_data'][n]['map5095']['mean']; first = next((b for b in BUDGETS if per[b]['map5095']['mean'] >= ref_mean - WITHIN), None)
            S['E21'][n] = dict(per_budget=per, same_set_reference_mean=ref_mean, first_budget_within_0_02_of_same_set_reference=first, note='descriptive; the pre-registered 16-day criterion was decided on FIXED-AUDITED-TEST')
            cfg = {}
            for c in CONFIGS:
                cfg[c] = {k: mean_sd([g(f'E22_{c}_s{s}', n, k) for s in (0, 1, 2)]) for k in ('map5095', 'map50')}
                cfg[c]['within_0_10_of_same_set_reference'] = cfg[c]['map5095']['mean'] >= ref_mean - E22_WITHIN
            S['E22'][n] = dict(per_config=cfg, same_set_reference_mean=ref_mean, note='descriptive; the frozen E22 rule was decided on FIXED-AUDITED-TEST')
        for f, b in pairs:
            d = {}
            for m in models:
                d[m['tag']] = dict(map5095=g(m['tag'], f) - g(m['tag'], b), map50=g(m['tag'], f, 'map50') - g(m['tag'], b, 'map50'))
            grp = {}
            for bud in BUDGETS:
                grp[f'E21_{bud}'] = mean_sd([d[f'E21_{bud}_s{s}']['map5095'] for s in (0, 1, 2)])
            for c in CONFIGS:
                grp[f'E22_{c}'] = mean_sd([d[f'E22_{c}_s{s}']['map5095'] for s in (0, 1, 2)])
            grp['E17_c'] = mean_sd([d[f'E17_c_s{s}']['map5095'] for s in (0, 1, 2)])
            S['cross_label_point_differences'][f'{f}|{b}'] = dict(per_model=d, per_group_seed_spread=grp, note='point differences FIXED minus BLIND on identical frames with the seed spread; not confidence intervals')
        (run_dir / 'results_S.json').write_text(json.dumps(S, indent=1))
        manifest['finished_at'] = time.strftime('%Y-%m-%dT%H%M%S%z'); (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=1))
    lib_log.close()
    files = sorted(p for p in run_dir.rglob('*') if p.is_file() and not p.is_symlink() and p.name != 'OUTPUT_SHA256SUMS')
    sums = ''.join(f'{sha(p)}  {p.relative_to(run_dir)}\n' for p in files); (run_dir / 'OUTPUT_SHA256SUMS').write_text(sums)
    say('DONE', run_dir, len(files), 'files', hashlib.sha256(sums.encode()).hexdigest())


if __name__ == '__main__':
    main()
