#!/usr/bin/env python3
"""E23: blind test-retest self-consistency of the single annotator (spec v2 section 5.5). Version 4.2.

Compares, on the 55 frames of INDEPENDENT-BLIND-RETEST-55, the retest boxes with the sealed first-pass boxes of
INDEPENDENT-BLIND-331 for the same images. Execution gates (all must pass before any box is compared):
  * E23_ANALYSIS_DEPS.json digests: spec, first-pass manifest + seal list + seal meta, retest manifest + meta;
  * both cohorts sealed: every file digest equals its seal list, the seal meta's aggregate equals the SHA256
    over the concatenated digests in ascending filename order, the sealed file set equals the manifest AND the
    JSON files present in the directory (no unregistered file; the only exception is the legacy first-pass
    directory, whose progress-gate report is tolerated by name AND digest recorded in the deps file), the meta's `files` map (seal v3) equals the list,
    its bound protocol / manifest digests equal the frozen values, and its off-machine receipt is verified AND
    its snapshot manifest carries every sealed file with the sealed digest; the
    first-pass seal (v2 format) is accepted through its frozen aggregate digest only;
  * every payload: complete, imagePath = manifest image, payload protocol / manifest digests = the bound values,
    image digest = manifest, width/height = the actual image, tz-aware completed_at >= started_at, payload
    eligible_from = manifest (retest), rectangles valid (0 <= x1 < x2 <= W, 0 <= y1 < y2 <= H), no duplicates;
  * eligibility: manifest eligible_from == first-pass completed_at + 14 days, retest started_at >= eligible_from.
Matching is the rule of the unblinding part D, copied verbatim (greedy, sorted((iou, i, j), reverse=True),
IoU >= 0.5). Reported (spec 5.5): matching F1 = 2M / (N_first + N_retest), both unmatched counts, matched-box
IoU (conditional on matching), signed and absolute per-frame count differences with the net change, the
first-pass partially-buried subgroup (conditional; the first pass is not a gold standard), frozen geometry of
matched pairs (retest/first width, height and area ratios; centre offset normalised by the first box), the
completion order (by completed_at; not a browsing trajectory) versus manifest order and the per-frame interval
since the first pass; frame-cluster
bootstrap 95% CIs (2,000, seed 0) with separate valid-replicate counts for F1 and IoU. Descriptive: no threshold
is asserted. The package (results, machine-readable pairs, per-frame rows) is written to a new run directory
with OUTPUT_SHA256SUMS over every file. --self-check runs the code on synthetic data in memory only.
"""
import argparse, csv, hashlib, json, platform, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

MATCH_IOU = 0.5; BOOT = 2000; SEED = 0; ELIGIBILITY_DAYS = 14
HERE = Path(__file__).resolve().parent; DEPS = HERE / 'E23_ANALYSIS_DEPS.json'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def read_seal(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d, n = line.split(None, 1); out[n.strip()] = d
    return out


def check_seal(out_dir, seal_list, seal_meta, expected_names, protocol_sha=None, manifest_sha=None, expected_aggregate=None, root=Path('.'), legacy_extra=None):
    seal = read_seal(seal_list); meta = json.loads(Path(seal_meta).read_text())
    if set(seal) != set(expected_names):
        raise SystemExit(f'{seal_list}: sealed file set differs from the manifest ({len(seal)} vs {len(expected_names)})')
    present = {p.name for p in Path(out_dir).glob('*.json')}
    if legacy_extra and expected_aggregate is not None:  # legacy cohort only: a named extra file with a bound digest
        n = legacy_extra['name']
        if n in present and (Path(out_dir) / n).exists() and sha(Path(out_dir) / n) == legacy_extra['sha256']:
            present.discard(n)
    if present != set(seal):
        raise SystemExit(f'{out_dir}: JSON files present differ from the sealed set (unregistered or missing: {sorted(present ^ set(seal))[:5]})')
    agg = hashlib.sha256(''.join(seal[n] for n in sorted(seal)).encode()).hexdigest()
    if agg != meta['seal_sha256_of_file_digests'] or meta.get('n_files', meta.get('frame_count')) != len(seal):
        raise SystemExit(f'{seal_meta}: aggregate digest or file count differs from the seal list')
    if expected_aggregate is not None and agg != expected_aggregate:
        raise SystemExit(f'{seal_meta}: aggregate differs from the frozen expected value')
    if 'files' in meta:  # seal v3 (seal_cohort.py): per-file map, bound hashes, off-machine receipt
        if dict(meta['files']) != seal:
            raise SystemExit(f'{seal_meta}: files map differs from the seal list')
        bh = meta.get('bound_hashes') or {}
        if protocol_sha is not None and bh.get('protocol') != protocol_sha:
            raise SystemExit(f'{seal_meta}: bound protocol digest differs')
        if manifest_sha is not None and manifest_sha not in (bh.get('manifest'), bh.get('row_manifest_sha256')):
            raise SystemExit(f'{seal_meta}: bound manifest digest differs')
        rc = meta.get('offmachine_receipt')
        if not rc:
            raise SystemExit(f'{seal_meta}: no off-machine receipt bound')
        rc_path = Path(root) / rc
        receipt = json.loads(rc_path.read_text())
        if receipt.get('status') != 'verified' or not receipt.get('restore_verified'):
            raise SystemExit(f'{seal_meta}: off-machine receipt not verified')
        # the receipt must prove that THIS cohort's sealed files were snapshotted with these digests
        mf = receipt.get('manifest_file')
        if not mf:
            raise SystemExit(f'{rc}: receipt names no snapshot manifest')
        mf_path = rc_path.parent / mf
        snap_manifest = json.loads(mf_path.read_text())
        # annotation_backup.py records the canonical digest of the manifest OBJECT, not of the file bytes
        canonical = hashlib.sha256(json.dumps(snap_manifest, sort_keys=True).encode('utf-8')).hexdigest()
        if receipt.get('manifest_sha256') and canonical != receipt['manifest_sha256']:
            raise SystemExit(f'{mf}: snapshot manifest differs from the digest recorded in the receipt')
        snap = {e['path']: e['sha256'] for e in snap_manifest['files']}
        prefix = Path(out_dir).resolve().relative_to(Path(root).resolve()).as_posix()
        missing = [n for n in seal if snap.get(f'{prefix}/{n}') != seal[n]]
        if missing:
            raise SystemExit(f'{rc}: snapshot manifest misses or contradicts {len(missing)} sealed file(s) of {prefix} (e.g. {missing[:3]})')
    elif expected_aggregate is None:
        raise SystemExit(f'{seal_meta}: legacy seal format without a frozen expected aggregate')
    bad = [n for n, d in seal.items() if not (Path(out_dir) / n).exists() or sha(Path(out_dir) / n) != d]
    if bad:
        raise SystemExit(f'{len(bad)} sealed outputs of {out_dir} differ from the seal list')
    return seal, meta


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def check_payload(p, row, image_wh, protocol_sha, manifest_sha, name):
    ra = p.get('revision_annotation') or {}
    if not ra.get('complete') or not ra.get('completed_at') or not ra.get('started_at'):
        raise SystemExit(f'{name}: not complete or timestamps missing')
    st, ct = datetime.fromisoformat(ra['started_at']), datetime.fromisoformat(ra['completed_at'])
    if st.tzinfo is None or ct.tzinfo is None or ct < st:
        raise SystemExit(f'{name}: timestamps naive or completed before started')
    if row.get('eligible_from') and ra.get('eligible_from') != row['eligible_from']:
        raise SystemExit(f'{name}: payload eligible_from differs from the manifest')
    if p.get('imagePath') != row['image_name'] or ra.get('anonymous_id') != row['anonymous_id'] or ra.get('label_set_id') != row['label_set_id']:
        raise SystemExit(f'{name}: identity differs from the manifest')
    if ra.get('protocol_sha256') != protocol_sha or ra.get('manifest_sha256') != manifest_sha or ra.get('image_sha256') != row['image_sha256']:
        raise SystemExit(f'{name}: protocol / manifest / image digest differs from the bound values')
    W, H = int(p.get('imageWidth', 0)), int(p.get('imageHeight', 0))
    if W <= 0 or H <= 0 or (image_wh is not None and (W, H) != tuple(image_wh)):
        raise SystemExit(f'{name}: image size differs from the actual image')
    seen = set(); boxes = []; buried = []
    for s in p.get('shapes', []):
        if s.get('label') != 'car' or s.get('shape_type') != 'rectangle' or len(s.get('points') or []) != 2:
            raise SystemExit(f'{name}: unsupported shape')
        (x1, y1), (x2, y2) = [tuple(map(float, q)) for q in s['points']]
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            raise SystemExit(f'{name}: rectangle out of bounds or degenerate')
        key = tuple(round(v, 2) for v in (x1, y1, x2, y2))
        if key in seen:
            raise SystemExit(f'{name}: duplicate rectangle')
        seen.add(key); boxes.append([x1, y1, x2, y2]); buried.append(bool((s.get('flags') or {}).get('partially_buried')))
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(buried, dtype=bool), ra


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-9, None)


def greedy_match(a, b, thr=MATCH_IOU):
    """Unblinding part D rule, verbatim: candidates sorted by (iou, i, j) descending, each box used once."""
    M = iou_matrix(a, b); matched_b = set(); used_a = set(); pairs = []
    for v, i, j in sorted(((M[i, j], i, j) for i in range(len(a)) for j in range(len(b)) if M[i, j] >= thr), reverse=True):
        if i in used_a or j in matched_b:
            continue
        used_a.add(i); matched_b.add(j); pairs.append((int(i), int(j), float(v)))
    return pairs


def pair_geometry(fa, rb):
    wf, hf = fa[2] - fa[0], fa[3] - fa[1]; wr, hr = rb[2] - rb[0], rb[3] - rb[1]
    return dict(width_ratio=wr / wf, height_ratio=hr / hf, area_ratio=(wr * hr) / (wf * hf),
                dx_norm=((rb[0] + rb[2]) - (fa[0] + fa[2])) / 2 / wf, dy_norm=((rb[1] + rb[3]) - (fa[1] + fa[3])) / 2 / hf)


def per_frame_stats(first, retest, extra=None):
    """first/retest: dict image -> (boxes, buried[, meta]). Rows in sorted image order; pair details kept."""
    rows = []
    for name in sorted(first):
        a, ba = first[name][:2]; b, _ = retest[name][:2]; pairs = greedy_match(a, b); mi = {i for i, _, _ in pairs}
        geo = [pair_geometry(a[i], b[j]) for i, j, _ in pairs]
        rows.append(dict(image=name, n_first=int(len(a)), n_retest=int(len(b)), matched=len(pairs), signed_diff=int(len(b) - len(a)),
                         ious=[v for _, _, v in pairs], buried_first=int(ba.sum()), buried_matched=int(sum(1 for i in mi if ba[i])),
                         buried_ious=[v for i, _, v in pairs if ba[i]], geometry=geo,
                         pairs=[dict(first_index=i, retest_index=j, iou=v, first_box=a[i].tolist(), retest_box=b[j].tolist(), first_buried=bool(ba[i])) for i, j, v in pairs],
                         unmatched_first=[a[i].tolist() for i in range(len(a)) if i not in mi], unmatched_retest=[b[j].tolist() for j in range(len(b)) if j not in {j for _, j, _ in pairs}],
                         **(extra.get(name, {}) if extra else {})))
    return rows


def summarize(rows):
    nf = sum(r['n_first'] for r in rows); nr = sum(r['n_retest'] for r in rows); m = sum(r['matched'] for r in rows)
    ious = np.array([v for r in rows for v in r['ious']]); bi = np.array([v for r in rows for v in r['buried_ious']])
    bf = sum(r['buried_first'] for r in rows); bm = sum(r['buried_matched'] for r in rows)
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else None
    geo = {k: [g[k] for r in rows for g in r['geometry']] for k in ('width_ratio', 'height_ratio', 'area_ratio', 'dx_norm', 'dy_norm')}
    sd = [r['signed_diff'] for r in rows]
    return dict(frames=len(rows), boxes_first=nf, boxes_retest=nr, matched=m,
                matching_f1=(2 * m / (nf + nr)) if nf + nr else None, recall_of_first=(m / nf) if nf else None, precision_of_retest=(m / nr) if nr else None,
                unmatched_first=nf - m, unmatched_retest=nr - m,
                matched_iou=dict(note='conditional on the matched pairs', n=int(len(ious)), median=q(ious, .5), mean=float(ious.mean()) if len(ious) else None, q10=q(ious, .1), q25=q(ious, .25), q75=q(ious, .75),
                                 frac_ge_075=float((ious >= .75).mean()) if len(ious) else None, frac_ge_090=float((ious >= .9).mean()) if len(ious) else None),
                count_difference=dict(net_change=int(sum(sd)), mean_signed=float(np.mean(sd)), mean_abs=float(np.mean(np.abs(sd))), frames_equal=int(sum(1 for d in sd if d == 0)),
                                      frames_retest_more=int(sum(1 for d in sd if d > 0)), frames_retest_fewer=int(sum(1 for d in sd if d < 0)), per_frame_signed=sd),
                partially_buried_first=dict(note='first-pass flag; conditional, the first pass is not a gold standard', n=bf, matched=bm, matched_fraction=(bm / bf) if bf else None,
                                            matched_iou_median=q(bi, .5), matched_iou_mean=float(bi.mean()) if len(bi) else None),
                matched_geometry={k: dict(median=q(np.array(v), .5), q25=q(np.array(v), .25), q75=q(np.array(v), .75), mean=float(np.mean(v)) if v else None) for k, v in geo.items()})


def bootstrap(rows, reps=BOOT, seed=SEED):
    rng = np.random.default_rng(seed); n = len(rows); f1 = []; med = []; mean = []; f1_undef = 0; iou_undef = 0
    for _ in range(reps):
        sub = [rows[i] for i in rng.integers(0, n, n)]
        nf = sum(r['n_first'] for r in sub); nr = sum(r['n_retest'] for r in sub); m = sum(r['matched'] for r in sub)
        if nf + nr == 0:
            f1_undef += 1
        else:
            f1.append(2 * m / (nf + nr))
        ious = [v for r in sub for v in r['ious']]
        if ious:
            med.append(float(np.median(ious))); mean.append(float(np.mean(ious)))
        else:
            iou_undef += 1
    ci = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))] if x else None
    return dict(replicates=reps, seed=seed, scheme='frames resampled with replacement; F1 undefined only when both cohorts have zero boxes in the replicate; IoU undefined when no pair matched',
                f1_valid=len(f1), f1_undefined=f1_undef, iou_valid=len(med), iou_undefined=iou_undef,
                matching_f1_ci95=ci(f1), matched_iou_median_ci95=ci(med), matched_iou_mean_ci95=ci(mean))


def load_cohort(out_dir, seal_list, seal_meta, manifest, protocol_sha, manifest_sha, need_eligible=False, expected_aggregate=None, subset=None, legacy_extra=None):
    with open(manifest, newline='') as f:
        rows = list(csv.DictReader(f))
    names = [f"{r['anonymous_id']}.json" for r in rows]
    seal, meta = check_seal(out_dir, seal_list, seal_meta, names, protocol_sha, manifest_sha, expected_aggregate, legacy_extra=legacy_extra)
    data = {}
    for r, n in zip(rows, names):
        if subset is not None and r['image_name'] not in subset:
            continue
        p = json.loads((out_dir / n).read_text())
        boxes, buried, ra = check_payload(p, r, image_size(r['image_path']), protocol_sha, manifest_sha, n)
        if need_eligible and not r.get('eligible_from'):
            raise SystemExit(f'{n}: manifest row has no eligible_from')
        data[r['image_name']] = (boxes, buried, dict(anonymous_id=r['anonymous_id'], sequence=int(r['sequence']), started_at=ra['started_at'], completed_at=ra['completed_at'],
                                                    eligible_from=r.get('eligible_from'), active_seconds=ra.get('active_seconds')))
    return data, meta


def check_eligibility(first, retest):
    extra = {}; order = []
    for name, (_, _, rm) in retest.items():
        fc = datetime.fromisoformat(first[name][2]['completed_at']); ef = datetime.fromisoformat(rm['eligible_from']); rs = datetime.fromisoformat(rm['started_at'])
        if ef != fc + timedelta(days=ELIGIBILITY_DAYS):
            raise SystemExit(f"{rm['anonymous_id']}: eligible_from differs from first-pass completion + {ELIGIBILITY_DAYS} days")
        if rs < ef:
            raise SystemExit(f"{rm['anonymous_id']}: retest started before its eligible instant")
        extra[name] = dict(retest_id=rm['anonymous_id'], manifest_sequence=rm['sequence'], retest_started_at=rm['started_at'], retest_completed_at=rm['completed_at'],
                           interval_days=(rs - fc).total_seconds() / 86400.0)
        order.append((rm['completed_at'], rm['sequence']))
    order.sort(); actual = [s for _, s in order]
    inversions = sum(1 for i in range(len(actual)) for j in range(i + 1, len(actual)) if actual[i] > actual[j])
    ivals = [e['interval_days'] for e in extra.values()]
    return extra, dict(actual_completion_order_by_manifest_sequence=actual, is_manifest_order=actual == sorted(actual), pairwise_inversions=inversions,
                       interval_days=dict(min=min(ivals), median=float(np.median(ivals)), max=max(ivals)),
                       note='frames were presented in manifest order restricted to eligibility (blocked frames skipped and completed later); not an unconstrained random permutation')


def self_check():
    rng = np.random.default_rng(1); first = {}
    for k in range(5):
        n = int(rng.integers(3, 9)); x = rng.uniform(0, 1800, n); y = rng.uniform(0, 1200, n); w = rng.uniform(80, 200, n); h = rng.uniform(60, 150, n)
        first[f'f{k}'] = (np.stack([x, y, x + w, y + h], 1), rng.random(n) < .3)
    s = summarize(per_frame_stats(first, first)); assert s['matching_f1'] == 1.0 and s['matched_iou']['median'] == 1.0 and s['count_difference']['net_change'] == 0, s
    jit = {k: (b + np.array([0.1, 0, 0.1, 0]) * (b[:, 2:3] - b[:, 0:1]), f) for k, (b, f) in first.items()}
    s2 = summarize(per_frame_stats(first, jit)); assert s2['matching_f1'] == 1.0 and abs(s2['matched_iou']['median'] - 0.9 / 1.1) < 1e-9 and abs(s2['matched_geometry']['dx_norm']['median'] - 0.1) < 1e-9, s2
    drop = {k: (b[:-1], f[:-1]) for k, (b, f) in first.items()}
    s3 = summarize(per_frame_stats(first, drop)); nf = s3['boxes_first']; assert s3['matched'] == nf - 5 and abs(s3['matching_f1'] - 2 * (nf - 5) / (2 * nf - 5)) < 1e-12 and s3['count_difference']['net_change'] == -5, s3
    # tie regression (auditor's counter-example): part D rule must yield 2 pairs
    a = np.array([[0, 0, 1, 1], [0, 0, 3, 1]], float); b = np.array([[0, 0, 2, 1], [1, 0, 3, 1]], float)
    assert len(greedy_match(a, b)) == 2, greedy_match(a, b)
    # bootstrap: one frame, one box each, no match -> F1 = 0 valid in every replicate, IoU undefined
    rows = per_frame_stats({'x': (np.array([[0, 0, 10, 10.]]), np.array([False]))}, {'x': (np.array([[100, 100, 110, 110.]]), np.array([False]))})
    bs = bootstrap(rows, reps=20); assert bs['f1_valid'] == 20 and bs['matching_f1_ci95'] == [0.0, 0.0] and bs['iou_valid'] == 0 and bs['iou_undefined'] == 20, bs
    # eligibility gate: early start rejected; wrong offset rejected
    fr = {'x': (a, np.array([False, False]), dict(completed_at='2026-08-20T10:00:00+09:00'))}
    ok = {'x': (b, np.array([False, False]), dict(anonymous_id='R0001', sequence=1, started_at='2026-09-03T10:00:00+09:00', completed_at='2026-09-03T10:05:00+09:00', eligible_from='2026-09-03T10:00:00+09:00'))}
    extra, order = check_eligibility(fr, ok); assert abs(extra['x']['interval_days'] - 14.0) < 1e-9 and order['is_manifest_order']
    for bad in (dict(ok['x'][2], started_at='2026-09-03T09:59:59+09:00'), dict(ok['x'][2], eligible_from='2026-09-04T10:00:00+09:00')):
        try:
            check_eligibility(fr, {'x': (b, np.array([False, False]), bad)}); raise AssertionError('eligibility gate did not fire')
        except SystemExit:
            pass
    # payload gate: wrong protocol digest / out-of-bounds rectangle / incomplete / size mismatch / time order / eligible_from mismatch rejected
    row = dict(image_name='i.png', anonymous_id='R0001', label_set_id='L', image_sha256='s', eligible_from='2026-09-03T10:00:00+09:00')
    good = dict(imagePath='i.png', imageWidth=10, imageHeight=10, shapes=[dict(label='car', shape_type='rectangle', points=[[0, 0], [5, 5]], flags={})],
                revision_annotation=dict(complete=True, started_at='2026-09-03T10:00:00+09:00', completed_at='2026-09-03T10:05:00+09:00', anonymous_id='R0001', label_set_id='L', protocol_sha256='P', manifest_sha256='M', image_sha256='s', eligible_from='2026-09-03T10:00:00+09:00'))
    check_payload(good, row, (10, 10), 'P', 'M', 'ok')
    for mut in (lambda p: p['revision_annotation'].update(protocol_sha256='X'), lambda p: p['shapes'][0].update(points=[[0, 0], [11, 5]]), lambda p: p['revision_annotation'].update(complete=False),
                lambda p: p.update(imageWidth=4000, imageHeight=4000), lambda p: p['revision_annotation'].update(completed_at='2026-09-03T09:59:59+09:00'), lambda p: p['revision_annotation'].update(eligible_from='2026-09-02T10:00:00+09:00')):
        p = json.loads(json.dumps(good)); mut(p)
        try:
            check_payload(p, row, (10, 10), 'P', 'M', 'bad'); raise AssertionError('payload gate did not fire')
        except SystemExit:
            pass
    # seal gate on a temporary fixture: v3 meta with files map, bound hashes and receipt; negatives: unregistered file, files-map mismatch, receipt not verified
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); out = td / 'out'; out.mkdir(); (out / 'R0001.json').write_text('{}'); d = sha(out / 'R0001.json')
        (td / 'seal.sha256').write_text(f'{d}  R0001.json\n'); (td / 'receipts').mkdir()
        rm_obj = dict(files=[dict(path='out/R0001.json', sha256=d)])
        (td / 'receipts' / 'r.manifest.json').write_text(json.dumps(rm_obj, indent=1, sort_keys=True) + '\n')
        canon = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True).encode('utf-8')).hexdigest()
        (td / 'receipts' / 'r.json').write_text(json.dumps(dict(status='verified', restore_verified=True, manifest_file='r.manifest.json', manifest_sha256=canon(rm_obj))))
        agg = hashlib.sha256(d.encode()).hexdigest()
        meta = dict(files={'R0001.json': d}, n_files=1, seal_sha256_of_file_digests=agg, bound_hashes=dict(protocol='P', manifest='M'), offmachine_receipt='receipts/r.json')
        (td / 'meta.json').write_text(json.dumps(meta)); check_seal(out, td / 'seal.sha256', td / 'meta.json', ['R0001.json'], 'P', 'M', root=td)
        negatives = []
        # legacy exemption: only with expected_aggregate AND the bound digest; a shapes-bearing file of the same name is rejected
        (out / 'progress_gate_30.json').write_text('{"report": 1}'); ok_extra = dict(name='progress_gate_30.json', sha256=sha(out / 'progress_gate_30.json'))
        check_seal(out, td / 'seal.sha256', td / 'meta.json', ['R0001.json'], 'P', 'M', expected_aggregate=agg, root=td, legacy_extra=ok_extra)
        for bad_case in (dict(legacy_extra=ok_extra), dict(expected_aggregate=agg, legacy_extra=dict(name='progress_gate_30.json', sha256='0' * 64))):
            try:
                check_seal(out, td / 'seal.sha256', td / 'meta.json', ['R0001.json'], 'P', 'M', root=td, **bad_case); raise AssertionError('extra file accepted without a valid legacy binding')
            except SystemExit:
                pass
        (out / 'progress_gate_30.json').write_text(json.dumps(dict(shapes=[1])))
        try:
            check_seal(out, td / 'seal.sha256', td / 'meta.json', ['R0001.json'], 'P', 'M', expected_aggregate=agg, root=td, legacy_extra=ok_extra); raise AssertionError('shapes-bearing file accepted')
        except SystemExit:
            pass
        (out / 'progress_gate_30.json').unlink()
        (out / 'extra.json').write_text('{}'); negatives.append(lambda: check_seal(out, td / 'seal.sha256', td / 'meta.json', ['R0001.json'], 'P', 'M', root=td)); 
        try:
            negatives[0](); raise AssertionError('unregistered file accepted')
        except SystemExit:
            (out / 'extra.json').unlink()
        # negative: a receipt whose snapshot manifest does not carry this cohort's sealed digest
        bad_obj = dict(files=[dict(path='out/R0001.json', sha256='0' * 64)])
        (td / 'receipts' / 'bad.manifest.json').write_text(json.dumps(bad_obj, indent=1, sort_keys=True) + '\n')
        (td / 'receipts' / 'bad.json').write_text(json.dumps(dict(status='verified', restore_verified=True, manifest_file='bad.manifest.json', manifest_sha256=canon(bad_obj))))
        # negative: manifest object edited after the receipt was written (canonical digest no longer matches)
        (td / 'receipts' / 'tam.manifest.json').write_text(json.dumps(dict(files=[dict(path='out/R0001.json', sha256=d), dict(path='x', sha256='1' * 64)]), indent=1, sort_keys=True) + '\n')
        (td / 'receipts' / 'tam.json').write_text(json.dumps(dict(status='verified', restore_verified=True, manifest_file='tam.manifest.json', manifest_sha256=canon(rm_obj))))
        for m2 in (dict(meta, files={'R0001.json': '0' * 64}), dict(meta, bound_hashes=dict(protocol='X', manifest='M')), dict(meta, offmachine_receipt='receipts/none.json'),
                   dict(meta, offmachine_receipt='receipts/bad.json'), dict(meta, offmachine_receipt='receipts/tam.json')):
            (td / 'meta2.json').write_text(json.dumps(m2))
            try:
                check_seal(out, td / 'seal.sha256', td / 'meta2.json', ['R0001.json'], 'P', 'M', root=td); raise AssertionError('seal gate did not fire')
            except (SystemExit, FileNotFoundError):
                pass
        (td / 'receipts' / 'r.json').write_text(json.dumps(dict(status='failed', restore_verified=False)))
        try:
            check_seal(out, td / 'seal.sha256', td / 'meta.json', ['R0001.json'], 'P', 'M', root=td); raise AssertionError('unverified receipt accepted')
        except SystemExit:
            pass
    print('self-check passed: identity; 10%-width shift (IoU .818, dx_norm .1); dropped boxes; tie regression (2 pairs); zero-match bootstrap (F1 CI [0,0], IoU undefined); eligibility, payload (size/time/eligible_from) and seal (unregistered file, files map, bound hashes, receipt status, snapshot coverage, tampered snapshot manifest, legacy extra-file binding) gates fire')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-check', action='store_true')
    ap.add_argument('--retest-dir', type=Path, default=Path('annotation/outputs/detection_retest_55'))
    ap.add_argument('--retest-seal', type=Path, default=Path('annotation/seals/INDEPENDENT-BLIND-RETEST-55.sha256'))
    ap.add_argument('--retest-seal-meta', type=Path, default=Path('annotation/seals/INDEPENDENT-BLIND-RETEST-55.meta.json'))
    ap.add_argument('--first-dir', type=Path, default=Path('annotation/outputs/detection'))
    a = ap.parse_args()
    if a.self_check:
        self_check(); return
    deps = json.loads(DEPS.read_text())
    for rel, d in deps['items'].items():
        if sha(rel) != d:
            raise SystemExit(f'dependency digest differs: {rel}')
    protocol_sha = deps['items']['snow_parking_annotation_spec_v2.md']
    fm = Path('annotation/manifests/independent_blind_331.csv'); rm = Path('annotation/manifests/independent_blind_retest_55.csv')
    rmeta = json.loads(rm.with_suffix('.meta.json').read_text())
    if sha(rm) != rmeta['manifest_sha256'] or rmeta['source_manifest_sha256'] != deps['items'][str(fm)]:
        raise SystemExit('retest manifest / metadata linkage differs')
    retest, rseal_meta = load_cohort(a.retest_dir, a.retest_seal, a.retest_seal_meta, rm, protocol_sha, rmeta['manifest_sha256'], need_eligible=True)
    first, fseal_meta = load_cohort(a.first_dir, Path('annotation/seals/INDEPENDENT-BLIND-331.sha256'), Path('annotation/seals/INDEPENDENT-BLIND-331.meta.json'), fm, protocol_sha, deps['items'][str(fm)],
                                    expected_aggregate=deps['first_seal_aggregate_expected'], subset=set(retest), legacy_extra=deps.get('legacy_extra_file'))
    if len(retest) != 55 or len(first) != 55:
        raise SystemExit('retest frames are not a 55-frame subset of the sealed first pass')
    extra, order = check_eligibility(first, retest)
    rows = per_frame_stats(first, retest, extra); s = summarize(rows); b = bootstrap(rows)
    run = HERE / 'results' / time.strftime('%Y-%m-%dT%H%M%S%z'); run.mkdir(parents=True)
    out = dict(experiment='E23', version=4.2, script_sha256=sha(__file__), deps_sha256=sha(DEPS), created_at=time.strftime('%Y-%m-%dT%H%M%S%z'), host=platform.node(), numpy=np.__version__,
               inputs=dict(first_seal_meta_note='seal v2 format accepted through its frozen aggregate digest', retest_seal_aggregate=rseal_meta['seal_sha256_of_file_digests'], first_seal_aggregate=fseal_meta['seal_sha256_of_file_digests'], retest_manifest_sha256=rmeta['manifest_sha256']),
               rule=dict(match='greedy, sorted((iou, i, j), reverse=True), IoU >= 0.5 (unblinding part D, verbatim)', f1='2 * matched / (boxes_first + boxes_retest)', eligibility=f'first-pass completed_at + {ELIGIBILITY_DAYS} days <= retest started_at (verified per frame)'),
               summary=s, bootstrap=b, presentation=order,
               per_frame=[{k: v for k, v in r.items() if k not in ('ious', 'buried_ious', 'geometry', 'pairs', 'unmatched_first', 'unmatched_retest')} for r in rows],
               notes=['descriptive self-consistency of one annotator on S1/S2 frames of the 331 cohort; no threshold asserted; not evidence about S0/S3',
                      'first-pass boxes were never displayed during the retest; matched-IoU and geometry are conditional on matching'])
    (run / 'test_retest_results.json').write_text(json.dumps(out, indent=1))
    (run / 'matched_pairs.json').write_text(json.dumps([dict(image=r['image'], retest_id=r['retest_id'], pairs=r['pairs'], unmatched_first=r['unmatched_first'], unmatched_retest=r['unmatched_retest']) for r in rows], indent=0))
    files = sorted(p for p in run.rglob('*') if p.is_file())
    (run / 'OUTPUT_SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(run)}\n' for p in files))
    print('DONE', run, len(files) + 1, 'files')


if __name__ == '__main__':
    main()
