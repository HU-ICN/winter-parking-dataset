#!/usr/bin/env python3
"""Cross-label (blind vs audited) matched-pair IoU restricted to the 55 test-retest frames.

Registers the comparison statistic quoted next to the test-retest result: the pooled median IoU of the
matched pairs of the sealed unblinding part D (as tabulated per pair in the geometry diagnostic v3, cohort
331) on exactly the 55 frames of INDEPENDENT-BLIND-RETEST-55. Inputs are checked against the expected digests in E23_ANALYSIS_DEPS.json and recorded; the
existing result packages are not modified. Descriptive only.
"""
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
GEO = Path('experiments/E17/unblind_results/geometry_diagnostic/geometry_diagnostic_v3.json'); MAN = Path('annotation/manifests/independent_blind_retest_55.csv')
meta = json.loads(MAN.with_suffix('.meta.json').read_text())
if sha(MAN) != meta['manifest_sha256']:
    raise SystemExit('retest manifest digest differs from its metadata')
stems = {r['image_name'][:-4] for r in csv.DictReader(MAN.open(newline=''))}
if len(stems) != 55:
    raise SystemExit('expected 55 frames')
deps = json.loads(Path('experiments/E23/E23_ANALYSIS_DEPS.json').read_text())
if sha(GEO) != deps['geometry_diagnostic_v3_sha256_expected']:
    raise SystemExit('geometry diagnostic v3 digest differs from the expected value in E23_ANALYSIS_DEPS.json')
geo = json.loads(GEO.read_text()); c = geo['cohorts']['331']
all_pairs = c['matched_pairs']; sub = [p for p in all_pairs if p['stem'] in stems]
frames_in_sub = {p['stem'] for p in sub}
out = dict(created_at=time.strftime('%Y-%m-%dT%H%M%S%z'), script_sha256=sha(__file__),
           inputs=dict(geometry_diagnostic_v3_sha256=sha(GEO), geometry_diagnostic_version=geo['version'], geometry_label_set_inputs=geo['inputs'], retest_manifest_sha256=meta['manifest_sha256']),
           selection='matched pairs of cohort 331 (FIXED-AUDITED-331 vs INDEPENDENT-BLIND-331, unblinding part D rule) whose frame stem is in the RETEST-55 manifest',
           subset_55=dict(frames=len(frames_in_sub), pairs=len(sub), iou_median=float(np.median([p['iou'] for p in sub])), iou_mean=float(np.mean([p['iou'] for p in sub]))),
           all_331=dict(pairs=len(all_pairs), iou_median=float(np.median([p['iou'] for p in all_pairs])), iou_mean=float(np.mean([p['iou'] for p in all_pairs]))),
           note='the two medians are different statistics (55-frame subset vs full 331); both round to 0.84')
dst = Path('experiments/E23/results/cross_label_subset_55.json'); dst.write_text(json.dumps(out, indent=1)); print(json.dumps({k: out[k] for k in ('subset_55', 'all_331')}))
