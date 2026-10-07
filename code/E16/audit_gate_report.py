#!/usr/bin/env python3
"""Protocol section 6.4 gate report for INDEPENDENT-BLIND-AUDIT-100 (v2, 2026-09-05).

Prerequisites enforced before any comparison (addendum audit_gate):
  --seal-list  per-file SHA256 list of the sealed batch (every completed output must match),
  --receipt    a backup receipt JSON with status=verified and restore_verified=true that
               post-dates the seal list.
Estimator: missed-box rate = weighted ratio  sum(w_i * miss_i) / sum(w_i * n_i), w_i = 1/pi_cum_i where
pi_cum = pi_total * revealed_in_stratum_at_CURRENT_cut / sampled_in_stratum (nested 300-frame design,
v4 manifests). revealed_in_stratum is counted over ALL manifests passed to this run, so earlier
batches are re-weighted at every cut (the per-row column cum_in_stratum_at_this_batch is NOT used).
Uncertainty: frame-cluster bootstrap resampling within STRATUM with the stratum's frame count fixed
(batches are pre-randomized slices, not sampling strata).
Sparse events (< 5 missed boxes): no bootstrap CI and no sample-size projection; the count is
reported with heuristic upper bounds; the pre-registered rule then is to reveal the next pre-drawn
25-frame batch and re-run this report, up to 300 frames, and to report the final count with its
heuristic bounds if events remain sparse at 300.
Heuristic bounds for sparse events: closed-form k=0 bound 1-(0.05)^(1/n_eff) and rule-of-three 3/n_eff
on n_eff = boxes / design effect (binomial model; coverage under clustering is NOT guaranteed).
All prerequisite checks (seal digests, receipt status, receipt manifest containing every sealed file with
the same digest, receipt after the newest output) run BEFORE any comparison with prelabels.
Missed box = blind box with no PRELABEL-SNAPSHOT box at IoU >= 0.5 (greedy, highest IoU first).
Compares blind outputs with HISTORICAL PRELABELS only; never with final test labels.
"""
import argparse, csv, glob, json, os, random, statistics as st, math, hashlib, datetime, collections

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2-x1)*max(0, y2-y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i; return i/u if u > 0 else 0.0

def rects(j):
    out = []
    for s in j.get('shapes', []):
        if s.get('shape_type', 'rectangle') != 'rectangle': continue
        xs = [p[0] for p in s['points']]; ys = [p[1] for p in s['points']]
        out.append(((min(xs), min(ys), max(xs), max(ys)), bool((s.get('flags') or {}).get('partially_buried'))))
    return out

def matched(blind, pre, t=0.5):
    pairs = sorted(((iou(b, p), i, j) for i, b in enumerate(blind) for j, p in enumerate(pre)), reverse=True); ub, up = set(), set()
    for v, i, j in pairs:
        if v < t: break
        if i in ub or j in up: continue
        ub.add(i); up.add(j)
    return ub

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default='annotation/outputs/detection_audit_100'); ap.add_argument('--manifests', nargs='+', default=['annotation/manifests/independent_blind_audit_100.csv'])
    ap.add_argument('--prelabels', required=True); ap.add_argument('--seal-list', required=True); ap.add_argument('--receipt', required=True); ap.add_argument('--receipt-manifest', help='local copy of the backup manifest (default: receipt[manifest_file] next to the receipt)')
    ap.add_argument('--gate', type=int, default=20); ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out')
    a = ap.parse_args()
    # ---- prerequisites: ALL before any comparison ----
    rc = json.load(open(a.receipt))
    if rc.get('status') != 'verified' or not rc.get('restore_verified'): raise SystemExit('receipt is not a verified, restore-checked backup')
    mpath = a.receipt_manifest or os.path.join(os.path.dirname(a.receipt), rc.get('manifest_file', ''))
    if not os.path.isfile(mpath): raise SystemExit('receipt has no local manifest copy; cannot bind sealed files to the backup')
    bm = json.load(open(mpath)); backed = {e['path']: e['sha256'] for e in bm['files']}
    if hashlib.sha256(json.dumps(bm, sort_keys=True).encode()).hexdigest() != rc.get('manifest_sha256'): raise SystemExit('local manifest copy does not match the receipt manifest_sha256')
    seal = {l.split()[1]: l.split()[0] for l in open(a.seal_list) if l.strip()}
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(a.manifests[0])), '..', '..'))
    pending = []; msize = collections.Counter(); mdone = collections.Counter()
    for bi, mf in enumerate(a.manifests, 1):
        for r in csv.DictReader(open(mf)):
            msize[bi] += 1
            p = os.path.join(a.outputs, r['anonymous_id'] + '.json')
            if not os.path.exists(p): continue
            j = json.load(open(p)); m = j['revision_annotation']
            if not m.get('complete'): continue
            mdone[bi] += 1
            name = os.path.basename(p); rel = os.path.relpath(os.path.abspath(p), root)
            if name not in seal: raise SystemExit(f'{name} is completed but not in the seal list; seal the batch first')
            dg = sha(p)
            if dg != seal[name]: raise SystemExit(f'{name} differs from its sealed digest')
            if backed.get(rel) != dg: raise SystemExit(f'{rel} is not in the verified backup manifest with the sealed digest; back up after sealing')
            pending.append((bi, r, j, m, p))
    if pending and datetime.datetime.fromisoformat(rc['completed_at']) < max(datetime.datetime.fromtimestamp(os.path.getmtime(p)).astimezone() for _, _, _, _, p in pending):
        raise SystemExit('receipt predates the newest sealed output; back up after sealing')
    frames = []
    revealed = collections.Counter(r['stratum'] for _, r, _, _, _ in pending)   # current cut, all included manifests
    for bi, r, j, m, p in pending:
        pre = [b for b, _ in rects(json.load(open(os.path.join(a.prelabels, r['image_name'][:-4] + '.json'))))]
        bl = rects(j); mb = matched([b for b, _ in bl], pre); miss = [i for i in range(len(bl)) if i not in mb]
        pi_cum = float(r['pi_total']) * revealed[r['stratum']] / int(r['sampled_in_stratum'])
        frames.append(dict(id=r['anonymous_id'], batch=bi, stratum=r['stratum'], n=len(bl), buried=sum(f for _, f in bl), miss=len(miss),
                           miss_buried=sum(bl[i][1] for i in miss), active=m['active_seconds'], w=1.0 / pi_cum))
    n = len(frames)
    if n < a.gate: print(f'only {n} completed frames; gate is {a.gate}'); return
    tot = sum(f['n'] for f in frames); miss = sum(f['miss'] for f in frames)
    def wrate(fs): 
        den = sum(f['w'] * f['n'] for f in fs); return sum(f['w'] * f['miss'] for f in fs) / den if den > 0 else 0.0
    est = wrate(frames)
    # ICC / design effect on box-level miss indicator
    groups = [[1]*f['miss'] + [0]*(f['n']-f['miss']) for f in frames if f['n'] > 0]; k = len(groups); m_bar = sum(len(g) for g in groups) / k; grand = miss / tot if tot else 0.0
    msb = sum(len(g) * (sum(g)/len(g) - grand)**2 for g in groups) / max(1, k-1)
    msw = sum(sum((x - sum(g)/len(g))**2 for x in g) for g in groups) / max(1, sum(len(g) for g in groups) - k)
    n0 = (sum(len(g) for g in groups) - sum(len(g)**2 for g in groups)/sum(len(g) for g in groups)) / max(1, k-1)
    icc_estimable = (msb + (n0 - 1) * msw) > 0  # with no miss events MSB = MSW = 0: the ICC is NOT estimable and the values below are fallbacks
    icc = max(0.0, (msb - msw) / (msb + (n0 - 1) * msw)) if icc_estimable else 0.0; deff = 1 + (m_bar - 1) * icc
    # stratified frame-cluster bootstrap: resample within (batch, stratum) cells, counts fixed
    cells = collections.defaultdict(list)
    for f in frames: cells[f['stratum']].append(f)
    rng = random.Random(a.seed); ests = []
    for _ in range(a.boot):
        s = []
        for c in cells.values(): s += [c[rng.randrange(len(c))] for _ in range(len(c))]
        ests.append(wrate(s))
    ests.sort(); lo, hi = ests[int(0.025*a.boot)], ests[int(0.975*a.boot)-1]; hw = (hi - lo) / 2
    sparse = miss < 5
    n_eff = tot / max(1.0, deff)
    n_eff_i = max(1, int(round(n_eff)))
    k0_upper = (1 - 0.05 ** (1.0 / n_eff_i)) if miss == 0 else None   # closed form, binomial model on n_eff; heuristic
    rule_of_three = None if miss > 0 else 3.0 / n_eff_i
    proj = None if sparse else {N: hw * math.sqrt(n / N) for N in range(100, 301, 25)}
    need = None if sparse else next((N for N in sorted(proj) if proj[N] <= 0.0075), None)
    rep = dict(completed_frames=n, manifests_passed=len(a.manifests), batches_started=sum(1 for bi in msize if mdone[bi] > 0), batches_completed=sum(1 for bi in msize if mdone[bi] >= msize[bi]), frames_completed_per_manifest={bi: mdone[bi] for bi in sorted(msize)}, boxes=tot, boxes_per_frame=dict(mean=tot/n, median=st.median(f['n'] for f in frames)),
               active_seconds=dict(mean=st.fmean(f['active'] for f in frames), median=st.median(f['active'] for f in frames)),
               missed_boxes=miss, missed_rate_unweighted=miss/max(1, tot), missed_rate_weighted_ratio=est, missed_frames=sum(1 for f in frames if f['miss']),
               buried_boxes=sum(f['buried'] for f in frames), buried_prevalence_unweighted=sum(f['buried'] for f in frames)/max(1, tot), missed_among_buried=sum(f['miss_buried'] for f in frames),
               icc_frame_cluster=icc, icc_estimable=icc_estimable, icc_note=None if icc_estimable else 'NOT estimable (all miss indicators zero: MSB = MSW = 0); icc/design_effect/effective_boxes are code fallbacks (0 / 1 / n), not measurements; no claim of independent effective samples', mean_cluster_size=m_bar, design_effect=deff, effective_boxes=n_eff,
               sparse_events=sparse, bootstrap_ci95=None if sparse else [lo, hi], ci95_halfwidth_now=None if sparse else hw,
               heuristic_upper_bounds=dict(k0_closed_form_on_n_eff=k0_upper, rule_of_three_on_n_eff=rule_of_three, note='HYPOTHETICAL reference only: binomial model on an effective size under the assumption DEFF = 1; not a cluster-valid confidence bound and not used to judge the 0.75 pp target; coverage under frame clustering is NOT guaranteed'), projected_halfwidth_by_N=proj, frames_needed_for_halfwidth_le_0_75pp=need, target_halfwidth=0.0075,
               bootstrap=dict(replicates=a.boot, seed=a.seed, cells=len(cells), scheme='resample frames within stratum, stratum counts fixed'),
               weights='1/pi_cum, pi_cum = pi_total * revealed_in_stratum_at_current_cut / sampled_in_stratum', seal_list=a.seal_list, receipt=a.receipt,
               note='compares blind boxes with historical prelabels only; not with final test labels; annotator receives aggregates only')
    print(json.dumps(rep, indent=1))
    if a.out: json.dump(rep, open(a.out, 'w'), indent=1)

if __name__ == '__main__': main()
