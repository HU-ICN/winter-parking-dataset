#!/usr/bin/env python3
"""Build INDEPENDENT-BLIND-AUDIT-100 manifest (protocol section 6.1).

v4 (2026-09-05): NESTED design. A single stratified two-stage probability sample of TOTAL frames
(default 300) is drawn once with exact per-frame inclusion probabilities pi_total; within each
(snow level x time-of-day) stratum the sampled frames are put in a random order and the batches
(base 100, then 8 x 25) are proportional per-stratum slices of that order. At any batch cut the
revealed set is a stratified sub-sample of the 300 with inclusion probability
pi_cum = pi_total * (revealed_in_stratum / sampled_in_stratum); no batch is drawn conditionally
on earlier results (fixes the biased merge of conditional extension draws found in audit).
k > D quotas give every day at least one frame (fixes zero-probability days).

v3 (2026-09-05): exact per-frame inclusion probabilities. v2 recorded one probability per stratum although frames were chosen by day rotation (audit counterexample: 1-frame and 9-frame days, 2 picks -> true probabilities 1 and 1/9, not 2/10). v2 (earlier today): day-level stratification added. Within each (snow level, time-of-day)
stratum the allocated frames are spread across capture days by systematic sampling over
the day-sorted list (one frame per day before any day repeats), so every stratum covers
as many days as its allocation allows. Inclusion probabilities per stratum are recorded
for probability-weighted estimation (miss rate must be estimated with these weights,
not the raw sample proportion, because S3 is enriched).

Cohort: 100 frames from the 2,794 model-prelabel frames, stratified by day-level snow
level and time-of-day, with S3 enrichment; frozen seed; randomized order; anonymous IDs
A0001-A0100. Also writes the paired PRELABEL-SNAPSHOT / FINAL-AUDITED counterpart lists.
Pre-registered extension batches of 25 (A0101...) are drawn from the same frozen stream.
"""
import argparse, csv, hashlib, json, os, random, glob, collections, datetime

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-dir', default=os.path.expanduser('~/Downloads/image3'))
    ap.add_argument('--day-index', default='snow_day_index.csv'); ap.add_argument('--out-dir', default='annotation/manifests')
    ap.add_argument('--base', type=int, default=100); ap.add_argument('--extensions', type=int, default=8); ap.add_argument('--batch', type=int, default=25)
    ap.add_argument('--seed', type=int, default=20260905)
    a = ap.parse_args()
    day_level = {r['date']: r['snow_label'] for r in csv.DictReader(open(a.day_index))}
    frames = []
    for p in sorted(glob.glob(os.path.join(a.image_dir, '*.json'))):
        j = json.load(open(p))
        if j.get('description') != 'AUTO_PRELABEL_yolo11s_2026-06-03': continue
        name = os.path.basename(p)[:-5] + '.png'; day = name[6:14]; hh = int(name[14:16])
        tod = 'morning' if hh < 10 else ('midday' if hh < 14 else 'afternoon')
        frames.append(dict(image_name=name, day=day, level=day_level.get(day, 'NA'), tod=tod))
    assert len(frames) == 2794, len(frames)
    rng = random.Random(a.seed)
    def draw(strata_pool, alloc, rng):
        """Two-stage design inside each (level, tod) stratum with EXACT per-frame inclusion probabilities.
          k <= D : select k days by SRS, then 1 frame by SRS within each selected day -> pi = (k/D) * (1/n_d)
          k >  D : every day gets quota >= 1; the remaining k-D by largest remainder of the proportional
                   share of the other frames (capped at n_d); SRS within day -> pi = q_d/n_d (> 0 for all)."""
        chosen, leftover, rec = [], [], {}
        for key in sorted(strata_pool):
            pool = strata_pool[key]; k = min(alloc.get(key, 0), len(pool))
            bydays = collections.defaultdict(list)
            for f in pool: bydays[f['day']].append(f)
            days = sorted(bydays); D = len(days); N = len(pool); picked = []
            if k == 0: pass
            elif k <= D:
                sel = rng.sample(days, k)
                for d in sel:
                    f = dict(rng.choice(bydays[d])); f['incl_prob'] = (k / D) * (1.0 / len(bydays[d])); picked.append(f)
            else:
                q = {d: 1 for d in days}; extra = k - D; rest = {d: len(bydays[d]) - 1 for d in days}; R = sum(rest.values())
                raw = {d: (extra * rest[d] / R if R else 0.0) for d in days}
                for d in days: q[d] += min(int(raw[d]), rest[d])
                rem = k - sum(q.values())
                for d in sorted(days, key=lambda d: (raw[d] - int(raw[d]), d), reverse=True):
                    if rem <= 0: break
                    if q[d] < len(bydays[d]): q[d] += 1; rem -= 1
                for d in days:
                    for f in rng.sample(bydays[d], q[d]):
                        f = dict(f); f['incl_prob'] = q[d] / len(bydays[d]); picked.append(f)
            ids = {f['image_name'] for f in picked}
            chosen += picked; leftover += [f for f in pool if f['image_name'] not in ids]
            rec[str(key)] = dict(stratum_size=N, days=D, sampled=len(picked), design='SRS-days then 1/day' if k <= D else 'per-day quota (>=1) SRS',
                                 min_incl_prob=min((f['incl_prob'] for f in picked), default=None), max_incl_prob=max((f['incl_prob'] for f in picked), default=None))
        return chosen, leftover, rec

    # ---- nested design: one probability sample of TOTAL frames, revealed in batches ----
    total = a.base + a.extensions * a.batch
    strata = collections.defaultdict(list)
    for f in frames: strata[(f['level'], f['tod'])].append(f)
    levels = collections.Counter(f['level'] for f in frames)
    s3_target = max(int(round(0.20 * total)), int(round(total * levels['S3'] / len(frames))))
    s3_keys = [k for k in strata if k[0] == 'S3']; non_s3 = [k for k in strata if k[0] != 'S3']
    s3_avail = sum(len(strata[k]) for k in s3_keys); s3_target = min(s3_target, s3_avail); rest = total - s3_target
    n_non = sum(len(strata[k]) for k in non_s3); n_s3 = s3_avail
    alloc = {k: rest * len(strata[k]) / n_non for k in non_s3}; alloc.update({k: s3_target * len(strata[k]) / n_s3 for k in s3_keys})
    floor = {k: min(int(v), len(strata[k])) for k, v in alloc.items()}
    for k in strata:
        if floor.get(k, 0) == 0 and strata[k]: floor[k] = 1
    rem = total - sum(floor.values())
    for k in sorted(alloc, key=lambda k: alloc[k] - floor[k], reverse=True):
        if rem <= 0: break
        if floor[k] < len(strata[k]): floor[k] += 1; rem -= 1
    while sum(floor.values()) > total:
        kmax = max((k for k in floor if floor[k] > 1), key=lambda k: floor[k] - alloc.get(k, 0)); floor[kmax] -= 1
    sample, _leftover, incl = draw(strata, floor, rng)
    # random order within each stratum, then proportional per-stratum slices for base and extensions
    by_stratum = collections.defaultdict(list)
    for f in sample: by_stratum[(f['level'], f['tod'])].append(f)
    for k in by_stratum: rng.shuffle(by_stratum[k])
    sizes = [a.base] + [a.batch] * a.extensions
    keys = sorted(by_stratum); remaining = {k: len(by_stratum[k]) for k in keys}; cuts = {k: [] for k in keys}
    for sz in sizes:  # exact batch sizes: per-stratum proportional share of what remains, largest remainder to hit sz
        R = sum(remaining.values()); shares = {k: (sz * remaining[k] / R if R else 0.0) for k in keys}
        fl = {k: min(int(shares[k]), remaining[k]) for k in keys}
        if sz == sizes[0]:  # base batch: every stratum with frames is represented at least once
            for k in keys:
                if fl[k] == 0 and remaining[k] > 0: fl[k] = 1
        r_ = sz - sum(fl.values())
        while r_ < 0:
            kmax = max((k for k in keys if fl[k] > 1), key=lambda k: fl[k] - shares[k]); fl[kmax] -= 1; r_ += 1
        for k in sorted(keys, key=lambda k: shares[k] - int(shares[k]), reverse=True):
            if r_ <= 0: break
            if fl[k] < remaining[k]: fl[k] += 1; r_ -= 1
        for k in keys: cuts[k].append(fl[k]); remaining[k] -= fl[k]
    batches = [[] for _ in sizes]; pos = collections.Counter()
    for k, lst in by_stratum.items():
        o = 0
        for i, n_i in enumerate(cuts[k]):
            for f in lst[o:o + n_i]:
                f = dict(f); f['batch'] = i; f['sampled_in_stratum'] = len(lst); f['cum_in_stratum'] = sum(cuts[k][:i + 1]); batches[i].append(f)
            o += n_i
    for b in batches: rng.shuffle(b)
    base = batches[0]; ext = batches[1:]
    ext_rec = [dict(batch=i + 1, frames=len(b), composition=collections.Counter(f['level'] for f in b), per_stratum_cut={str(k): cuts[k][i + 1] for k in cuts}) for i, b in enumerate(ext)]
    os.makedirs(a.out_dir, exist_ok=True)
    def write(rows, fname, label_set, prefix, start):
        with open(os.path.join(a.out_dir, fname), 'w', newline='') as fh:
            w = csv.writer(fh, lineterminator='\n'); w.writerow(['sequence', 'anonymous_id', 'image_name', 'image_path', 'image_sha256', 'label_set_id', 'pi_total', 'sampled_in_stratum', 'cum_in_stratum_at_this_batch', 'stratum', 'batch'])
            for i, f in enumerate(rows, start):
                ip = os.path.join(a.image_dir, f['image_name']); w.writerow([i - start + 1, f'{prefix}{i:04d}', f['image_name'], ip, sha(ip), label_set, f'{f["incl_prob"]:.10f}', f['sampled_in_stratum'], f['cum_in_stratum'], f'{f["level"]}|{f["tod"]}', f['batch']])
    write(base, 'independent_blind_audit_100.csv', 'INDEPENDENT-BLIND-AUDIT-100', 'A', 1)
    for e, rows in enumerate(ext, 1):
        fn = f'independent_blind_audit_ext{e:02d}.csv'
        write(rows, fn, f'INDEPENDENT-BLIND-AUDIT-100-EXT{e:02d}', 'A', a.base + (e-1)*a.batch + 1)
        emeta = dict(batch=e, label_set_id=f'INDEPENDENT-BLIND-AUDIT-100-EXT{e:02d}', frames=len(rows), seed=a.seed, drawn_from='leftover pool after base allocation, shuffled once with the same RNG stream',
                     composition=collections.Counter(f['level'] for f in rows), days=len({f['day'] for f in rows}), slice=ext_rec[e-1], nested_design=True, csv_sha256=sha(os.path.join(a.out_dir, fn)))
        json.dump(emeta, open(os.path.join(a.out_dir, fn[:-4] + '.meta.json'), 'w'), indent=2, default=dict)
    meta = dict(created_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'), seed=a.seed, base=len(base), extensions=[len(x) for x in ext],
                source_frames=len(frames), source_rule="image3 JSON description == 'AUTO_PRELABEL_yolo11s_2026-06-03' (2,794 model-prelabel frames)",
                stratification='day-level snow label x time-of-day (<10h / 10-14h / >=14h), proportional allocation with S3 enriched to >=20% of base, largest-remainder rounding',
                s3_target=s3_target, base_composition=collections.Counter(f['level'] for f in base), base_tod=collections.Counter(f['tod'] for f in base),
                base_days=len({f['day'] for f in base}), day_index_sha256=sha(a.day_index), interface='blank-image blind, identical rules to INDEPENDENT-BLIND-331',
                stratum_design=incl, total_sample=total, per_frame_columns='pi_total (exact inclusion probability in the 300-frame two-stage sample), sampled_in_stratum, cum_in_stratum_at_this_batch; at any batch cut pi_cum = pi_total * cum_in_stratum / sampled_in_stratum',
                min_pi_total=min(f['incl_prob'] for f in sample), zero_probability_days_check='every day with frames in a stratum receives quota >= 1 when k > D; when k <= D days are selected by SRS (positive probability for every frame)',
                estimator='missed-box rate = weighted ratio sum(w*miss)/sum(w*boxes), w = 1/pi_cum (ratio estimator; approximately design-unbiased); CI = frame-cluster bootstrap within stratum with fixed counts; no conditional extension draws',
                progress_gate=20, manifest_version=4)
    json.dump(meta, open(os.path.join(a.out_dir, 'independent_blind_audit_100.meta.json'), 'w'), indent=2, default=dict)
    print(json.dumps(meta, indent=1, default=dict))

if __name__ == '__main__': main()
