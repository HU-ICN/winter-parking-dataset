#!/usr/bin/env python3
"""E15: within-day scene/vehicle diversity from final vehicle boxes.

Label set: BOX-DERIVED from FINAL-AUDITED labels on all 5,217 retained frames
(temporal-diversity use only; not independent occupancy ground truth).

Outputs (JSON): per-day rows and pooled distributions for
  (a) adjacent-retained-frame max box IoU (pooled over source boxes; nominal 10-min
      spacing but gaps vary -- gap statistics and a strict <=15-min subset are reported),
  (b) adjacent-frame box persistence (IoU>=0.5 greedy match; pooled over valid pairs,
      pairs whose first frame is empty are excluded and counted),
  (c) first->last frame unmatched fraction per day (days whose first frame is empty are
      excluded and counted; describes disappearance/change of initial boxes only),
  (d) geometric track segments per day by IoU-chained linking, relative to mean boxes
      present (segments are NOT vehicle identities: same-stall replacement can merge,
      occlusion/gaps can split),
  (e) stall-level: fraction of registered stalls whose box-derived state (box centre in
      polygon, no per-frame offset correction) changes within the day.
  Aggregation weights differ: (a) per box, (b) per pair, (c)-(e) per day.
"""
import argparse, glob, json, os, re, collections, statistics as st, hashlib
W, H = 2304, 1536

def load(f):
    b = []
    for l in open(f):
        p = l.split()
        if len(p) < 5: continue
        cx, cy, w, h = map(float, p[1:5]); b.append((cx*W - w*W/2, cy*H - h*H/2, cx*W + w*W/2, cy*H + h*H/2))
    return b

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2-x1)*max(0, y2-y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i; return i/u if u > 0 else 0.0

def greedy(A, B, t=0.5):
    used = set(); m = 0
    for a in A:
        best, bi = -1, -1
        for j, b in enumerate(B):
            if j in used: continue
            v = iou(a, b)
            if v > best: best, bi = v, j
        if best >= t: used.add(bi); m += 1
    return m

def pip(x, y, poly):
    inside = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1: inside = not inside
    return inside

def stall_states(boxes, slots):
    occ = [False]*len(slots)
    for b in boxes:
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        for k, s in enumerate(slots):
            if pip(cx, cy, s): occ[k] = True; break
    return occ

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--labels-dir', required=True); ap.add_argument('--slots', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.labels_dir, '*', '*.txt')) or glob.glob(os.path.join(a.labels_dir, '*.txt')))
    slots_json = json.load(open(a.slots)); slots = [s['polygon'] if isinstance(s, dict) and 'polygon' in s else s for s in slots_json['slots']]
    frames = collections.defaultdict(list)
    for f in files:
        m = re.search(r'image_(\d{8})(\d{6})', os.path.basename(f)); frames[m.group(1)].append((m.group(2), f))
    cons, persist, turnover, rows = [], [], [], []
    cons_strict, persist_strict, gaps = [], [], []; pairs_empty_first = 0; days_empty_first = 0
    def tsec(t): return int(t[:2])*3600 + int(t[2:4])*60 + int(t[4:6])
    for d in sorted(frames):
        seq = sorted(frames[d]); boxes = [load(f) for _, f in seq]
        if len(seq) < 2: rows.append(dict(day=d, frames=len(seq), note='single frame')); continue
        for i in range(len(seq)-1):
            A, B = boxes[i], boxes[i+1]; gap = tsec(seq[i+1][0]) - tsec(seq[i][0]); gaps.append(gap)
            if not A: pairs_empty_first += 1; continue
            mx = [max((iou(x, y) for y in B), default=0.0) for x in A]; pr = greedy(A, B)/len(A)
            cons += mx; persist.append(pr)
            if gap <= 900: cons_strict += mx; persist_strict.append(pr)
        A, Z = boxes[0], boxes[-1]; tl = None
        if A: tl = 1 - greedy(A, Z)/len(A); turnover.append(tl)
        else: days_empty_first += 1
        tracks = 0; active = []
        for bs in boxes:
            newact = []; used = set()
            for x in active:
                best, bi = -1, -1
                for j, y in enumerate(bs):
                    if j in used: continue
                    v = iou(x, y)
                    if v > best: best, bi = v, j
                if best >= 0.5: used.add(bi); newact.append(bs[bi])
            for j, y in enumerate(bs):
                if j not in used: newact.append(y); tracks += 1
            active = newact
        states = [stall_states(bs, slots) for bs in boxes]
        changed = sum(1 for k in range(len(slots)) if len({s[k] for s in states}) > 1)
        nb = [len(b) for b in boxes]
        rows.append(dict(day=d, frames=len(seq), boxes=sum(nb), mean_present=st.fmean(nb), track_segments=tracks,
                         segments_over_mean=tracks/max(1e-9, st.fmean(nb)), turnover_first_last=tl,
                         stalls_changed=changed, stalls_changed_frac=changed/len(slots)))
    valid = [r for r in rows if 'stalls_changed_frac' in r]
    summary = dict(days=len(frames), frames=len(files), days_with_ge2_frames=len(valid), n_slots=len(slots),
        consecutive_iou=dict(n=len(cons), median=st.median(cons), mean=st.fmean(cons), frac_lt_0_5=sum(x < 0.5 for x in cons)/len(cons), frac_zero=sum(x == 0 for x in cons)/len(cons)),
        persistence=dict(median=st.median(persist), mean=st.fmean(persist)),
        turnover_first_last=dict(n=len(turnover), median=st.median(turnover), mean=st.fmean(turnover)),
        segments_over_mean=dict(median=st.median([r['segments_over_mean'] for r in valid]), mean=st.fmean([r['segments_over_mean'] for r in valid])),
        adjacent_gap_seconds=dict(n=len(gaps), median=st.median(gaps), gt_15min=sum(g > 900 for g in gaps), gt_30min=sum(g > 1800 for g in gaps), gt_1h=sum(g > 3600 for g in gaps), max=max(gaps)),
        strict_le_15min=dict(consecutive_iou_n=len(cons_strict), consecutive_iou_median=st.median(cons_strict), persistence_mean=st.fmean(persist_strict), persistence_median=st.median(persist_strict)),
        exclusions=dict(pairs_with_empty_first_frame=pairs_empty_first, multi_frame_days_with_empty_first_frame=days_empty_first),
        stalls_changed_frac=dict(median=st.median([r['stalls_changed_frac'] for r in valid]), mean=st.fmean([r['stalls_changed_frac'] for r in valid])),
        label_set='BOX-DERIVED-STALL-5217 / FINAL-AUDITED boxes', stall_rule='box-centre point-in-polygon on registered stall polygons, no per-frame offset (temporal diversity only; camera shift or box jitter can flip a state)', weighting='(a) per source box, (b) per valid adjacent pair, (c)-(e) per day', identity_caveat='track segments are geometric, not vehicle identities',
        inputs_sha256=dict(slots=hashlib.sha256(open(a.slots,'rb').read()).hexdigest(), n_label_files=len(files)))
    json.dump(dict(summary=summary, per_day=rows), open(a.out, 'w'), indent=1)
    print(json.dumps(summary, indent=1))

if __name__ == '__main__': main()
