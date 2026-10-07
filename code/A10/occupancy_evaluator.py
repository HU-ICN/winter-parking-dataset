#!/usr/bin/env python3
"""A10: versioned per-stall occupancy evaluator (v1 candidate rules; frozen after golden reproduction).

Inputs: slots.json (82 stall polygons on the reference frame), frame_offsets.json (per-frame
[dx, dy] shift applied to stall polygons), predictions {frame: [[x1,y1,x2,y2], ...]} (detector boxes,
original image pixels), ground truth {frame: [occupied slot_ids]} (independent labels).
Rules:
  centroid : box centre -> nearest stall centroid (after offset); assigned only if distance <= R px;
             each stall occupied if at least one box is assigned to it (a box marks at most one stall).
  pip      : box centre inside stall polygon (after offset); first matching polygon in slot order.
Metrics: micro per-stall accuracy over frames x 82 stalls, occupied-class precision/recall/F1,
per-frame accuracy, counts of empty predictions. Frames with no GT entry are refused (no silent skip).
"""
import argparse, json, math, statistics as st

def centroid(poly):
    x = [p[0] for p in poly]; y = [p[1] for p in poly]; return (sum(x)/len(x), sum(y)/len(y))

def pip(x, y, poly):
    inside = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1: inside = not inside
    return inside

def occupied_set(boxes, slots, offset, rule, radius):
    dx, dy = offset; occ = set()
    polys = [[(x+dx, y+dy) for x, y in s['polygon']] for s in slots]; cents = [centroid(p) for p in polys]
    for b in boxes:
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        if rule == 'centroid':
            k, d = min(((i, math.hypot(cx-c[0], cy-c[1])) for i, c in enumerate(cents)), key=lambda t: t[1])
            if d <= radius: occ.add(slots[k]['slot_id'])
        else:
            for i, p in enumerate(polys):
                if pip(cx, cy, p): occ.add(slots[i]['slot_id']); break
    return occ

def validate_gt(entry, ids, frame):
    occ = entry['occupied'] if isinstance(entry, dict) else entry; unres = entry.get('unresolved', []) if isinstance(entry, dict) else []
    for name, lst in (('occupied', occ), ('unresolved', unres)):
        if len(set(lst)) != len(lst): raise SystemExit(f'{frame}: duplicate stall ids in {name}')
        bad = [x for x in lst if x not in ids]
        if bad: raise SystemExit(f'{frame}: unknown stall ids in {name}: {bad}')
    if set(occ) & set(unres): raise SystemExit(f'{frame}: stall marked both occupied and unresolved: {sorted(set(occ) & set(unres))}')
    return set(occ), set(unres)


def evaluate(pred, gt, slots, offsets, rule, radius, require_offsets=True):
    ids = [s['slot_id'] for s in slots]; tp = fp = fn = tn = 0; per_frame = {}; empty_pred = 0; n_unresolved = 0; frames_all_unresolved = 0
    if len(set(ids)) != len(ids): raise SystemExit('duplicate slot ids in slots.json')
    for f in sorted(gt):
        if f not in pred: raise SystemExit(f'no predictions for {f}')
        if require_offsets and f not in offsets: raise SystemExit(f'no frame offset for {f}')
        boxes = pred[f]; empty_pred += (len(boxes) == 0)
        o = occupied_set(boxes, slots, offsets.get(f, [0, 0]), rule, radius)
        g, unres = validate_gt(gt[f], ids, f); n_unresolved += len(unres); c = 0; n_res = 0
        for sid in ids:
            if sid in unres: continue   # unresolved stalls are excluded from the denominator, never mapped to empty
            n_res += 1
            p, t = sid in o, sid in g
            tp += p and t; fp += p and not t; fn += (not p) and t; tn += (not p) and (not t); c += (p == t)
        if n_res == 0: frames_all_unresolved += 1; per_frame[f] = None
        else: per_frame[f] = c / n_res
    n = tp + fp + fn + tn
    if n == 0: raise SystemExit('no resolved stall states to score (all unresolved)')
    prec = tp/(tp+fp) if tp+fp else 0.0; rec = tp/(tp+fn) if tp+fn else 0.0; scored = [v for v in per_frame.values() if v is not None]
    return dict(rule=rule, radius_px=radius if rule == 'centroid' else None, frames=len(gt), stalls=len(ids), states=n,
                accuracy=(tp+tn)/n, precision=prec, recall=rec, f1=(2*prec*rec/(prec+rec) if prec+rec else 0.0), tp=tp, fp=fp, fn=fn, tn=tn,
                per_frame_accuracy_mean=st.fmean(scored), frames_scored=len(scored), frames_all_unresolved=frames_all_unresolved, unresolved_states=n_unresolved, empty_prediction_frames=empty_pred,
                f1_note='occupied-class F1; if tp+fp+fn == 0 (no occupied stall in GT or prediction) F1 is reported as 0.0 with zero denominators and flagged' , f1_zero_denominator=(tp + fp + fn == 0))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--slots', default='slots.json'); ap.add_argument('--offsets', default='frame_offsets.json')
    ap.add_argument('--pred', default='occ_pred.json'); ap.add_argument('--gt', default='occupancy_gt.json'); ap.add_argument('--rule', default='centroid'); ap.add_argument('--radius', type=float, default=90)
    ap.add_argument('--scan', action='store_true', help='scan both rules and radii 40..150 to identify the historical rule'); ap.add_argument('--out')
    a = ap.parse_args(); slots = json.load(open(a.slots))['slots']; offsets = json.load(open(a.offsets)); pred = json.load(open(a.pred)); gt = json.load(open(a.gt))
    if a.scan:
        for rule, radii in (('pip', [None]), ('centroid', [40, 50, 60, 70, 80, 90, 100, 110, 120, 150])):
            for r in radii:
                e = evaluate(pred, gt, slots, offsets, rule, r or 0); print(f"{rule:8s} R={str(r):4s} acc={e['accuracy']:.6f} f1={e['f1']:.5f} P={e['precision']:.4f} R={e['recall']:.4f} tp/fp/fn={e['tp']}/{e['fp']}/{e['fn']}")
        return
    e = evaluate(pred, gt, slots, offsets, a.rule, a.radius); print(json.dumps(e, indent=1))
    if a.out: json.dump(e, open(a.out, 'w'), indent=1)

if __name__ == '__main__': main()
