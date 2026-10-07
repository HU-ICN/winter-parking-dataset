#!/usr/bin/env python3
"""E16 part 1: full difference between PRELABEL-SNAPSHOT-2794 and FINAL-AUDITED-2794.

Per frame: prelabel boxes, final boxes, matched pairs (greedy IoU>=0.5, highest IoU first),
'auditor-added' (final box with no prelabel match), 'auditor-removed' (prelabel box with no
final match), and geometry edits among matches (IoU<0.9 or centre shift>8px or AREA change>10%).
Also counts frames with ANY coordinate/count change (no threshold) separately from frames
reaching a preset difference threshold. Greedy matching describes the difference; it is not
a maximum matching and does not recover the editing history.
'auditor-added' is NOT a true miss rate; the independent blind audit provides that.
"""
import argparse, glob, json, os, statistics as st, hashlib, collections

def boxes(j):
    out = []
    for s in j.get('shapes', []):
        if s.get('shape_type', 'rectangle') != 'rectangle': continue
        xs = [p[0] for p in s['points']]; ys = [p[1] for p in s['points']]
        out.append((min(xs), min(ys), max(xs), max(ys), s.get('score')))
    return out

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2-x1)*max(0, y2-y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i; return i/u if u > 0 else 0.0

def match(P, F, t=0.5):
    pairs = sorted(((iou(p, f), i, j) for i, p in enumerate(P) for j, f in enumerate(F)), reverse=True)
    up, uf, m = set(), set(), []
    for v, i, j in pairs:
        if v < t: break
        if i in up or j in uf: continue
        up.add(i); uf.add(j); m.append((i, j, v))
    return m, [i for i in range(len(P)) if i not in up], [j for j in range(len(F)) if j not in uf]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--prelabels', required=True); ap.add_argument('--final', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = []; added = removed = matched = edited = pre_total = fin_total = 0; iou_matched = []; added_sizes = []; removed_scores = []; frames_changed = 0
    frames_any_change = 0; identical_matches = 0; coord_changed_matches = 0
    for pf in sorted(glob.glob(os.path.join(a.prelabels, '*.json'))):
        name = os.path.basename(pf); ff = os.path.join(a.final, name)
        if not os.path.exists(ff): raise SystemExit(f'final label missing for {name}: refusing to count it in the denominator')
        P = boxes(json.load(open(pf))); F = boxes(json.load(open(ff)))
        m, up, uf = match(P, F)
        e = 0; anychange = bool(uf or up)
        for i, j, v in m:
            p, f = P[i], F[j]; cp = ((p[0]+p[2])/2, (p[1]+p[3])/2); cf = ((f[0]+f[2])/2, (f[1]+f[3])/2)
            shift = ((cp[0]-cf[0])**2 + (cp[1]-cf[1])**2) ** 0.5; ap_ = (p[2]-p[0])*(p[3]-p[1]); af = (f[2]-f[0])*(f[3]-f[1])
            if v < 0.9 or shift > 8 or abs(af-ap_)/max(ap_, 1) > 0.10: e += 1
            if p[:4] == f[:4]: identical_matches += 1
            else: coord_changed_matches += 1; anychange = True
            iou_matched.append(v)
        added += len(uf); removed += len(up); matched += len(m); edited += e; pre_total += len(P); fin_total += len(F)
        added_sizes += [((F[j][2]-F[j][0])*(F[j][3]-F[j][1])) ** 0.5 for j in uf]; removed_scores += [P[i][4] for i in up if P[i][4] is not None]
        ch = bool(uf or up or e); frames_changed += ch; frames_any_change += anychange
        rows.append(dict(frame=name, prelabel=len(P), final=len(F), matched=len(m), added=len(uf), removed=len(up), edited=e, reached_threshold=ch, any_change=anychange))
    n = len(rows)
    summary = dict(frames=n, prelabel_boxes=pre_total, final_boxes=fin_total, matched=matched, auditor_added=added, auditor_removed=removed, geometry_edited=edited,
                   frames_reaching_threshold=frames_changed, frames_reaching_threshold_frac=frames_changed/n, frames_with_any_change=frames_any_change, frames_with_any_change_frac=frames_any_change/n, matched_identical=identical_matches, matched_coord_changed=coord_changed_matches,
                   added_per_frame_mean=added/n, removed_per_frame_mean=removed/n,
                   added_frac_of_final=added/max(1, fin_total), removed_frac_of_prelabel=removed/max(1, pre_total), edited_frac_of_matched=edited/max(1, matched),
                   matched_iou_median=st.median(iou_matched) if iou_matched else None,
                   added_box_sqrt_area_median=st.median(added_sizes) if added_sizes else None,
                   removed_prelabel_score_median=st.median(removed_scores) if removed_scores else None,
                   frames_with_added=sum(1 for r in rows if r.get('added')), frames_with_removed=sum(1 for r in rows if r.get('removed')),
                   label_sets=['PRELABEL-SNAPSHOT-2794', 'FINAL-AUDITED-2794'], match_rule='greedy IoU>=0.5 highest-first', edit_rule='IoU<0.9 or centre shift>8px or area change>10%',
                   caveat='auditor_added is an auditor-corrected-miss count, not a true miss rate (see independent blind audit); a low change rate does not by itself show how carefully the prelabels were reviewed')
    json.dump(dict(summary=summary, per_frame=rows), open(a.out, 'w'), indent=1); print(json.dumps(summary, indent=1))

if __name__ == '__main__': main()
