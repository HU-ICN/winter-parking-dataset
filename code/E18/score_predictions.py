#!/usr/bin/env python3
"""Offline scoring of YOLO-format predictions (with conf) against YOLO labels, single class.

Uses ultralytics.utils.metrics.ap_per_class with the same TP matching as
DetectionValidator (IoU thresholds 0.50:0.95, greedy by IoU), so numbers are
comparable with model.val() for a single-class dataset. Works for any label set
(FIXED-AUDITED, INDEPENDENT-BLIND-...) as long as labels are YOLO txt.
"""
import argparse, glob, os, json, numpy as np
from ultralytics.utils.metrics import ap_per_class, box_iou
import torch

def read(p, conf):
    if not os.path.exists(p): return np.zeros((0, 5 if conf else 4))
    rows = []
    for l in open(p):
        v = l.split()
        if len(v) < 5: continue
        cx, cy, w, h = map(float, v[1:5]); b = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        rows.append(b + ([float(v[5])] if conf else []))
    return np.array(rows) if rows else np.zeros((0, 5 if conf else 4))

def match(pred, gt, iouv):
    """Return tp matrix [n_pred, n_iou] like DetectionValidator.match_predictions."""
    tp = np.zeros((len(pred), len(iouv)), dtype=bool)
    if len(pred) == 0 or len(gt) == 0: return tp
    iou = box_iou(torch.tensor(gt[:, :4]), torch.tensor(pred[:, :4])).numpy()  # [gt, pred]
    for i, t in enumerate(iouv):
        m = np.argwhere(iou >= t)
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            tp[m[:, 1], i] = True
    return tp

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pred-dir'); ap.add_argument('--pred-json', help='ultralytics val predictions.json (COCO xywh pixels)')
    ap.add_argument('--json-categories', nargs='*', type=int, help='keep only these category_id values (val save_json uses COCO91 ids: car=3, bus=6, truck=8); default keep all')
    ap.add_argument('--img-w', type=int, default=2304); ap.add_argument('--img-h', type=int, default=1536)
    ap.add_argument('--label-dir', required=True)
    ap.add_argument('--names', nargs='*', help='restrict to these image stems (e.g. a manifest subset)'); ap.add_argument('--out')
    a = ap.parse_args()
    iouv = np.linspace(0.5, 0.95, 10); tps, confs, n_gt, n_img = [], [], 0, 0
    from collections import defaultdict
    jpred = None
    if a.pred_json:
        jpred = defaultdict(list)
        for d in json.load(open(a.pred_json)):
            if a.json_categories and d['category_id'] not in a.json_categories: continue
            x, y, w, h = d['bbox']; jpred[str(d['image_id'])].append([x / a.img_w, y / a.img_h, (x + w) / a.img_w, (y + h) / a.img_h, d['score']])
    stems = a.names or [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(a.label_dir, '*.txt')))]
    for s in stems:
        gt = read(os.path.join(a.label_dir, s + '.txt'), False)
        pr = np.array(jpred.get(s, [])) if jpred is not None else read(os.path.join(a.pred_dir, s + '.txt'), True)
        if len(pr) == 0: pr = np.zeros((0, 5))
        if len(pr): pr = pr[pr[:, 4].argsort()[::-1]]
        tps.append(match(pr, gt, iouv)); confs.append(pr[:, 4] if len(pr) else np.zeros(0)); n_gt += len(gt); n_img += 1
    tp = np.concatenate(tps); conf = np.concatenate(confs)
    pc = np.zeros(len(conf), dtype=int); tc = np.zeros(n_gt, dtype=int)
    r = ap_per_class(tp, conf, pc, tc, plot=False)
    # ultralytics returns (tp, fp, p, r, f1, ap, unique_classes, ...) ; ap shape [nc, 10]
    apm = r[5]; p = r[2]; rr = r[3]
    res = dict(images=n_img, gt_boxes=n_gt, predictions=len(conf), map50=float(apm[:, 0].mean()), map5095=float(apm.mean()),
               precision=float(p.mean()), recall=float(rr.mean()), iou_thresholds=[float(x) for x in iouv])
    print(json.dumps(res, indent=1))
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)

if __name__ == '__main__': main()
