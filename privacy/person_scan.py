"""Privacy scan: COCO-pretrained YOLO11m, person class only, over the 5,217 dataset frames.
Writes one JSON line per frame with person boxes (xyxy, conf) at imgsz 1280, conf 0.25."""
import json, sys, hashlib
from pathlib import Path
from ultralytics import YOLO
weights, imgdir, listfile, out = sys.argv[1:5]
names = [l.strip() for l in open(listfile) if l.strip()]
model = YOLO(weights)
sha = hashlib.sha256(open(weights,'rb').read()).hexdigest()
with open(out, 'w') as f:
    f.write(json.dumps(dict(meta=dict(weights=weights, sha256=sha, imgsz=1280, conf=0.25, classes=[0], n_frames=len(names))))+'\n')
    for i in range(0, len(names), 16):
        batch = [str(Path(imgdir)/n) for n in names[i:i+16]]
        for r in model.predict(batch, imgsz=1280, conf=0.25, classes=[0], verbose=False, device=0):
            b = r.boxes
            f.write(json.dumps(dict(frame=Path(r.path).name, persons=[dict(xyxy=[round(v,1) for v in bb], conf=round(c,3)) for bb,c in zip(b.xyxy.tolist(), b.conf.tolist())]))+'\n')
        if i % 800 == 0: print(i, flush=True)
print('done', flush=True)
