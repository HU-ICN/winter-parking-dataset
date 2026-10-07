# Privacy check of the 5,217 released frames (2026-10-02, revised after audit)

Purpose: support the manuscript statement (Sect. 3) that license plates and faces are *generally* not legible at the camera's distance and resolution (Logitech C920, 2304x1536, eleventh-floor window). This is a detector-based screen plus manual review of size-selected crops; it is not a per-pixel guarantee and does not establish anonymity.

Scope: the 5,217 frames of the released dataset (train/val/test union; file list = `_ds_meta/labels_yolo`). Images were read from the full-capture directory on the 4090 (`custom_dataset/image3`). Environment: Ultralytics 8.3.186 (base env), weights `snowpark_yolo/yolo11m.pt`, sha256 d5ffc1a674953a08... (full value in the JSONL meta lines).

## 1. Vehicle size and an order-of-magnitude plate estimate
- All 154,412 audited boxes: width p5 99 px, median 143 px, p95 191 px, max 435 px; height median 73 px, max 322 px (percentiles: nearest-rank).
- A 330 mm plate on a ~4.5 m vehicle spans ~7% of the vehicle length, i.e. roughly 10 px at the median box width and ~30 px at the widest box. This is an order-of-magnitude estimate only: box width is not vehicle length, and plates sit on near-vertical faces that are foreshortened in this overhead view.
- Manual review: the 12 widest boxes, the 12 tallest boxes and the 12 largest-area boxes (one per frame), cropped at native resolution and shown at 2x nearest-neighbour (`controlled/largest_vehicles_x2.jpg`, `controlled/vehicles_by_height_area_x2.jpg`). Mostly trucks, snow-removal vehicles and a wheel loader seen from above. No plate characters were resolvable in the reviewed crops. Company livery (name/logo) is visible on the sides of a few trucks; no personal name or contact detail was seen.

## 2. Pedestrian screen
Scan 1 (`person_scan.py`): YOLO11m, COCO person class, imgsz 1280, conf 0.25. 24 of 5,217 frames with at least one detection; 27 detections (11 at conf >= 0.5). Detection height (linear-interpolated percentiles): median 36.7 px, p95 51.5 px; excluding one 360 px false positive (bare asphalt), median 36.5 px, p95 47.2 px, max 53.1 px. All 27 candidates were reviewed at 3x nearest-neighbour (`controlled/all_27_person_candidates.jpg`).

Scan 2 (`person_scan_lowthr.py`): same model, imgsz 2304 (native long side), conf 0.05, chosen in advance as a supplementary low-threshold screen (not a validated safety threshold). 122 frames, 148 detections (14 at conf >= 0.25, 41 at 0.10-0.25, 93 below 0.10). Reviewed: every detection in a frame not flagged by scan 1 with conf >= 0.10, plus every detection taller than 60 px (105 crops; `controlled/lowthr_review_{1,2,3}.jpg`). 105 of the 148 candidates were reviewed under this rule. The additional true pedestrians among them are 30-50 px tall (including one worker on a snow blower); among the reviewed candidates, false positives included asphalt patches, snow piles, vehicle roofs and snow-removal machinery.

Negative-frame sample: 10 frames without any detection in scan 1, one per (month, am/pm) stratum, viewed at half resolution (`controlled/negative_sample_half.jpg`); no pedestrian noticed. Detector recall was not measured.

## 3. Conclusion (wording for the release notes)
We screened all 5,217 images for person detections and manually inspected size-selected vehicle crops and person-detection candidates. No legible license-plate characters or discernible facial features were observed in the reviewed crops. The screening may miss instances and does not establish anonymity or exclude re-identification through contextual information (e.g., company livery combined with time and place). Access and use are governed by the research-use agreement, which prohibits re-identification and redistribution.

## 4. Records
- Public: this file, `person_scan.py`, `person_scan_lowthr.py`, `crop_remote.py`.
- Controlled (derived images and per-frame, time-stamped detection records; kept with the controlled dataset package, not published): `controlled/` with sha256 values in `controlled/SHA256SUMS.json`. Image files are excluded from git by the repository's ignore rules; their hashes are recorded.
