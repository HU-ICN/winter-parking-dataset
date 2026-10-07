# Winter Parking-Lot Dataset (Sapporo, 2023-2024)

A season-long, fixed-camera parking-lot image dataset for vehicle detection and per-stall
occupancy estimation under snow, with per-day snow-severity labels, independently
re-annotated evaluation subsets and an evaluation protocol. Collected at Hokkaido
University, Sapporo, Japan. The dataset is the intellectual property of Hokkaido University.

This repository contains everything except the images: labels, splits, polygons, snow-severity
labels, occupancy labels, the released detector, result records and code. The images (5,217 PNG
frames, 2304x1536, 20.6 GB) are provided for research use on request; see
[Access to the images](#access-to-the-images).

Paper: M. Li, A. W. Chimkono, K. Shimizu, H. Tsutsui, and T. Ohgane, "A winter parking-lot dataset
and label-efficient site adaptation for vehicle occupancy detection under snow," IEICE Trans.
Fundamentals (Special Section on Smart Multimedia & Communication Systems), under review; see
`CITATION.cff`. Status note: the paper is under review; this repository will be updated with the
final reference.

## Contents

| Path | What it is |
|---|---|
| `labels/audited/` | 154,412 vehicle boxes for all 5,217 frames, YOLO format (`class cx cy w h`, normalized; single class `car`). These are the model-assisted, human-corrected labels used for training and as the "audited" test labels. |
| `splits/` | Day-based train / val / test frame lists (3,868 / 626 / 723 frames; 73 / 13 / 13 days). |
| `labels/blind/` | Independent "blind" re-annotations of 419 test frames (331 S1/S2 frames and 88 S0/S3 frames), YOLO format, with the label-set manifests (frame lists and hashes). |
| `labels/blind_records/` | The annotation-tool records of the blind passes (box geometry, `partially_buried` flags, revision metadata): first pass (331 and 88 frames), the 55-frame test-retest pass, and the 125-frame blind audit of the historical pre-labels with its sampling manifests (inclusion probabilities). Embedded image data removed. |
| `occupancy/` | Independent per-stall occupancy labels for 120 test frames (`occupancy_120.csv`: frame, stall id 1-82, state), the per-frame stall-layout translation used for registration (`layout_offsets_120.csv`), and the 82 stall polygons (`stall_polygons.json`). |
| `snow_levels_by_day.csv` | Snow-severity level (S0-S4) and frame count for each of the 99 retained days. |
| `day_budgets/` | The sampled training-day lists of the label-efficiency experiment (1, 2, 4, 8, 16, 32 days x three samplings). |
| `models/` | The released YOLO11s detector (`yolo11s_winter_parking_released.pt`, SHA256 in `models/SHA256SUMS`). |
| `results/` | Machine-readable result records of the main experiments (same-site two-label-source evaluation, day budgets, occupancy on 120 frames, identical-image control, test-retest). Machine paths are redacted; original file hashes are in `RESULTS_PROVENANCE.json`. |
| `code/` | Training and evaluation scripts, annotation tools and analysis scripts. See `code/README.md`. |
| `privacy/` | The privacy check of the released frames (method, scripts, conclusion). |
| `SHA256SUMS` | Hashes of every file in this repository. |

Frame names are `image_YYYYMMDDhhmmss` (capture time, JST). The same names are used in the image
package, the labels, the splits and all records.

## Snow-severity levels

| Level | Criterion (judged per day from the parking surface) | Frames |
|---|---|---:|
| S0 | No snow; bare/wet pavement; markings visible | 401 |
| S1 | Light/patchy snow; markings still discernible | 2,397 |
| S2 | Ground fully white; markings hidden; cars distinct | 2,021 |
| S3 | Heavy snow; vehicle edges softened/partly buried | 336 |
| S4 | Vehicles nearly covered, or heavy snowfall/whiteout | 62 |

Days are the unit of sampling and splitting: frames of one day are never split across train, val
and test. The held-out test split contains no S4 day.

## Evaluation protocol

mAP@0.5 saturates on this dataset (0.995 for same-site models) and should not be used as the
primary metric. Report:

1. mAP@[.5:.95] on the audited test labels **and** on the blind labels of the same frames
   (`labels/blind/`); the two label sources give different absolute values.
2. Per-stall occupancy accuracy and occupied-class precision / recall / F1 on the 120 independently
   labeled frames (`occupancy/`), with frame-clustered confidence intervals. Use the released
   detector's working point (confidence 0.25, NMS IoU 0.7) and the centre-to-centroid assignment
   rule with a 90 px radius after applying the per-frame layout offset; see
   `code/A10/occupancy_evaluator.py`.
3. For label-efficiency studies, the number of labeled **days** as the primary unit, with frame
   counts secondary; do not re-draw the splits inside a budget.

Reference results of the released detector (audited labels of the 723 test frames): mAP@0.5 0.995,
mAP@[.5:.95] 0.896, recall 0.998; on the blind labels of the 331 S1/S2 frames mAP@[.5:.95] 0.722
(0.849 on the audited labels of the same frames); occupancy accuracy 0.955 and occupied-class F1
0.919 on the 120 frames.

## Access to the images

The images are provided for non-commercial research under the Dataset Research Use Agreement
(`AGREEMENT.md`). Request procedure: https://docs.google.com/forms/d/e/1FAIpQLSd8KILl9Gc-wRFBLGOQzDiLdmlxrCcq5ws660spaO-hSScd0w/viewform. Approved applicants receive a download link
by e-mail. The agreement prohibits redistribution of the images and any attempt to identify
vehicles or persons. A privacy check of the frames is documented in `privacy/PRIVACY_CHECK.md`.

## Licenses

Components of this repository are licensed separately; see `LICENSES.md`:

- labels, splits, polygons, snow-severity and occupancy labels, day-budget lists and result
  records: CC BY 4.0;
- annotation tools and analysis scripts that do not depend on Ultralytics: MIT;
- scripts that call Ultralytics YOLO, and the released weights (fine-tuned from Ultralytics
  COCO weights): AGPL-3.0, as required by the upstream license;
- the images: Dataset Research Use Agreement (not included here).

Third-party data used in the paper (PKLot, PUCPR+) are not redistributed; `code/E24` contains
the preparation scripts and the official sources are cited in the paper.

## Citation

See `CITATION.cff`. Please cite the paper when you use the dataset, the labels or the detector.

## Contact

Hiroshi Tsutsui (corresponding author of the paper); dataset requests: mingyang.li.t6@elms.hokudai.ac.jp.
