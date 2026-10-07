# Code

Scripts are copied from the laboratory's experiment repository as used for the paper. They are
grouped by experiment:

| Directory | Scripts | Purpose |
|---|---|---|
| `annotation/` | `blind_detection_tool.py`, `blind_occupancy_tool.py`, `pklot_paired_box_tool.py`, `output_lock.py`, `validate_detection_outputs.py`, `prepare_*_manifest.py` | Blind annotation tools (no existing label or prediction is shown; randomized manifests with anonymous frame identifiers; outputs locked on completion) and manifest preparation, including the stratified probability sample of the pre-label audit. |
| `common/` | `eval_valnative.py`, `prepare_labelsets.py` | Detection evaluation with the Ultralytics validator at input size 1280 (confidence 0.001, NMS IoU 0.7, at most 300 detections per image) on named label sets; label-set preparation. |
| `A10/` | `occupancy_evaluator.py`, `occupancy_eval_120.py`, `occupancy_by_level.py` | Per-stall occupancy evaluation: detections at confidence 0.25, per-frame layout offset, centre-to-centroid assignment within 90 px, one box per stall; frame-bootstrap confidence intervals; per-level breakdown. |
| `E15/` | `e15_within_day_diversity.py` | Within-day diversity statistics (consecutive-frame IoU, persistence, first-last turnover, stall-state changes). |
| `E16/` | `e16_snapshot_diff.py`, `audit_gate_report.py` | Pre-label snapshot vs final labels (greedy IoU >= 0.5 matching; edit rule IoU < 0.9 / centre shift > 8 px / area change > 10%), and the weighted miss-rate estimate with its stopping rule. |
| `E18/` | `run_e18_zero_shot.py`, `score_predictions.py` | Zero-shot COCO-pretrained YOLO11 baselines (car-only and vehicle-class mappings). |
| `E21/` | `train_budget.py`, `budget_blind_eval.py` | Day-budget training (frozen hyper-parameters, one training seed per sampling) and evaluation under both label sources. |
| `E23/` | `test_retest_analysis.py`, `cross_label_subset_55.py` | Test-retest agreement of the 55 re-annotated frames and the cross-label comparison on the same frames. |
| `E24/` | `match_pucpr_plus.py`, `build_e24_datasets.py`, `train_e24.py`, `eval_e24.py`, `per_level_recall.py` | Identical-image control: PUCPR+ vehicle boxes vs PKLot occupied-stall boxes on the same 100 PUCPR frames (PKLot and PUCPR+ must be obtained from their official sources). |

Environment used for the paper: Python 3.9/3.10, Ultralytics 8.3.186, PyTorch 2.7.0 (CUDA 12.6),
one NVIDIA RTX 4090. Training: YOLO11s from COCO weights, input 1280, 60 epochs, patience 15,
batch 16, deterministic, seeds 0/1/2 where three replicates are reported.

Paths. The following scripts contain the laboratory's absolute data and run paths as defaults or
constants and must be adapted before use: `annotation/pklot_paired_box_tool.py`,
`common/eval_valnative.py`, `A10/occupancy_eval_120.py`, `E18/run_e18_zero_shot.py`,
`E21/train_budget.py`, `E21/budget_blind_eval.py`, `E24/match_pucpr_plus.py`,
`E24/build_e24_datasets.py`, `E24/train_e24.py`, `E24/eval_e24.py`. The expected directory layout
is: images in one directory named `image_YYYYMMDDhhmmss.png`, labels as in `labels/`, and label-set
directories as in `labels/blind/` (one YOLO `.txt` per frame plus `LABELSET_MANIFEST.json`).

Licenses are per component; see `../LICENSES.md`.
