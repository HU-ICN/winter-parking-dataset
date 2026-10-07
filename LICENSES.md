# Licenses

The components of this repository are licensed separately. The images are not part of this
repository and are governed by the Dataset Research Use Agreement (`AGREEMENT.md`).

| Component | Paths | License |
|---|---|---|
| Labels and metadata: vehicle boxes, blind re-annotations and records, splits, stall polygons, layout offsets, occupancy labels, snow-severity labels, day-budget lists, result records | `labels/`, `splits/`, `occupancy/`, `snow_levels_by_day.csv`, `day_budgets/`, `results/` | Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ |
| Annotation tools and analysis scripts without an Ultralytics dependency | `code/annotation/`, `code/E15/`, `code/E16/`, `code/E23/`, `code/A10/occupancy_evaluator.py`, `code/A10/occupancy_by_level.py`, `code/common/prepare_labelsets.py`, `privacy/crop_remote.py` | MIT (text below) |
| Scripts that import or call Ultralytics YOLO, and the released weights fine-tuned from Ultralytics COCO weights | `code/common/eval_valnative.py`, `code/A10/occupancy_eval_120.py`, `code/E18/`, `code/E21/`, `code/E24/`, `privacy/person_scan*.py`, `models/` | GNU Affero General Public License v3.0 (AGPL-3.0), https://www.gnu.org/licenses/agpl-3.0.html, following the Ultralytics license (https://www.ultralytics.com/license). Ultralytics is not redistributed here. |

Attribution for CC BY 4.0 material: "Winter Parking-Lot Dataset, Hokkaido University, Li et al.,
IEICE Trans. Fundamentals" (see `CITATION.cff`).

## MIT License (for the components marked MIT)

Copyright (c) 2026 Hokkaido University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Third-party notices

- Ultralytics YOLO11 (https://github.com/ultralytics/ultralytics), AGPL-3.0; version 8.3.186 was used.
- PKLot (de Almeida et al., 2015) and PUCPR+ (Hsieh et al., 2017) were used in the paper's
  cross-domain experiments under their own terms; neither is redistributed here.
