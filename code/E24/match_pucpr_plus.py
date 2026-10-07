#!/usr/bin/env python3
"""E24 step 1: match every PUCPR+ image to its PKLot PUCPR source frame by decoded-pixel digest.

PUCPR+ (Hsieh et al. 2017) re-annotated 125 frames of the PKLot PUCPR lot with per-car boxes but renamed the files
(e.g. 0_Cloudy.jpg). We recover the source frame so that the SAME images can be trained with PKLot stall labels
(condition C2) and PUCPR+ car labels (condition C1). Matching rule (frozen): exact equality of the SHA256 of the
decoded RGB pixel buffer; if no exact match, the nearest frame by mean absolute pixel difference at 160x90 is
reported with its distance but NOT accepted unless the distance is below --accept-mad (default 2.0/255) AND the
runner-up is at least 3x farther. Writes E24_pucpr_plus_match.json (mapping, digests, distances, unmatched list).
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from PIL import Image


def pix_sha(p):
    return hashlib.sha256(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest()


def thumb(p):
    return np.asarray(Image.open(p).convert('L').resize((160, 90), Image.BILINEAR), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plus', type=Path, default=Path('/mnt/ssd/datasets/pucpr_plus/PUCPR+_devkit/data'))
    ap.add_argument('--pklot', type=Path, default=Path('/mnt/ssd/PKLot/PUCPR'))
    ap.add_argument('--out', type=Path, default=Path('/mnt/ssd/datasets/pucpr_plus/E24_pucpr_plus_match.json'))
    ap.add_argument('--accept-mad', type=float, default=2.0)
    a = ap.parse_args()
    plus = sorted((a.plus / 'Images').glob('*.jpg')); src = sorted(a.pklot.rglob('*.jpg'))
    print('PUCPR+ images', len(plus), 'PKLot PUCPR frames', len(src))
    t0 = time.time(); src_sha = {}; src_thumb = {}
    for p in src:
        src_sha.setdefault(pix_sha(p), []).append(str(p))
    print('pixel digests of PKLot frames done in %.0f s; distinct %d' % (time.time() - t0, len(src_sha)))
    rows = []; exact = 0; near = 0; unmatched = 0; thumbs = None
    for p in plus:
        h = pix_sha(p); rec = dict(plus=p.name, plus_pixel_sha256=h, plus_file_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        if h in src_sha and len(src_sha[h]) == 1:
            rec.update(match='exact', source=src_sha[h][0]); exact += 1
        elif h in src_sha:
            rec.update(match='exact_ambiguous', candidates=src_sha[h]); unmatched += 1
        else:
            if thumbs is None:
                thumbs = np.stack([thumb(s) for s in src])
            d = np.abs(thumbs - thumb(p)[None]).mean(axis=(1, 2)); o = np.argsort(d)
            rec.update(nearest=str(src[o[0]]), nearest_mad=float(d[o[0]]), runner_up_mad=float(d[o[1]]))
            if d[o[0]] < a.accept_mad and d[o[1]] >= 3 * d[o[0]]:
                rec.update(match='near', source=str(src[o[0]])); near += 1
            else:
                rec.update(match='none'); unmatched += 1
        rows.append(rec)
    splits = {s: [l.strip() for l in (a.plus / 'ImageSets' / f'{s}.txt').read_text().splitlines() if l.strip()] for s in ('train', 'test')}
    out = dict(created_at=time.strftime('%Y-%m-%dT%H%M%S%z'), rule=__doc__.strip(), n_plus=len(plus), n_pklot_pucpr=len(src), exact=exact, near=near, unmatched=unmatched,
               splits={s: len(v) for s, v in splits.items()}, matched_per_split={s: sum(1 for r in rows if r.get('source') and r['plus'][:-4] in set(v)) for s, v in splits.items()},
               duplicate_sources=len([1 for h, v in src_sha.items() if len(v) > 1]), rows=rows)
    a.out.write_text(json.dumps(out, indent=1)); print({k: v for k, v in out.items() if k not in ('rows', 'rule')}); print('->', a.out)


if __name__ == '__main__':
    main()
