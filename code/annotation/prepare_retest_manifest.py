#!/usr/bin/env python3
"""Build the frozen INDEPENDENT-BLIND-RETEST-55 manifest (spec v2 section 5, blind test-retest).

Selection: 55 frames drawn uniformly at random (seeded) from the sealed INDEPENDENT-BLIND-331 cohort; a new
random order and new anonymous ids (R0001..R0055). Eligibility: each frame carries eligible_from =
first-pass completed_at + 14 days (spec 5.2, "14 full days after its first-pass completion timestamp"),
enforced by blind_detection_tool 1.0.12. Identity checks before anything is written: the source manifest
digest equals the registry value, every first-pass output digest equals the sealed digest list, every
first-pass output is complete. Nothing about the first-pass boxes is read or copied. Only aggregate
metadata is printed; no manifest row is ever printed.
"""
import argparse, csv, hashlib, json, os, random, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-manifest', type=Path, default=Path('annotation/manifests/independent_blind_331.csv'))
    ap.add_argument('--source-manifest-sha256', default='bf33ff8f3faa5f075300fd8a4564a039bf32887077aec7b1aed4643628bddcdd')
    ap.add_argument('--first-pass-dir', type=Path, default=Path('annotation/outputs/detection'))
    ap.add_argument('--seal-list', type=Path, default=Path('annotation/seals/INDEPENDENT-BLIND-331.sha256'))
    ap.add_argument('--output', type=Path, default=Path('annotation/manifests/independent_blind_retest_55.csv'))
    ap.add_argument('--seed', type=int, default=20260909)
    ap.add_argument('--count', type=int, default=55)
    ap.add_argument('--eligibility-days', type=int, default=14)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f'{a.output} exists; frozen manifests are never overwritten')
    if sha256_file(a.source_manifest) != a.source_manifest_sha256:
        raise SystemExit('source manifest digest differs from the registry')
    sealed = {}
    for line in a.seal_list.read_text().splitlines():
        if line.strip():
            d, n = line.split(None, 1); sealed[n.strip()] = d
    with a.source_manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 331 or len({r['image_name'] for r in rows}) != 331:
        raise SystemExit('source manifest is not the 331-frame cohort')
    completed_at = {}
    for r in rows:
        name = f"{r['anonymous_id']}.json"; p = a.first_pass_dir / name
        if name not in sealed or not p.exists() or sha256_file(p) != sealed[name]:
            raise SystemExit(f'first-pass output {name} missing or differs from the seal list')
        meta = json.loads(p.read_text())['revision_annotation']
        if not meta.get('complete') or not meta.get('completed_at'):
            raise SystemExit(f'first-pass output {name} is not complete')
        completed_at[r['image_name']] = datetime.fromisoformat(meta['completed_at'])
    rng = random.Random(a.seed)
    chosen = rng.sample(rows, a.count)
    rng.shuffle(chosen)
    fields = ['sequence', 'anonymous_id', 'image_name', 'image_path', 'image_sha256', 'label_set_id', 'eligible_from']
    eligible_from = {}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', newline='', dir=a.output.parent, delete=False, suffix='.tmp') as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        for i, r in enumerate(chosen, start=1):
            image_path = Path(r['image_path'])
            if not image_path.is_file() or sha256_file(image_path) != r['image_sha256']:
                raise SystemExit('image missing or digest differs from the source manifest')
            e = (completed_at[r['image_name']] + timedelta(days=a.eligibility_days)).isoformat(timespec='seconds')
            eligible_from[r['image_name']] = e
            w.writerow(dict(sequence=i, anonymous_id=f'R{i:04d}', image_name=r['image_name'], image_path=str(image_path),
                            image_sha256=r['image_sha256'], label_set_id='INDEPENDENT-BLIND-RETEST-55', eligible_from=e))
        tmp = Path(h.name)
    os.replace(tmp, a.output)
    now = datetime.now(timezone.utc).astimezone()
    el = sorted(eligible_from.values())
    meta = dict(label_set_id='INDEPENDENT-BLIND-RETEST-55', created_at=now.isoformat(timespec='seconds'), random_seed=a.seed,
                frame_count=a.count, source_label_set_id='INDEPENDENT-BLIND-331', source_manifest_sha256=a.source_manifest_sha256,
                source_seal_list_sha256=sha256_file(a.seal_list),
                selection_rule='uniform random sample without replacement from the 331 sealed first-pass frames; new random order; new anonymous ids R0001..R0055',
                eligibility_rule=f'eligible_from = first-pass completed_at + {a.eligibility_days} days (spec v2 section 5.2); enforced per frame by blind_detection_tool >= 1.0.12',
                eligible_at_creation=sum(1 for e in el if datetime.fromisoformat(e) <= now), earliest_eligible_from=el[0], latest_eligible_from=el[-1],
                manifest_sha256=sha256_file(a.output), script_sha256=sha256_file(__file__))
    a.output.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
