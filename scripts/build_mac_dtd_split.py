#!/usr/bin/env python3
"""
Build a MAC/CoOp-compatible DTD split JSON from the local torchvision-format DTD.

Why: MAC's official code loads CoOp-style datasets via `BaseJsonDataset`, which
expects `<root>/dtd/images/<class>/<file>.jpg` plus a
`<root>/dtd/split_zhou_DescribableTextures.json` with entries [relpath, label].

We only have the raw torchvision DTD (dtd/dtd/images + dtd/dtd/labels/{split}1.txt).
This script converts partition-1 labels into the CoOp JSON schema so that BOTH the
official MAC code and the NAC-harness port evaluate on the *identical* sample set.

Labels index into MAC's `dtd_classes` order (alphabetical, 47 texture names) so that
the class->text-prompt mapping ("A photo of a {}.") matches MAC's evaluation.

Output: <out_json> (default: F:/codex/NAC/data/dtd/split_zhou_DescribableTextures.json)
"""
import argparse
import json
import os

# MAC's data/cls_to_names.py: dtd_classes (alphabetical, index == label)
DTD_CLASSES = [
    'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed',
    'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked',
    'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced',
    'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley',
    'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed',
    'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified',
    'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled',
    'zigzagged',
]


def read_split(labels_path):
    """Parse a DTD label file: each line is '<class>/<file>.jpg'."""
    entries = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls = line.split('/')[0]
            entries.append([line, DTD_CLASSES.index(cls)])
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dtd_root', default='F:/codex/NAC/data/dtd')
    ap.add_argument('--partition', type=int, default=1)
    ap.add_argument('--out_json',
                    default='F:/codex/NAC/data/dtd/split_zhou_DescribableTextures.json')
    args = ap.parse_args()

    labels_dir = os.path.join(args.dtd_root, 'dtd', 'labels')
    splits = {}
    for mode in ('train', 'val', 'test'):
        lp = os.path.join(labels_dir, f'{mode}{args.partition}.txt')
        splits[mode] = read_split(lp)
        print(f'{mode}: {len(splits[mode])} entries from {lp}')

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(splits, f)
    print(f'wrote {args.out_json}')


if __name__ == '__main__':
    main()