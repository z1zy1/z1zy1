#!/usr/bin/env python3
"""Fuse ChangeFormer/BIT probability maps into masks and confidence maps."""

import argparse
import os

import numpy as np
from imageio.v2 import imwrite


def files(root, split):
    path = os.path.join(root, split)
    return sorted(name for name in os.listdir(path) if name.lower().endswith('.npy'))


def load_probability(path):
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError('Expected [H,W] probability map at %s, got %s.' % (path, value.shape))
    return np.nan_to_num(np.clip(value, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)


def main(args):
    os.makedirs(args.mask_root, exist_ok=True)
    os.makedirs(args.confidence_root, exist_ok=True)
    os.makedirs(args.uncertainty_root, exist_ok=True)
    total = 0
    for split in [value.strip() for value in args.splits.split(',') if value.strip()]:
        cf_names = files(args.changeformer_root, split)
        bit_names = set(files(args.bit_root, split))
        if not cf_names:
            raise FileNotFoundError('No ChangeFormer maps under %s/%s.' % (args.changeformer_root, split))
        missing = [name for name in cf_names if name not in bit_names]
        if missing:
            raise FileNotFoundError('BIT maps missing for %s: %s' % (split, missing[:5]))
        for name in cf_names:
            cf = load_probability(os.path.join(args.changeformer_root, split, name))
            bit = load_probability(os.path.join(args.bit_root, split, name))
            if cf.shape != bit.shape:
                raise ValueError('Shape mismatch for %s/%s: %s vs %s' % (split, name, cf.shape, bit.shape))
            ensemble = 0.5 * (cf + bit)
            agreement = np.abs(cf - bit)
            high_change = (cf >= args.high_threshold) & (bit >= args.high_threshold)
            moderate_change = (ensemble >= args.mask_threshold) & (agreement <= args.max_disagreement)
            mask = high_change | moderate_change
            agreement_confidence = 1.0 - agreement
            confidence = np.where(
                mask,
                ensemble * agreement_confidence,
                (1.0 - ensemble) * agreement_confidence,
            ).astype(np.float32)
            uncertainty = (~high_change & ~((ensemble <= args.low_threshold) & (agreement <= args.max_disagreement))).astype(np.uint8)
            stem = os.path.splitext(name)[0]
            mask_path = os.path.join(args.mask_root, split, stem + '.png')
            confidence_path = os.path.join(args.confidence_root, split, stem + '.png')
            uncertainty_path = os.path.join(args.uncertainty_root, split, stem + '.png')
            for path in (mask_path, confidence_path, uncertainty_path):
                if os.path.exists(path) and not args.overwrite:
                    raise FileExistsError('Refusing to overwrite %s; pass --overwrite.' % path)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            os.makedirs(os.path.dirname(confidence_path), exist_ok=True)
            os.makedirs(os.path.dirname(uncertainty_path), exist_ok=True)
            imwrite(mask_path, mask.astype(np.uint8) * 255)
            imwrite(confidence_path, np.round(confidence * 255.0).astype(np.uint8))
            imwrite(uncertainty_path, uncertainty * 255)
            total += 1
        print('Fused %s: %d maps' % (split, len(cf_names)))
    print('Finished ensemble fusion: %d maps' % total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--changeformer_root', required=True)
    parser.add_argument('--bit_root', required=True)
    parser.add_argument('--mask_root', required=True)
    parser.add_argument('--confidence_root', required=True)
    parser.add_argument('--uncertainty_root', required=True)
    parser.add_argument('--splits', default='train,val,test')
    parser.add_argument('--high_threshold', type=float, default=0.65)
    parser.add_argument('--low_threshold', type=float, default=0.25)
    parser.add_argument('--mask_threshold', type=float, default=0.50)
    parser.add_argument('--max_disagreement', type=float, default=0.15)
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
