import argparse
import os

import numpy as np
from imageio.v2 import imread, imwrite


def list_pngs(path):
    return sorted(fn for fn in os.listdir(path) if fn.lower().endswith('.png'))


def build_mask(before_path, after_path):
    before = imread(before_path)
    after = imread(after_path)
    if before.shape != after.shape:
        raise ValueError(
            'Shape mismatch for %s and %s: %s vs %s'
            % (before_path, after_path, before.shape, after.shape)
        )
    diff = before != after
    if diff.ndim == 3:
        diff = np.any(diff, axis=2)
    return diff.astype(np.uint8) * 255


def main(args):
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    total = 0
    for split in splits:
        before_dir = os.path.join(args.dataset_root, split, args.before_phase)
        after_dir = os.path.join(args.dataset_root, split, args.after_phase)
        output_dir = os.path.join(args.output_root, split)
        if not os.path.isdir(before_dir):
            raise FileNotFoundError('Missing before directory: %s' % before_dir)
        if not os.path.isdir(after_dir):
            raise FileNotFoundError('Missing after directory: %s' % after_dir)
        os.makedirs(output_dir, exist_ok=True)

        filenames = list_pngs(before_dir)
        if args.max_images is not None:
            filenames = filenames[:args.max_images]
        for index, filename in enumerate(filenames, start=1):
            before_path = os.path.join(before_dir, filename)
            after_path = os.path.join(after_dir, filename)
            if not os.path.exists(after_path):
                raise FileNotFoundError('Missing matching after image: %s' % after_path)
            mask = build_mask(before_path, after_path)
            imwrite(os.path.join(output_dir, filename), mask)
            if args.log_interval > 0 and index % args.log_interval == 0:
                print('Processed %s: %d / %d' % (split, index, len(filenames)))
        print('Finished %s: %d masks' % (split, len(filenames)))
        total += len(filenames)
    print('Finished all splits: %d masks' % total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build binary pseudo masks by differencing paired semantic maps.'
    )
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--before_phase', default=os.path.join('sem', 'A'))
    parser.add_argument('--after_phase', default=os.path.join('sem', 'B'))
    parser.add_argument('--splits', default='train,val,test')
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=500)
    main(parser.parse_args())