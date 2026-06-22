import argparse
import os
import subprocess
import sys


def parse_csv(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def count_files(path, suffix):
    if not os.path.isdir(path):
        return 0
    return len([fn for fn in os.listdir(path) if fn.lower().endswith(suffix)])


def run_extract(args, split, phase):
    input_dir = os.path.join(args.input_root, split, phase)
    output_dir = os.path.join(args.output_root, split, phase)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError('Missing image directory: %s' % input_dir)

    png_count = count_files(input_dir, '.png')
    npy_count = count_files(output_dir, '.npy')
    if not args.force and png_count > 0 and npy_count >= png_count:
        print('Skipping %s/%s: %d feature files already exist' % (split, phase, npy_count))
        return

    command = [
        sys.executable,
        os.path.join('scripts', 'extract_features.py'),
        '--input_image_dir', input_dir,
        '--output_dir', output_dir,
        '--batch_size', str(args.batch_size),
        '--image_height', str(args.image_height),
        '--image_width', str(args.image_width),
        '--model', args.model,
        '--model_stage', str(args.model_stage),
    ]
    if args.max_images is not None:
        command.extend(['--max_images', str(args.max_images)])

    print('Running: %s' % ' '.join(command))
    subprocess.run(command, check=True)


def main(args):
    for split in parse_csv(args.splits):
        for phase in parse_csv(args.phases):
            run_extract(args, split, phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract ResNet features for paired change-caption datasets by split and phase.'
    )
    parser.add_argument('--input_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--splits', default='train,val,test')
    parser.add_argument('--phases', default='A,B')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)
    parser.add_argument('--model', default='resnet101')
    parser.add_argument('--model_stage', default=3, type=int)
    parser.add_argument('--max_images', default=None, type=int)
    parser.add_argument('--force', action='store_true')
    main(parser.parse_args())