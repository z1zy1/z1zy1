import argparse
import json
import os
import random
import sys

import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.semantic_label import (
    ACTION_VOCAB,
    OBJECT_VOCAB,
    build_semantic_targets,
    is_no_change_caption_set,
)


SPECIAL_TOKEN_IDS = {0, 3}


def load_levir_metadata(root):
    data_root = os.path.join(root, 'Levir-CC')
    paths = {
        'splits': os.path.join(data_root, 'splits.json'),
        'vocab': os.path.join(data_root, 'transformer_levir_vocab.json'),
        'labels': os.path.join(data_root, 'transformer_levir_labels.h5'),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError('%s file not found: %s' % (name, path))

    with open(paths['splits'], 'r', encoding='utf-8') as f:
        splits = json.load(f)
    with open(paths['vocab'], 'r', encoding='utf-8') as f:
        word_to_idx = json.load(f)
    idx_to_word = {int(idx): word for word, idx in word_to_idx.items()}
    return splits, idx_to_word, paths['labels']


def decode_caption(token_ids, idx_to_word):
    words = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in SPECIAL_TOKEN_IDS:
            break
        word = idx_to_word.get(token_id)
        if word is None:
            continue
        if word.startswith('<') and word.endswith('>'):
            continue
        words.append(word)
    return ' '.join(words)


def get_reference_captions(h5_file, img_idx, idx_to_word):
    ix1 = int(h5_file['label_start_idx'][img_idx])
    ix2 = int(h5_file['label_end_idx'][img_idx])
    max_seq_length = h5_file['labels'].shape[1]
    return [
        decode_caption(h5_file['labels'][ix, :max_seq_length], idx_to_word)
        for ix in range(ix1, ix2 + 1)
    ]


def format_positive(names, vector):
    positives = [
        '%s=1' % name
        for name, value in zip(names, vector)
        if float(value) > 0.5
    ]
    return ', '.join(positives) if positives else 'all zero'


def format_relations(relations):
    positives = []
    for action_idx, action_name in enumerate(ACTION_VOCAB):
        for object_idx, object_name in enumerate(OBJECT_VOCAB):
            if float(relations[action_idx][object_idx]) > 0.5:
                positives.append('%s-%s=1' % (action_name, object_name))
    return positives


def select_indices(split_indices, num_samples, start_index, random_sample, seed):
    if random_sample:
        rng = random.Random(seed)
        copied = list(split_indices)
        rng.shuffle(copied)
        return copied[:num_samples]
    return split_indices[start_index:start_index + num_samples]


def run_levir_check(args):
    splits, idx_to_word, label_path = load_levir_metadata(args.root)
    if args.split not in splits:
        raise ValueError('Unknown split "%s". Available splits: %s' % (args.split, sorted(splits.keys())))

    split_indices = [int(x) for x in splits[args.split]]
    selected_indices = select_indices(
        split_indices,
        args.num_samples,
        args.start_index,
        args.random,
        args.seed,
    )
    if not selected_indices:
        raise ValueError('No samples selected for split "%s".' % args.split)

    empty_objects = 0
    empty_actions = 0
    empty_relations = 0
    all_zero = 0
    no_change_samples = 0
    all_zero_non_no_change = 0
    total_object_pos = 0.0
    total_action_pos = 0.0
    total_relation_pos = 0.0

    with h5py.File(label_path, 'r') as h5_file:
        for sample_idx, img_idx in enumerate(selected_indices, 1):
            filename = splits['idx_to_filename'][str(img_idx)]
            captions = get_reference_captions(h5_file, img_idx, idx_to_word)
            targets = build_semantic_targets(captions)
            is_no_change = is_no_change_caption_set(captions)

            object_count = float(targets['objects'].sum())
            action_count = float(targets['actions'].sum())
            relation_count = float(targets['relations'].sum())
            total_object_pos += object_count
            total_action_pos += action_count
            total_relation_pos += relation_count
            empty_objects += int(object_count == 0)
            empty_actions += int(action_count == 0)
            empty_relations += int(relation_count == 0)
            all_zero += int(object_count == 0 and action_count == 0 and relation_count == 0)
            no_change_samples += int(is_no_change)
            all_zero_non_no_change += int(
                object_count == 0 and action_count == 0 and relation_count == 0 and not is_no_change
            )

            print('=' * 80)
            print('sample %d/%d | split=%s | img_idx=%d | file=%s' % (
                sample_idx,
                len(selected_indices),
                args.split,
                img_idx,
                filename,
            ))
            print('no_change_caption_set: %s' % is_no_change)
            print('captions:')
            for cap_idx, caption in enumerate(captions, 1):
                print('  %d. "%s"' % (cap_idx, caption))
            print('objects:')
            print('  ' + format_positive(OBJECT_VOCAB, targets['objects']))
            print('actions:')
            print('  ' + format_positive(ACTION_VOCAB, targets['actions']))
            print('relations:')
            relation_lines = format_relations(targets['relations'])
            if relation_lines:
                for relation in relation_lines:
                    print('  ' + relation)
            else:
                print('  all zero')

    sample_count = len(selected_indices)
    print('=' * 80)
    print('summary:')
    print('  samples: %d' % sample_count)
    print('  total_object_positive: %.0f' % total_object_pos)
    print('  total_action_positive: %.0f' % total_action_pos)
    print('  total_relation_positive: %.0f' % total_relation_pos)
    print('  object_empty_samples: %d / %d' % (empty_objects, sample_count))
    print('  action_empty_samples: %d / %d' % (empty_actions, sample_count))
    print('  relation_empty_samples: %d / %d' % (empty_relations, sample_count))
    print('  all_zero_samples: %d / %d' % (all_zero, sample_count))
    print('  no_change_caption_samples: %d / %d' % (no_change_samples, sample_count))
    print('  all_zero_non_no_change_samples: %d / %d' % (all_zero_non_no_change, sample_count))

    checked_non_no_change = sample_count - no_change_samples
    all_zero_ratio = 0.0 if checked_non_no_change == 0 else float(all_zero_non_no_change) / float(checked_non_no_change)
    if all_zero_ratio >= args.warn_all_zero_ratio:
        print(
            'WARNING: %.1f%% non-no-change samples are all-zero. Expand object/action synonyms before training.'
            % (100.0 * all_zero_ratio)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='LEVIR_CC', choices=['LEVIR_CC'])
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--root', default='.', help='CARD project root.')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--random', action='store_true', help='Randomly sample examples instead of taking the first N.')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--warn_all_zero_ratio', type=float, default=0.5)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError('--num_samples must be positive.')

    run_levir_check(args)


if __name__ == '__main__':
    main()
