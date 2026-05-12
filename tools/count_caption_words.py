import argparse
import json
import os
import sys
from collections import Counter

import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SPECIAL_TOKEN_IDS = {0, 3}
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'before', 'by', 'for',
    'from', 'has', 'have', 'in', 'into', 'is', 'it', 'no', 'of', 'on', 'or',
    'the', 'there', 'this', 'to', 'two', 'was', 'were', 'with',
}

OBJECT_CANDIDATES = [
    'building', 'buildings', 'house', 'houses', 'villa', 'villas',
    'residential', 'residential area', 'road', 'roads', 'street', 'streets',
    'tree', 'trees', 'forest', 'woods', 'woodland', 'vegetation',
    'green land', 'field', 'fields', 'farmland', 'bareland', 'bare land',
    'land', 'soil', 'ground', 'water', 'river', 'pond', 'lake', 'bridge',
    'parking', 'parking lot', 'parking area', 'factory', 'factories',
    'playground', 'playgrounds',
]

ACTION_CANDIDATES = [
    'built', 'build', 'constructed', 'construction', 'added', 'add',
    'appear', 'appeared', 'appears', 'new', 'removed', 'remove',
    'disappeared', 'disappear', 'gone', 'missing', 'replaced', 'replace',
    'converted', 'changed', 'change', 'expanded', 'expand', 'extended',
    'increased', 'larger',
]


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
        words.append(word.lower())
    return words


def iter_split_captions(h5_file, split_indices, idx_to_word):
    max_seq_length = h5_file['labels'].shape[1]
    for img_idx in split_indices:
        ix1 = int(h5_file['label_start_idx'][img_idx])
        ix2 = int(h5_file['label_end_idx'][img_idx])
        for label_idx in range(ix1, ix2 + 1):
            yield decode_caption(h5_file['labels'][label_idx, :max_seq_length], idx_to_word)


def add_ngrams(counter, tokens, n):
    if len(tokens) < n:
        return
    for idx in range(len(tokens) - n + 1):
        counter[' '.join(tokens[idx:idx + n])] += 1


def count_phrase(counter, phrase):
    return counter.get(phrase.lower(), 0)


def print_counter(title, counter, topk, min_count=1):
    print(title)
    printed = 0
    for word, count in counter.most_common():
        if count < min_count:
            continue
        print('  %s\t%d' % (word, count))
        printed += 1
        if printed >= topk:
            break
    if printed == 0:
        print('  <empty>')


def print_candidate_counts(title, candidates, unigram_counter, phrase_counter):
    print(title)
    for term in candidates:
        if ' ' in term:
            count = count_phrase(phrase_counter, term)
        else:
            count = unigram_counter.get(term.lower(), 0)
        print('  %s\t%d' % (term, count))


def run_count(args):
    splits, idx_to_word, label_path = load_levir_metadata(args.root)
    if args.split not in splits:
        raise ValueError('Unknown split "%s". Available splits: %s' % (args.split, sorted(splits.keys())))

    split_indices = [int(x) for x in splits[args.split]]
    word_counter = Counter()
    content_word_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()
    phrase_counter = Counter()
    caption_count = 0
    token_count = 0

    with h5py.File(label_path, 'r') as h5_file:
        for tokens in iter_split_captions(h5_file, split_indices, idx_to_word):
            caption_count += 1
            token_count += len(tokens)
            word_counter.update(tokens)
            content_word_counter.update([token for token in tokens if token not in STOPWORDS])
            add_ngrams(bigram_counter, tokens, 2)
            add_ngrams(trigram_counter, tokens, 3)
            add_ngrams(phrase_counter, tokens, 2)
            add_ngrams(phrase_counter, tokens, 3)

    print('dataset: LEVIR_CC')
    print('split: %s' % args.split)
    print('images: %d' % len(split_indices))
    print('captions: %d' % caption_count)
    print('tokens: %d' % token_count)
    print('')
    print_counter('top_%d_words:' % args.topk, word_counter, args.topk, min_count=args.min_count)
    print('')
    print_counter('top_%d_content_words_without_common_stopwords:' % args.topk, content_word_counter, args.topk, min_count=args.min_count)
    print('')
    print_candidate_counts('object_candidate_counts:', OBJECT_CANDIDATES, word_counter, phrase_counter)
    print('')
    print_candidate_counts('action_candidate_counts:', ACTION_CANDIDATES, word_counter, phrase_counter)
    print('')
    print_counter('top_%d_bigrams:' % args.ngram_topk, bigram_counter, args.ngram_topk, min_count=args.min_count)
    print('')
    print_counter('top_%d_trigrams:' % args.ngram_topk, trigram_counter, args.ngram_topk, min_count=args.min_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='LEVIR_CC', choices=['LEVIR_CC'])
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--topk', type=int, default=200)
    parser.add_argument('--ngram_topk', type=int, default=50)
    parser.add_argument('--min_count', type=int, default=1)
    parser.add_argument('--root', default='.', help='CARD project root.')
    args = parser.parse_args()

    if args.topk <= 0:
        raise ValueError('--topk must be positive.')
    if args.ngram_topk <= 0:
        raise ValueError('--ngram_topk must be positive.')

    run_count(args)


if __name__ == '__main__':
    main()
