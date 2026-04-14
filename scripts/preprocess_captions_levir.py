import argparse
import json
from collections import Counter

import h5py
import numpy as np


SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<UNK>': 1,
    '<START>': 2,
    '<END>': 3,
    '<SEP>': 4,
}


def normalize_tokens(sentence_obj):
    if 'tokens' in sentence_obj and sentence_obj['tokens']:
        return [str(t).strip().lower() for t in sentence_obj['tokens'] if str(t).strip()]
    raw = str(sentence_obj.get('raw', '')).strip().lower()
    raw = raw.replace('.', ' ').replace('?', ' ').replace(',', ' ').replace(';', ' ')
    return [tok for tok in raw.split() if tok]


def build_vocab(sequences, min_token_count=1):
    counter = Counter()
    for seq in sequences:
        counter.update(seq)

    word_to_idx = dict(SPECIAL_TOKENS)
    for token in sorted(counter.keys()):
        if counter[token] >= min_token_count:
            word_to_idx[token] = len(word_to_idx)
    return word_to_idx


def encode(tokens, word_to_idx, allow_unk=True):
    output = []
    for token in tokens:
        if token in word_to_idx:
            output.append(word_to_idx[token])
        elif allow_unk:
            output.append(word_to_idx['<UNK>'])
        else:
            raise KeyError(f'Token "{token}" not in vocab')
    return output


def build_coco_annotations(captions_by_image):
    info_dict = {
        'contributor': 'CARD-LevirCC',
        'date_created': 'N/A',
        'description': 'Levir-CC captions reformatted for COCO caption evaluation',
        'url': 'N/A',
        'version': '1.0',
        'year': '2024',
    }
    annotations = {
        'info': info_dict,
        'licenses': [info_dict],
        'type': 'captions',
        'images': [],
        'annotations': [],
    }
    ann_id = 0
    for image_name, captions in captions_by_image.items():
        annotations['images'].append({'id': image_name, 'file_name': image_name})
        for cap in captions:
            annotations['annotations'].append(
                {'id': ann_id, 'image_id': image_name, 'caption': cap}
            )
            ann_id += 1
    return annotations


def main(args):
    with open(args.input_captions_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    images = sorted(raw_data['images'], key=lambda x: x['imgid'])

    num_images = len(images)
    assert num_images > 0, 'No image annotations found.'
    assert images[0]['imgid'] == 0 and images[-1]['imgid'] == num_images - 1, \
        'Expected contiguous imgid starting from 0.'

    captions_by_image = {}
    splits = {'train': [], 'val': [], 'test': [], 'idx_to_filename': {}, 'idx_to_split': {}}
    train_caption_tokens = []
    all_caption_tokens = []

    for item in images:
        imgid = int(item['imgid'])
        filename = item['filename']
        split = item['split']
        filepath = item.get('filepath', split)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown split "{split}" in annotation json.')

        sent_tokens_list = [normalize_tokens(s) for s in item['sentences']]
        sent_tokens_list = [tokens for tokens in sent_tokens_list if tokens]
        if not sent_tokens_list:
            raise ValueError(f'Image {filename} has no valid captions.')

        captions_by_image[filename] = [' '.join(tokens) for tokens in sent_tokens_list]
        splits[split].append(imgid)
        splits['idx_to_filename'][str(imgid)] = filename
        splits['idx_to_split'][str(imgid)] = filepath

        all_caption_tokens.extend(sent_tokens_list)
        if split == 'train':
            train_caption_tokens.extend(sent_tokens_list)

    print(f'Total images: {num_images}')
    print(f'Train/Val/Test: {len(splits["train"])}/{len(splits["val"])}/{len(splits["test"])}')
    print(f'Total captions: {len(all_caption_tokens)}')

    if args.input_vocab_json:
        print(f'Loading vocab from: {args.input_vocab_json}')
        with open(args.input_vocab_json, 'r', encoding='utf-8') as f:
            word_to_idx = json.load(f)
    else:
        print('Building vocab from train split captions...')
        word_to_idx = build_vocab(train_caption_tokens, min_token_count=args.word_count_threshold)

    encoded_by_image = {}
    max_length = 0
    for imgid in range(num_images):
        filename = splits['idx_to_filename'][str(imgid)]
        cap_list = captions_by_image[filename]
        encoded_caps = []
        for cap in cap_list:
            tokens = ['<START>'] + cap.split() + ['<END>']
            max_length = max(max_length, len(tokens))
            encoded_caps.append(tokens)
        encoded_by_image[imgid] = encoded_caps

    label_arrays = []
    label_start_idx = -np.ones(num_images, dtype=np.int64)
    label_end_idx = -np.ones(num_images, dtype=np.int64)
    label_length = []
    caption_counter = 0

    for imgid in range(num_images):
        token_seqs = encoded_by_image[imgid]
        n = len(token_seqs)
        Li = np.zeros((n, max_length), dtype=np.int64)
        for j, tokens in enumerate(token_seqs):
            tokens_encoded = encode(tokens, word_to_idx, allow_unk=args.allow_unk == 1)
            Li[j, :len(tokens_encoded)] = tokens_encoded
            label_length.append(len(tokens_encoded))
        label_arrays.append(Li)
        label_start_idx[imgid] = caption_counter
        label_end_idx[imgid] = caption_counter + n - 1
        caption_counter += n

    labels = np.concatenate(label_arrays, axis=0)
    label_length = np.array(label_length, dtype=np.int64)

    print(f'Encoded captions array size: {labels.shape}')
    print(f'Vocabulary size: {len(word_to_idx)}')
    print(f'Max sequence length: {max_length}')

    with open(args.output_vocab_json, 'w', encoding='utf-8') as f:
        json.dump(word_to_idx, f, ensure_ascii=False, indent=2)

    with open(args.output_split_json, 'w', encoding='utf-8') as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    with open(args.output_caption_json, 'w', encoding='utf-8') as f:
        json.dump(captions_by_image, f, ensure_ascii=False, indent=2)

    coco_anno = build_coco_annotations(captions_by_image)
    with open(args.output_coco_anno, 'w', encoding='utf-8') as f:
        json.dump(coco_anno, f, ensure_ascii=False, indent=2)

    with h5py.File(args.output_h5, 'w') as f:
        f.create_dataset('labels', data=labels)
        f.create_dataset('label_start_idx', data=label_start_idx)
        f.create_dataset('label_end_idx', data=label_end_idx)
        f.create_dataset('label_length', data=label_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_captions_json', default='./Levir-CC/LevirCCcaptions.json')
    parser.add_argument('--input_vocab_json', default='')
    parser.add_argument('--output_split_json', default='./Levir-CC/splits.json')
    parser.add_argument('--output_caption_json', default='./Levir-CC/levir_cc_captions.json')
    parser.add_argument('--output_vocab_json', default='./Levir-CC/transformer_levir_vocab.json')
    parser.add_argument('--output_h5', default='./Levir-CC/transformer_levir_labels.h5')
    parser.add_argument('--output_coco_anno', default='./Levir-CC/levir_cc_captions_reformat.json')
    parser.add_argument('--word_count_threshold', default=1, type=int)
    parser.add_argument('--allow_unk', default=1, type=int)
    main(parser.parse_args())
