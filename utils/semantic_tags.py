import os
import re

import numpy as np


_SPACE_RE = re.compile(r'\s+')
_NON_WORD_RE = re.compile(r'[^a-z0-9_]+')

_PHRASE_REPLACEMENTS = [
    ('parking lots', 'parking_lot'),
    ('parking lot', 'parking_lot'),
    ('bare lands', 'bare_land'),
    ('bare land', 'bare_land'),
    ('green lands', 'vegetation'),
    ('green land', 'vegetation'),
    ('newly built', 'construct'),
]

_TOKEN_REPLACEMENTS = {
    'buildings': 'building',
    'roads': 'road',
    'trees': 'tree',
    'waters': 'water',
    'grasses': 'grass',
    'lands': 'land',
    'houses': 'house',
    'parking_lots': 'parking_lot',
    'playgrounds': 'playground',
    'farmlands': 'farmland',
    'plants': 'vegetation',
    'plant': 'vegetation',
    'built': 'construct',
    'constructed': 'construct',
    'constructing': 'construct',
    'construction': 'construction construct',
    'new': 'appear',
    'newly': 'appear',
    'add': 'appear',
    'adds': 'appear',
    'added': 'appear',
    'additional': 'appear',
    'appeared': 'appear',
    'appears': 'appear',
    'appearing': 'appear',
    'removed': 'remove',
    'removing': 'remove',
    'disappeared': 'remove',
    'disappears': 'remove',
    'disappearing': 'remove',
    'missing': 'remove',
    'gone': 'remove',
    'changed': 'change',
    'changes': 'change',
    'changing': 'change',
    'larger': 'increase',
    'increases': 'increase',
    'increased': 'increase',
    'increasing': 'increase',
    'smaller': 'decrease',
    'decreases': 'decrease',
    'decreased': 'decrease',
    'decreasing': 'decrease',
    'expanded': 'expand',
    'expansion': 'expand',
    'expansions': 'expand',
    'extend': 'expand',
    'extended': 'expand',
    'extending': 'expand',
    'replaced': 'replace',
    'replacing': 'replace',
    'converted': 'replace',
    'conversion': 'replace',
}


def _collapse_spaces(text):
    return _SPACE_RE.sub(' ', text).strip()


def _replace_phrase(text, phrase, replacement):
    pattern = r'\b' + re.escape(phrase).replace(r'\ ', r'\s+') + r'\b'
    return re.sub(pattern, replacement, text)


def read_semantic_tags(tag_file):
    if not os.path.exists(tag_file):
        raise FileNotFoundError(
            'Semantic tag file not found: %s' % os.path.abspath(tag_file)
        )
    with open(tag_file, 'r', encoding='utf-8') as f:
        tags = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    if not tags:
        raise ValueError('Semantic tag file is empty: %s' % os.path.abspath(tag_file))
    return tags


def normalize_caption_for_semantic_tags(text, normalize_synonyms=True):
    text = text.lower()
    text = _NON_WORD_RE.sub(' ', text)
    text = _collapse_spaces(text)
    if not normalize_synonyms:
        return text

    for phrase, replacement in _PHRASE_REPLACEMENTS:
        text = _replace_phrase(text, phrase, replacement)

    normalized_tokens = []
    for token in text.split():
        replacement = _TOKEN_REPLACEMENTS.get(token, token)
        normalized_tokens.extend(replacement.split())
    return _collapse_spaces(' '.join(normalized_tokens))


def build_semantic_label(captions, semantic_tags, normalize_synonyms=True):
    tokens = set()
    for caption in captions:
        normalized = normalize_caption_for_semantic_tags(
            caption,
            normalize_synonyms=normalize_synonyms,
        )
        tokens.update(normalized.split())

    label = np.zeros(len(semantic_tags), dtype=np.float32)
    for idx, tag in enumerate(semantic_tags):
        normalized_tag = normalize_caption_for_semantic_tags(
            tag,
            normalize_synonyms=False,
        )
        if tag in tokens or normalized_tag in tokens:
            label[idx] = 1.0
    return label
