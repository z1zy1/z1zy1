import os
import re

import numpy as np


_SPACE_RE = re.compile(r'\s+')
_NON_WORD_RE = re.compile(r'[^a-z0-9_]+')

_PHRASE_REPLACEMENTS = [
    ('construction areas', 'construction_area'),
    ('construction area', 'construction_area'),
    ('parking lots', 'parking_lot'),
    ('parking lot', 'parking_lot'),
    ('parking areas', 'parking_lot'),
    ('parking area', 'parking_lot'),
    ('car parks', 'parking_lot'),
    ('car park', 'parking_lot'),
    ('bare lands', 'bare_land'),
    ('bare land', 'bare_land'),
    ('open spaces', 'bare_land'),
    ('open space', 'bare_land'),
    ('green lands', 'vegetation'),
    ('green land', 'vegetation'),
    ('grass lands', 'grass'),
    ('grass land', 'grass'),
    ('showed up', 'appear'),
    ('shows up', 'appear'),
    ('show up', 'appear'),
    ('newly built', 'appear construct'),
    ('no longer', 'remove'),
    ('pulled down', 'remove'),
    ('knocked down', 'remove'),
    ('cut down', 'remove'),
]

_TOKEN_REPLACEMENTS = {
    'building': 'building',
    'buildings': 'building',
    'warehouse': 'building',
    'warehouses': 'building',
    'factory': 'building',
    'factories': 'building',
    'road': 'road',
    'roads': 'road',
    'roadside': 'road',
    'roadsides': 'road',
    'street': 'road',
    'streets': 'road',
    'highway': 'road',
    'highways': 'road',
    'path': 'road',
    'paths': 'road',
    'tree': 'tree',
    'trees': 'tree',
    'forest': 'tree vegetation',
    'forests': 'tree vegetation',
    'woods': 'tree vegetation',
    'woodland': 'tree vegetation',
    'woodlands': 'tree vegetation',
    'water': 'water',
    'waters': 'water',
    'pond': 'water',
    'ponds': 'water',
    'lake': 'water',
    'lakes': 'water',
    'river': 'water',
    'rivers': 'water',
    'reservoir': 'water',
    'reservoirs': 'water',
    'pool': 'water',
    'pools': 'water',
    'grass': 'grass',
    'grasses': 'grass',
    'grassland': 'grass',
    'grasslands': 'grass',
    'land': 'land',
    'lands': 'land',
    'bareland': 'bare_land',
    'barelands': 'bare_land',
    'ground': 'land',
    'grounds': 'land',
    'house': 'house',
    'houses': 'house',
    'villa': 'house',
    'villas': 'house',
    'mansion': 'house',
    'mansions': 'house',
    'parking_lots': 'parking_lot',
    'playground': 'playground',
    'playgrounds': 'playground',
    'farmland': 'farmland',
    'farmlands': 'farmland',
    'field': 'farmland',
    'fields': 'farmland',
    'farm': 'farmland',
    'farms': 'farmland',
    'vegetation': 'vegetation',
    'plants': 'vegetation',
    'plant': 'vegetation',
    'greenery': 'vegetation',
    'shrubs': 'vegetation',
    'shrub': 'vegetation',
    'bushes': 'vegetation',
    'bush': 'vegetation',
    'construction_area': 'construction_area',
    'appear': 'appear',
    'appears': 'appear',
    'appearing': 'appear',
    'appeared': 'appear',
    'new': 'appear',
    'newly': 'appear',
    'add': 'appear',
    'adds': 'appear',
    'added': 'appear',
    'additional': 'appear',
    'emerge': 'appear',
    'emerges': 'appear',
    'emerged': 'appear',
    'remove': 'remove',
    'removed': 'remove',
    'removing': 'remove',
    'disappear': 'remove',
    'disappeared': 'remove',
    'disappears': 'remove',
    'disappearing': 'remove',
    'missing': 'remove',
    'gone': 'remove',
    'demolished': 'remove',
    'cleared': 'remove',
    'clearing': 'remove',
    'construct': 'construct',
    'constructs': 'construct',
    'built': 'construct',
    'build': 'construct',
    'builds': 'construct',
    'constructed': 'construct',
    'constructing': 'construct',
    'construction': 'construct',
    'reconstructed': 'construct change',
    'increase': 'increase',
    'increases': 'increase',
    'increased': 'increase',
    'increasing': 'increase',
    'more': 'increase',
    'larger': 'increase',
    'lusher': 'increase',
    'decrease': 'decrease',
    'decreases': 'decrease',
    'decreased': 'decrease',
    'decreasing': 'decrease',
    'less': 'decrease',
    'smaller': 'decrease',
    'expand': 'expand',
    'expands': 'expand',
    'expanded': 'expand',
    'expanding': 'expand',
    'extension': 'expand',
    'extensions': 'expand',
    'expansion': 'expand',
    'expansions': 'expand',
    'extend': 'expand',
    'extended': 'expand',
    'replace': 'replace',
    'replaces': 'replace',
    'replaced': 'replace',
    'replacing': 'replace',
    'converted': 'replace',
    'conversion': 'replace',
    'turn': 'replace',
    'turned': 'replace',
    'change': 'change',
    'changed': 'change',
    'changes': 'change',
    'changing': 'change',
    'modified': 'change',
}

_NO_CHANGE_PHRASES = [
    'no change',
    'no changes',
    'no difference',
    'unchanged',
    'nothing has changed',
    'same as before',
    'the scene is the same as before',
    'two scenes seem identical',
    'scenes seem identical',
    'seem identical',
    'are identical',
]


def _collapse_spaces(text):
    return _SPACE_RE.sub(' ', text).strip()


def _replace_phrase(text, phrase, replacement):
    pattern = r'\b' + re.escape(phrase).replace(r'\ ', r'\s+') + r'\b'
    return re.sub(pattern, replacement, text)


def _contains_phrase(text, phrase):
    pattern = r'\b' + re.escape(phrase).replace(r'\ ', r'\s+') + r'\b'
    return re.search(pattern, text) is not None


def _is_no_change_caption(text):
    return any(_contains_phrase(text, phrase) for phrase in _NO_CHANGE_PHRASES)


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
    if _is_no_change_caption(text):
        return 'no_change'

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
