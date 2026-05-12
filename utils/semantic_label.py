import json
import os
import re

import numpy as np
import torch


OBJECT_VOCAB = [
    "building",
    "house",
    "road",
    "tree",
    "vegetation",
    "bareland",
    "field",
    "water",
    "bridge",
    "parking_lot",
]

ACTION_VOCAB = [
    "added",
    "built",
    "constructed",
    "removed",
    "disappeared",
    "replaced",
    "changed",
    "expanded",
]

OBJECT_SYNONYMS = {
    "building": [
        "building",
        "buildings",
        "house",
        "houses",
        "villa",
        "villas",
        "residential building",
        "residential buildings",
        "bungalow",
        "bungalows",
        "cottage",
        "cottages",
        "warehouse",
        "warehouses",
        "factory",
        "factories",
    ],
    "house": ["house", "houses", "villa", "villas", "bungalow", "bungalows", "cottage", "cottages"],
    "road": ["road", "roads", "crossroad", "crossroads", "cross road", "street", "streets", "highway", "highways", "path", "paths"],
    "tree": ["tree", "trees", "forest", "forests", "woods", "woodland", "woodlands"],
    "vegetation": ["vegetation", "plants", "plant", "grass", "grasses", "grassland", "grasslands", "greenery", "lawn", "bush", "bushes", "shrubs"],
    "bareland": ["bare land", "bareland", "barelands", "open space", "land", "lands", "ground", "barren", "wasteland", "wastelands"],
    "field": ["field", "fields", "farmland", "farmlands", "crop field", "farm", "farms"],
    "water": ["water", "river", "rivers", "pond", "ponds", "lake", "lakes", "reservoir", "reservoirs", "pool", "pools"],
    "bridge": ["bridge", "bridges", "overpass", "overpasses", "flyover", "viaduct", "viaducts"],
    "parking_lot": ["parking lot", "parking lots", "parking area", "parking areas", "car park", "car parks"],
}

ACTION_SYNONYMS = {
    "added": ["added", "add", "additional", "appeared", "appear", "appears", "appearing", "new", "newly", "showed up", "shows up", "emerged", "emerge", "emerges"],
    "built": ["built", "build", "constructed", "developed", "erected", "established", "created", "formed", "made"],
    "constructed": ["constructed", "built", "newly built", "reconstructed", "construction"],
    "removed": ["removed", "remove", "cleared", "clearing", "cut down", "demolished", "eliminated", "knocked down", "razed", "pulled down", "wiped"],
    "disappeared": ["disappeared", "disappear", "disappears", "gone", "missing", "no longer", "is no longer", "are no longer", "vanished", "vanish", "vanishes"],
    "replaced": ["replaced", "replace", "replaces", "converted", "turned into", "changed into", "transformed"],
    "changed": ["changed", "change", "changes", "modified", "reshaped"],
    "expanded": ["expanded", "expand", "expands", "extended", "extends", "larger", "increased", "increase", "widened", "wider", "broadened", "enlarged"],
}

_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9_]+")
_CONTENT_TOKEN_STOPWORDS = {
    "a",
    "an",
    "are",
    "be",
    "by",
    "down",
    "in",
    "into",
    "is",
    "new",
    "newly",
    "no",
    "of",
    "the",
    "to",
    "up",
    "was",
    "were",
}


def _normalize_text(text):
    text = text.lower().replace("_", " ")
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _phrase_to_pattern(phrase):
    phrase = _normalize_text(phrase)
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.compile(r"\b" + escaped + r"\b")


def _match_any(text, phrases):
    return any(_phrase_to_pattern(phrase).search(text) for phrase in phrases)


def extract_caption_semantics(caption):
    """Return object/action indices mentioned in one caption."""
    normalized = _normalize_text(caption)
    object_ids = set()
    action_ids = set()

    for obj_idx, obj_name in enumerate(OBJECT_VOCAB):
        if _match_any(normalized, OBJECT_SYNONYMS[obj_name]):
            object_ids.add(obj_idx)
    for act_idx, act_name in enumerate(ACTION_VOCAB):
        if _match_any(normalized, ACTION_SYNONYMS[act_name]):
            action_ids.add(act_idx)

    return object_ids, action_ids, normalized


def _add_rule_relations(normalized_caption, relation):
    """Lightweight relation rules for common RSICC wording."""
    obj_index = {name: idx for idx, name in enumerate(OBJECT_VOCAB)}
    act_index = {name: idx for idx, name in enumerate(ACTION_VOCAB)}

    rule_pairs = [
        (["tree", "trees", "forest", "woods"], ["removed", "cleared", "cut down", "disappeared"], "removed", "tree"),
        (["building", "buildings", "house", "houses"], ["built", "constructed", "newly built"], "built", "building"),
        (["house", "houses", "villa", "villas"], ["appear", "appeared", "new"], "added", "house"),
        (["road", "roads", "street", "streets"], ["built", "constructed"], "built", "road"),
    ]
    for obj_terms, act_terms, action_name, object_name in rule_pairs:
        if _match_any(normalized_caption, obj_terms) and _match_any(normalized_caption, act_terms):
            relation[act_index[action_name], obj_index[object_name]] = 1.0

    if _match_any(normalized_caption, ["bare land", "bareland", "open space", "land"]):
        if _match_any(normalized_caption, ["replaced by building", "replaced by buildings", "turned into building", "turned into buildings", "changed into building", "changed into buildings"]):
            relation[act_index["replaced"], obj_index["bareland"]] = 1.0
            relation[act_index["built"], obj_index["building"]] = 1.0
            relation[act_index["added"], obj_index["building"]] = 1.0


def build_semantic_targets(captions):
    """Build object/action/relation multi-hot targets from reference captions."""
    object_targets = np.zeros(len(OBJECT_VOCAB), dtype=np.float32)
    action_targets = np.zeros(len(ACTION_VOCAB), dtype=np.float32)
    relation_targets = np.zeros((len(ACTION_VOCAB), len(OBJECT_VOCAB)), dtype=np.float32)

    caption_object_sets = []
    caption_action_sets = []
    normalized_captions = []
    for caption in captions:
        object_ids, action_ids, normalized = extract_caption_semantics(caption)
        caption_object_sets.append(object_ids)
        caption_action_sets.append(action_ids)
        normalized_captions.append(normalized)
        for obj_idx in object_ids:
            object_targets[obj_idx] = 1.0
        for act_idx in action_ids:
            action_targets[act_idx] = 1.0
        for act_idx in action_ids:
            for obj_idx in object_ids:
                relation_targets[act_idx, obj_idx] = 1.0
        _add_rule_relations(normalized, relation_targets)

    # Caption sets describe one image pair, so union-level co-occurrence is a useful weak label.
    union_objects = set().union(*caption_object_sets) if caption_object_sets else set()
    union_actions = set().union(*caption_action_sets) if caption_action_sets else set()
    for act_idx in union_actions:
        for obj_idx in union_objects:
            relation_targets[act_idx, obj_idx] = 1.0

    return {
        "objects": object_targets,
        "actions": action_targets,
        "relations": relation_targets,
    }


def semantic_targets_to_tensors(targets):
    return {
        "objects": torch.as_tensor(targets["objects"], dtype=torch.float32),
        "actions": torch.as_tensor(targets["actions"], dtype=torch.float32),
        "relations": torch.as_tensor(targets["relations"], dtype=torch.float32),
    }


def save_semantic_target_cache(cache_path, targets_by_key):
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    serializable = {}
    for key, targets in targets_by_key.items():
        serializable[str(key)] = {
            "objects": np.asarray(targets["objects"], dtype=np.float32).tolist(),
            "actions": np.asarray(targets["actions"], dtype=np.float32).tolist(),
            "relations": np.asarray(targets["relations"], dtype=np.float32).tolist(),
        }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def load_semantic_target_cache(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    targets_by_key = {}
    for key, targets in raw.items():
        targets_by_key[str(key)] = {
            "objects": np.asarray(targets["objects"], dtype=np.float32),
            "actions": np.asarray(targets["actions"], dtype=np.float32),
            "relations": np.asarray(targets["relations"], dtype=np.float32),
        }
    return targets_by_key


def build_content_word_token_ids(word_to_idx):
    """Map object/action canonical words and synonyms to word-level vocab ids."""
    content_words = set()
    for vocab_name, synonyms in list(OBJECT_SYNONYMS.items()) + list(ACTION_SYNONYMS.items()):
        content_words.update(_normalize_text(vocab_name).split())
        for phrase in synonyms:
            for token in _normalize_text(phrase).split():
                if token and token not in _CONTENT_TOKEN_STOPWORDS:
                    content_words.add(token)

    token_ids = set()
    for word in content_words:
        candidates = {word, word.replace(" ", "_")}
        for candidate in candidates:
            if candidate in word_to_idx:
                token_ids.add(int(word_to_idx[candidate]))
    return sorted(token_ids)
