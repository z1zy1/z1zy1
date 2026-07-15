import os


DATASET_ALIASES = {
    'levir_cc': 'rcc_dataset_transformer_levir',
    'levir-cc': 'rcc_dataset_transformer_levir',
    'levir_mci': 'rcc_dataset_transformer_levir',
    'levir-mci': 'rcc_dataset_transformer_levir',
    'second_cc': 'rcc_dataset_transformer_levir',
    'second-cc': 'rcc_dataset_transformer_levir',
}


def canonical_dataset_name(dataset_name):
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def _norm(path):
    return os.path.normpath(path)


def _join(*parts):
    return _norm(os.path.join(*parts))


def _configure_levir_cc(cfg, root, feature_root):
    root = _norm(root)
    cfg.data.data_root = root
    cfg.data.default_feature_dir = _norm(feature_root or _join(root, 'features'))
    cfg.data.semantic_feature_dir = _norm(feature_root or _join(root, 'features'))
    cfg.data.default_img_dir = _join(root, 'images')
    cfg.data.semantic_img_dir = _join(root, 'images')
    cfg.data.default_phase = 'A'
    cfg.data.semantic_phase = 'B'
    cfg.data.pseudo_mask_root = _join(root, 'pseudo_masks')
    cfg.data.caption_json = _join(root, 'LevirCCcaptions.json')
    cfg.data.changeflag_json = cfg.data.caption_json
    cfg.data.eval_anno_path = _join(root, 'levir_cc_captions_reformat.json')
    cfg.data.splits_json = _join(root, 'splits.json')
    cfg.data.vocab_json = _join(root, 'transformer_levir_vocab.json')
    cfg.data.h5_label_file = _join(root, 'transformer_levir_labels.h5')


def _configure_levir_mci(cfg, root, feature_root):
    root = _norm(root)
    cfg.data.data_root = root
    cfg.data.default_feature_dir = _norm(feature_root or _join(root, 'features'))
    cfg.data.semantic_feature_dir = _norm(feature_root or _join(root, 'features'))
    cfg.data.default_img_dir = _join(root, 'images')
    cfg.data.semantic_img_dir = _join(root, 'images')
    cfg.data.default_phase = 'A'
    cfg.data.semantic_phase = 'B'
    cfg.data.mask_label_dir = _join(root, 'images')
    cfg.data.mask_label_rgb_dir = _join(root, 'images')
    cfg.data.pseudo_mask_root = _join(root, 'images')
    cfg.data.pseudo_mask_phase = 'label'
    cfg.data.caption_json = _join(root, 'LevirCCcaptions.json')
    cfg.data.changeflag_json = cfg.data.caption_json
    cfg.data.eval_anno_path = _join(root, 'levir_mci_captions_reformat.json')
    cfg.data.splits_json = _join(root, 'splits.json')
    cfg.data.vocab_json = _join(root, 'transformer_levir_mci_vocab.json')
    cfg.data.h5_label_file = _join(root, 'transformer_levir_mci_labels.h5')
    cfg.data.num_mask_classes = int(getattr(cfg.data, 'num_mask_classes', 1) or 1)
    cfg.model.num_mask_classes = cfg.data.num_mask_classes


def _configure_second_cc(cfg, root, feature_root):
    root = _norm(root)
    cfg.data.data_root = root
    cfg.data.default_feature_dir = _norm(feature_root or _join(root, 'features'))
    cfg.data.semantic_feature_dir = _norm(feature_root or _join(root, 'features'))
    cfg.data.default_img_dir = root
    cfg.data.semantic_img_dir = root
    cfg.data.default_phase = os.path.join('rgb', 'A')
    cfg.data.semantic_phase = os.path.join('rgb', 'B')
    cfg.data.semantic_map_root = root
    cfg.data.semantic_before_phase = os.path.join('sem', 'A')
    cfg.data.semantic_after_phase = os.path.join('sem', 'B')
    cfg.data.pseudo_mask_root = _join(root, 'pseudo_masks')
    cfg.data.pseudo_mask_phase = ''
    cfg.data.caption_json = _join(root, 'SECOND-CC-AUG.json')
    cfg.data.changeflag_json = cfg.data.caption_json
    cfg.data.eval_anno_path = _join(root, 'second_cc_aug_captions_reformat.json')
    cfg.data.splits_json = _join(root, 'splits.json')
    cfg.data.vocab_json = _join(root, 'transformer_second_cc_aug_vocab.json')
    cfg.data.h5_label_file = _join(root, 'transformer_second_cc_aug_labels.h5')
    if not int(getattr(cfg.data, 'num_semantic_classes', 0) or 0):
        cfg.data.num_semantic_classes = 7
    cfg.model.num_semantic_classes = cfg.data.num_semantic_classes


def _infer_levir_mci_root(cfg):
    img_dir = _norm(cfg.data.default_img_dir)
    if os.path.basename(img_dir) == 'images':
        return os.path.dirname(img_dir)
    return './LEVIR-MCI-dataset'


def _infer_second_cc_root(cfg):
    return _norm(cfg.data.default_img_dir or './SECOND-CC-AUG')


def apply_dataset_cli_overrides(args, cfg):
    dataset = getattr(args, 'dataset', None)
    data_root = getattr(args, 'data_root', None)
    levir_mci_root = getattr(args, 'levir_mci_root', None)
    second_cc_root = getattr(args, 'second_cc_root', None)
    feature_root = getattr(args, 'feature_root', None)

    if dataset is not None:
        cfg.data.dataset = dataset
    elif levir_mci_root:
        cfg.data.dataset = 'levir_mci'
    elif second_cc_root:
        cfg.data.dataset = 'second_cc'

    if data_root:
        if cfg.data.dataset in ('levir_cc', 'levir-cc'):
            pass
        elif cfg.data.dataset in ('levir_mci', 'levir-mci'):
            levir_mci_root = data_root
        elif cfg.data.dataset in ('second_cc', 'second-cc'):
            second_cc_root = data_root
        else:
            cfg.data.data_root = _norm(data_root)

    should_rewrite_layout = dataset is not None or data_root or levir_mci_root or second_cc_root or feature_root
    if cfg.data.dataset in ('levir_cc', 'levir-cc') and should_rewrite_layout:
        cfg.data.dataset = 'levir_cc'
        _configure_levir_cc(cfg, data_root or './Levir-CC', feature_root)
    elif cfg.data.dataset in ('levir_mci', 'levir-mci') and should_rewrite_layout:
        cfg.data.dataset = 'levir_mci'
        _configure_levir_mci(cfg, levir_mci_root or _infer_levir_mci_root(cfg), feature_root)
    elif cfg.data.dataset in ('second_cc', 'second-cc') and should_rewrite_layout:
        cfg.data.dataset = 'second_cc'
        _configure_second_cc(cfg, second_cc_root or _infer_second_cc_root(cfg), feature_root)
    elif feature_root:
        cfg.data.default_feature_dir = _norm(feature_root)
        cfg.data.semantic_feature_dir = _norm(feature_root)

    for name in ('use_change_mask', 'use_semantic_maps', 'eval_change_nochange_split'):
        if getattr(args, name, False):
            setattr(cfg.data, name, True)
    if getattr(args, 'mask_type', None):
        cfg.data.mask_type = args.mask_type
    if getattr(args, 'num_mask_classes', None) is not None:
        cfg.data.num_mask_classes = args.num_mask_classes
        cfg.model.num_mask_classes = args.num_mask_classes
    if getattr(args, 'semantic_input_mode', None):
        cfg.model.semantic_input_mode = args.semantic_input_mode
    if getattr(args, 'num_semantic_classes', None) is not None:
        cfg.data.num_semantic_classes = args.num_semantic_classes
        cfg.model.num_semantic_classes = args.num_semantic_classes


def feature_extraction_command(cfg, batch_size=64):
    phases = '%s,%s' % (cfg.data.default_phase, cfg.data.semantic_phase)
    return (
        'python scripts/extract_change_dataset_features.py '
        '--input_root "%s" --output_root "%s" --phases "%s" --batch_size %d'
        % (cfg.data.default_img_dir, cfg.data.default_feature_dir, phases, batch_size)
    )
