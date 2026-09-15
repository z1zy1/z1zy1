"""Frozen configuration and initialization for the B/D/C0/G controls.

Legacy protocols deliberately do not use this module's initialization.
"""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

PROTOCOL = 'p1_semantic_controls_20260915'
P1_PROTOCOLS = ('p1_rsaca_20260913', PROTOCOL)
ARMS = ('card', 'plain_fusion', 'rsaca', 'fixed_gate')
DATASETS = ('levir_cc', 'levir_mci', 'second_cc')
SEEDS = (1111, 2222, 3333)
METRICS = ('Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')
REFERENCE = dict(zip(METRICS, (0.4375, 0.3377, 0.6942, 1.2299, 0.2607)))
ROOTS = dict(zip(DATASETS, ('./Levir-CC', './LEVIR-MCI-dataset', './SECOND-CC-AUG')))
ENV_ROOTS = dict(zip(DATASETS, ('LEVIR_CC_ROOT', 'LEVIR_MCI_ROOT', 'SECOND_CC_ROOT')))
ENV_FEATURES = dict(zip(DATASETS, ('LEVIR_CC_FEATURE_ROOT', 'LEVIR_MCI_FEATURE_ROOT', 'SECOND_CC_FEATURE_ROOT')))


def configuration(project, run_root, dataset, seed, arm, environment):
    """Use existing dataset layout helpers; never infer host-specific paths."""
    import yaml
    from configs.config_transformer import cfg as defaults
    from utils.attr_dict import AttrDict
    from utils.dataset_config import apply_dataset_cli_overrides
    from utils.experiment_tracking import sync_wcsg_config_aliases
    if dataset not in DATASETS or seed not in SEEDS or arm not in ARMS:
        raise ValueError('Unknown controls dataset, seed or arm')
    cfg = copy.deepcopy(defaults)

    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict):
                merge(target.setdefault(key, AttrDict()), value)
            else:
                target[key] = value

    # All core settings originate from ONE template, with only text-interface
    # defaults taken from the dataset-specific file (runtime vocabulary wins).
    config_dir = Path(project) / 'configs/dynamic'
    merge(cfg, yaml.safe_load((config_dir/'transformer_levir_cc_sgc_card.yaml').read_text(encoding='utf-8')))
    dataset_file = {'levir_cc': 'transformer_levir_cc_sgc_card.yaml',
                    'levir_mci': 'transformer_levir_mci_sgc_card.yaml',
                    'second_cc': 'transformer_second_cc_aug_sgc_card.yaml'}[dataset]
    decoder = yaml.safe_load((config_dir/dataset_file).read_text(encoding='utf-8'))['model']['transformer_decoder']
    for key in ('vocab_size', 'seq_length'):
        cfg.model.transformer_decoder[key] = decoder[key]
    root = environment.get(ENV_ROOTS[dataset]) or ROOTS[dataset]
    features = environment.get(ENV_FEATURES[dataset]) or str(Path(root)/'features')
    apply_dataset_cli_overrides(SimpleNamespace(dataset=dataset, data_root=root, feature_root=features), cfg)
    cfg.exp_dir = str(Path(run_root)/arm)
    cfg.exp_name = '%s_%s_seed%d' % (arm, dataset, seed)
    cfg.gpu_id = [int(environment.get('PYTORCH_GPU') or 0)]
    cfg.data.num_workers = int(environment.get('NUM_WORKERS') or 8)
    cfg.data.seed_workers = cfg.data.isolate_loader_rng = True
    cfg.data.train.batch_size = 32
    cfg.data.train.seq_per_img = 1
    cfg.data.num_semantic_classes = cfg.model.num_semantic_classes = 7
    cfg.data.num_mask_classes = cfg.model.num_mask_classes = 1
    cfg.data.mask_type = 'binary'
    cfg.data.use_change_mask = False
    cfg.data.use_semantic_maps = arm != 'card'
    for key in ('semantic_map_root', 'semantic_before_phase', 'semantic_after_phase',
                'semantic_diff_root', 'semantic_diff_phase', 'semantic_diff_confidence_root',
                'semantic_diff_confidence_phase'):
        cfg.data[key] = ''
    cfg.data.semantic_diff_only = dataset != 'second_cc' and arm != 'card'
    cfg.data.semantic_diff_binary = dataset == 'levir_cc' and arm != 'card'
    cfg.data.semantic_unknown_change_class = 6
    cfg.data.allow_missing_pseudo_mask = dataset == 'levir_cc'
    if arm != 'card':
        if dataset == 'levir_cc':
            cfg.data.semantic_diff_root = str(Path(root)/'pseudo_masks')
        elif dataset == 'levir_mci':
            cfg.data.semantic_diff_root = str(Path(root)/'images')
            cfg.data.semantic_diff_phase = 'label'
        else:
            cfg.data.semantic_map_root = root
            cfg.data.semantic_before_phase, cfg.data.semantic_after_phase = 'sem/A', 'sem/B'
    cfg.model.enable_aux_mask = cfg.model.use_aux_mask = False
    cfg.model.type = 'sgc_card'
    cfg.model.semantic_input_mode = 'none' if arm == 'card' else 'cross_attention'
    cfg.model.semantic_fusion_heads = 8
    cfg.model.semantic_fusion_dropout = 0.1
    cfg.model.semantic_fusion_gamma_init = 0.01
    cfg.model.semantic_fusion_gamma_max = 0.5
    cfg.model.semantic_fusion_norm_mode = 'legacy_post_norm'
    cfg.model.semantic_fusion_sparse_change_tokens = arm in ('rsaca', 'fixed_gate')
    cfg.model.semantic_fusion_reliability_gate = arm in ('rsaca', 'fixed_gate')
    cfg.model.semantic_fusion_global_token = arm in ('rsaca', 'fixed_gate')
    cfg.model.semantic_fusion_gate_whole_adapter = arm in ('rsaca', 'fixed_gate')
    cfg.model.semantic_fusion_fixed_nonempty_gate = arm == 'fixed_gate'
    cfg.model.semantic_fusion_global_token_mode = 'all_mean'
    cfg.model.semantic_fusion_reliability_gate_bias = -1.5
    cfg.model.semantic_fusion_detach_reliability_inputs = False
    cfg.model.semantic_fusion_visual_consistency_gate = False
    cfg.model.semantic_fusion_visual_fallback = False
    cfg.model.semantic_fusion_warmup_steps = 0
    for key in ('use_aux_mask', 'use_semantic_aux', 'use_aux_semantic', 'use_feature_reweight',
                'use_semantic_hard_gate', 'use_weak_mask_prior', 'use_relation_aux',
                'use_content_word_weight', 'use_content_word_weighted_ce', 'finetune_decoder_only',
                'use_semantic_detach'):
        cfg.train[key] = False
    cfg.train.use_semantic_cross_attention = arm != 'card'
    cfg.train.use_partial_detach = cfg.train.use_semantic_partial_detach = arm != 'card'
    cfg.train.semantic_detach_ratio = 0.0 if arm == 'card' else 0.5
    cfg.train.lambda_mask = cfg.train.lambda_semantic = 0.0
    cfg.train.init_checkpoint = ''
    cfg.train.protocol_id = PROTOCOL
    cfg.train.seed = seed
    cfg.train.validation_greedy = cfg.train.paper_selection_mode = True
    cfg.train.selection_strategy = 'five_metric_equal_weight_log'
    cfg.train.max_iter = 10000
    cfg.train.total_steps = 10000
    cfg.train.finetune_steps = 0
    cfg.train.snapshot_interval = cfg.train.save_interval = cfg.train.eval_interval = 1000
    cfg.train.grad_clip = -1.0
    cfg.train.optim.update(type='adam', lr=0.0002, alpha=0.9, beta=0.999,
                           epsilon=1e-8, weight_decay=0.0, scheduler='warmup_cosine',
                           warmup_steps=500, min_lr_ratio=0.05, pretrained_lr_scale=1.0)
    sync_wcsg_config_aliases(cfg)
    return cfg


def parameter_summary(module):
    return {name: {'shape': list(p.shape), 'dtype': str(p.dtype),
                   'sha256': hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for name, p in module.named_parameters()}


def build_control_models(cfg, card_factory, speaker_factory):
    """Copy random initial tensors, including constructor-wide Xavier resets."""
    from utils.seed import seeded_initialization
    if cfg.train.protocol_id != PROTOCOL or cfg.train.init_checkpoint:
        raise ValueError('Controls initialization requires fresh controls protocol')
    seed = int(cfg.train.seed)
    public_cfg = copy.deepcopy(cfg)
    public_cfg.model.semantic_input_mode = 'none'
    public_cfg.data.use_semantic_maps = False
    public_cfg.train.use_semantic_cross_attention = False
    with seeded_initialization(seed + 1001):
        public = card_factory(public_cfg)
    public_state = public.state_dict()
    if cfg.model.semantic_input_mode == 'none':
        detector = public
    else:
        template_cfg = copy.deepcopy(cfg)
        for key in ('semantic_fusion_sparse_change_tokens', 'semantic_fusion_reliability_gate',
                    'semantic_fusion_global_token', 'semantic_fusion_gate_whole_adapter'):
            template_cfg.model[key] = True
        template_cfg.model.semantic_fusion_fixed_nonempty_gate = False
        with seeded_initialization(seed + 3001):
            template = card_factory(template_cfg)
        with seeded_initialization(seed + 4001):
            detector = card_factory(cfg)
        template_state = template.state_dict()
        state = detector.state_dict()
        for name, value in state.items():
            source = public_state[name] if name in public_state else template_state[name]
            if source.shape != value.shape:
                raise ValueError('Shared initialization shape mismatch: ' + name)
            state[name] = source.clone()
        detector.load_state_dict(state, strict=True)
        for name, value in public_state.items():
            if not value.equal(detector.state_dict()[name]):
                raise RuntimeError('Public initialization mismatch: ' + name)
    with seeded_initialization(seed + 2001):
        public_speaker = speaker_factory(cfg)
    with seeded_initialization(seed + 5001):
        speaker = speaker_factory(cfg)
    speaker.load_state_dict({name: value.clone() for name, value in public_speaker.state_dict().items()}, strict=True)
    return detector, speaker
