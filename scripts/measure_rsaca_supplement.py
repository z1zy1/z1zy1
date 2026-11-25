#!/usr/bin/env python3
"""Count model parameters and measure matched downstream CUDA inference."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils import rsaca_supplement as supplement
from utils.semantic_control_audit import read, write
from utils.semantic_controls import ABLATIONS, ARMS, DATASETS, SEEDS, supplement_configuration


def parameter_counts(detector, speaker):
    groups = defaultdict(lambda: dict(total=0, requires_grad=0))
    seen = set()
    for root, module in (('detector', detector), ('decoder', speaker)):
        for name, param in module.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            if root == 'decoder':
                group = root
            elif name.startswith(('semantic_embedding.', 'semantic_cross_fusion.class_embedding.',
                                  'semantic_cross_fusion.diff_projection.')):
                group = 'semantic_encoding'
            elif name.startswith('semantic_cross_fusion.'):
                group = 'adapter'
            else:
                group = 'card_core'
            for key in (group, 'total'):
                groups[key]['total'] += param.numel()
                groups[key]['requires_grad'] += param.numel() if param.requires_grad else 0
    return dict(groups)


def profile_macs(modules, operation):
    """Structural MACs. The coverage fields delimit the proxy."""
    import torch.nn as nn
    counts = defaultdict(int)
    hooks = []

    def count(module, inputs, output):
        if isinstance(module, nn.MultiheadAttention):
            q, k, v = inputs[:3]
            batch_first = getattr(module, 'batch_first', False)
            n = q.shape[0 if batch_first else 1]
            ql, kl = q.shape[1 if batch_first else 0], k.shape[1 if batch_first else 0]
            d = module.embed_dim
            counts['multihead_attention'] += n*(ql*(q.shape[-1]+d)+kl*(k.shape[-1]+v.shape[-1]))*d
            counts['multihead_attention'] += 2*n*ql*kl*d
        elif isinstance(module, nn.Linear):
            counts['linear'] += output.numel()*module.in_features
        elif isinstance(module, nn.Conv2d):
            counts['conv2d'] += output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]

    for root in modules:
        for name, module in root.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)) and not name.endswith('.out_proj'):
                hooks.append(module.register_forward_hook(count))
    try:
        operation()
    finally:
        for hook in hooks:
            hook.remove()
    total = sum(counts.values())
    return dict(mac=total, flops_2_per_mac=2*total, components=dict(counts))


def measure(operation, torch, device, warmup, repeats, batch):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter()-start)
    mean = statistics.mean(times)
    peak = torch.cuda.max_memory_allocated(device)
    return dict(mean_ms=mean*1000, median_ms=statistics.median(times)*1000,
                individual_ms=[v*1000 for v in times], pairs_per_second=batch/mean,
                peak_allocated_bytes=peak, extra_peak_allocated_bytes=peak-before)


def benchmark(cfg, item, ids, args, detector, speaker):
    import torch
    from datasets.datasets import create_dataset
    from utils.checkpointing import load_checkpoint_file, split_model_states
    from utils.checkpoint_integrity import sha256_file
    from utils.utils import decode_sequence_transformer

    if sha256_file(item['checkpoint']) != item['checkpoint_sha256']:
        raise ValueError('Benchmark checkpoint differs from frozen run')
    checkpoint = load_checkpoint_file(item['checkpoint'])
    detector_state, speaker_state = split_model_states(checkpoint)
    detector.load_state_dict(detector_state, strict=True)
    speaker.load_state_dict(speaker_state, strict=True)
    device = torch.device('cuda', cfg.gpu_id[0])
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for real timings')
    detector.to(device).eval()
    speaker.to(device).eval()
    cfg.data.test.batch_size = args.batch
    cfg.data.test.max_samples = args.batch
    cfg.data.num_workers = 0
    cfg.data.return_dict = True
    dataset, loader = create_dataset(cfg, 'test')
    batch = next(iter(loader))
    actual_ids = list(map(str, batch['image_id']))
    if list(map(os.path.basename, actual_ids)) != ids:
        raise ValueError('Batch differs from the preselected sample IDs')
    first, second = (batch[key].to(device) for key in ('feature_before', 'feature_after'))
    semantics = {}
    for key in ('semantic_before', 'semantic_after', 'semantic_diff', 'semantic_confidence'):
        if batch.get(key) is not None:
            semantics[key] = batch[key].to(device)
    detector.set_global_step(cfg.train.total_steps)

    def e2e(fixed_steps=None):
        memory = detector(first, second, **semantics)[0]
        return speaker.sample(memory, sample_max=1, fixed_steps=fixed_steps)[0]

    result = {'sample_ids': ids, 'step': item['step'], 'checkpoint_sha256': item['checkpoint_sha256']}
    with torch.no_grad():
        generated = decode_sequence_transformer(dataset.get_idx_to_word(), e2e()[:, 1:])
        result['generated_lengths'] = [len(text.split()) for text in generated]
        for name, steps in (('greedy', None), ('fixed_steps', args.fixed_steps)):
            def work():
                return e2e(steps)
            result[name] = measure(work, torch, device, args.warmup, args.repeats, args.batch)
            result[name]['operator_proxy'] = profile_macs((detector, speaker), work)
        adapter = detector.semantic_cross_fusion
        if adapter is not None:
            recorded = []
            hook = adapter.register_forward_pre_hook(lambda module, values: recorded.append(values[0]))
            try:
                detector(first, second, **semantics)
            finally:
                hook.remove()

            def adapter_only():
                return adapter(recorded[0], spatial_size=tuple(first.shape[-2:]),
                               detach_ratio=cfg.train.semantic_detach_ratio, **semantics)

            result['adapter'] = measure(adapter_only, torch, device, args.warmup, args.repeats, args.batch)
            result['adapter']['operator_proxy'] = profile_macs((adapter,), adapter_only)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--series-root', type=Path, help='O, S or new-full frozen series directory')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=1111)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--fixed-steps', type=int, default=10)
    args = parser.parse_args()
    if min(args.batch, args.warmup, args.repeats, args.fixed_steps) < 1:
        parser.error('Batch, warm-up, repetitions and fixed steps must be positive')
    from models.CARD import CARD
    from models.transformer_decoder import DynamicSpeaker

    frozen = supplement.read_frozen(args.series_root.resolve()) if args.series_root else None
    rows = []
    for arm in ARMS+ABLATIONS:
        item = None if frozen is None else frozen['runs'].get(
            'runs/%s/%s_%s_seed%d' % (arm, arm, args.dataset, args.seed))
        cfg = supplement.node(item['config']) if item else supplement_configuration(
            PROJECT, PROJECT/'experiments', args.dataset, args.seed, arm, os.environ)
        if item:
            run = supplement.training_run(cfg)
            supplement.admission_details(PROJECT, cfg, item['checkpoint'], run/'captions/test.json')
        actual = supplement.node(supplement.adapted_configuration(cfg))
        detector, speaker = CARD(actual), DynamicSpeaker(actual)
        row = dict(arm=arm, parameters=parameter_counts(detector, speaker),
                   vocab_size=actual.model.transformer_decoder.vocab_size,
                   sequence_length=actual.model.transformer_decoder.seq_length)
        if item:
            manifest = read(item['manifest'])
            ids = supplement.split_ids(manifest, 'test')[:args.batch]
            row['timing'] = benchmark(actual, item, ids, args, detector, speaker)
        rows.append(row)
    baseline = rows[0]['parameters']['total']
    for row in rows:
        row['delta_from_card'] = {k: row['parameters']['total'][k]-v for k, v in baseline.items()}
    write(args.output, dict(dataset=args.dataset, seed=args.seed, runtime=supplement.runtime(),
        batch=args.batch, precision='float32', warmup=args.warmup, repetitions=args.repeats,
        fixed_steps=args.fixed_steps, frozen_series=frozen['series'] if frozen else None, models=rows,
        cost_notes=['MAC proxy covers module Linear, Conv2d and MultiheadAttention projections and dot products.',
                    '2 FLOPs per MAC. Functional operations, loss, backward, optimizer, normalization, softmax and activation excluded.',
                    'GPU timing uses preloaded cached features and semantic tensors. I/O and host-to-device transfer excluded.',
                    'ResNet-101 feature extraction and upstream external map production unmeasured.',
                    'Local restrictions use dense attention. Fixed gate retains the unused gate MLP.']))


if __name__ == '__main__':
    main()
