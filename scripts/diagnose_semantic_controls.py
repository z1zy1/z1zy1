#!/usr/bin/env python
"""Fixed validation sample forward diagnostics; never a training ablation."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys

PROJECT = Path(os.environ.get('PROJECT_DIR') or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--sample-seed', type=int, default=1111)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    os.chdir(PROJECT)
    import torch
    from configs.config_transformer import cfg, merge_cfg_from_file
    from datasets.rcc_dataset_transformer_levir import RCCDataset
    from models.CARD import CARD
    from utils.checkpointing import load_checkpoint_file, split_model_states
    from utils.checkpoint_integrity import sha256_file
    from utils.semantic_control_audit import write
    from utils.seed import seeded_initialization
    if args.count < 1 or Path(args.output).exists():
        raise ValueError('Need positive sample count and a new output file')
    merge_cfg_from_file(args.cfg)
    cfg.data.return_dict = True
    dataset = RCCDataset(cfg, 'val')
    cfg.model.transformer_decoder.vocab_size = dataset.get_vocab_size()
    cfg.model.transformer_decoder.seq_length = dataset.get_max_seq_length()
    with seeded_initialization(args.sample_seed):
        model = CARD(cfg).to(args.device).eval()
    state, _ = split_model_states(load_checkpoint_file(args.checkpoint))
    model.load_state_dict(state, strict=True)
    records, active = [], {}

    def observe(module, inputs, output):
        query = inputs[0]
        active.update(gamma=float(module.gamma.detach()),
            gate=module.last_reliability_gate.cpu().tolist(),
            coverage=module.last_change_coverage.cpu().tolist(),
            residual_l2=float((output-query).norm()), query_l2=float(query.norm()),
            relative_residual=float((output-query).norm()/query.norm().clamp_min(1e-12)))

    handles = [m.register_forward_hook(observe) for m in model.modules()
               if m.__class__.__name__ == 'SemanticCrossAttentionFusion']
    indices = sorted(random.Random(args.sample_seed).sample(range(len(dataset)), min(args.count, len(dataset))))
    with torch.no_grad():
        for index in indices:
            active = {}
            sample = dataset[index]
            def tensor(key):
                value = sample.get(key)
                return value.unsqueeze(0).to(args.device) if torch.is_tensor(value) else None
            model(tensor('feature_before'), tensor('feature_after'),
                  semantic_before=tensor('semantic_before'), semantic_after=tensor('semantic_after'),
                  semantic_diff=tensor('semantic_diff'), semantic_confidence=tensor('semantic_confidence'))
            records.append({'sample_id': sample['image_id'], 'dataset_index': index, **active})
    for handle in handles:
        handle.remove()
    write(args.output, {'split': 'val', 'sample_seed': args.sample_seed,
          'checkpoint_sha256': sha256_file(args.checkpoint), 'config_sha256': sha256_file(args.cfg),
          'records': records, 'interpretation': 'Forward behavior only; not a retrained ablation'}, immutable=True)


if __name__ == '__main__':
    main()
