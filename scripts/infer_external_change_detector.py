#!/usr/bin/env python3
"""Run a binary remote-sensing change detector from an external repository."""

import argparse
import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class PairDataset(Dataset):
    def __init__(self, root, split, image_size):
        self.root = root
        self.split = split
        self.image_size = int(image_size)
        self.before_root = os.path.join(root, 'images', split, 'A')
        self.after_root = os.path.join(root, 'images', split, 'B')
        self.names = sorted(
            name for name in os.listdir(self.before_root)
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        )
        missing = [name for name in self.names if not os.path.exists(os.path.join(self.after_root, name))]
        if missing:
            raise FileNotFoundError('Missing paired B images, first examples: %s' % missing[:5])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        values = []
        for base in (self.before_root, self.after_root):
            image = Image.open(os.path.join(base, name)).convert('RGB')
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            values.append((tensor - 0.5) / 0.5)
        return values[0], values[1], name


def import_external_model(repo, model_name, checkpoint, device, embed_dim):
    repo = os.path.abspath(repo)
    sys.path.insert(0, repo)
    try:
        networks = importlib.import_module('models.networks')
        args = SimpleNamespace(net_G=model_name, embed_dim=int(embed_dim), n_class=2)
        model = networks.define_G(args=args, gpu_ids=[])
        payload = torch.load(checkpoint, map_location=device)
        state = payload.get('model_G_state_dict', payload.get('state_dict', payload))
        if any(key.startswith('module.') for key in state):
            state = {key.replace('module.', '', 1): value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError('Missing checkpoint keys for %s: %s' % (model_name, missing[:5]))
        if unexpected:
            print('Warning: ignored unexpected checkpoint keys: %s' % unexpected[:5])
        return model.to(device).eval()
    finally:
        sys.path.remove(repo)


def logits_to_probability(output):
    if isinstance(output, (list, tuple)):
        output = output[-1]
    if output.dim() != 4 or output.size(1) < 2:
        raise ValueError('Expected binary detector logits [B,2,H,W], got %s.' % (tuple(output.shape),))
    return F.softmax(output[:, :2], dim=1)[:, 1]


def main(args):
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    device = torch.device(args.device)
    model = import_external_model(args.repo, args.model, args.checkpoint, device, args.embed_dim)
    for split in [value.strip() for value in args.splits.split(',') if value.strip()]:
        dataset = PairDataset(args.dataset_root, split, args.image_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        output_dir = os.path.join(args.output_root, split)
        os.makedirs(output_dir, exist_ok=True)
        with torch.inference_mode():
            for before, after, names in loader:
                probabilities = logits_to_probability(model(before.to(device), after.to(device))).cpu().numpy()
                for probability, name in zip(probabilities, names):
                    output_path = os.path.join(output_dir, os.path.splitext(name)[0] + '.npy')
                    if os.path.exists(output_path) and not args.overwrite:
                        raise FileExistsError('Refusing to overwrite %s; pass --overwrite.' % output_path)
                    np.save(output_path, np.clip(probability.astype(np.float32), 0.0, 1.0))
        print('Finished %s: %d probability maps -> %s' % (split, len(dataset), output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset_root', default='./Levir-CC')
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--splits', default='train,val,test')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
