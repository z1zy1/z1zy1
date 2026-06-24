import argparse
import json
import os
import random
import textwrap


DATASET_CHOICES = ('levir_mci', 'second_cc')


def parse_args():
    parser = argparse.ArgumentParser(description='Create qualitative paper case sheets for LEVIR-MCI / SECOND-CC.')
    parser.add_argument('--dataset', required=True, choices=DATASET_CHOICES)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--result_json', required=True)
    parser.add_argument('--caption_json', default=None, help='Dataset JSON with ground-truth captions/changeflag.')
    parser.add_argument('--split', default='test')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--num_cases', type=int, default=12)
    parser.add_argument('--group', choices=['all', 'change', 'nochange'], default='all')
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


def load_results(path):
    with open(path, encoding='utf-8') as f:
        rows = json.load(f)
    return {os.path.basename(str(row.get('image_id'))): row.get('caption', '') for row in rows}


def load_caption_meta(path, split=None):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    items = data.get('images', data) if isinstance(data, dict) else data
    meta = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_split = item.get('split') or item.get('filepath')
        if split and item_split and str(item_split) != str(split):
            continue
        filename = item.get('filename') or item.get('image_id')
        if not filename:
            continue
        captions = []
        for sent in item.get('sentences', []):
            raw = sent.get('raw') if isinstance(sent, dict) else None
            if raw:
                captions.append(raw.strip())
        meta[os.path.basename(str(filename))] = {
            'changeflag': item.get('changeflag'),
            'captions': captions,
        }
    return meta


def candidate_paths(dataset, data_root, split, filename):
    if dataset == 'levir_mci':
        base = os.path.join(data_root, 'images', split)
        return {
            'before': os.path.join(base, 'A', filename),
            'after': os.path.join(base, 'B', filename),
            'aux': os.path.join(base, 'label_rgb', filename),
            'aux_fallback': os.path.join(base, 'label', filename),
        }
    base = os.path.join(data_root, split)
    return {
        'before': os.path.join(base, 'rgb', 'A', filename),
        'after': os.path.join(base, 'rgb', 'B', filename),
        'aux': os.path.join(base, 'sem', 'B', filename),
        'aux_fallback': os.path.join(data_root, 'pseudo_masks', split, filename),
    }


def resolve_existing(path, fallback=None):
    if path and os.path.exists(path):
        return path
    if fallback and os.path.exists(fallback):
        return fallback
    return None


def select_cases(results, meta, group, num_cases, seed):
    items = []
    for filename, caption in results.items():
        flag = meta.get(filename, {}).get('changeflag')
        if group == 'change' and flag != 1:
            continue
        if group == 'nochange' and flag != 0:
            continue
        items.append(filename)
    random.Random(seed).shuffle(items)
    return items[:num_cases]


def make_sheet(dataset, data_root, split, filename, generated, meta, output_path):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError('Pillow is required for visualization: %s' % exc)

    paths = candidate_paths(dataset, data_root, split, filename)
    before_path = resolve_existing(paths['before'])
    after_path = resolve_existing(paths['after'])
    aux_path = resolve_existing(paths['aux'], paths.get('aux_fallback'))
    if before_path is None or after_path is None:
        raise FileNotFoundError('Missing before/after image for %s' % filename)

    tile_size = (256, 256)
    images = []
    labels = []
    for label, path in [('Before', before_path), ('After', after_path), ('Label/Semantic', aux_path)]:
        if path is None:
            img = Image.new('RGB', tile_size, (245, 245, 245))
        else:
            img = Image.open(path).convert('RGB').resize(tile_size)
        images.append(img)
        labels.append(label)

    text_width = 360
    header_h = 28
    sheet_w = tile_size[0] * 3 + text_width
    sheet_h = tile_size[1] + header_h
    sheet = Image.new('RGB', (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('arial.ttf', 14)
        bold = ImageFont.truetype('arial.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
        bold = font

    for idx, img in enumerate(images):
        x = idx * tile_size[0]
        draw.text((x + 8, 6), labels[idx], fill=(20, 20, 20), font=bold)
        sheet.paste(img, (x, header_h))

    x = tile_size[0] * 3 + 14
    draw.text((x, 8), filename, fill=(20, 20, 20), font=bold)
    flag = meta.get(filename, {}).get('changeflag', 'N/A')
    y = 36
    draw.text((x, y), 'changeflag: %s' % flag, fill=(70, 70, 70), font=font)
    y += 28
    draw.text((x, y), 'Generated:', fill=(20, 20, 20), font=bold)
    y += 22
    for line in textwrap.wrap(generated or '', width=46)[:5]:
        draw.text((x, y), line, fill=(20, 20, 20), font=font)
        y += 18
    refs = meta.get(filename, {}).get('captions', [])
    if refs:
        y += 10
        draw.text((x, y), 'Reference:', fill=(20, 20, 20), font=bold)
        y += 22
        for line in textwrap.wrap(refs[0], width=46)[:4]:
            draw.text((x, y), line, fill=(50, 50, 50), font=font)
            y += 18

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sheet.save(output_path)


def main():
    args = parse_args()
    output_dir = os.path.normpath(args.output_dir or os.path.join('experiments', 'paper_visualizations', args.dataset))
    results = load_results(args.result_json)
    meta = load_caption_meta(args.caption_json, args.split)
    selected = select_cases(results, meta, args.group, args.num_cases, args.seed)
    manifest = []
    for filename in selected:
        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, '%s_%s.jpg' % (args.group, stem))
        try:
            make_sheet(args.dataset, args.data_root, args.split, filename, results.get(filename, ''), meta, output_path)
            status = 'ok'
            error = ''
        except Exception as exc:
            status = 'failed'
            error = str(exc)
        manifest.append({
            'filename': filename,
            'output_path': output_path,
            'status': status,
            'error': error,
        })
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({'output_dir': output_dir, 'manifest': manifest_path, 'cases': len(manifest)}, indent=2))


if __name__ == '__main__':
    main()
