import argparse, os, json
import numpy as np
# from scipy.misc import imread, imresize
from imageio import imread
import cv2
# from scipy.misc.pilutil import imread, imresize
# from matplotlib.pyplot import imread, imresize

import torch
import torchvision
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--recursive', action='store_true',
                    help='recursively scan input_image_dir and preserve relative paths in output_dir')

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def build_model(args):
  if not hasattr(torchvision.models, args.model):
    raise ValueError('Invalid model "%s"' % args.model)
  if not 'resnet' in args.model:
    raise ValueError('Feature extraction only supports ResNets')
  cnn = getattr(torchvision.models, args.model)(pretrained=True)
  layers = [
    cnn.conv1,
    cnn.bn1,
    cnn.relu,
    cnn.maxpool,
  ]
  for i in range(args.model_stage):
    name = 'layer%d' % (i + 1)
    layers.append(getattr(cnn, name))
  model = torch.nn.Sequential(*layers)
  # model = torchvision.models.resnet101()
  # model = torch.nn.Sequential(*(list(model.children())[:-2]))
  model.to(device)
  model.eval()
  return model


def run_batch(cur_batch, model):
  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
  std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

  image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
  image_batch = (image_batch / 255.0 - mean) / std
  image_batch = torch.FloatTensor(image_batch).to(device)

  with torch.no_grad():
      feats = model(image_batch)
  feats = feats.cpu().clone().numpy()

  return feats


def main(args):
  input_paths = []
  if args.recursive:
    for root, _, files in os.walk(args.input_image_dir):
      for fn in files:
        if not fn.lower().endswith('.png'):
          continue
        full_path = os.path.join(root, fn)
        rel_path = os.path.relpath(full_path, args.input_image_dir)
        input_paths.append((full_path, rel_path))
  else:
    for fn in os.listdir(args.input_image_dir):
      if not fn.lower().endswith('.png'):
        continue
      full_path = os.path.join(args.input_image_dir, fn)
      input_paths.append((full_path, fn))

  input_paths.sort(key=lambda x: x[1])
  if args.max_images is not None:
    input_paths = input_paths[:args.max_images]
  print(input_paths[0])
  print(input_paths[-1])

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  model = build_model(args)

  img_size = (args.image_height, args.image_width)

  i0 = 0
  cur_batch = []
  cur_path_batch = []
  for i, (path, rel_path) in enumerate(input_paths):
    # img = imread(path, pilmode='RGB')
    img = imread(path, format='tiff-pil') # read tiff images
    # img = cv2.resize(img, img_size, interp='bicubic')
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    img = img.transpose(2, 0, 1)[None]
    cur_batch.append(img)
    cur_path_batch.append(path)
    if len(cur_batch) == args.batch_size:
      feats = run_batch(cur_batch, model)
      for img_full_path, feat in zip(cur_path_batch, feats):
        feat = feat.squeeze()
        rel_output = os.path.relpath(img_full_path, args.input_image_dir)
        output_path = os.path.join(args.output_dir, rel_output)
        output_dirname = os.path.dirname(output_path)
        if output_dirname and not os.path.exists(output_dirname):
          os.makedirs(output_dirname, exist_ok=True)
        np.save(output_path, feat)

      i1 = i0 + len(cur_batch)
      i0 = i1
      print('Processed %d / %d images' % (i1, len(input_paths)))
      cur_batch = []
      cur_path_batch = []
  if len(cur_batch) > 0:
    feats = run_batch(cur_batch, model)
    for img_full_path, feat in zip(cur_path_batch, feats):
      feat = feat.squeeze()
      rel_output = os.path.relpath(img_full_path, args.input_image_dir)
      output_path = os.path.join(args.output_dir, rel_output)
      output_dirname = os.path.dirname(output_path)
      if output_dirname and not os.path.exists(output_dirname):
        os.makedirs(output_dirname, exist_ok=True)
      np.save(output_path, feat)
    i1 = i0 + len(cur_batch)
    print('Processed %d / %d images' % (i1, len(input_paths)))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
