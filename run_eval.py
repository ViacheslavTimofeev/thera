import json
import pickle
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from jax.image import resize
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from args.eval import parser
from data import ImageFolder
from model import build_thera
from utils import make_grid, compute_metrics
from vendor.matlab_bicubic import imresize as matlab_imresize

# Compatibility patch for older checkpoints pickled with older JAX.
import jax._src.core as jax_core
original_shaped_array_new = jax_core.ShapedArray.__new__


def patched_shaped_array_new(cls, shape, dtype, weak_type=False, **kwargs):
    kwargs.pop('named_shape', None)
    return original_shaped_array_new(cls, shape, dtype, weak_type, **kwargs)


jax_core.ShapedArray.__new__ = patched_shaped_array_new

MEAN = np.array([.4488, .4371, .4040])
VAR = np.array([.25, .25, .25])


def prepare_batch(target, scale):
    target = jnp.asarray(target)
    target = target.transpose((0, 2, 3, 1))

    source_h, source_w = int(target.shape[1] / scale), int(target.shape[2] / scale)
    target = target[:, :source_h * scale, :source_w * scale]
    target_t = jnp.float32(scale**(-2))[None]

    source_lr = matlab_imresize(target[0], output_shape=(source_h, source_w))[None]
    source = source_lr
    source_up = resize(source, target.shape, 'nearest')
    source = jax.nn.standardize(source, mean=MEAN, variance=VAR)

    return source, source_up, source_lr, target_t, target


def summarize_times(times):
    if not times:
        return {
            'count': 0,
            'min_seconds': None,
            'max_seconds': None,
            'avg_seconds': None,
            'total_seconds': 0.0,
        }

    return {
        'count': len(times),
        'min_seconds': float(np.min(times)),
        'max_seconds': float(np.max(times)),
        'avg_seconds': float(np.mean(times)),
        'total_seconds': float(np.sum(times)),
    }


def evaluate(val_loader, model, params, scale, border_crop,
             do_ensemble, save_dir: Optional[Path] = None, y_only=False,
             patch_size_dec=256, lr_save_dir: Optional[Path] = None):

    apply_encoder = jit(model.apply_encoder)
    apply_decoder = jit(model.apply_decoder)

    metrics = defaultdict(list)
    image_times = []
    for i_img, target in enumerate(tqdm(val_loader)):
        image_start = time.perf_counter()
        source, source_up, source_lr, target_t, target = prepare_batch(target, scale)

        if lr_save_dir is not None:
            lr_save_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.rint(np.array(source_lr[0] * 255)).astype(np.uint8))\
                .save(lr_save_dir / f'{i_img}.png')

        # memory scales in patch_size * scale, so we keep that factor constant
        patch_size = patch_size_dec // scale
        if patch_size < 1:
            raise ValueError('patch_size_dec must be at least as large as scale')
        if patch_size > min(source.shape[1:3]):
            patch_size = min(source.shape[1:3])

        target_coords = jnp.tile(make_grid(patch_size * scale), (target.shape[0], 1, 1, 1))

        outs = []

        for i_rot in range(4 if do_ensemble else 1):
            source_ = jnp.rot90(source, k=i_rot, axes=(-3, -2))
            source_up_ = jnp.rot90(source_up, k=i_rot, axes=(-3, -2))
            encoding = apply_encoder(params, source_)
            assert encoding.shape[:-1] == source_.shape[:-1]

            num_patches_h = (source_.shape[1] // patch_size) + 1
            num_patches_w = (source_.shape[2] // patch_size) + 1
            out = np.full_like(source_up_, np.nan, dtype=np.float32)

            for i, j in product(range(num_patches_h), range(num_patches_w)):
                h_min = min(i * patch_size, source_.shape[1] - patch_size)
                h_max = min((i + 1) * patch_size, source_.shape[1])
                w_min = min(j * patch_size, source_.shape[2] - patch_size)
                w_max = min((j + 1) * patch_size, source_.shape[2])
                encoding_p = encoding[:, h_min:h_max, w_min:w_max, :]
                out_p = apply_decoder(params, encoding_p, target_coords, target_t)
                out[:, scale * h_min:scale * h_max, scale * w_min:scale * w_max, :] = out_p

            assert not np.isnan(out).any()
            out = out * np.sqrt(VAR)[None, None, None] + MEAN[None, None, None]
            out += source_up_
            outs.append(np.rot90(out, k=i_rot, axes=(-2, -3)))

        out = np.stack(outs).mean(0).clip(0., 1.)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        image_times.append(time.perf_counter() - image_start)

        if save_dir is not None:
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.rint(np.array(out[0] * 255)).astype(np.uint8))\
                .save(save_dir / f'{i_img}.png')

        s = border_crop
        batch_metrics = compute_metrics(out[:, s:-s, s:-s], target[:, s:-s, s:-s], y_only=y_only)
        for k, v in batch_metrics.items():
            metrics[k] += [v.item()]

    return {k: np.mean(v) for k, v in metrics.items()}, {
        **summarize_times(image_times),
        'post_warmup': summarize_times(image_times[1:]),
    }


def main(args):
    data_set = ImageFolder(Path(args.data_dir) / args.eval_set, in_memory=False)
    data_loader = DataLoader(data_set, batch_size=1, num_workers=0, shuffle=False)

    with open(args.checkpoint, 'rb') as fh:
        check = pickle.load(fh)
        params, backbone, size = check['model'], check['backbone'], check['size']

    model = build_thera(3, backbone, size)
    report = {
        'checkpoint': str(args.checkpoint),
        'backbone': backbone,
        'size': size,
        'data_dir': str(args.data_dir),
        'eval_set': args.eval_set,
        'eval_scale': args.eval_scale,
        'save_dir': str(args.save_dir) if args.save_dir else None,
        'save_lr_dir': str(args.save_lr_dir) if args.save_lr_dir else None,
        'patch_size_dec': args.patch_size_dec,
        'geo_ensemble': not args.no_geo_ensemble,
        'y_only': args.y_only,
    }

    scale = args.eval_scale
    border_crop = scale + 6 if 'DIV2K' in args.eval_set else scale
    save_dir = (Path(args.save_dir) / ('ours_' + args.eval_set + '_' + backbone) / str(scale)) \
        if args.save_dir else None
    lr_save_dir = (Path(args.save_lr_dir) / args.eval_set / str(scale)) \
        if args.save_lr_dir else None

    metrics, timing = evaluate(data_loader, model, params, scale, border_crop,
        not args.no_geo_ensemble, save_dir, args.y_only, args.patch_size_dec, lr_save_dir)

    metrics = {k: np.round(v, 5) for k, v in metrics.items()}
    report.update({
        'border_crop': border_crop,
        'run_save_dir': str(save_dir) if save_dir else None,
        'lr_save_dir': str(lr_save_dir) if lr_save_dir else None,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'timing': timing,
    })
    print(f'[{args.eval_set} x{scale}] ' + ' '.join([f'{k}: {v}' for k, v in metrics.items()]))

    if args.report_file:
        report_file = Path(args.report_file)
    elif args.save_dir:
        report_file = Path(args.save_dir) / 'eval_report.json'
    else:
        report_file = Path('eval_report.json')

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f'Eval report written to {report_file}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
