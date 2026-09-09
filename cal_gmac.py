"""
Calculate GMACs (and params) of a PaddleDetection model from a config + .pdparams checkpoint.

Usage:
    python cal_gmac.py -c configs/picodet/your_config.yml \
        -o weights=output/your_config/model_final.pdparams \
        --shape 320 480

If you omit -o weights=..., it uses the `weights:` field already set in the yml.
--shape takes H W (must match your Reader's target_size / Pad.size).
"""

import argparse
import sys

import paddle
import paddle.nn as nn

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.checkpoint import load_weight


class FlopsWrapper(nn.Layer):
    """
    paddle.flops() calls net(dummy_tensor) with a single positional tensor input.
    PaddleDetection architectures expect a dict: {'image', 'im_shape', 'scale_factor'}.
    This wrapper builds that dict from the single input tensor paddle.flops gives us.
    """

    def __init__(self, model, im_h, im_w):
        super().__init__()
        self.model = model
        self.im_h = im_h
        self.im_w = im_w

    def forward(self, image):
        n = image.shape[0]
        inputs = {
            'image': image,
            'im_shape': paddle.to_tensor(
                [[self.im_h, self.im_w]] * n, dtype='float32'),
            'scale_factor': paddle.to_tensor(
                [[1.0, 1.0]] * n, dtype='float32'),
        }
        return self.model(inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='path to yml config')
    parser.add_argument(
        '-o', '--opt', nargs='*', default=[],
        help="override config options, e.g. -o weights=path/to/model.pdparams")
    parser.add_argument(
        '--shape', nargs=2, type=int, metavar=('H', 'W'), default=None,
        help='input H W; defaults to TestReader.inputs_def.image_shape[-2:] from config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.opt:
        override = {}
        for item in args.opt:
            k, v = item.split('=', 1)
            override[k.strip()] = v.strip()
        merge_config(override)
        cfg = load_config(args.config)  # reload merged
        merge_config(override)

    if args.shape:
        h, w = args.shape
    else:
        try:
            h, w = cfg['TestReader']['inputs_def']['image_shape'][-2:]
        except KeyError:
            print("Could not infer input shape from config; pass --shape H W")
            sys.exit(1)

    print(f"Using input shape: 1x3x{h}x{w}")

    model = create(cfg.architecture)
    model.eval()

    weights_path = cfg.get('weights', None)
    if not weights_path:
        print("No weights found in config or -o override; computing "
              "GMACs on randomly-initialized weights (architecture-only, "
              "GMAC count is unaffected by weight values, so this is fine "
              "if you only care about compute cost, not accuracy).")
    else:
        load_weight(model, weights_path)
        print(f"Loaded weights from: {weights_path}")

    wrapper = FlopsWrapper(model, h, w)
    wrapper.eval()

    flops = paddle.flops(
        wrapper,
        input_size=[1, 3, h, w],
        print_detail=False,
    )

    # paddle.flops() reports FLOPs (1 MAC ~= 2 FLOPs for typical conv/linear layers)
    gflops = flops / 1e9
    gmacs = flops / 2 / 1e9

    print(f"\nInput size: 1x3x{h}x{w}")
    print(f"GFLOPs: {gflops:.4f}")
    print(f"GMACs:  {gmacs:.4f}")


if __name__ == '__main__':
    main()
