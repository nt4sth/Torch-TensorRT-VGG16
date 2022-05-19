import os
from random import shuffle
import argparse
import numpy as np
import torch_tensorrt

from vgg16 import vgg16
from utils.PTQ import calibrate_model
from utils.models import load_fp_model


def make_parser():
    parser = argparse.ArgumentParser(description='VGG16 PTQ deploy')
    parser.add_argument('--ckpt-file',
                        default='/hy-nas/ckpt_epoch100.pth',
                        type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--output-dir',
                        default='/hy-nas/',
                        type=str,
                        help='Path to save PTQ model')
    parser.add_argument('--cache-file',
                        default='./calibration.cache',
                        type=str,
                        help='Path of cache file')
    return parser


def main():
    args = make_parser().parse_args()
    model = load_fp_model(args.ckpt_file)
    trt_model, _ = calibrate_model(
        model=model,
        cache_file=args.cache_file,
        subsample_ratio=0.1,
    )

    ptq_file = os.path.join(args.output_dir, 'ptq_vgg16.ts')
    print("Saving quantized model to ", ptq_file)
    trt_model.save(ptq_file)


if __name__ == '__main__':
    main()
