import os
import torch
import torch_tensorrt

import argparse


def make_parser():
    parser = argparse.ArgumentParser(description='VGG16 PTQ deploy')
    parser.add_argument('--ptq-file',
                        default='/hy-nas/ptq_vgg16.ts',
                        type=str,
                        help='Path of ptq model')
    parser.add_argument('--output-dir',
                        default='./ckpts/',
                        type=str,
                        help='Path to save PTQ model')
    return parser


def main():
    args = make_parser().parse_args()
    print('Loading from ', args.ptq_file)
    trt_model = torch.jit.load(args.ptq_file)
    print(trt_model)


if __name__ == '__main__':
    main()
