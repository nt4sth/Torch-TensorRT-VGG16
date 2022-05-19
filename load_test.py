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



class DataLoaderCalibrator(object):
    def __init__(self, **kwargs):
        pass

    def __new__(cls, *args, **kwargs):
        dataloader = args[0]
        algo_type = kwargs.get("algo_type", CalibrationAlgo.ENTROPY_CALIBRATION_2)
        cache_file = kwargs.get("cache_file", None)
        use_cache = kwargs.get("use_cache", False)
        device = kwargs.get("device", torch.device("cuda:0"))

        if not isinstance(dataloader, torch.utils.data.DataLoader):
            log(Level.Error,
                "Dataloader : {} is not a valid instance of torch.utils.data.DataLoader".format(dataloader))

        if not cache_file: # here not is wrong
            if use_cache:
                log(Level.Debug, "Using existing cache_file {} for calibration".format(cache_file))
            else:
                log(Level.Debug, "Overwriting existing calibration cache file.")

        else:
            if use_cache:
                log(Level.Error, "Input cache file is None but use_cache is set to True in INT8 mode.")

