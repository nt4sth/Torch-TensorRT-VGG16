import os
from random import shuffle
import time
import argparse
import torch
import torchvision
import numpy as np
import torch_tensorrt

from vgg16 import vgg16
from torchvision import transforms
# from torch.utils.data.sampler import SubsetRandomSampler


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


def load_model(ckpt_file):
    model = vgg16(num_classes=10, init_weights=False)
    print('Loading from checkpoint {}\n'.format(ckpt_file))
    assert (os.path.isfile(ckpt_file))
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt["model_state_dict"])
    model.cuda()
    model.eval()
    return model


def get_calib_loader(
    batch_size,
    subsample_ratio,
    num_workers=4,
):
    print("Creating calibration dataloader.\n")
    train_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ])
    )
    num_train = len(train_set)
    calib_idx = list(range(int(subsample_ratio * num_train)))
    calib_set = torch.utils.data.Subset(train_set, calib_idx)
    calib_loader = torch.utils.data.DataLoader(
            dataset=calib_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
    return calib_loader


def calibrate_model(model, cache_file, subsample_ratio):
    calib_loader = get_calib_loader(
        batch_size=1,
        subsample_ratio=subsample_ratio,
    )

    if os.path.isfile(cache_file):
        print('Using cache file\n')
        calibrator = torch_tensorrt.ptq.CacheCalibrator(
            cache_file,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        )
    else:
        print('No cache file available.\n')
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            calib_loader,
            cache_file=cache_file,
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device('cuda:0')
        )

    compile_spec = {
             "inputs": [torch_tensorrt.Input((1, 3, 32, 32))],
             "enabled_precisions": {torch.float, torch.half, torch.int8},
             "calibrator": calibrator,
             "device": {
                 "device_type": torch_tensorrt.DeviceType.GPU,
                 "gpu_id": 0,
                 "dla_core": 0,
                 "allow_gpu_fallback": False,
                 "disable_tf32": False
            }
    }
    print("Start tracing model.\n")
    traced_model = torch.jit.trace(model, torch.empty([1, 3, 32, 32]).to("cuda"))

    print("Start calibrating.\n")
    torch.cuda.synchronize()
    start_time = time.time()
    trt_model = torch_tensorrt.compile(traced_model, **compile_spec)
    torch.cuda.synchronize()
    end_time = time.time()
    consumed_time = end_time - start_time
    print(f"Calibrating finished. {consumed_time:.2f}(s) consumed.\n")

    return trt_model, consumed_time


def main():
    args = make_parser().parse_args()
    model = load_model(args.ckpt_file)
    trt_model = calibrate_model(
        model=model,
        cache_file=args.cache_file,
        subsample_ratio=0.1,
    )

    ptq_file = os.path.join(args.output_dir, 'ptq_vgg16.ts')
    print("Saving quantized model to ", ptq_file)
    trt_model.save(ptq_file)


if __name__ == '__main__':
    main()
