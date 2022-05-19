import os
import argparse
import torch
import torchvision
import torch_tensorrt

from vgg16 import vgg16
from torchvision import transforms


def make_parser():
    parser = argparse.ArgumentParser(description='VGG16 PTQ deploy')
    parser.add_argument('--ckpt-file',
                        default='./vgg16_ckpts/ckpt_epoch10.pth',
                        type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--output-dir',
                        default='./',
                        type=str,
                        help='Path to save PTQ model')
    parser.add_argument('--cache-file',
                        default='./calibration.cache',
                        type=str,
                        help='Path of cache file')
    return parser


def load_model(ckpt_file):
    model = vgg16(num_classes=10, init_weights=False)
    print('Loading from checkpoint {}'.format(ckpt_file))
    assert (os.path.isfile(ckpt_file))
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to('cuda')
    model.eval()
    model.cuda()
    return model


def calibrate_model(model, cache_file):
    calib_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ])
    )
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        calib_loader,
        cache_file=cache_file,
        # DataLoadCalibrator can not detect cache file if `use_cache` is set to True
        use_cache=True,
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
    traced_model = torch.jit.trace(model, torch.empty([1, 3, 32, 32]).to("cuda"))
    trt_model = torch_tensorrt.compile(traced_model, **compile_spec)
    return trt_model


def main():
    args = make_parser().parse_args()
    model = load_model(args.ckpt_file)
    trt_model = calibrate_model(model=model, cache_file=args.cache_file)
    trt_model.save(os.path.join(args.output_dir, 'ptq_vgg16.ts'))


if __name__ == '__main__':
    main()
