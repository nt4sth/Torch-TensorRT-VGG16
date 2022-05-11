import torch
import torchvision
import torch_tensorrt

import os
import argparse

from vgg16 import vgg16
from torchvision import transforms

PARSER = argparse.ArgumentParser(description="VGG16 example to use with Torch-TensorRT PTQ")
PARSER.add_argument('--ckpt-dir',
                    default="./vgg16_ckpts/",
                    type=str,
                    help="Path to save checkpoints (saved every 10 epochs)")

args = PARSER.parse_args()

num_classes = 10

testing_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])
)

testing_dataloader = torch.utils.data.DataLoader(
    testing_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1
)

calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    testing_dataloader,
    cache_file='./calibration.cache',
    use_cache=False,
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    evice=torch.device('cuda:0')
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

model = vgg16(num_classes=num_classes, init_weights=False)
ckpt_file = args.ckpt_dir + '/ckpt_epoch' + 100 + '.pth'
print('Loading from checkpoint {}'.format(ckpt_file))
assert (os.path.isfile(ckpt_file))
ckpt = torch.load(ckpt_file)
model.load_state_dict(ckpt["model_state_dict"])
trt_mod = torch_tensorrt.compile(model, compile_spec)

