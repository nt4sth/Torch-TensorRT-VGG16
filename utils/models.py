import torch
from .vgg16 import vgg16


def load_ptq_model(ckpt_file):
    print('Loading from {}'.format(ckpt_file))
    trt_model = torch.jit.load(ckpt_file)
    trt_model.cuda()
    trt_model.eval()
    return trt_model


def load_fp_model(ckpt_file):
    print('Loading from {}'.format(ckpt_file))
    model = vgg16(num_classes=10, init_weights=False)
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt["model_state_dict"])
    model.cuda()
    model.eval()
    return model
    