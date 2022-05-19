import time
import torch
import torch_tensorrt
import argparse
import numpy as np
import torch.nn.functional as F

from vgg16 import vgg16
from torchvision import datasets, transforms
from utils.models import load_fp_model, load_ptq_model


def make_parser():
    parser = argparse.ArgumentParser(description='VGG16 PTQ evaluation')
    parser.add_argument(
        '--fp-ckpt-file', default='/hy-nas/ckpt_epoch100.pth',
        type=str, help='Path to load fp checkpoints'
        )
    parser.add_argument(
        '--ptq-ckpt-file', default='/hy-nas/ptq_vgg16.ts',
        type=str, help='Path to load quantized checkpoints'
        )
    return parser


def evaluate(model, dataloader, crit):
    global classes
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []

    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []

    with torch.no_grad():
        for data, labels in dataloader:
            start_time = time.time()
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = model(data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)

            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            class_probs.append([F.softmax(i, dim=0) for i in out])
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    # test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    # test_preds = torch.cat(class_preds)
    print(f'loss:{loss/total}\taccuracy: {correct / total}\t'
          f'average batch time (ms): {np.mean(timings)*1000:.2f}')


def main():
    args = make_parser().parse_args()

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ]))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    crit = torch.nn.CrossEntropyLoss()

    print('Evaluating float-point model:')
    fp_model = load_fp_model(args.fp_ckpt_file)
    evaluate(model=fp_model, dataloader=test_dataloader, crit=crit)    

    print('Evaluating ptq model:')
    ptq_model = load_ptq_model(args.ptq_ckpt_file)
    evaluate(model=ptq_model, dataloader=test_dataloader, crit=crit)


if __name__ == '__main__':
    main()
