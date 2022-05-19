import os
import time
import json
import torch
import argparse
import numpy as np

from torchvision import datasets, transforms
from utils.models import load_fp_model
from utils.PTQ import calibrate_model


def make_parser():
    parser = argparse.ArgumentParser(description='VGG16 sub-sample ratio test')
    parser.add_argument(
        '--fp-ckpt-file', default='/hy-nas/ckpt_epoch100.pth',
        type=str, help='Path to load float-point checkpoints'
        )
    parser.add_argument(
        "--output-dir", default="/hy-nas/",
        type=str, help="Path to save trt model."
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
    acc = correct / total
    avg_batch_time = np.mean(timings)*1000
    print(f'loss:{loss/total}\taccuracy: {acc}\t'
          f'average batch time (ms): {avg_batch_time:.2f}')
    return acc, avg_batch_time


def get_test_loader():
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
    return test_dataloader


def main():
    args = make_parser().parse_args()
    res = {
        "sample_ratio": [],
        "calib_time": [],
        "avg_batch_time": [],
        "test_acc": [],
    }
    sample_ratio = np.range(0, 1, 0.1) + 0.1
    model = load_fp_model(args.fp_ckpt_file)
    test_loader = get_test_loader()
    for i, ratio in enumerate(sample_ratio):
        print(f"[INFO] Exp {i+1} of {len(sample_ratio)}\tsub-sample ratio: {ratio}\n")
        trt_model, calib_time = calibrate_model(
            model=model,
            subsample_ratio=ratio,
        )
        acc, avg_batch_time = evaluate(
            model=trt_model,
            dataloader=test_loader,
        )
        res["sample_ratio"].append(ratio)
        res["calib_time"].append(calib_time)
        res["avg_batch_time"].append(avg_batch_time)
        res["test_acc"].append(acc)
    res_file = os.path.join(args.output_dir, res_file)
    log = open(res_file, "w")
    json.dump(res, log)
    log.close()
    print("Saving results to ", res_file)
    

if __name__ == "__main__":
    main()
