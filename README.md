# VGG16 Trained on CIFAR10

## Prequisites

```bash
pip3 install -r requirements.txt --user
```

Out:

     RuntimeError:
      ###########################################################################################
      The package you are trying to install is only a placeholder project on PyPI.org repository.
      This package is hosted on NVIDIA Python Package Index.
      
      This package can be installed as:
      ```
      $ pip install nvidia-pyindex
      $ pip install pytorch-quantization
      ```
      ###########################################################################################

Install above two packages with commands

Install from `requirments.txt` again, completed.

## Training

```bash
# link data
cd /path-to-your-cifar10
for a in * ; do ln -s /path-to-your-cifar10$a /project-root-path/data/$a ; done
for a in * ; do ln -s /hy-nas/datasets/CIFAR10/$a /hy-tmp/vgg16/data/$a ; done
```
The following recipe should get somewhere between 89-92% accuracy on the CIFAR10 testset
```bash
python3 main.py --lr 0.01 --batch-size 128 --drop-ratio 0.15 --ckpt-dir $(pwd)/vgg16_ckpts --epochs 100
```

