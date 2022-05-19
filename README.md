# Torch-TensorRT Deployment on VGG16 as CIFAR10 Classifier 

## Installation

### Create virtual environment

```bash
pip install virtualenv
virtualenv venv
source ./venv/bin/activate
```

### Install PyTorch

Go to pytorch.org -> Install -> Start locally, select preferences and install using given `pip` command.

### Install TensorRT

```bash
# update setuptools
python3 -m pip install --upgrade setuptools pip

# update index
python3 -m pip install nvidia-pyindex

# install TensorRT
python3 -m pip install --upgrade nvidia-tensorrt
```

**Test**:

```python
python3
>>> import tensorrt
>>> print(tensorrt.__version__)
>>> assert tensorrt.Builder(tensorrt.Logger())
```

### Install Torch-TensorRT

dependencies: PyTorch, cuDNN, CUDA, TensorRT

```bash
pip3 install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases
```

**Test**:

```python
python3
>>> import tensorrt
>>> print(tensorrt.__version__)
```

### Load Repository

```bash
git clone https://github.com/QQQQ00243/Torch-TensorRT-VGG16.git
```

### Install required packages

```bash
pip3 install -r requirements.txt --user
```

### Prepare dataset

```bash
# link data
cd /path-to-your-cifar10
for a in * ; do ln -s /path-to-your-cifar10$a /project-root-path/data/$a ; done

# for example
cd /hy-nas/datasets/CIFAR10/
for a in * ; do ln -s /hy-nas/datasets/CIFAR10/$a /hy-tmp/VGG16/data/$a ; done
```

## Training

The following recipe should get somewhere between 89-92% accuracy on the CIFAR10 testset
```bash
python3 main.py --lr 0.01 --batch-size 128 --drop-ratio 0.15 --ckpt-dir $(pwd)/vgg16_ckpts --epochs 100
```

## Load pretrained model

model is stored in `/hy-nas/ckpt_epoch100.pth`, `ckpt-dir` is `/hy-nas/`

## QAT

```bash
python3 finetune_qat.py --lr 0.01 --batch-size 128 --drop-ratio 0.15 --ckpt-dir /hy-nas/ckpts/ --start-from 100 --epochs 110
```

## PTQ

You can set `cache-file` if you already have calibration file.

```bash
python3 PTQ.py --ckpt-file <path-to-your checkpoint file> --output-dir <directory-to-save-quantized-model> --cache-file <path-to-your-calibration-cache-file>
```

### Updates

* [2022/05/19] Evaluation result of VGG16 calibrated on test set of CIFAR10

| model | accuracy | average batch time (ms) |
| :---: | :------: | :---------------------: |
|  fp   |  0.9288  |          2.43           |
|  ptq  |  0.9296  |          1.61           |

**[Warning]** Model should not see test dataset before testing. Hence, it is better to subsample training dataset as calibration dataset.

* [2022/05/19] Add calibrating by sub-sampling training dataset



## Appendix: Common Issues

**Problem**: `libnvinfer_plugin.so.8: cannot open shared object file: No such file or directory` when importing `torch_tensorrt`

**Reason**: `libvinfer_plugin.so.8 ` suggests that there is something wrong with TensorRT 8.x, either it is incorrctly installed or the path is not properly linked

**Solution**: Find `libvinfer.so.8`, which is installed under `tensorrt` package.

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/hy-nas/venv/lib/python3.8/site-packages/tensorrt"
```



**Problem**:

evaluation batch size must be the same as that of calibration.

````bash
ERROR: [Torch-TensorRT] - 3: [executionContext.cpp::setBindingDimensions::943] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::setBindingDimensions::943, condition: profileMaxDims.d[i] >= dimensions.d[i]. Supplied binding dimension [32,3,32,32] for bindings[0] exceed min ~ max range at index 0, maximum dimension in profile is 1, minimum dimension in profile is 1, but supplied dimension is 32.
)
Traceback (most recent call last):
  File "eval_ptq.py", line 113, in <module>
    main()
  File "eval_ptq.py", line 109, in main
    evaluate(model=ptq_model, dataloader=test_dataloader, crit=crit)
  File "eval_ptq.py", line 64, in evaluate
    loss += crit(out, labels)
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/loss.py", line 1163, in forward
    return F.cross_entropy(input, target, weight=self.weight,
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/functional.py", line 2996, in cross_entropy
    return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
ValueError: Expected input batch_size (1) to match target batch_size (32).
````

**Reason**:



**Solution**:







**Problem**:

```bash
RuntimeError: 
Unknown type name '__torch__.torch.classes.tensorrt.Engine':
Serialized   File "code/__torch__/vgg16.py", line 4
  __parameters__ = []
  __buffers__ = []
  __torch___vgg16_VGG_trt_engine_ : __torch__.torch.classes.tensorrt.Engine
                                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
  def forward(self_1: __torch__.vgg16.VGG_trt,
    input_0: Tensor) -> Tensor:
```

**Solution**:

```bash
# import torch_tensorrt to change Runtime type
```

**Problem**: `ERROR: [Torch-TensorRT] - Input cache file is None but use_cache is set to True in INT8 mode.`

**Reason**: in source code

```python
class DataLoaderCalibrator(object):
    """
    Constructs a calibrator class in TensorRT and uses pytorch dataloader to load/preproces
    data which is passed during calibration.
    Args:
        dataloader: an instance of pytorch dataloader which iterates through a given dataset.
        algo_type: choice of calibration algorithm.
        cache_file: path to cache file.
        use_cache: flag which enables usage of pre-existing cache.
        device: device on which calibration data is copied to.
    """

[docs]    def __init__(self, **kwargs):
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
```

 ### Description

Recently I used Torch-TensorRT to quantized YOLOx. I calibrated the model using PTQ and saved cache file. However, when I want to use this cache file in `DataLoaderCalibrator`, an error occured:

```
ERROR: [Torch-TensorRT] - Input cache file is None but use_cache is set to True in INT8 mode.
```

I believe I have set the right path to cache file so I checked the source code of  `DataLoaderCalibrator`. I found out that in `__new__()` method the following lines were used to determine  whether to use cache file or not:

```python
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

```

and I think that `if not cache_file:` should be `if cache_file:` instead.

### Environment

**TensorRT Version**:  8.4.0.6
**GPU Type**:   NVIDIA GeForce RTX 3090
**Nvidia Driver Version**:  470.57.02
**CUDA Version**:  11.3
**CUDNN Version**:  8.4.0
**Operating System + Version**:  Ubuntu 20.04
**Python Version (if applicable)**:  3.8
**TensorFlow Version (if applicable)**: 
**PyTorch Version (if applicable)**:  1.11.0
**Baremetal or Container (if container which image + tag)**: 



###  Relevant Files

link to source code of  `DataLoaderCalibrator` https://pytorch.org/TensorRT/_modules/torch_tensorrt/ptq.html#CacheCalibrator


### Steps To Reproduce

<!-- Craft a minimal bug report following this guide - https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports -->

Please include:
  * Exact steps/commands to build your repro
  * Exact steps/commands to run your repro
  * Full traceback of errors encountered

```bash
git clone https://github.com/QQQQ00243/Torch-TensorRT-VGG16.git
python main.py --epochs 10
python issue.py 
```



## Appendix: Resource Usage

<img src="/home/qqqq/.config/Typora/typora-user-images/image-20220518182828820.png" alt="image-20220518182828820" style="zoom:25%;" />

<img src="/home/qqqq/.config/Typora/typora-user-images/image-20220518182847782.png" alt="image-20220518182847782" style="zoom:33%;" />
