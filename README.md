<h1 align="center">
Reproduction of Paper: Spike-driven Transformer
</h1>
<p align="center">
    Project of AI3610 Brain-inspired Intelligence, 2023 Fall, SJTU
    <br />
    <a href="https://github.com/Ark-ike"><strong>Yi Ai</strong></a>
    &nbsp;
    <a href="https://github.com/xxyQwQ"><strong>Xiangyuan Xue</strong></a>
    &nbsp;
    <a href="https://github.com/YsmmsY"><strong>Shengmin Yang</strong></a>
    <br />
</p>

## Requirements

To ensure the code runs correctly, following packages are required:

* `python`
* `pytorch`
* `hydra`
* `einops`
* `timm`
* `spikingjelly`

You can install them following the instructions below.

* Create a new conda environment and activate it:
  
    ```bash
    conda create -n spikingjelly python=3.10
    conda activate spikingjelly
    ```

* Install [pytorch](https://pytorch.org/get-started/previous-versions/) with appropriate CUDA version, e.g.
  
    ```bash
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

* Install `hydra`, `einops`, `timm`, and `spikingjelly`:
  
    ```bash
    pip install hydra-core
    pip install einops
    pip install timm
    pip install spikingjelly
    ```

Latest version is recommended for all the packages, but make sure that your CUDA version is compatible with your `pytorch`.

## Experiments

### CIFAR-10

The CIFAR-10 dataset is supported by `torchvision`, which can be automatically downloaded. Run the following command for experiments on the CIFAR-10 dataset:

```bash
python training.py dataset=cifar-10 model=light
```

### CIFAR-100

The CIFAR-100 dataset is also supported by `torchvision`. Run the following command for experiments on the CIFAR-100 dataset:

```bash
python training.py dataset=cifar-100 model=light
```

### CIFAR-10-DVS

The CIFAR-10-DVS dataset is loaded by `spikingjelly` and can be automatically downloaded. Run the following command for experiments on the CIFAR-10-DVS dataset:

```bash
python training.py dataset=cifar-10-dvs model=light
```

Since neuromorphic datasets require larger time steps, a smaller batch size is recommended.

### DVS-128-Gesture

The DVS-128-Gesture dataset is also loaded by `spikingjelly`, but you need to download the dataset manually. Run the following command for experiments on the DVS-128-Gesture dataset:

```bash
python training.py dataset=dvs-128-gesture model=light
```

It is likely that `spikingjelly` will raise an error at the first time you run the experiment. Follow the instructions to download the dataset correctly and try again. Similarly, a smaller batch size is recommended.
