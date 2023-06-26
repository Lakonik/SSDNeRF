# SSDNeRF

Official PyTorch implementation of the paper:

**Single-Stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction**
<br>
[Hansheng Chen](https://lakonik.github.io/)<sup>1,</sup>\*, [Jiatao Gu](https://jiataogu.me/)<sup>2</sup>, [Anpei Chen](https://apchenstu.github.io/)<sup>3</sup>, [Wei Tian](https://scholar.google.com/citations?user=aYKQn88AAAAJ&hl=en)<sup>1</sup>, [Zhuowen Tu](https://pages.ucsd.edu/~ztu/)<sup>4</sup>, [Lingjie Liu](https://lingjie0206.github.io/)<sup>5</sup>, [Hao Su](https://cseweb.ucsd.edu/~haosu/)<sup>4</sup><br>
<sup>1</sup>Tongji University, <sup>2</sup>Apple, <sup>3</sup>ETH Zürich, <sup>4</sup>UCSD, <sup>5</sup>University of Pennsylvania
<br>
\*Work done during a remote internship with UCSD.

[[project page](https://lakonik.github.io/ssdnerf)] [[paper](https://arxiv.org/pdf/2304.06714.pdf)]

Part of this codebase is based on [torch-ngp](https://github.com/ashawkey/torch-ngp) and [MMGeneration](https://github.com/open-mmlab/mmgeneration).
<br>

https://github.com/Lakonik/Diffusion-NeRF-internal/assets/53893837/53337f0b-aa82-4d0c-a985-fdaeccaf2ef3

## Highlights

- Code to reproduce ALL the experiments in the paper and supplementary material.
- New features including support for tiled triplanes (rollout layout) and 16-bit caching (to save memory).
- A simple GUI demo (modified from [torch-ngp](https://github.com/ashawkey/torch-ngp)).

<img src="ssdnerf_gui.png" width="500" alt=""/>

## Installation

### Prerequisites

The code has been tested in the environment described as follows:

- Linux (tested on Ubuntu 18.04/20.04 LTS)
- Python 3.7
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 11
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 1.12.1
- [MMCV](https://github.com/open-mmlab/mmcv) 1.6.0
- [MMGeneration](https://github.com/open-mmlab/mmgeneration) 0.7.2

Also, this codebase should be able to work on Windows systems as well (tested in the inference mode).

Other dependencies can be installed via `pip install -r requirements.txt`. 

An example of commands for installing the Python packages is shown below:

```bash
# Export the PATH of CUDA toolkit
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

# Create conda environment
conda create -y -n ssdnerf python=3.7
conda activate ssdnerf

# Install PyTorch (this script is for CUDA 11.3)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install MMCV and MMGeneration
pip install -U openmim
mim install mmcv-full==1.6
git clone https://github.com/open-mmlab/mmgeneration && cd mmgeneration && git checkout v0.7.2
pip install -v -e .
cd ..

# Clone this repo and install other dependencies
git clone <this repo> && cd <repo folder> && git checkout ssdnerf-dev
pip install -r requirements.txt
```

### Compile CUDA packages

There are two CUDA packages from [torch-ngp](https://github.com/ashawkey/torch-ngp) that need to be built locally.

```bash
cd lib/ops/raymarching/
pip install -e .
cd ../shencoder/
pip install -e .
cd ../../..
```

## Data preparation

Download `srn_cars.zip` and `srn_chairs.zip` from [here](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR).
Unzip them to `./data/shapenet`.

Download `abo_tables.zip` from [here](https://drive.google.com/file/d/1lzw3uYbpuCxWBYYqYyL4ZEFomBOUN323/view?usp=share_link). Unzip it to `./data/abo`. For convenience I have converted the ABO dataset into PixelNeRF's SRN format.

Finally you should have the following folder tree:

```
./
├── configs/
├── data/
│   ├── shapenet/
│   │   ├── cars_test/
│   │   ├── cars_train/
│   │   ├── cars_val/
│   │   ├── chairs_test/
│   │   ├── chairs_train/
│   │   └── chairs_val/
│   └── abo/
│       ├── tables_train/
│       └── tables_test/
├── demo/
├── lib/
├── tools/
…

```

For FID and KID evaluation, run the following commands to extract the Inception features of the real images. (This script will use all the available GPUs on your machine, so remember to set `CUDA_VISIBLE_DEVICES`.)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/inception_stat.py configs/ssdnerf_cars_uncond.py
CUDA_VISIBLE_DEVICES=0 python tools/inception_stat.py configs/ssdnerf_chairs_recons1v.py
CUDA_VISIBLE_DEVICES=0 python tools/inception_stat.py configs/ssdnerf_abotables_uncond.py
```

### Todos

- [ ] Add KITTI Cars dataset preparation instructions.

## About the configs

Naming convention:
    
```
ssdnerf_cars3v_uncond
   │      │      └── testing data: test unconditional generation
   │      └── training data: train on Cars dataset, using 3 views per scene
   └── training method: single-stage diffusion nerf training
  
stage2_cars_recons1v
   │     │      └── testing data: test 3D reconstruction from 1 view
   │     └── training data: train on Cars dataset, using all views per scene
   └── training method: stage 2 of two-stage training
```

### Todos

- [ ] Add descriptions for each config file.
- [ ] Add multi-view testing configs.

## Training

Run the following command to train a model:

```bash
python train.py /PATH/TO/CONFIG --gpu-ids 0 1
```

Note that the total batch size is determined by the number of GPUs you specified. All our models are trained using 2 GPUs.

Since we adopt the density-based NeRF pruning trategy in [torch-ngp](https://github.com/ashawkey/torch-ngp), training would start slow and become faster later, so the initial esitamtion of remaining time is usually over twice as much as the actual training time.

Model checkpoints will be saved into `./work_dirs`. Scene caches will be saved into `./cache`.

## Testing and evaluation

```bash
python test.py /PATH/TO/CONFIG /PATH/TO/CHECKPOINT --gpu-ids 0 1  # you can specify any number of GPUs here
```
Some trained models can be downloaded from [here](https://drive.google.com/drive/folders/13z4C13TsofPkBuqMqQjRp5yDck7CjCiZ?usp=sharing) for testing.

## Visualization

By default, during training or testing, the visualizations will be saved into `./work_dirs`. 

A GUI tool is provided for visualizing the results (currently only supports unconditional generation). Run the following command to start the GUI:

```bash
python tools/ssdnerf_gui.py /PATH/TO/CONFIG /PATH/TO/CHECKPOINT
```

## Citation

If you find this project useful in your research, please consider citing:

```
@misc{ssdnerf,
    title={Single-Stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction}, 
    author={Hansheng Chen and Jiatao Gu and Anpei Chen and Wei Tian and Zhuowen Tu and Lingjie Liu and Hao Su},
    year={2023},
    archivePrefix={arXiv},
    eprint={2304.06714},
    primaryClass={cs.CV}
}
```
