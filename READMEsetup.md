# Setup (Athena)

## 1. Create conda environment

```
conda create -y -n ssdnerf python=3.7
```

## 2. Enter the cluster with srun
important: set --mem=50G

```
srun -p plgrid-gpu-a100 --mem=50G --gres=gpu:1 -N 1 --ntasks-per-node=1 -n 1 -A <grantname>-gpu-a100 --pty /bin/bash -l
conda activate ssdnerf
```

## 3. Install PyTorch 
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c conda-forge cudatoolkit-dev
```

## 4. Install MMCV and MMGeneration
```
pip install -U openmim
mim install mmcv-full==1.6
git clone https://github.com/open-mmlab/mmgeneration && cd mmgeneration && git checkout v0.7.2
pip install -v -e .
cd ..
```

## 5. Clone this repo and install other dependencies
```
git clone <this repo> && cd <repo folder>
pip install -r requirements.txt
```

## 6. Set CUDA_HOME variable
```
export CUDA_HOME=$CONDA_PREFIX
```

## 7. Install other dependecies 
```
cd lib/ops/raymarching/
pip install -e .
cd ../shencoder/
pip install -e .
cd ../../..
```

## 8. Prepare work_dirs (for specific config)
```
CUDA_VISIBLE_DEVICES=0
python tools/inception_stat.py configs/paper_cfgs/ssdnerf_cars_uncond.py
```

## 8. You can run code now
one GPU
```
python train.py /PATH/TO/CONFIG --gpu-ids 0 
```
two GPUs etc.
```
python train.py /PATH/TO/CONFIG --gpu-ids 0 1
```
