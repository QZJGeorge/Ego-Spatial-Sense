# Vision-Language Spatial Sense Model

## Prerequisites
Make sure you have [Anaconda](https://docs.anaconda.com/getting-started/) installed before proceeding.

## 1. Create a Virtual Conda Environment
Create a new environment named `vlssm` with python 3.10:

```bash
# create conda envivornment
conda create --name vlssm python=3.10

# activate conda environment
conda activate vlssm
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Running Examples

```bash
# To generate raycast data from front camera images
# This should create a folder called raycast
python3 run_raycast.py

# To render images with spatial sense field
# This should create a folder called sense
python3 run_sense.py