# 2D Escaping Ball

## Prerequisites
Make sure you have [Anaconda](https://docs.anaconda.com/getting-started/) installed before proceeding.

## 1. Create a Virtual Conda Environment
Create a new environment named `escaping_ball` with python 3.10:

```bash
# create conda envivornment
conda create --name escaping_ball python=3.10

# activate conda environment
conda activate escaping_ball
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Running Examples

`--model_path`: path to load a pretrained model.

`--random_seed`: define a fixed initial condition. 

`--num_balls`: define the number of background balls. 

`--device`: specify the runtime device (`cpu` or `cuda`). 

`--render`: enable animation rendering. 


```bash
# Ego spatial sense model
python3 evaluate.py --model_path "pretrained/es2.pth" --random_seed 42 --num_balls 10 --device "cuda" --render

# MLP model
python3 evaluate.py --model_path "pretrained/mlp.pth" --random_seed 42 --num_balls 10 --device "cuda" --render

# Transformer model
python3 evaluate.py --model_path "pretrained/transformer.pth" --random_seed 42 --num_balls 10 --device "cuda" --render
```

## 4. Training Examples

For additional training parameters, please refer to the training script.

```bash
# To train a new ego spatial sense model:
python train_es2.py --device "cuda" --learning_rate 0.001 --num_epochs 500

# To train a new mlp model:
python train_mlp.py --device "cuda" --learning_rate 0.001 --num_epochs 500

# To train a new transformer model:
python train_transformer.py --device "cuda" --learning_rate 0.001 --num_epochs 500
```

## 5. Generate Training Data

Run the potential field algorithm to generate expert training data. Press Ctrl+C at any time to save and exit.

```bash
python expert.py
```