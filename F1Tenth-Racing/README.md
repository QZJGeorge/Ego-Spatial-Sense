# F1Tenth Racing Simulation

## Prerequisites
Make sure you have [Anaconda](https://docs.anaconda.com/getting-started/) installed before proceeding.

## 1. Create a Virtual Conda Environment
Create a new environment named `f1tenth` with python 3.10:

```bash
# create conda envivornment
conda create --name f1tenth python=3.10

# activate conda environment
conda activate f1tenth
```

## 2. Install F1tenth Gym
Navigate to the `f1tenth_gym` directory and install the package:

```bash
cd f1tenth_gym && pip install . && cd ..
```

## 3. Running Examples

`--model_path`: path to load a pretrained model.  

`--device`: specify the runtime device (`cpu` or `cuda`). 
 
`--render`: enable animation rendering.

Map selection: navigate to `config.yaml` and uncomment the corresponding map.


```bash
# Ego spatial sense model
python3 evaluate.py --model_path "pretrained/es2_0.pth" --device "cuda" --render

# MLP model
python3 evaluate.py --model_path "pretrained/mlp_0.pth" --device "cuda" --render

# Transformer model
python3 evaluate.py --model_path "pretrained/transformer_0.pth" --device "cuda" --render

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

Run the gap_follow algorithm to generate expert training data. Press Ctrl+C at any time to save and exit.

```bash
python gap_follow.py
```