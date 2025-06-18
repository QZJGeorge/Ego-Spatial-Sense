# F1Tenth Racing Simulation

## Prerequisites
We recommend using [Anaconda](https://docs.anaconda.com/getting-started/) to manage the environment. Alternatively, you may use a standard Python 3.10 setup. 

## 1. Create a Virtual Conda Environment (Optional)
Create a new environment named `f1tenth`:

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

## 3. Evaluation Examples

The following are example evaluation scripts with corresponding configuration parameters. The game ends upon completing a lap or encountering a collision, boundary violation, or stalling, with distance traveled recorded as the score. For additional evaluation settings, please refer to the script.

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

The following are example training scripts with corresponding configuration parameters. For additional training parameters, please refer to the script.

```bash
# To train a new ego spatial sense model:
python3 train_es2.py --data_path "dataset/austin.csv" --device "cuda" --learning_rate 0.001 --num_epochs 500

# To train a new mlp model:
python3 train_mlp.py --data_path "dataset/austin.csv" --device "cuda" --learning_rate 0.001 --num_epochs 500

# To train a new transformer model:
python3 train_transformer.py --data_path "dataset/austin.csv" --device "cuda" --learning_rate 0.001 --num_epochs 500
```

## 5. Generate Training Data

Run the gap_follow algorithm to generate expert training data. Press Ctrl+C at any time to save and exit. Record the dataset path for use during model training.

```bash
python3 gap_follow.py
```
