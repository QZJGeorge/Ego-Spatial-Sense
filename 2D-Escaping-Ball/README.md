# 2D Escaping Ball

## Prerequisites
We recommend using [Anaconda](https://docs.anaconda.com/getting-started/) to manage the environment. Alternatively, you may use a standard Python 3.10 setup. 

## 1. Create a Virtual Conda Environment (Optional)
Create a new environment named `escaping_ball`:

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

## 3. Evaluation Examples

The following are example evaluation scripts with corresponding configuration parameters. The game ends upon the first collision or after a fixed number of time steps, with survival duration serving as the final score. For additional evaluation settings, please refer to the script.

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

The following are example training scripts with corresponding configuration parameters. For additional training parameters, please refer to the script.

```bash
# To train a new ego spatial sense model:
python3 train_es2.py --data_path "dataset/data_1x.csv" --device "cuda" --learning_rate 0.001 --num_epochs 500

# To train a new mlp model:
python3 train_mlp.py --data_path "dataset/data_1x.csv" --device "cuda" --learning_rate 0.001 --num_epochs 500

# To train a new transformer model:
python3 train_transformer.py --data_path "dataset/data_1x.csv" --device "cuda" --learning_rate 0.001 --num_epochs 500
```

## 5. Generate Training Data

Run the potential field algorithm to generate expert training data. Press Ctrl+C at any time to save and exit. Record the dataset path for use during model training.

```bash
python3 expert.py
```