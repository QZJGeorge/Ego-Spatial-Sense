# Ego Spatial Sense

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04-orange.svg)](https://releases.ubuntu.com/22.04/)
[![VLM](https://img.shields.io/badge/VLM-GPT_4o-blueviolet.svg)](#)


## Introduction

This project contains the source code for the paper titled "Overcoming the Long-tail Problems for Safety-Critical
Autonomy with Ego Spatial Sense". It includes three primary experiments: the 2D Escaping Ball Game, the F1Tenth Racing Simulation, and the Vision-Language Spatial Sense Model.

## Description of Experiments

1. [**2D Escaping Ball Game**](/2D-Escaping-Ball/README.md): The 2D escaping ball game simulates an ego ball navigating within a square window to avoid collisions with independently moving background balls. The game ends upon the first collision or after a fixed number of time steps, with survival duration serving as the final score.
  
    ![2D Ball Game](/figures/GIF1.gif)

2. [**F1Tenth Racing Simulation**](/F1Tenth-Racing/README.md): The F1Tenth Racing experiment uses the F1Tenth Gym simulator to evaluate an ego car navigating a 2D track with static obstacles using LiDAR-based perception. The game ends upon completing a lap or encountering a collision, boundary violation, or stalling, with distance traveled recorded as the score.
   
   ![F1Tenth](/figures/GIF2.gif)

3. [**Vision-Language Spatial Sense Model (VLSS)**](/VLSS-Model/README.md): The Vision-Language Spatial Sense Model combines the ego spatial sense model with a pre-trained VLM to generate context-aware, spatially informed trajectories from front-view images in previously unseen driving scenarios. The VLM extracts commonsense knowledge, while the spatial sense module performs structured spatial reasoning, with training and testing based on nuScenes data.
   
   ![VLSS](/figures/GIF3.gif)

## Code structure

```
Ego-Spatial-Sense
|- 2D-Escaping-Ball: source code and instructions for the 2D Escaping Ball Game.
|- F1Tenth-Racing: source code and instructions for the F1Tenth Racing Simulation.
|- VLSS-Model: source code and instructions for the Vision-Language Spatial Sense Model.
|- README.md
|- LICENSE.txt
```


## Requirements

The hardware and software requirements for this repository are flexible and broadly compatible across various platforms. However, we recommend **not** using macOS, as certain dependencies and functionalities may not perform reliably. We also advise against running the code on remote servers, as visualization features may not function properly. The following environments have been thoroughly validated:

- **Hardware**:
  - CPU with x86 architecture  
  - NVIDIA GPU with CUDA (optional but recommended for accelerated model training and testing)

- **Operating System**:  
  - Ubuntu 22.04  
  - Windows 11

- **Python Environment**:  
  - Python 3.10 (via native installation or within an Anaconda virtual environment).

## Installation

Detailed installation instructions are available in each experiment's respective directory. In general, the installation process is straightforward and typically requires approximately 5 to 10 minutes on a standard computing system. Each experiment is self-contained and can be executed independently.

## Usage

Detailed usage instructions are provided within each experiment's respective directory, covering expert demonstration, data collection, model training, and model evaluation. Additionally, pretrained models are available for immediate testing purposes. Each experiment is self-contained and can be executed independently.


## Developer

- Zhijie Qiao: zhijieq@umich.edu

- Zhong Cao: zhcao@umich.edu


## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.txt).

## Contact

- Henry Liu: henryliu@umich.edu
