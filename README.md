### Overview
This project implements and compares Quantization-Aware Training (QAT) applied to Deep Q-Networks (DQN) against standard DQN models. The research focuses on maintaining performance while achieving model compression through quantization techniques.

### Installation
#### 1. Clone repository
$ git clone https://github.com/jkw0701/qat_example.git

$ cd qat_example

#### 2. Create and activate conda environment
$ conda env create -f environment.yaml

$ conda activate qat_rl_env

#### Verify installation
$ python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

 
### Usage
#### Basic Training Pipeline
1. Train Normal DQN (baseline)
$ python test.py --mode train_normal

2. QAT Transfer Learning (recommended)
$ python test.py --mode transfer_qat

3. Comprehensive Analysis
$ python test.py --mode compare

4. Real-time Simulation
$ python test.py --mode simulate

#### Advanced Options
##### Extended training
$ python test.py --mode transfer_qat --episodes 1000

##### Animation comparison
$ python test.py --mode animate

##### Train both models from scratch
$ python test.py --mode train_both

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Author
Keunwoo Jang (Center for Humanoid Research, KIST)
