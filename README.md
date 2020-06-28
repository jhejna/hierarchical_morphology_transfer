# Hierarchically Decoupled Imitation for Morphological Transfer
Code for the paper [Hierarchically Decoupled Imitation for Morphological Transfer](https://arxiv.org/abs/2003.01709) by Donald (Joey) Hejna, Pieter Abbeel, and Lerrel Pinto.

## Setup

1. Install the required python dependencies using the requirements.txt file. Note that if you want to use GPU, you must swap tensorflow for tensorflow-gpu
2. Install the bot_transfer package by running `pip install -e .` from the root of the repository. This will use the `setup.py` file.

## Usage

All training, testing, and rendering can be completed through the scripts in the `scripts` folder.

Below are example scripts for training models in our framework.

1. Pre-train Low Level
```
python scripts/train_model.py --env Ant_Low --alg SAC -t 2500000 --learning-rate 0.0008 \ 
                              --batch-size 100 --layers 400 300 --reset-prob 0.1 \
                              --buffer-size 1000000 --delta-max 4.0 --time-limit 100
```
2. Train High Level
```
python scripts/train_model.py --env Ant_High --alg SAC -t 200000 --time-limit 50
```

3. Train Low Level With Discirminative Imitation
```
python scripts/train_model.py --env Ant_Discriminator --alg DSAC -t 2500000 --learning-rate 0.0008 \
                              --batch-size 100 --layers 400 300 --reset-prob 0.1 \
                              --buffer-size 1000000 --delta-max 4.0 --time-limit 100 \
                              --discrim-learning-rate 0.0002 --discrim-stop 0.5 --discrim-decay true \
                              --discrim-online false --discrim-time-limit 32 \
                              --policy <path to model output folder>
```
Note that the default output location is `output/MM_DD_YY/env_alg_seed_num`

4. Finetune a High Level Policy with KL-Regularization
```
python scripts/finetune_high_level_kl.py --low <path to low level model> --high <path to high level model> --env <optional for Maze or Steps> \
                                      -t 100000 --learning-rate 0.01 \
                                      --kl-coef 0.01 --kl-stop 0.5 \
                                      --kl-decay true --kl-type regular
```
