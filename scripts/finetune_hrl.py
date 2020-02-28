#### This is a testing Script ####
from bot_transfer.utils.loader import ModelParams, load_hrl, BASE
from bot_transfer.utils.trainer import train_hrl
import pickle
import os
import bot_transfer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--low", "-l", type=str, required=True)
parser.add_argument("--high", "-m", type=str, required=True)
parser.add_argument("--env", "-e", type=str, required=True)
parser.add_argument("--high-level-skip", "-k", type=int)
parser.add_argument("--timesteps", "-t", type=int, default=2000)
parser.add_argument("--delta-max", "-dm", type=float, default=None)

args = parser.parse_args()

high_params = ModelParams.load(args.high)
low_params = ModelParams.load(args.low)

print("LOADED PARAMS", high_params)
print("LOADED PARAMS", low_params)

high_params['env'] = args.env

high_params['env_args']['k'] = args.high_level_skip
low_params['env_args']['k'] = args.high_level_skip

if args.delta_max:
    high_params['env_args']['delta_max'] = args.delta_max
    low_params['env_args']['delta_max'] = args.delta_max

model, _ = load_hrl(args.low, args.high, low_params, high_params, load_env=False)

train_hrl(low_params, high_params, model=model)