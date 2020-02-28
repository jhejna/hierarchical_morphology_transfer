#### This is a testing Script ####
from bot_transfer.utils.loader import ModelParams, load
from bot_transfer.utils.trainer import train
import pickle
import os
import bot_transfer
import argparse
import stable_baselines

parser = argparse.ArgumentParser()

parser.add_argument("--low", "-l", type=str, required=True)
parser.add_argument("--high", "-m", type=str, required=True)
parser.add_argument("--time-limit", "-k", type=int)
parser.add_argument("--finetune-time-limit", "-f", type=int)
parser.add_argument("--timesteps", "-t", type=int, default=2000)

args = parser.parse_args()


params = ModelParams.load(args.low)
assert params['env'].endswith("_Low")

params['env_wrapper_args']['policy'] = args.high

if args.finetune_time_limit:
    params['env_wrapper_args']['finetune_time_limit'] = args.finetune_time_limit
if args.time_limit:
    params['time_limit'] = args.time_limit

params['env'] = '_'.join([params['env'].split('_')[0], 'LowFinetune'])

model, _ = load(args.low, params, load_env=False, best=True)
if isinstance(model, stable_baselines.SAC):
    model.learning_starts = 0
train(params, model=model)