from bot_transfer.utils.loader import compose_params, load, BASE
from bot_transfer.utils.cmd_util import boolean
from bot_transfer.utils.trainer import train
import pickle
import os
import bot_transfer
import argparse
import stable_baselines

parser = argparse.ArgumentParser()

parser.add_argument("--low", "-l", type=str)
parser.add_argument("--high", "-m", type=str)
parser.add_argument("--env", "-e", type=str, default=None)
parser.add_argument("--high-level-skip", "-k", type=int)
parser.add_argument("--timesteps", "-t", type=int, default=100000)
parser.add_argument("--learning-rate", type=float, default=None)
parser.add_argument("--sample-goals", type=boolean, default=None)

args = parser.parse_args()

params = compose_params(args.low, args.high, env_name=args.env, k=args.high_level_skip)
assert params['alg'] == 'SAC', "Only SAC supported for finetuning"

params['timesteps'] = args.timesteps
params['alg_args']['learning_starts'] = 0
if not args.learning_rate is None:
    params['alg_args']['learning_rate'] = args.learning_rate
if not args.sample_goals is None:
    params['env_args']['sample_goals'] = args.sample_goals

model, _ = load(args.high, params, load_env=False, best=False)
if isinstance(model, stable_baselines.SAC):
    model.learning_starts = 0
train(params, model=model)