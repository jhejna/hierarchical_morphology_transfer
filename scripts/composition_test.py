import argparse
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from bot_transfer.utils.tester import test_composition, composition_sweep

parser = argparse.ArgumentParser()

parser.add_argument("--low", "-l", type=str, nargs='+')
parser.add_argument("--high", "-m", type=str, nargs='+')
parser.add_argument("--env", "-e", type=str)
parser.add_argument("--gif", "-g", type=int, default=0)
parser.add_argument("--high-level-skip", "-k", type=int, default=None)
parser.add_argument("--num-ep", "-n", type=int, default=None)

args = parser.parse_args()

if len(args.high) > 1 or len(args.low) > 1:
    print(args.high_level_skip)
    composition_sweep(args.low, args.high, env_name=args.env, k=args.high_level_skip, num_ep=args.num_ep)
else:
    test_composition(args.low[0], args.high[0], args.env, args.gif, k=args.high_level_skip, num_ep=args.num_ep)
