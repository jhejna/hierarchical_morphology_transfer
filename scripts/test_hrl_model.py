import argparse
from bot_transfer.utils.tester import test_hrl

parser = argparse.ArgumentParser()

parser.add_argument("--low", "-l", type=str)
parser.add_argument("--high", "-m", type=str)
parser.add_argument("--gif", "-g", type=int, default=0)
parser.add_argument("--high-level-skip", "-k", type=int, default=None)
# parser.add_argument("--fps", "-f", type=int, default=0)
args = parser.parse_args()

test_hrl(args.low, args.high, args.gif)
