import argparse
from bot_transfer.utils.tester import test

parser = argparse.ArgumentParser()

parser.add_argument("--path", "-p", type=str)
parser.add_argument("--gif", "-g", type=int, default=0)
# parser.add_argument("--fps", "-f", type=int, default=0)
args = parser.parse_args()

test(args.path, args.gif)
