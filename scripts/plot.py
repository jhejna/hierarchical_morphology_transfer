from bot_transfer.utils import plotter
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--path", "-p", type=str, nargs='+')
parser.add_argument("--legend", "-l", type=str, default=None, nargs='+')
parser.add_argument("--n", "-n", type=int, default=None)
parser.add_argument("--title", "-t", type=str, default=None)
parser.add_argument("--single", "-s", action='store_true', default=False)
parser.add_argument("--use-wall-time", "-w", action='store_true', default=False)
parser.add_argument("--use-episodes", "-e", action='store_true', default=False)
parser.add_argument("--individual", "-i", action='store_true', default=False)
parser.add_argument("--zero-shot", "-z", type=float, default=None)
parser.add_argument("--hiro", "-d", type=str, default=None)
parser.add_argument("--clip", "-c", type=float, default=None)
args = parser.parse_args()

if args.title:
    title = args.title
else:
    title = args.path

if args.use_wall_time:
    print("Using time")
    xaxis = plotter.X_WALLTIME
elif args.use_episodes:
    print("Using episodes")
    xaxis = plotter.X_EPISODES
else:
    print("Using timesteps")
    xaxis = plotter.X_TIMESTEPS

if args.single:
    plotter.plot_results(args.path, args.n, xaxis, title, legend_names=args.legend)
else:
    plotter.multi_seed_plot_results(args.path, args.n, xaxis, title, legend_names=args.legend, individual=args.individual, zero_shot=args.zero_shot, hiro=args.hiro, clip=args.clip)
plt.show()
