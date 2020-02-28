import argparse
from bot_transfer.utils.trainer import train_hrl
from bot_transfer.utils.cmd_util import train_hrl_parser, params_from_args_hrl

parser = train_hrl_parser()

args = parser.parse_args()

low_params, high_params = params_from_args_hrl(args)

train_hrl(low_params, high_params, high_training_starts=args.high_training_starts)