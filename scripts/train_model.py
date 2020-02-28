from bot_transfer.utils.cmd_util import train_parser, params_from_args
from bot_transfer.utils.trainer import train

parser = train_parser()
args = parser.parse_args()
params = params_from_args(args)

train(params)