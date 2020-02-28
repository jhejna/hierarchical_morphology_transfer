import argparse
from bot_transfer.utils.loader import ModelParams

'''
Implements An Arg parser consistent across all experiments.
TODO: change it so that I have a list of the param names for each seciton,
alg args, env args, etc. then use that to auto generate the parsers and the 
mappings to params
'''
def boolean(item):
    if item == 'true' or item == 'True':
        return True
    elif item == 'false' or item == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.') 

def cvt_kwargs_args(kwargs, arg_parser):
    arg_list = []
    for key in kwargs.keys():
        arg_list.append('--' + key.replace('_', '-'))
        if isinstance(kwargs[key], list):
            arg_list.extend([str(item) for item in kwargs[key]])
        else:
            arg_list.append(str(kwargs[key]))
    args, unknown_args = arg_parser.parse_known_args(arg_list)
    if len(unknown_args) > 0:
        print("############# ERROR #####################################")
        print("Unknown Arguments:", unknown_args)
        print("#########################################################")
    return args

def base_parser():
    '''
    Common Arguments across all training jobs.
    '''
    parser = argparse.ArgumentParser()

    # Required Arguments
    parser.add_argument("--env", "-e", type=str, required=True)
    parser.add_argument("--alg", "-a", type=str, required=True)

    # General Args
    parser.add_argument("--timesteps", "-t", type=int)
    parser.add_argument("--time-limit", "-l", type=int)
    parser.add_argument("--layers", "-ly", type=int, nargs='+')
    parser.add_argument("--num-proc", "-np", type=int)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--tensorboard", "-tb", default=None, type=boolean)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)

    # Environment Args
    parser.add_argument("--early-low-termination", "-elt", type=boolean)
    parser.add_argument("--skip", "-k", type=int, default=None)
    parser.add_argument("--normalize", "-norm", type=boolean, default=False)
    parser.add_argument("--delta-max", "-dm", type=float, default=None)
    parser.add_argument("--action-penalty", "-ap", default=None, type=float)
    parser.add_argument("--skill-penalty", "-sp", default=None, type=float)
    parser.add_argument("--agent-size", "-as", default=None, type=float)
    parser.add_argument("--reset-prob", "-lrp", default=None, type=float)
    parser.add_argument("--max-sequential-low", "-msl", default=None, type=int)
    parser.add_argument("--gear", default=None, type=int)
    parser.add_argument("--ant-density", default=None, type=boolean)
    parser.add_argument("--ant-mass", default=None, type=boolean)
    parser.add_argument("--include-contacts", default=None, type=boolean)
    return parser

def train_parser():
    parser = base_parser()

    # General Alg Args
    parser.add_argument("--gamma", "-g", type=float)
    parser.add_argument("--batch-size", "-b", type=int)
    parser.add_argument("--policy", "-p", type=str)
    parser.add_argument("--best", default=None, type=boolean)
    parser.add_argument("--params")

    # Off Policy Args
    parser.add_argument("--buffer-size", "-bs", type=int)
    parser.add_argument("--learning-starts", "-ls", type=int)
    parser.add_argument("--learning-rate", "-lr", type=float)
    parser.add_argument("--noise", type=float)
    parser.add_argument("--gradient-steps", type=int, default=None)

    # On Policy Args
    parser.add_argument("--actor-lr", "-alr", type=float)
    parser.add_argument("--critic-lr", "-clr", type=float)
    parser.add_argument("--nminibatches", "-nmb", type=int)
    parser.add_argument("--n-steps", "-ns", type=int)
    parser.add_argument("--noptepochs", "-noe", type=int)
    parser.add_argument("--optim_stepsize", "-os", type=float)

    # Discriminator args
    parser.add_argument("--discrim-buffer-size", "-dbs", default=None, type=int)
    parser.add_argument("--discrim-layers", "-dly", default=None, nargs='+', type=int)
    parser.add_argument("--discrim-learning-rate", "-dlr", default=None, type=float)
    parser.add_argument("--discrim-weight", "-dw", default=None, type=float)
    parser.add_argument("--discrim-clip", "-dc", default=None, type=float)
    parser.add_argument("--discrim-batch-size", "-db", default=None, type=float)
    parser.add_argument("--discrim-time-limit", "-dtl", default=None, type=int)
    parser.add_argument("--discrim-early-low-term", "-delt", default=None, type=boolean)
    parser.add_argument("--discrim-train-freq", "-dtf", default=None, type=int)
    parser.add_argument("--discrim-stop", "-ds", default=None, type=float)
    parser.add_argument("--discrim-coef", "-dcf", default=None, type=float)
    parser.add_argument("--discrim-decay", "-dd", default=None, type=boolean)
    parser.add_argument("--discrim-include-next-state", default=None, type=boolean)
    parser.add_argument("--discrim-include-skill", default=None, type=boolean)
    parser.add_argument("--discrim-online", default=None, type=boolean)
    parser.add_argument("--finetune-time-limit", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--kl-policy", default=None, type=str)
    parser.add_argument("--kl-type", default=None, type=str)
    parser.add_argument("--kl-coef", default=None, type=float)
    parser.add_argument("--kl-stop", default=None, type=float)
    parser.add_argument("--kl-decay", default=None, type=boolean)
    parser.add_argument("--random-exploration", default=None, type=float)
    parser.add_argument("--sample-goals", default=None, type=boolean)
    parser.add_argument("--use-relative", default=None, type=boolean)
    parser.add_argument("--remove-table", default=None, type=boolean)
    parser.add_argument("--tau", default=None, type=float)
    parser.add_argument("--intermediate-steps", default=None, type=boolean)
    parser.add_argument("--rand-low-init", default=None, type=boolean)
    parser.add_argument("--use-velocity", default=None, type=boolean)
    parser.add_argument("--add-extra-z", default=None, type=boolean)
    parser.add_argument("--vertical-bonus", default=None, type=boolean)
    parser.add_argument("--reach-task", default=None, type=boolean)
    return parser

def train_hrl_parser():
    parser = base_parser()

    # High Alg Args -- no low level counter parts
    parser.add_argument("--high-training-starts", type=int, default=0)
    parser.add_argument("--learning-starts", "-ls", type=int)
    parser.add_argument("--n-steps", "-ns", type=int)

    # High / Low Split
    parser.add_argument("--low-layers", "-lly", type=int, nargs='+')
    parser.add_argument("--low-learning-rate", "-llr", type=float)
    parser.add_argument("--low-batch-size", "-lb", type=int)
    parser.add_argument("--low-actor-lr", "-lalr", type=float)
    parser.add_argument("--low-critic-lr", "-lclr", type=float)
    parser.add_argument("--low-nminibatches", "-lnmb", type=int)
    parser.add_argument("--low-noptepochs", "-lnoe", type=int)
    parser.add_argument("--low-n-steps", "-lns", type=int)
    parser.add_argument("--low-buffer-size", "-lbs", type=int)
    parser.add_argument("--low-gradient-steps", "-lgs", type=int)

    parser.add_argument("--high-layers", "-hly", type=int, nargs='+')
    parser.add_argument("--high-learning-rate", "-hlr", type=float)
    parser.add_argument("--high-batch-size", "-hb", type=int)
    parser.add_argument("--high-actor-lr", "-halr", type=float)
    parser.add_argument("--high-critic-lr", "-hclr", type=float)
    parser.add_argument("--high-nminibatches", "-hnmb", type=int)
    parser.add_argument("--high-noptepochs", "-hnoe", type=int)
    parser.add_argument("--high-n-steps", "-hns", type=int)
    parser.add_argument("--high-buffer-size", "-hbs", type=int)
    parser.add_argument("--high-gradient-steps", "-hgs", type=int)

    return parser

def params_from_args(args):
    # Required Arguments
    params = ModelParams(args.env, args.alg)
    # Optional Arguments
    if not args.name is None:
        params['name'] = args.name
    if not args.tensorboard is None:
        params['tensorboard'] = args.tensorboard
    if not args.timesteps is None:
        params['timesteps'] = args.timesteps
    if not args.checkpoint_freq is None:
        params['checkpoint_freq'] = args.checkpoint_freq
    if not args.eval_freq is None:
        params['eval_freq'] = args.eval_freq
    if not args.time_limit is None:
        params["time_limit"] = args.time_limit
    if not args.seed is None:
        params["seed"] = args.seed
    if not args.policy is None:
        params['env_wrapper_args']['policy'] = args.policy
    if not args.params is None:
        params['env_wrapper_args']['params'] = args.params
    if not args.buffer_size is None:
        params['alg_args']['buffer_size'] = args.buffer_size
    if not args.layers is None:
        params['policy_args']['layers'] = args.layers
    if not args.skip is None:
        params['env_args']['k'] = args.skip
    if not args.learning_starts is None:
        params['alg_args']['learning_starts'] = args.learning_starts
    if not args.gradient_steps is None:
        params['alg_args']['gradient_steps'] = args.gradient_steps
    if not args.num_proc is None:
        params['num_proc'] = args.num_proc
    if not args.normalize is None:
        params['normalize'] = args.normalize
    if not args.delta_max is None:
        params['env_args']['delta_max'] = args.delta_max
    if not args.early_low_termination is None:
        params['env_args']['early_low_termination'] = args.early_low_termination
    if not args.action_penalty is None:
        params['env_args']['action_penalty'] = args.action_penalty
    if not args.skill_penalty is None:
        params['env_args']['skill_penalty'] = args.skill_penalty
    if not args.agent_size is None:
        params['env_args']['agent_size'] = args.agent_size
    if not args.reset_prob is None:
        params['env_args']['reset_prob'] = args.reset_prob
    if not args.gear is None:
        params['env_args']['gear'] = args.gear
    if not args.max_sequential_low is None:
        params['env_args']['max_sequential_low'] = args.max_sequential_low
    if not args.ant_density is None:
        params['env_args']['ant_density'] = args.ant_density
    if not args.ant_mass is None:
        params['env_args']['ant_mass'] = args.ant_mass
    if not args.include_contacts is None:
        params['env_args']['include_contacts'] = args.include_contacts
        print("#####################")
        print("Contacts arg was", args.include_contacts)
    if not args.use_relative is None:
        params['env_args']['use_relative'] = args.use_relative
    if not args.remove_table is None:
        params['env_args']['remove_table'] = args.remove_table
    if not args.tau is None:
        params['alg_args']['tau'] = args.tau
    # Discrim args
    if not args.discrim_buffer_size is None:
        params['alg_args']['discrim_buffer_size'] = args.discrim_buffer_size
    if not args.discrim_layers is None:
        params['alg_args']['discrim_layers'] = args.discrim_layers
    if not args.discrim_learning_rate is None:
        params['alg_args']['discrim_learning_rate'] = args.discrim_learning_rate
    if not args.discrim_weight is None:
        params['alg_args']['discrim_weight'] = args.discrim_weight
    if not args.discrim_clip is None:
        params['alg_args']['discrim_clip'] = args.discrim_clip
    if not args.discrim_batch_size is None:
        params['alg_args']['discrim_batch_size'] = args.discrim_batch_size
    if not args.discrim_time_limit is None:
        params['env_wrapper_args']['discrim_time_limit'] = args.discrim_time_limit
    if not args.discrim_early_low_term is None:
        params['env_wrapper_args']['discrim_early_low_term'] = args.discrim_early_low_term
    if not args.discrim_train_freq is None:
        params['alg_args']['discrim_train_freq'] = args.discrim_train_freq
    if not args.discrim_stop is None:
        params['alg_args']['discrim_stop'] = args.discrim_stop
    if not args.discrim_coef is None:
        params['alg_args']['discrim_coef'] = args.discrim_coef
    if not args.discrim_decay is None:
        params['alg_args']['discrim_decay'] = args.discrim_decay
    if not args.discrim_include_next_state is None:
        params['alg_args']['discrim_include_next_state'] = args.discrim_include_next_state
    if not args.discrim_include_skill is None:
        params['alg_args']['discrim_include_skill'] = args.discrim_include_skill
    if not args.discrim_online is None:
        params['env_wrapper_args']['discrim_online'] = args.discrim_online
    if not args.finetune_time_limit is None:
        params['env_wrapper_args']['finetune_time_limit'] = args.finetune_time_limit
    if not args.random_exploration is None:
        params['alg_args']['random_exploration'] = args.random_exploration
    if not args.sample_goals is None:
        params['env_args']['sample_goals'] = args.sample_goals
    if not args.intermediate_steps is None:
        params['env_wrapper_args']['intermediate_steps'] = args.intermediate_steps
    if not args.rand_low_init is None:
        params['env_args']['rand_low_init'] = args.rand_low_init
    if not args.use_velocity is None:
        params['env_args']['use_velocity'] = args.use_velocity
    if not args.add_extra_z is None:
        params['env_args']['add_extra_z'] = args.add_extra_z
    # KL Policy Args
    if not args.kl_policy is None:
        params['alg_args']['kl_policy'] = args.kl_policy
    if not args.kl_type is None:
        params['alg_args']['kl_type'] = args.kl_type
    if not args.kl_coef is None:
        params['alg_args']['kl_coef'] = args.kl_coef
    if not args.kl_decay is None:
        params['alg_args']['kl_decay'] = args.kl_decay
    if not args.kl_stop is None:
        params['alg_args']['kl_stop'] = args.kl_stop

    # Previously searchable args
    if not args.optim_stepsize is None:
        params['alg_args']['optim_stepsize'] = args.optim_stepsize
    if not args.learning_rate is None:
        params['alg_args']['learning_rate'] = args.learning_rate
    if not args.actor_lr is None:
        params['alg_args']['actor_lr'] = args.actor_lr
    if not args.critic_lr is None:
        params['alg_args']['critic_lr'] = args.critic_lr
    if not args.batch_size is None:
        params['alg_args']['batch_size'] = args.batch_size
    if not args.gamma is None:
        params['alg_args']['gamma'] = args.gamma
    if not args.noise is None:
        params['noise'] = args.noise
    if not args.nminibatches is None:
        params['alg_args']['nminibatches'] = args.nminibatches
    if not args.n_steps is None:
        params['alg_args']['n_steps'] = args.n_steps
    if not args.noptepochs is None:
        params['alg_args']['noptepochs'] = args.noptepochs
    if not args.vertical_bonus is None:
        params['env_args']['vertical_bonus'] = args.vertical_bonus
    if not args.reach_task is None:
        params['env_args']['reach_task'] = args.reach_task

    return params

def params_from_args_hrl(args):
    # Required Arguments
    low_params = ModelParams(args.env + '_Low', args.alg)
    high_params = ModelParams(args.env + '_High', args.alg)

    # Optional Arguments
    if not args.name is None:
        low_params['name'] = args.name
        high_params['name'] = args.name
    if not args.tensorboard is None:
        low_params['tensorboard'] = args.tensorboard
        high_params['tensorboard'] = args.tensorboard
    if not args.timesteps is None:
        high_params['timesteps'] = args.timesteps
        if not args.skip is None:
            low_params['timesteps'] = args.timesteps * args.skip
    if not args.time_limit is None:
        high_params["time_limit"] = args.time_limit
        if not args.skip is None:
            low_params["time_limit"] = args.skip
    if not args.num_proc is None:
        low_params['num_proc'] = args.num_proc
        high_params['num_proc'] = args.num_proc

    # Env Arguments
    if not args.delta_max is None:
        low_params['env_args']['delta_max'] = args.delta_max
        high_params['env_args']['delta_max'] = args.delta_max
    if not args.early_low_termination is None:
        low_params['env_args']['early_low_termination'] = args.early_low_termination
        high_params['env_args']['early_low_termination'] = args.early_low_termination
    if not args.action_penalty is None:
        low_params['env_args']['action_penalty'] = args.action_penalty
        high_params['env_args']['action_penalty'] = args.action_penalty
    if not args.skill_penalty is None:
        low_params['env_args']['skill_penalty'] = args.skill_penalty
        high_params['env_args']['skill_penalty'] = args.skill_penalty
    if not args.skip is None:
        low_params['env_args']['k'] = args.skip
        high_params['env_args']['k'] = args.skip
    if not args.agent_size is None:
        low_params['env_args']['agent_size'] = args.agent_size
        high_params['env_args']['agent_size'] = args.agent_size
    if not args.reset_prob is None:
        low_params['env_args']['reset_prob'] = args.reset_prob
        high_params['env_args']['reset_prob'] = args.reset_prob
    if not args.gear is None:
        low_params['env_args']['gear'] = args.gear
        high_params['env_args']['gear'] = args.gear
    if not args.max_sequential_low is None:
        low_params['env_args']['max_sequential_low'] = args.max_sequential_low
        high_params['env_args']['max_sequential_low'] = args.max_sequential_low
    if not args.ant_density is None:
        low_params['env_args']['ant_density'] = args.ant_density
        high_params['env_args']['ant_density'] = args.ant_density
    if not args.ant_mass is None:
        low_params['env_args']['ant_mass'] = args.ant_mass
        high_params['env_args']['ant_mass'] = args.ant_mass

    # Alg Args
    if not args.learning_starts is None:
        low_params['alg_args']['learning_starts'] = args.learning_starts
        high_params['alg_args']['learning_starts'] = args.learning_starts
    if not args.n_steps is None:
        low_params['alg_args']['n_steps'] = args.n_steps
        high_params['alg_args']['n_steps'] = args.n_steps
    # Low Alg Args
    if not args.low_layers is None:
        low_params['policy_args']['layers'] = args.low_layers
    if not args.low_learning_rate is None:
        low_params['alg_args']['learning_rate'] = args.low_learning_rate
    if not args.low_actor_lr is None:
        low_params['alg_args']['actor_lr'] = args.low_actor_lr
    if not args.low_critic_lr is None:
        low_params['alg_args']['critic_lr'] = args.low_critic_lr
    if not args.low_batch_size is None:
        low_params['alg_args']['batch_size'] = args.low_batch_size
    if not args.low_nminibatches is None:
        low_params['alg_args']['nminibatches'] = args.low_nminibatches
    if not args.low_noptepochs is None:
        low_params['alg_args']['noptepochs'] = args.low_noptepochs
    if not args.low_buffer_size is None:
        low_params['alg_args']['buffer_size'] = args.low_buffer_size
    if not args.low_gradient_steps is None:
        low_params['alg_args']['gradient_steps'] = args.low_gradient_steps
    # High Alg Args
    if not args.high_layers is None:
        high_params['policy_args']['layers'] = args.high_layers
    if not args.high_learning_rate is None:
        high_params['alg_args']['learning_rate'] = args.high_learning_rate
    if not args.high_actor_lr is None:
        high_params['alg_args']['actor_lr'] = args.high_actor_lr
    if not args.high_critic_lr is None:
        high_params['alg_args']['critic_lr'] = args.high_critic_lr
    if not args.high_batch_size is None:
        high_params['alg_args']['batch_size'] = args.high_batch_size
    if not args.high_nminibatches is None:
        high_params['alg_args']['nminibatches'] = args.high_nminibatches
    if not args.high_noptepochs is None:
        high_params['alg_args']['noptepochs'] = args.high_noptepochs
    if not args.high_buffer_size is None:
        high_params['alg_args']['buffer_size'] = args.high_buffer_size
    if not args.high_gradient_steps is None:
        high_params['alg_args']['gradient_steps'] = args.high_gradient_steps

    return low_params, high_params