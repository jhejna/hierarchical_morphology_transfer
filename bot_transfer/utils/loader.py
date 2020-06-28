import os
import pickle
import stable_baselines
import bot_transfer
import copy
from datetime import date
import json

BASE = os.path.dirname(os.path.dirname(bot_transfer.__file__)) + '/output'
LOGS = os.path.dirname(os.path.dirname(bot_transfer.__file__)) + '/logs'

# TODO: Do something to enforce consistency of delta_max value for envs.

class ModelParams(dict):

    def __init__(self, env : str, alg : str):
        super(ModelParams, self).__init__()
        # Construction Specification
        self['alg'] = alg
        self['env'] = env
        self['policy'] = 'MlpPolicy' # TODO: Support different types of policies!
        # Arg Dicts
        self['env_args'] = dict()
        self['env_wrapper_args'] = dict()
        self['alg_args'] = dict()
        self['policy_args'] = dict()
        # Env Wrapper Arguments
        self['early_reset'] = True
        self['normalize'] = False
        self['time_limit'] = None
        # Training Args
        self['seed'] = None
        self['timesteps'] = 250000
        # Logistical Args
        self['log_interval'] = 10
        self['name'] = None
        self['tensorboard'] = None
        self['num_proc'] = 1 # Default to single process
        self['eval_freq'] = 100000
        self['checkpoint_freq'] = None

    def get_save_name(self):
        save_name = self['env'] + '_' + self['alg']
        if self['name'] != None:
            save_name += '_' + self['name']
        if not self['seed'] is None:
            save_name += '_s' + str(self['seed'])
        return save_name

    def save(self, path : str):
        if path.endswith('.json'):
            save_path = path
        else:
            save_path = os.path.join(path, 'params.json')
        with open(save_path, 'w') as fp:
            json.dump(self, fp, indent=4)
    
    @classmethod
    def load(cls, path):
        if not path.startswith('/'):
            path = os.path.join(BASE, path)
        # CASE 1: Path is a path to the directory containing the parameters.
        if os.path.isdir(path):
            param_files = [f for f in os.listdir(path) if f.startswith('params')]
            if len(param_files) != 1:
                raise ValueError("Supposed to be one params file per directory. Found " + str(len(param_files)))
            path = os.path.join(path, param_files[0])
        # Case 2: It is the full path to the params file.
        else:
            assert os.path.exists(path)
        
        if os.path.splitext(path)[1] == '.pkl':
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(path, 'r') as fp:
                data = json.load(fp)
        if 'alg' not in data.keys():
            data['alg'] = data['algorithm']
        params = cls(data['env'], data['alg'])
        params.update(data)
        # Fixes for backwards compatibility:
        if 'low_level_policy' in params:
            params['env_wrapper_args']['policy'] = 'pre_update/' + params['low_level_policy']
            del params['low_level_policy']
        elif 'low_level_policy' in params['env_wrapper_args']:
            params['env_wrapper_args']['policy'] = params['env_wrapper_args']['low_level_policy']
            del params['env_wrapper_args']['low_level_policy']
        if 'low_reset_prob' in params['env_args']:
            params['env_args']['reset_prob'] = params['env_args']['low_reset_prob']
            del params['env_args']['low_reset_prob']
        print("Loaded Parameters:", params)
        return params

def merge_hrl_params(low_params, high_params, env_name=None, env_args=None):
    '''
    Merge two sets of parameters for HRL Training.
    This is done as follows:
    1. Pass on the alg and policy args to a joint params file
    2. Convert the algorithm to the hierarchical version
    3. Pass on the general params from the high level env
    4. Set the environment params as specified, if None, default to 
        the ones from the high level.
    '''
    # Check that we can merge them based on algorithm
    assert low_params['alg'] == high_params['alg']
    if env_name is None:
        assert high_params['env'].endswith('_High')
        env_name = high_params['env'][:-5] # Remove "_High" from the end

    alg_name = 'H' + low_params['alg']
    env_extension = {
        'HPPO1' : 'JointAC1',
        'HPPO2' : 'JointAC2',
        'HSAC' : 'JointOP'
    }[alg_name]

    merged_params = copy.copy(high_params)

    merged_params['alg'] = alg_name
    merged_params['env'] = env_name + '_' + env_extension
    del merged_params['policy_args']
    del merged_params['alg_args']
    assert high_params['policy'] == low_params['policy'], "Differing policy types not supported"

    # If we specify env args, set them here, otherwise default to the ones given by the high level policy.
    if env_args:
        merged_params['env_args'] = env_args
    if 'policy' in merged_params['env_wrapper_args']:
        del merged_params['env_wrapper_args']['policy']
    
    # Set High and Low Parameters
    merged_params['high_alg_args'] = high_params['alg_args']
    merged_params['high_policy_args'] = high_params['policy_args']
    merged_params['high_policy'] = high_params['policy']
    merged_params['low_alg_args'] = low_params['alg_args']
    merged_params['low_policy_args'] = low_params['policy_args']
    merged_params['low_policy'] = low_params['policy']

    return merged_params

def split_hrl_params(hrl_params):

    alg_name = hrl_params['alg']
    assert alg_name.startswith('H')
    env_extension = {
        'HPPO1' : 'JointAC1',
        'HPPO2' : 'JointAC2',
        'HSAC' : 'JointOP'
    }[alg_name]
    assert hrl_params['env'].endswith('_' + env_extension)
    env_name = hrl_params['env'][:-len('_' + env_extension)]
    alg_name = alg_name[1:] # get the regular version of the algorithm

    low_params = ModelParams(env_name + '_Low', alg_name)
    high_params = ModelParams(env_name + '_High', alg_name)

    for param_key in hrl_params.keys():
        if not (param_key.startswith('low') or param_key.startswith('high') or param_key in ['alg', 'env', 'env_wrapper_args']):
            low_params[param_key] = hrl_params[param_key]
            high_params[param_key] = hrl_params[param_key]

    if 'k' in hrl_params['env_args']:
        low_params['timesteps'] = hrl_params['env_args']['k'] * hrl_params['timesteps']
        low_params['time_limit'] = hrl_params['env_args']['k']
    
    low_params['alg_args'] = hrl_params['low_alg_args']
    low_params['policy'] = hrl_params['low_policy']
    low_params['policy_args'] = hrl_params['low_policy_args']

    high_params['alg_args'] = hrl_params['high_alg_args']
    high_params['policy'] = hrl_params['high_policy']
    high_params['policy_args'] = hrl_params['high_policy_args']

    return low_params, high_params
    
def get_paths(params, path=None):
    print("CWD", os.getcwd())
    if path:
        base_dir = path
    else:
        date_prefix = date.today().strftime('%m_%d_%y')
        base_dir = os.path.join(BASE, date_prefix)
    save_name = params.get_save_name()

    def get_comparison_name(save_name):
        components = save_name.split('_')
        # Remove the number extension
        save_name = '_'.join(save_name.split('_')[:-1])
        if save_name.endswith('Low'):
            save_name = save_name[:-4]
        elif save_name.endswith('High'):
            save_name = save_name[:-5]
        return save_name

    if os.path.isdir(base_dir):
        candidates = [f_name for f_name in os.listdir(base_dir) if get_comparison_name(f_name) == save_name]
        if len(candidates) == 0:
            save_name += '_0'
        else:
            num = max([int(dirname[-1]) for dirname in candidates]) + 1
            save_name += '_' + str(num)
    else:
        save_name += '_0'
    
    save_path = os.path.join(base_dir, save_name)
    tb_path = os.path.join(LOGS, date_prefix, save_name) if params['tensorboard'] else None
    return save_path, tb_path


def get_env(params: ModelParams):
    env_names = params['env'].split('_')
    try:
        env_cls = vars(bot_transfer.envs)[env_names[0]]
        env = env_cls(**params['env_args'])
        if len(env_names) == 2:
            print("ENV WRAPPER ARGS", params['env_wrapper_args'])
            env = vars(bot_transfer.envs)[env_names[1]](env, **params['env_wrapper_args'])
        if params['time_limit']:
            from gym.wrappers import TimeLimit
            env = TimeLimit(env, params['time_limit'])
    except:
        # If we don't get the env, then we assume its a gym environment
        import gym
        env = gym.make(params['env'])
    return env    

def get_alg(params: ModelParams):
    alg_name = params['alg']
    try:
        alg = vars(bot_transfer)[alg_name]
    except:
        alg = vars(stable_baselines)[alg_name]
    return alg

def get_policy(params: ModelParams):
    policy_name = params['policy']
    try:
        policy = vars(bot_transfer.policies)[policy_name]
        return policy
    except:
        alg_name = params['alg']
        if alg_name.endswith('SAC'):
            search_location = stable_baselines.sac.policies
        elif alg_name == 'DDPG':
            search_location = stable_baselines.ddpg.policies
        elif alg_name == 'DQN':
            search_location = stable_baselines.dqn.policies
        elif alg_name == 'TD3':
            search_location = stable_baselines.td3.policies
        else:
            search_location = stable_baselines.common.policies
        policy = vars(search_location)[policy_name]
        return policy

def load_from_name(path, best=False, load_env=True):
    if not path.startswith('/'):
        path = os.path.join(BASE, path)
    params = ModelParams.load(path)
    print("LOAD PATH", path)
    return load(path, params, best=best, load_env=load_env)

def load(path: str, params : ModelParams, best=False, load_env=True, log_dir=None):
    if not path.startswith('/'):
        path = os.path.join(BASE, path)
    files = os.listdir(path)
    if not 'final_model.zip' in files and 'best_model.zip' in files:
        model_path = path + '/best_model.zip'
    elif 'best_model.zip' in files and best:
        model_path = path + '/best_model.zip'
    elif 'final_model.zip' in files:
        model_path = path + '/final_model.zip'
    else:
        raise ValueError("Cannot find a model for name: " + path)
    # get model
    alg = get_alg(params)
    model = alg.load(model_path, tensorboard_log=log_dir, **params['alg_args'])
    if load_env:
        env = get_env(params)
    else:
        env = None
    return model, env

def load_hrl_from_name(low_path, high_path, load_env=True, best=False):
    if not low_path.startswith('/'):
        low_path = os.path.join(BASE, low_path)
    if not high_path.startswith('/'):
        high_path = os.path.join(BASE, high_path)
    low_params = ModelParams.load(low_path)
    high_params = ModelParams.load(high_path)
    return load_hrl(low_path, high_path, low_params, high_params, load_env=load_env, best=best)

def load_hrl(low_path, high_path, low_params : ModelParams, high_params : ModelParams, log_dir=None, load_env=True, best=False):
    params = merge_hrl_params(low_params, high_params)

    if not low_path.startswith('/'):
        low_path = os.path.join(BASE, low_path)
    low_files = os.listdir(low_path)
    if not 'final_model.zip' in low_files and 'best_model.zip' in low_files:
        low_model_path = low_path + '/best_model.zip'
    elif 'best_model.zip' in low_files and best:
        low_model_path = low_path + '/best_model.zip'
    elif 'final_model.zip' in low_files:
        low_model_path = low_path + '/final_model.zip'
    else:
        raise ValueError("Cannot find a model for name: " + low_model_path)

    if not high_path.startswith('/'):
        high_path = os.path.join(BASE, high_path)
    high_files = os.listdir(high_path)
    if not 'final_model.zip' in high_files and 'best_model.zip' in high_files:
        high_model_path = high_path + '/best_model.zip'
    elif 'best_model.zip' in high_files and best:
        high_model_path = high_path + '/best_model.zip'
    elif 'final_model.zip' in high_files:
        high_model_path = high_path + '/final_model.zip'
    else:
        raise ValueError("Cannot find a model for name: " + high_model_path)

    alg = get_alg(params)
    model = alg.load(low_model_path, high_model_path, tensorboard_log=log_dir)
    if load_env:
        env = get_env(params)
    else:
        env = None
    return model, env

def create_high_level_finetune_params(low_name, high_name, env_name, k=None, t=None, dm=None):
    high_params = ModelParams.load(high_name)
    low_params = ModelParams.load(low_name)
    
    high_params['env'] = env_name
    if 'delta_max' in high_params['env_args']:
        low_params['env_args']['delta_max'] = high_params['env_args']['delta_max']
    high_params['env_args'] = low_params['env_args']

    if k:
        high_params['env_args']['k'] = k
    if t:
        high_params['timesteps'] = t
    if dm:
        high_params['env_args']['delta_max'] = dm
    high_params['env_wrapper_args']['policy'] = low_name
    return high_params

def compose_params(low_name, high_name, env_name=None, k=None, t=None):
    print("############# COMPOSING", low_name, high_name)
    high_params = ModelParams.load(high_name)
    low_params = ModelParams.load(low_name)        
    print("Low env args", low_params['env_args'])
    print("High env args", high_params['env_args'])
    
    # Get Skill space arguments
    if 'delta_max' in high_params['env_args']:
        dm = high_params['env_args']['delta_max']
    elif 'delta_max' in low_params['env_args']:
        dm = low_params['env_args']['delta_max']
    else:
        dm = None
    if k is None:
        k = low_params['env_args']['k'] if 'k' in low_params['env_args'] else high_params['env_args']['k']

    # Get the actual environment name for the composed version.
    if env_name == "Maze":
        low_prefix = low_params['env'].split('_')[0]
        if low_prefix.startswith('Point'):
            env_name = 'PointMaze_High'
        else:
            env_name = low_prefix + 'Maze_High'
        high_params['env'] = env_name
    elif env_name == "Steps":
        low_prefix = low_params['env'].split('_')[0]
        if low_prefix.startswith('Point'):
            env_name = 'PointSteps_High'
        else:
            env_name = 'AntSteps_High'
        high_params['env'] = env_name
    elif not env_name is None:
        high_params['env'] = env_name
    else:
        if 'Hard' in high_params['env']:
            high_params['env'] = low_params['env'].split('_')[0] + 'Hard_High'
        else:
            high_params['env'] = low_params['env'].split('_')[0] + '_High'

    high_params['env_args'] = high_params['env_args'].copy()

    # Check For args to remove for assessment
    if 'rand_low_init' in high_params['env_args']:
        del high_params['env_args']['rand_low_init']
    if 'remove_table' in high_params['env_args']:
        del high_params['env_args']['remove_table']
    if 'agent_size' in high_params['env_args']:
        del high_params['env_args']['agent_size']
    if 'gear' in high_params['env_args']:
        del high_params['env_args']['gear']
    if 'random_exploration' in high_params['alg_args']:
        high_params['alg_args']['random_exploration'] = 0.0
    if 'include_contacts' in high_params['env_args']:
        del high_params['env_args']['include_contacts']
    # NOTE: Turn off sampling goals for evaluations.
    # In the paper this line nwas sometimes commented out for evaluation.
    if 'sample_goals' in high_params['env_args']:
        high_params['env_args']['sample_goals'] = False

    # Check for agent specific arguments to add in
    if 'agent_size' in low_params['env_args']:
        high_params['env_args']['agent_size'] = low_params['env_args']['agent_size']
    if 'gear' in low_params['env_args']:
        high_params['env_args']['gear'] = low_params['env_args']['gear']
    if 'include_contacts' in low_params['env_args']:
        high_params['env_args']['include_contacts'] = low_params['env_args']['include_contacts']
    
    # Add necesary args back
    high_params['env_args']['k'] = k
    if dm:
        high_params['env_args']['delta_max'] = dm
    high_params['env_wrapper_args']['policy'] = low_name

    # high_params['name'] = 'composition_' + low_name
    del low_params
    return high_params
