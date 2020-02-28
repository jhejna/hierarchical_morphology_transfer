import os
from bot_transfer.utils.loader import get_env, get_paths, get_alg, get_policy, ModelParams
from bot_transfer.utils.loader import merge_hrl_params, split_hrl_params
from stable_baselines.bench import Monitor
from stable_baselines import logger
import numpy as np
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.misc_util import mpi_rank_or_zero
from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals, data_dir, freq=None, low_level_data_dir=None, checkpoint_freq=None):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    if not freq:
        freq = 100000
    global n_steps, best_mean_reward
    # Print stats every freq calls
    if (n_steps + 1) % freq == 0:
        if low_level_data_dir:
            x, y = ts2xy(load_results(data_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-200:])
                print(x[-1], 'timesteps')
                print("Best 200 mean reward: {:.2f} - Last 2000 mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model.")
                    _locals['self'].save(low_level_data_dir + '/best_model', data_dir + '/best_model')
        else:
            params = ModelParams.load(data_dir)
            env = get_env(params)
            ep_rewards = list()
            for _ in range(4):
                rewards = list()
                obs = env.reset()
                while True:
                    ac = _locals['self'].predict(obs)
                    obs, reward, done, _ = env.step(ac[0])
                    rewards.append(reward)
                    if done:
                        break
                ep_rewards.append(sum(rewards))
            
            mean_reward = sum(ep_rewards) / 100.0
            print("Best 100 mean reward: {:.2f} -  Last mean 100 Ep reward: {:.2f}".format(best_mean_reward, mean_reward))
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model.")
                _locals['self'].save(data_dir + '/best_model')
            del env

        '''
        # Evaluate policy training performance
        x, y = ts2xy(load_results(data_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-200:])
            print(x[-1], 'timesteps')
            print("Best 200 mean reward: {:.2f} - Last 2000 mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                if low_level_data_dir:
                    _locals['self'].save(low_level_data_dir + '/best_model', data_dir + '/best_model')
                else:
                    _locals['self'].save(data_dir + '/best_model')
        '''
    if not checkpoint_freq is None and (n_steps + 1) % checkpoint_freq == 0:    
        print("Saving Model Checkpoint")
        name = "/checkpoint_" + str(n_steps + 1)
        if low_level_data_dir:
            _locals['self'].save(low_level_data_dir + name, data_dir + name)
        else:
            _locals['self'].save(data_dir + name)

    n_steps += 1
    return True

def create_training_callback(data_dir, freq=None, low_level_data_dir=None, checkpoint_freq=None):
    return lambda _locals, _globals: callback(_locals, _globals, data_dir, freq=freq, low_level_data_dir=low_level_data_dir, checkpoint_freq=checkpoint_freq)

def train(params, model=None, path=None):
    if model: # indicate in filename that this is a finetune
        if params['name']:
            params['name'] += '_Finetune'
        else:
            params['name'] = 'Finetune'
    
    data_dir, tb_path = get_paths(params, path=path)
    print("Training Parameters: ", params)
    os.makedirs(data_dir, exist_ok=True)
    # Save parameters immediatly
    params.save(data_dir)

    rank = mpi_rank_or_zero()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    
    def make_env(i):
        env = get_env(params)
        env = Monitor(env, data_dir + '/' + str(i), allow_early_resets=params['early_reset'])
        return env

    use_her = params['env_args']['use_her'] if 'use_her' in params['env_args'] else False

    if use_her:
        env = make_env(0)
        goal_selection_strategy = 'future'
    else:
        env = DummyVecEnv([(lambda n: lambda: make_env(n))(i) for i in range(params['num_proc'])])

    if model: # indicate in filename that this is a finetune
        print("Model action space", model.action_space, model.action_space.low)
        print("Env action space", env.action_space, env.action_space.low)
    if params['normalize']:
        env = VecNormalize(env)
    if params['seed']:
        seed = params['seed'] + 100000 * rank
        set_global_seeds(seed)
        params['alg_args']['seed'] = seed
    if 'noise' in params and params['noise']:
        from stable_baselines.ddpg import OrnsteinUhlenbeckActionNoise
        n_actions = env.action_space.shape[-1]
        params['alg_args']['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(params['noise'])*np.ones(n_actions))
    
    if model is None:
        alg = get_alg(params)
        policy = get_policy(params)
        if use_her:
            from stable_baselines import HER
            model = HER(policy, env, alg, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy, verbose=1, 
                            tensorboard_log=tb_path, policy_kwargs=params['policy_args'], **params['alg_args'])
        else:
            model = alg(policy,  env, verbose=1, tensorboard_log=tb_path, policy_kwargs=params['policy_args'], **params['alg_args'])
    else:
        model.set_env(env)

    model.learn(total_timesteps=params['timesteps'], log_interval=params['log_interval'], callback=create_training_callback(data_dir, 
                                                    freq=params['eval_freq'], checkpoint_freq=params['checkpoint_freq']))
    print("######## SAVING MODEL TO", data_dir)
    model.save(data_dir +'/final_model')
    if params['normalize']:
        env.save(data_dir + '/normalized_environment.env')
    env.close()

def train_hrl(low_params, high_params, high_training_starts=0, model=None, path=None):
    if model: # indicate in filename that this is a finetune
        if low_params['name']:
            low_params['name'] += '_Finetune'
        else:
            low_params['name'] = 'Finetune'
        if high_params['name']:
            high_params['name'] += '_Finetune'
        else:
            high_params['name'] = 'Finetune'

    params = merge_hrl_params(low_params, high_params)
    data_dir, tb_path = get_paths(params, path=path)

    data_dir_components = data_dir.split('_')
    data_dir_components.insert(-1, 'Low')
    low_data_dir = '_'.join(data_dir_components)
    data_dir_components[-2] = 'High'
    high_data_dir =  '_'.join(data_dir_components)

    os.makedirs(high_data_dir, exist_ok=True)
    os.makedirs(low_data_dir, exist_ok=True)
    # Enforce consistency across params by using the split function.
    low_params, high_params = split_hrl_params(params)
    print("HRL PARAMS")
    print("High Params", high_params)
    print("low Params", low_params)
    high_params['env_wrapper_args']['policy'] = '/'.join(low_data_dir.split('/')[-2:])

    low_params.save(low_data_dir)
    high_params.save(high_data_dir)
    
    def make_env(i):
        env = get_env(params)
        print("ENVIRONMENT", env)
        env = Monitor(env, high_data_dir + '/' + str(i), allow_early_resets=params['early_reset'],
                    info_keywords=('low_ep_info',))
        return env

    env = DummyVecEnv([(lambda n: lambda: make_env(n))(i) for i in range(params['num_proc'])])

    if params['normalize']:
        env = VecNormalize(env)

    seed = params['seed']
    if seed:
        set_global_seeds(seed)
        params['alg_args']['seed'] = seed
    
    if model is None:
        alg = get_alg(params)
        policy = get_policy(params)
        model = alg(policy, policy, env, verbose=1, tensorboard_log=tb_path, high_policy_kwargs=params['high_policy_args'],
                                            low_policy_kwargs=params['low_policy_args'], 
                                            **{'low_' + key : value for key, value in params['low_alg_args'].items()},
                                            **{'high_' + key : value for key, value in params['high_alg_args'].items()})
    else:
        model.set_env(env)

    model.learn(total_timesteps=params['timesteps'], log_interval=int(params['log_interval']/4), 
                callback=create_training_callback(high_data_dir, low_level_data_dir=low_data_dir, freq=params['eval_freq'],
                checkpoint_freq=params['checkpoint_freq']), high_training_starts=high_training_starts)
    
    model.save(low_data_dir +'/final_model', high_data_dir + '/final_model')
    if params['normalize']:
        env.save(data_dir + '/normalized_environment.env')
    env.close()