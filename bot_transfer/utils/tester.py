from bot_transfer.utils.loader import load_from_name, ModelParams, load, compose_params, load_hrl_from_name, BASE
import bot_transfer
import os
from bot_transfer.algs import hrl
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv

RENDERS = os.path.dirname(os.path.dirname(bot_transfer.__file__)) + '/renders'
BASE = bot_transfer.utils.loader.BASE

def test(obj, gif_steps):
    if isinstance(obj, str):
        model, env = load_from_name(obj, best=True)
    else:
        raise ValueError("Unexpected input type to load")
    
    # print("ENV", env)
    # env.env.env.early_low_termination = True
    env = DummyVecEnv([lambda: env])
    
    obs = env.reset()
    if gif_steps == 0:
        total_reward = 0
        ep_len = 0
        for _ in range(10000):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            ep_len += 1
            if done:
                print("EP length:", ep_len, " reward:", total_reward)
                total_reward = 0
                ep_len = 0
                obs = env.reset()
            frame = env.render()
    else:
        gif_frames = list()
        rewards = list()
        for i in range(gif_steps):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if 'frames' in info[0]:
                gif_frames.extend(info[0]['frames'])
        
            frame = env.render(mode='rgb_array')
            gif_frames.append(frame)
            rewards.append(reward)

        print("REWARD", sum(rewards))
        import imageio
        render_path = os.path.join(RENDERS, obj + '.gif')
        print("saving to ", render_path)
        os.makedirs(os.path.dirname(render_path), exist_ok=True)
        imageio.mimsave(render_path, gif_frames[::5], subrectangles=True, duration=0.05)

def test_hrl(low_name, high_name, g):
    model, env = load_hrl_from_name(low_name, high_name)
    # env = DummyVecEnv([lambda: env])
    obs = env.reset()
    total_reward = 0
    for _ in range(10000):
        # TODO: make it work for PPO
        high_action, _states = model.predict_skill(obs)
        obs, reward, done, info = env.step( (high_action, lambda x: (None, model.predict(x)[0])) )
        total_reward += reward
        if done:
            print("TOTAL EP REWARD", total_reward)
            total_reward = 0
            obs = env.reset()
        env.render()

def test_composition(low_name, high_name, env_name, g, k=None, num_ep=100):
    params = compose_params(low_name, high_name, env_name, k=k)
    model, env = load(high_name, params, best=True)
    print("COMPOSED PARAMS", params)
    print("ENV", env)
    ep_rewards = list()
    rewards = list()
    gif_frames = list()
    obs = env.reset()
    for _ in range(g):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if 'frames' in info:
            gif_frames.extend(info['frames'])
        # env.render()
        # frame = env.render(mode='rgb_array')
        # gif_frames.append(frame)
        rewards.append(reward)
        if done:
            ep_rewards.append(sum(rewards))
            print("REWARD", sum(rewards), len(rewards), "Ep to go:", num_ep, "cur avg", np.mean(ep_rewards))
            num_ep -= 1
            rewards = []
            if num_ep == 0:
                break
            obs = env.reset()

    # print('AVG EP REWARD', np.mean(ep_rewards))

    import imageio
    render_path = os.path.join(RENDERS, 'composition_' + low_name + '2.gif')
    os.makedirs(os.path.dirname(render_path), exist_ok=True)
    print("saving to ", render_path)
    imageio.mimsave(render_path, gif_frames[::4], subrectangles=True, duration=0.05)
    print("completed saving")


def composition_sweep(low_names, high_names, env_name=None, k=None, num_ep=100, success=False):
    import time
    f = open(os.getcwd() + '/data/comparison.results' + str(int(round(time.time() * 10000000))), 'w+')
    for high_name in high_names:
        data = list()
        for low_name in low_names:

            # Determine if we are dealing with a single seed directory or multiple seeds.
            try:
                test_load = ModelParams.load(low_name)
                test_load = ModelParams.load(high_name)
                multi_seed = False
            except ValueError:
                multi_seed = True

            if multi_seed:
                file_location_low = os.path.join(BASE, low_name) if not low_name.startswith('/') else low_name
                file_location_high = os.path.join(BASE, high_name) if not high_name.startswith('/') else high_name
                low_runs = sorted([os.path.join(low_name, run) for run in os.listdir(file_location_low) if not run.endswith('.log')])
                high_runs = sorted([os.path.join(high_name, run) for run in os.listdir(file_location_high) if not run.endswith('.log')])
                print(low_runs)
                print(high_runs)
                assert len(low_runs) == len(high_runs)
                run_list = zip(low_runs, high_runs)
            else:
                run_list = [(low_name, high_name)]

            seed_rewards = list()
            f.write("----------------------------------------------\n")
            for run_low, run_high in run_list:
                print("Composing", run_low, "with", run_high)
                params = compose_params(run_low, run_high, env_name=env_name, k=k)
                print("COMPOSED PARAMS", params)
                
                ep_rewards = list()
                model, env = load(run_high, params, best=True)
                obs = env.reset()
                for _ in range(num_ep):
                    rewards = list()
                    while True:
                        action, _states = model.predict(obs)
                        obs, reward, done, info = env.step(action)
                        rewards.append(reward)
                        if done:
                            if success:
                                val_to_add = 1.0 if sum(rewards) > 0 else 0.0
                            else:
                                val_to_add = sum(rewards)
                            ep_rewards.append(val_to_add)
                            obs = env.reset()
                            break
                env.close()
                del model
                del env
                seed_rew_mean = np.mean(ep_rewards)
                seed_rewards.append(seed_rew_mean)

                print("==============================")
                print("Run:", run_low, run_high, ":", seed_rew_mean)
                write_str = run_low + "\t" + run_high + "\t" + str(seed_rew_mean) + "\n"
                f.write(write_str)

            data.append((low_name, np.mean(seed_rewards), np.std(seed_rewards)))

        # print the resulting output
        print("=================================================")
        print("Results for high policy" + high_name)
        for name, score, std in data:
            print('{:<60} {:.2f} {:.2f}'.format(name[-55:], score, std))
        # Write it to a file
        f.write("==================== FINAL RESULTS ==========================\n")
        f.write("== Results for High Level: " + high_name + "\n")
        for name, score, std in data:
            f.write('{:<60} {:.2f} {:.2f}\n'.format(name[-55:], score, std))

    f.close()
    
    

    

    