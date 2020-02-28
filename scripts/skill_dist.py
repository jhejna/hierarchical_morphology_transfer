from bot_transfer.utils.loader import ModelParams, load, BASE
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str)
parser.add_argument("--num", "-n", type=int, default=100)
parser.add_argument("--limit", "-l", type=int, default=100)

args = parser.parse_args()

params = ModelParams.load(args.path)

model, env = load(args.path, params, best=False)

skills = list()
for _ in range(args.num):
    obs = env.reset()
    rewards = list()
    for i in range(args.limit):
        action, _ = model.predict(obs, deterministic=False)
        print(action)
        skills.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            print("episode finished", sum(rewards), len(rewards))
            break

skills = np.array(skills)
plt.title('Skill Histogram: ' + args.path)
plt.hist2d(skills[:, 0], skills[:,1], bins=(10,10))
plt.colorbar()
plt.show()