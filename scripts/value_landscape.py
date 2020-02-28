from bot_transfer.utils.loader import load_from_name
from bot_transfer.envs.hierarchical import LowLevelEnv
import stable_baselines
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str)
parser.add_argument("--num", "-n", type=int, default=100)
args = parser.parse_args()


model, env = load_from_name(args.path)

if isinstance(model, stable_baselines.PPO2):
    value_fn = model.value
elif isinstance(model, stable_baselines.SAC):
    def sac_value_fn(x, model):
        val = model.sess.run(model.value_fn, {model.observations_ph : x})
        return val
    value_fn = lambda x: sac_value_fn(x, model)[0]

init_state = env.reset()
skills = list()
values = list()
for _ in range(args.num):
    skill = env.env.env.skill_space().sample()
    env.env.env.set_skill(skill)
    state = env.env.env.agent_state_func(env.env.env.state())
    skills.append(skill)
    values.append(value_fn(np.expand_dims(state, axis=0)))

x = [skill[0] for skill in skills]
y = [skill[1] for skill in skills]
z = [value[0] for value in values]
print(len(x), len(y), len(z))
print(x[0], y[0], z[0])
ax = plt.gca()
ax.tricontour(x, y, z, levels=14)
cntr2 = ax.tricontourf(x, y, z, levels=14)

plt.colorbar(cntr2, ax=ax)
ax.plot(x, y, 'ko', ms=0.75)
plt.title('Values: ' + args.path)
plt.show()




