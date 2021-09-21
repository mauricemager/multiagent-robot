# calculate scores for self-play experiment

import torch
import numpy as np
from algorithms.maddpg import MADDPG
from make_env import make_env
from torch.autograd import Variable

#TODO: make sure seed is the same
# make vectors of strings
eps_steps = 200

path_seed1 = "models/collaborative_task3/aug25/run17/model.pt"
path_seed2 = "models/collaborative_task3/aug25/run18/model.pt"
path_seed3 = "models/collaborative_task3/aug25/run19/model.pt"
paths = [path_seed1, path_seed2, path_seed3]

PI = []
# PI.append(MADDPG.init_2_from_save(path_seed1, path_seed2))
for i in range(len(paths)):
    for j in range(len(paths)):
        PI.append(MADDPG.init_2_from_save(paths[i], paths[j]))

# print(f"lengths of PI = {len(PI)}")


# pi_1 = MADDPG.init_from_save(path_seed1)
# pi_2 = MADDPG.init_from_save(path_seed2)
# pi_3 = MADDPG.init_from_save(path_seed3)

# PI_mix = MADDPG.init_2_from_save(path_seed1, path_seed2)

# pi_3.prep_rollouts(device="cpu")


XP_returns = np.ones(len(PI)) * np.inf

for run in range(len(PI)):
    print(f'Cross play run: {run}')
    PI[run].prep_rollouts(device="cpu")
    env = make_env('collaborative_task3')
    obs = env.reset()
    env.render('human')
    rewards = np.ones(eps_steps) * np.inf
    for step in range(eps_steps):
        torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1), requires_grad=False)
                     for i in range(len(env.agents))]
        torch_actions = PI[run].step(torch_obs, explore=False)
        actions = [ac.data.numpy().flatten() for ac in torch_actions]
        obs, reward, dones, infos = env.step(actions)
        env.render('human')
        rewards[step] = reward[0]
    XP_returns[run] = sum(rewards) / eps_steps
    # env.close()


#
# for step in range(eps_steps):
#     torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1), requires_grad=False)
#                  for i in range(len(env.agents))]
#     torch_actions = pi_3.step(torch_obs, explore=False)
#     actions = [ac.data.numpy().flatten() for ac in torch_actions]
#     obs, reward, dones, infos = env.step(actions)
#     rewards[step] = reward[0]
#     print(f'rewards = {reward[0]}')
#     env.render('human')
#
# # print(f"rewards = {rewards}")
# print(f"\nundiscounted reward = {sum(rewards)/eps_steps}")
env.close()
XP_returns += 1
print(XP_returns)

print(f'SP score = {(XP_returns[0]+XP_returns[4]+XP_returns[8]) / 3}')
print(f'XP score = {sum(XP_returns) / 9}')


