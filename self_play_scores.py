import argparse, time, torch, imageio
import numpy as np
from pathlib import Path
from algorithms.maddpg import MADDPG
from make_env import make_env
from torch.autograd import Variable

path_seed1 = "models/collaborative_task1/aug24/run6/model.pt"

pol_seed1 = MADDPG.init_from_save(path_seed1)
pol_seed1.prep_rollouts(device="cpu")
env = make_env('coltasks')
print('f')

obs = env.reset()
env.render('human')

for step in range(200):
    torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1), requires_grad=False)
                 for i in range(len(env.agents))]
    torch_actions = pol_seed1.step(torch_obs, explore=False)
    actions = [ac.data.numpy().flatten() for ac in torch_actions]
    obs, rewards, dones, infos = env.step(actions)
    # env.render('human')

env.close()
