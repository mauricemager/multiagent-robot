import argparse, time, torch, imageio
import numpy as np
from pathlib import Path
from algorithms.maddpg import MADDPG
from make_env import make_env
from torch.autograd import Variable

task1 = 'trained_models/task1/model.pt'
pol = MADDPG.init_from_save(task1)
print(pol)