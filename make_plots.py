import argparse, time, torch, imageio
import numpy as np
from pathlib import Path
from algorithms.maddpg import MADDPG
from make_env import make_env
from torch.autograd import Variable

task1 = 'trained_models/task1/model.pt'
pol = MADDPG.init_from_save(task1)
print(pol)

# !/usr/bin/env python3

'''
This script exctracts training variables from all logs from 
tensorflow event files ("event*"), writes them to Pandas 
and finally stores in long-format to a CSV-file including
all (readable) runs of the logging directory.
The magic "5" infers there are only the following v.tags:
[lr, loss, acc, val_loss, val_acc]
'''

import tensorflow as tf
import glob
import os
import pandas as pd

# Get all event* runs from logging_dir subdirectories
logging_dir = 'models/collaborative_task1/aug24/run3/logs/agent0/rewards/mean_episode_rewards'
event_paths = glob.glob(os.path.join(logging_dir, "*", "event*"))


# Extraction function
def sum_log(path):
    runlog = pd.DataFrame(columns=['metric', 'value'])
    try:
        for e in tf.train.summary_iterator(path):
            for v in e.summary.value:
                r = {'metric': v.tag, 'value': v.simple_value}
                runlog = runlog.append(r, ignore_index=True)

    # Dirty catch of DataLossError
    except:
        print('Event file possibly corrupt: {}'.format(path))
        return None

    runlog['epoch'] = [item for sublist in [[i] * 5 for i in range(0, len(runlog) // 5)] for item in sublist]

    return runlog


# Call & append
all_log = pd.DataFrame()
for path in event_paths:
    log = sum_log(path)
    if log is not None:
        if all_log.shape[0] == 0:
            all_log = log
        else:
            all_log = all_log.append(log)

# Inspect
print(all_log.shape)
all_log.head()

# Store
all_log.to_csv('all_training_logs_in_one_file.csv', index=None)