import argparse, time, torch, imageio
import numpy as np
from pathlib import Path
from algorithms.maddpg import MADDPG
from make_env import make_env
from torch.autograd import Variable



def run(config):
    # task1 = 'models/task1/june23/run1/model.pt'
    # task2 = 'models/task2/june23/run18/model.pt'
    # task3 = 'models/task3/june23/run3/model.pt'
    # task4 = 'models/task4/june24/run1/model.pt'

    task1 = 'trained_models/task1/model.pt'
    task2 = 'trained_models/task2/model.pt'
    task3 = 'trained_models/task3/model.pt'
    task4 = 'trained_models/task4/model.pt'

    task_paths = [task1, task2, task3, task4]

    model_path = (Path('./models') / config.env_id / config.model_name) / 'model.pt'
    policies = [MADDPG.init_from_save(path) for path in task_paths]

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(parents=True, exist_ok=True)

    env = make_env(config.env_id)
    for pol in policies:
        pol.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        env.render('human')
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(len(env.agents))]
            torch_actions = policies[choose_task(torch_obs) - 1].step(torch_obs, explore=False)

            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)

    env.close()

def choose_task(obs):
    """Hierarchical policy determines task"""
    # relative agent gripper position from torch obvervation
    agent0 = obs[0][0][2:5]
    agent1 = obs[1][0][2:5]
    # object position relative to respective agent
    obj0 = torch.cat((obs[0][0][5:7], torch.tensor([1.0])))
    obj1 = torch.cat((obs[1][0][5:7], torch.tensor([1.0])))
    # boolean to show if respective agent is picking up the object
    agent0_picked = (sum(agent0 == obj0) == 3).item()
    agent1_picked = (sum(agent1 == obj1) == 3).item()
    # object distances to agent end effector in local perspective
    obj_grip_agent0 = np.linalg.norm(agent0[0:2] - obj0[0:2])
    # object distance to agent base in local perspective
    obj_base_agent0 = np.linalg.norm(obj0[0:2])

    if obj_base_agent0 <= 0.5 or obj_grip_agent0 <= 0.2 :
        if agent0_picked:
            print(f"Do task 4: left agent brings object to goal")
            return 4
        else:
            print(f"Do task 3: transfer object from right agent to left agent")
            return 3
    else:
        if agent1_picked:
            print(f"Do task 2: bring object to the left agent")
            return 2
        else:
            print(f"Do task 1: let right agent pick up object")
            return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    # parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_gifs", default=True,
                        action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--n_episodes", default=5, type=int)
    parser.add_argument("--episode_length", default=250, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    run(config)