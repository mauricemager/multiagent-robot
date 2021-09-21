import argparse, time, torch, imageio
from pathlib import Path
from algorithms.maddpg import MADDPG
from make_env import make_env
from torch.autograd import Variable



def run(config):
    pick_path = 'models/pick_up/june17/run1/model.pt'
    drop_path = 'models/drop_off/june17/run1/model.pt'
    model_path = (Path('./models') / config.env_id / config.model_name) / 'model.pt'
    pick = MADDPG.init_from_save(pick_path)
    drop = MADDPG.init_from_save(drop_path)
    # maddpg = MADDPG.init_from_save(model_path)

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(parents=True, exist_ok=True)

    env = make_env(config.env_id)
    # maddpg.prep_rollouts(device='cpu')
    pick.prep_rollouts(device='cpu')
    drop.prep_rollouts(device='cpu')
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
                         for i in range(pick.nagents)]

            if drop_off_task(torch_obs):
                torch_actions = drop.step(torch_obs, explore=False)
            else:
                torch_actions = pick.step(torch_obs, explore=False)

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

def drop_off_task(obs):
    """Hierarchical policy based on the current observation, choose either to pick up or drop off the object"""
    gripper_obs = obs[0][0][2:5]
    object_obs = torch.cat((obs[0][0][5:7], torch.tensor([1.0])))
    if (sum(gripper_obs == object_obs) == 3).item():
        print(f'Dropping the object off now')
        return True
    else:
        print(f'Picking up the object!')
        return False

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
    parser.add_argument("--episode_length", default=150, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    run(config)