import argparse
import torch
import time
import os
import imageio
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# from utils.make_env import make_env
from make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from subprocess import Popen
import webbrowser

USE_CUDA = torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run_evaluation(config, ep_i, gif_path, ifi, maddpg):
    print(f'Evaluating episode {ep_i + 1}...')
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
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
                     for i in range(maddpg.nagents)]
        # get actions as torch Variables
        torch_actions = maddpg.step(torch_obs, explore=False)
        # convert actions to numpy arrays
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
        # imageio.core.format.Reader.close()
        env.viewers[0].close()
        env.close()


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    # rew_dir = run_dir / 'rewards'
    os.makedirs(log_dir)
    # os.makedirs(rew_dir)
    logger = SummaryWriter(str(log_dir))

    if config.save_gifs:
        gif_path = run_dir / 'gifs'
        gif_path.mkdir(exist_ok=True)
    ifi = 1 / config.fps  # inter-frame interval

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  gamma=0.95,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    # print(f"Is the action discrete {maddpg.discrete_action}")
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0

    # start up tensorboard
    cmd = "tensorboard --logdir='models/" + config.env_id + "/" + config.model_name + "/" + curr_run + "/logs'"
    Popen(cmd, shell=True)
    url = 'http://localhost:6012/' # still have to manually change this

    evaluation_eps = []
    if config.n_evaluations != 0:
        # create a list of evaluation episodes dependent on configuration
        interval = int(config.n_episodes / config.n_rollout_threads / config.n_evaluations)
        evaluation_eps = list(range(0, config.n_episodes, config.n_rollout_threads))
        evaluation_eps = evaluation_eps[-1:0:-interval]


    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    # print(f'now training at ep_i = {ep_i} and et_i = {et_i}')
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
            if et_i == 0: logger.add_scalars('agent0/rewards/',
                                            {'random_agent_reward': rewards.mean()}, ep_i)

        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalars('agent%i/rewards/' % a_i,
                              {'mean_episode_rewards': a_ep_rew}, ep_i)
            # logger.add_scalar('agent%i/random_agent_reward' % a_i, init_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

        # if (ep_i/config.n_rollout_threads) % rel_interval == rel_interval - 1:
        #     run_evaluation(config, ep_i, gif_path, ifi, maddpg)
        #     print(f'this is ep_i = {ep_i}')

        if ep_i in evaluation_eps:
            run_evaluation(config, ep_i, gif_path, ifi, maddpg)
            print(f'this is ep_i = {ep_i}')

        # show tensorboard in browser
        if ep_i == config.save_interval:
            webbrowser.open_new_tab(url)




    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=4, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=5000, type=int)
    parser.add_argument("--init_noise_scale", default=1.0, type=float)
    parser.add_argument("--final_noise_scale", default=1.0, type=float)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    # add for evaluation
    parser.add_argument("--save_gifs", default=True,
                        action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--n_evaluations", default=0, type=int)
    parser.add_argument("--fps", default=30, type=int)
    # parser.add_argument("--eval", default=True, type=bool)

    config = parser.parse_args()

    run(config)
