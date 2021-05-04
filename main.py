#!/usr/bin/env python
# fork multiagent-particle-envs from openai and uncomment the following:
# line 7 in multiagent.multi_discrete.py: # from gym.spaces import prng
# line 14 in multiagent.rendering.py: # from gym.utils import reraise

import argparse
from robot.robot_policy import RobotPolicy
from robot.robot_environment import RobotEnv
# import multiagent.scenarios as scenarios
import robot.robot_scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='robotarm.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = RobotEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [RobotPolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # print(f'observation_n = {obs_n}')
        # print(f" joint0 pos = {type(world.agents[0].get_joint_pos(0))}"
        #       f" joint1 pos = {world.agents[0].get_joint_pos(1)}"
        #       f" joint2 pos = {world.agents[0].get_joint_pos(2)}")

        # print(f"goal angles = {world.goals[0].state.angles}"
        #       f" and agent angle = {world.agents[0].state.angles}"
        #       f" reward = {reward_n}")
        print(f'reward = {reward_n}')
        # print(f'objects pos = {world.objects[0].state.p_pos} '
        #       f'and goals pos = {world.goals[0].state.p_pos} ')
        # print(f"Theta agent = {env.world.agents[0].state.angles}"
        #       f"and Theta goal = {env.world.goals[0].state.angles}"
        #       f"and reward = {env._get_reward(env.world.agents[0])}")
        # render all agent view
        env.render()
        # display rewards
        # for agent in env.world.agents:
           # print(agent.name + " reward: %0.3f" % env._get_reward(agent))


