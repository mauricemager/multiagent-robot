import numpy as np
import math
from robot.robot_core import Robot, Robotworld, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        # define scenario properties
        num_agents = 1
        num_objects = 0
        num_joints = 1
        arm_length = 0.35

        # create world
        world = Robotworld()

        # add agents
        world.agents = [Robot() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True

        # add objects
        world.objects = [Landmark() for i in range(num_objects)]
        for i, object in enumerate(world.objects):
            object.name = 'object %d' % i

        # add goals
        world.goals = [Robot() for i in range(1)]
        for i, goal in enumerate(world.goals):
            goal.name = 'end_pos'

        # add world specifications
        world.num_joints = num_joints
        world.arm_length = arm_length

        # reset world
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set agent properties
        origins = world.robot_position(len(world.agents))
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
            agent.state.lengths = world.arm_length * np.ones(world.num_joints)
            agent.state.angles = (2 * np.random.rand(world.num_joints) - 1) * math.pi
            agent.state.p_pos = np.array(origins[i][:])

        # set properties for goal
        world.goals[0].color = np.array([0.5, 0.1, 0.1])
        world.goals[0].state.p_pos = np.array(origins[i][:])
        world.goals[0].state.lengths = world.arm_length * np.ones(world.num_joints)
        world.goals[0].state.angles = (2 * np.random.rand(world.num_joints) - 1) * math.pi

    def reward(self, agent, world):
        # reward = 0.0
        # for i in range(world.num_joints):
        #     # ang_g = world.goals[0].state.angles[i]
        #     # ang_a = agent.state.angles[i]
        #     # reward += np.square(math.cos(ang_g) - math.cos(ang_a)) + np.square(math.sin(ang_g) - math.sin(ang_a))
        #     reward += np.absolute(world.goals[0].state.angles[i] - agent.state.angles[i])
        #     if reward > math.pi: reward = 2 * math.pi - reward
        reward = 0.0
        for i in range(len(agent.state.p_pos)):
            reward += np.square(agent.position_end_effector()[i] - world.goals[0].position_end_effector()[i])
        return -reward

    def observation(self, agent, world):
        # initialize observation variables
        state_observations = (agent.state.angles / math.pi).tolist()
        state_observations = agent.position_end_effector().tolist()
        # goal_observation = (world.goals[0].state.angles / math.pi).tolist()
        goal_observation = world.goals[0].position_end_effector().tolist()
        return state_observations + goal_observation


