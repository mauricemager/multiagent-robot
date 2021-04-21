import numpy as np
import math
from robot.robot_core import Robot, Robotworld, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        # define scenario properties
        num_agents = 1
        num_objects = 1
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
        world.goals = [Landmark() for i in range(1)]
        for i, goal in enumerate(world.goals):
            goal.name = 'goal'

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

        # set properties for object
        for i, object in enumerate(world.objects):
            object.color = np.array([0, 0, 0])
            object.state.angles = (2 * np.random.rand() - 1) * math.pi
            object.state.p_pos = world.arm_length * np.array([math.cos(object.state.angles),
                                                              math.sin(object.state.angles)])

        # set properties for goal
        for i, goal in enumerate(world.goals):
            goal.color = np.array([1, 0, 0])
            goal.state.angles = (2 * np.random.rand() - 1) * math.pi
            goal.state.p_pos = world.arm_length * np.array([math.cos(goal.state.angles),
                                                            math.sin(goal.state.angles)])

    def reward(self, agent, world): # make suitable for multiple objects
        # reward = np.absolute(world.goals[0].state.angles - world.objects[0].state.angles) % math.pi
        reward = 0.0
        for i in range(len(agent.state.p_pos)):
            reward += (world.goals[0].state.p_pos[i] - world.objects[0].state.p_pos[i]) ** 2
        return -reward

    def observation(self, agent, world):
        # initialize observation variables
        state_observations = (agent.state.angles / math.pi).tolist()
        # state_observations = agent.position_end_effector().tolist()
        goal_observation = [world.goals[0].state.angles / math.pi]
        # goal_observation = world.goals[0].state.p_pos.tolist()
        return state_observations + goal_observation


