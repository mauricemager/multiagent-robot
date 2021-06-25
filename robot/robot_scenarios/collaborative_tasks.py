import math
import numpy as np
from multiagent.scenario import BaseScenario
from robot.robot_core import Robot, Robotworld, Landmark

# set random seed
np.random.seed(2)

class CollScenario(BaseScenario):
    """Define the parent collaborative scenario."""

    def make_world(self):
        # define scenario properties
        num_agents = 2
        num_objects = 1
        num_goals = 1
        num_joints = 2
        arm_length = 0.35

        # create world
        world = Robotworld()

        # add robot agents
        world.agents = [Robot() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i

        # add objects with landmark properties
        world.objects = [Landmark() for i in range(num_objects)]
        for object in world.objects:
            object.name = 'object %d' % i

        # add goals with landmark properties
        world.goals = [Landmark() for i in range(num_goals)]
        for goal in world.goals:
            goal.name = 'goal'

        # add scenario parameters to world
        world.num_joints = num_joints
        world.arm_length = arm_length

        # reset world
        self.reset_world(world)

        return world

    def reset_world(self, world):
        """Reset the world state """

        # set agent properties
        origins = world.robot_position(len(world.agents))
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
            agent.state.lengths = world.arm_length * np.ones(world.num_joints)
            agent.state.angles = (2 * np.random.rand(world.num_joints) - 1) * math.pi
            agent.state.p_pos = np.array(origins[i][:])

        # set properties for objects
        for object in world.objects:
            object.color = np.array([0, 0, 1])
            object.state.p_pos = world.object_position(world.agents[1].state.p_pos, radius=2*world.arm_length)

        # set goal properties
        for goal in world.goals:
            goal.state.p_pos = np.array([-1.0, 0.0])
            goal.color = np.array([1, 0, 0])


    def reward(self, agent, world):
        """Reward function for total collaborative robotics task."""

        # reward based on object's distance to goal
        reward = np.linalg.norm(world.goals[0].state.p_pos - world.objects[0].state.p_pos)

        return -reward


    def observation(self, agent, world):
        """Observation function for multi agent collaborative task."""

        # initialize observation variables
        state_obs, object_obs, partner_obs, goal_obs = [], [], [], []
        grasp_obs = [agent.state.grasp]
        # update agent state observation
        for i in range(1, world.num_joints + 1):
            # state_obs += agent.get_joint_pos(i).tolist()
            state_obs += agent.local_joint_pos(i).tolist()
        # update object state observation
        for i in range(len(world.objects)):
            object_obs += np.ndarray.tolist(world.objects[i].state.p_pos - agent.state.p_pos)
        # update partner observation, if any (centralized observation, not relative)
        if len(world.agents) > 1:
            for partner in world.agents:
                # only for partner agents
                if agent.name == partner.name: continue
                # partner state observations
                partner_obs += np.ndarray.tolist(partner.get_joint_pos(world.num_joints) - agent.state.p_pos)
                # for i in range(world.num_joints + 1):
                #     partner_obs += partner.get_joint_pos(i).tolist()
                # add partner grasp observation
                partner_obs += [partner.state.grasp]
        # update goal observation
        if len(world.goals) > 0:
            goal_obs = np.ndarray.tolist(world.goals[0].state.p_pos - agent.state.p_pos)

        return state_obs + grasp_obs + object_obs + partner_obs + goal_obs


