import numpy as np
import math
from robot.robot_core import Robot, Robotworld, Landmark
# from multiagent.core import Landmark
from multiagent.scenario import BaseScenario

# np.random.seed(2)

class Scenario(BaseScenario):
    def make_world(self):
        # define scenario properties
        num_agents = 2
        num_objects = 0
        num_joints = 2
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
            object.collide = True
            object.movable = True

        # add goals
        world.goals = [Landmark() for i in range(1)]
        for i, goal in enumerate(world.goals):
            goal.name = 'goal'
            goal.collide = False
            goal.movable = False

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
            # agent.state.angles = (np.array([0.0,0.5])) * math.pi
            agent.state.p_pos = np.array(origins[i][:])

        # set goal properties
        world.goals[0].state.p_pos = np.array([-1.0, 0.0])
        world.goals[0].color = np.array([1, 0, 0])
        world.touched = False

        # manual adjustments
        # world.agents[0].state.angles = np.array([0, 0])
        # world.agents[1].state.angles = np.array([math.pi/2, 0])

    def reward(self, agent, world):
        r_goal = np.linalg.norm(world.goals[0].state.p_pos - world.agents[0].position_end_effector())
        r_touch = np.linalg.norm(world.agents[0].position_end_effector() - world.agents[1].position_end_effector())
        if world.touched:
            return -r_goal
        else:
            return -r_touch -r_goal

    def observation(self, agent, world):
        # initialize observation variables
        state_obs, partner_obs  = [], []
        # update agent state observation
        for i in range(2, world.num_joints + 1):
            # state_obs += agent.get_joint_pos(i).tolist()
            state_obs += agent.local_joint_pos(i).tolist()
        # update object state observation
        if len(world.agents) > 1:
            for partner in world.agents:
                # only for partner agents
                if agent.name == partner.name: continue
                # partner state observations
                partner_obs += np.ndarray.tolist(partner.get_joint_pos(world.num_joints) - agent.state.p_pos)

                # endpoint_diff = partner.get_joint_pos(world.num_joints) - agent.get_joint_pos(world.num_joints)
                # obs = [np.linalg.norm(endpoint_diff)]

        touched_obs = [1.0] if world.touched else [0.0]

        return state_obs + partner_obs + touched_obs


