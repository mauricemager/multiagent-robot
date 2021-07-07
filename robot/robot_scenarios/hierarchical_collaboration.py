import numpy as np
from robot.robot_scenarios.collaborative_tasks import CollScenario
from robot.robot_core import Robot, Robotworld, Landmark, Hierworld, Hieragent


# set random seed
np.random.seed(2)

class Scenario(CollScenario):
    """Nothing yet here"""

    def make_world(self):
        # define scenario properties
        num_agents = 2
        num_objects = 1
        num_goals = 1
        num_joints = 2
        arm_length = 0.35

        # create world
        world = Hierworld()

        world.agents = [Hieragent()]


        # add robot agents
        world.roboworld.agents = [Robot() for i in range(num_agents)]
        for i, agent in enumerate(world.roboworld.agents):
            agent.name = 'agent %d' % i

        # add objects with landmark properties
        world.roboworld.objects = [Landmark() for i in range(num_objects)]
        for object in world.roboworld.objects:
            object.name = 'object %d' % i

        # add goals with landmark properties
        world.roboworld.goals = [Landmark() for i in range(num_goals)]
        for goal in world.roboworld.goals:
            goal.name = 'goal'

        # add scenario parameters to world
        world.roboworld.num_joints = num_joints
        world.roboworld.arm_length = arm_length

        # reset world
        self.reset_world(world)

        return world


    def reset_world(self, world):
        """Reset the world state """

        # set agent properties
        origins = world.roboworld.robot_position(len(world.roboworld.agents))
        for i, agent in enumerate(world.roboworld.agents):
            agent.color = np.array([0.25,0.25,0.25])
            agent.state.lengths = world.roboworld.arm_length * np.ones(world.roboworld.num_joints)
            agent.state.angles = (2 * np.random.rand(world.roboworld.num_joints) - 1) * np.pi
            agent.state.p_pos = np.array(origins[i][:])

        # set properties for objects
        for object in world.roboworld.objects:
            object.color = np.array([0, 0, 1])
            object.state.p_pos = world.roboworld.object_position(world.roboworld.agents[1].state.p_pos,
                                                                 radius=2*world.roboworld.arm_length)

        # set goal properties
        for goal in world.roboworld.goals:
            goal.state.p_pos = np.array([-1.0, 0.0])
            goal.color = np.array([1, 0, 0])

    def observation(self, agent, world):
        # """Observation function for multi agent collaborative task."""
        #
        # # initialize observation variables
        # state_obs, object_obs, partner_obs, goal_obs = [], [], [], []
        # grasp_obs = [agent.state.grasp]
        # # update agent state observation
        # for i in range(1, world.roboworld.num_joints + 1):
        #     # state_obs += agent.get_joint_pos(i).tolist()
        #     state_obs += agent.local_joint_pos(i).tolist()
        # # update object state observation
        # for i in range(len(world.objects)):
        #     object_obs += np.ndarray.tolist(world.objects[i].state.p_pos - agent.state.p_pos)
        # # update partner observation, if any (centralized observation, not relative)
        # if len(world.agents) > 1:
        #     for partner in world.agents:
        #         # only for partner agents
        #         if agent.name == partner.name: continue
        #         # partner state observations
        #         partner_obs += np.ndarray.tolist(partner.get_joint_pos(world.num_joints) - agent.state.p_pos)
        #         # for i in range(world.num_joints + 1):
        #         #     partner_obs += partner.get_joint_pos(i).tolist()
        #         # add partner grasp observation
        #         partner_obs += [partner.state.grasp]
        # # update goal observation
        # if len(world.goals) > 0:
        #     goal_obs = np.ndarray.tolist(world.goals[0].state.p_pos - agent.state.p_pos)

        return []