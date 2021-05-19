import numpy as np
import math
from robot.robot_core import Robot, Robotworld, Landmark
from multiagent.scenario import BaseScenario

np.random.seed(2)

class Scenario(BaseScenario):
    def make_world(self):
        # define scenario properties
        num_agents = 1
        num_objects = 1
        num_joints = 2
        arm_length = 0.35
        world_res = 16
        # create world
        world = Robotworld()
        world.discrete_world = True

        # add agents
        world.agents = [Robot() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            # agent.collide = True
            # agent.silent = True

        # add objects
        world.objects = [Landmark() for i in range(num_objects)]
        for i, object in enumerate(world.objects):
            object.name = 'object %d' % i

        # add goals
        world.goals = [Landmark() for i in range(1)]
        for i, goal in enumerate(world.goals):
            goal.name = 'goal'

        world.num_joints = num_joints
        world.arm_length = arm_length
        world.resolution = world_res

        # reset world
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set agent properties
        origins = world.robot_position(len(world.agents))
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
            agent.state.lengths = world.arm_length * np.ones(world.num_joints)
            agent.state.angles = np.random.randint(world.resolution, size=world.num_joints)
            # agent.state.angles = np.array([0, 0.5]) * math.pi
            agent.state.p_pos = np.array(origins[i][:])

        # set properties for objects
        for i, object in enumerate(world.objects):
            object.color = np.array([0, 0, 1])
            # object.state.p_pos = world.random_object_pos()
            # object.state.p_pos = np.array([0.4, 0.4])
            object.state.angles = np.random.randint(world.resolution, size=world.num_joints)
            object.state.p_pos = world.get_position(object)
            # print(f" test = {world.get_position(object)}")

        # set properties for goal
        # world.goals[0].state.p_pos = world.random_object_pos()
        world.goals[0].state.angles = np.random.randint(world.resolution, size=world.num_joints)
        world.goals[0].state.p_pos = world.get_position(world.goals[0])
        world.goals[0].color = np.array([1, 0, 0])

    def reward(self, agent, world):




        r_grab = np.linalg.norm(world.objects[0].state.p_pos - agent.get_joint_pos(world.num_joints))
        r_goal = np.linalg.norm(world.goals[0].state.p_pos - world.objects[0].state.p_pos)
        return -r_grab - 2 * r_goal


    def observation(self, agent, world):
        state_obs = []
        for i in range(world.num_joints):
            # state_obs += agent.get_joint_pos(i).tolist()
            state_obs += agent.local_joint_pos(i).tolist()

        grasp_obs = [agent.state.grasp]
        return state_obs + grasp_obs
        # # initialize observation variables
        # state_obs, object_obs, partner_obs, goal_obs = [], [], [], []
        # grasp_obs = [agent.state.grasp]
        # # update agent state observation
        # for i in range(1, world.num_joints + 1):
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
        # goal_obs = np.ndarray.tolist(world.goals[0].state.p_pos - agent.state.p_pos)

        # return state_obs + grasp_obs + object_obs + partner_obs + goal_obs
        return []

        #
        # # initialize observation variables
        # # state_observations = (agent.state.angles / math.pi).tolist() + [agent.state.grasp]
        # goal_obs = world.goals[0].state.p_pos.tolist() # cartesian goal obs
        # # goal_obs =
        # object_obs = world.objects[0].state.p_pos.tolist() # cartesian obs
        # object_dist = []
        # partners = []
        # state_obs = []
        # grasp_obs = [agent.state.grasp]
        # for joint in range(1, world.num_joints + 1): # cartesian observation
        #     state_obs += agent.get_joint_pos(joint).tolist()
        #
        # # for joint in range(world.num_joints):
        # #     state_obs += [agent.state.angles[joint]]
        #
        #
        # # fill in object observation for every object in the environment
        # for object in world.objects:
        #     # determine relative distance to every object in the environment
        #     dist = np.sum(np.square(object.state.p_pos - agent.position_end_effector()))
        #     object_dist += [dist]
        # # print(f'object_dist test = {object_dist}')
        # # when partner agents available, obtain their information
        # if len(world.agents) > 1:
        #     for partner in world.agents:
        #         # only for partner agents
        #         if agent.name != partner.name:
        #             # determine relative distance to other agent's end effector
        #             diff = partner.position_end_effector() - agent.position_end_effector()
        #             partners += [np.linalg.norm(diff)] + [partner.state.grasp]
        # # combine observations to a single numpy array
        # return state_obs + grasp_obs + object_obs + partners + goal_obs
        # # return state_obs + grasp_obs + object_dist + partners + goal_obs
        # # return state_observations + object_pos + partners + goal_obs



