import numpy as np
import math
from robot.robot_core import Robot, Robotworld, Landmark
# from multiagent.core import Landmark
from multiagent.scenario import BaseScenario

np.random.seed(2)

class Scenario(BaseScenario):
    def make_world(self):
        # define scenario properties
        num_agents = 2
        num_objects = 1
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

        # set properties for objects
        # for i, object in enumerate(world.objects):
        #     object.color = np.array([0, 0, 1])
        #     object.state.p_pos = 0.2 * np.random.randn(world.dim_p) + np.random.choice([-1, 1]) * np.array([0.5, 0.0])

        # set properties for goal
        # world.goals[0].state.p_pos = - world.objects[0].state.p_pos
        # world.goals[0].color = np.array([1, 0, 0])

        # manual adjustments
        # world.objects[0].state.p_pos = np.array([1.2, 0])
        # world.goals[0].state.p_pos = np.array([-.4, 0])
        # world.agents[1].state.angles = np.array([0.0, 0.0])
        # world.agents[0].state.angles = np.array([0.0, 0.0])

    def reward(self, agent, world):
        # reward = 0.0 # reward not collective !!
        # print(f" test agent.state.p_pos = {type(agent.state.p_pos)} and object.state.p_pos = {world.objects[0].state.p_pos}")
        # for object in world.objects: # dit gaat alleen goed voor 1 object anders overwrite hij
        #     object_dist = np.sum(np.square(agent.state.p_pos - object.state.p_pos)) # check of dit goed gaat
        #     goal_dist = np.sum(np.square(object.state.p_pos - world.goals[0].state.p_pos)) # check of dit goed gaat
        #     print(f"agent {agent.name} has object_dist {object_dist} and goal_dist {goal_dist}")
        # reward = object_dist + 1.5 * goal_dist
        # return -reward


        # reward for agent's distance and object to goal
        # reward = 0.0
        # for agent in world.agents:
        #     reward += np.linalg.norm(world.objects[0].state.p_pos - agent.get_joint_pos(world.num_joints))
        # reward += 2 * np.linalg.norm(world.goals[0].state.p_pos - world.objects[0].state.p_pos)
        # return -reward

        reward = np.linalg.norm(world.objects[0].state.p_pos - agent.get_joint_pos(world.num_joints))
        reward += 3 * np.linalg.norm(world.goals[0].state.p_pos - world.objects[0].state.p_pos)
        # if reward <= 0.10: reward /= 3
        return -reward

        # reward = np.linalg.norm(world.goals[0].state.p_pos - world.objects[0].state.p_pos)
        # if reward <= 0.05: reward /= 3
        # return -reward * 10


        # reward = 0.0
        #
        # #
        # # for object in world.objects:
        # #     dist2 = np.sum(np.square(object.state.p_pos - world.goals[0].state.p_pos))
        # #     reward += dist2
        # return -reward

    def observation(self, agent, world):
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
        goal_obs = np.ndarray.tolist(world.goals[0].state.p_pos - agent.state.p_pos)

        return state_obs + grasp_obs + object_obs + partner_obs + goal_obs




        # print(f' test partner obs = {partner_obs}')

                    # determine relative distance to other agent's end effector


                    # diff = partner.position_end_effector() - agent.position_end_effector()
                    # partner_obs += [np.linalg.norm(diff)] + [partner.state.grasp]

        # print(f' test state_obs = {state_obs}')
        # print(f" agent state test = {agent.state.angles}")
        #
        # state_observations = (agent.state.angles / math.pi).tolist() + [agent.state.grasp]
        # object_pos = []
        # partners = []
        # # fill in object observation for every object in the environment
        # for object in world.objects:
        #     # determine relative distance to every object in the environment
        #     dist = np.sum(np.square(object.state.p_pos - agent.position_end_effector()))
        #     object_pos += [dist]
        # # when partner agents available, gain their information
        # if len(world.agents) > 1:
        #     for partner in world.agents:
        #         # only for partner agents
        #         if agent.name != partner.name:
        #             # determine relative distance to other agent's end effector
        #             diff = partner.position_end_effector() - agent.position_end_effector()
        #             partners += [np.linalg.norm(diff)] + [partner.state.grasp]
        # combine observations to a single numpy array
        # return [0,0,0,0,0,0,0,0,0]


