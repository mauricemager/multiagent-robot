import numpy as np
from robot.robot_core import Robot, Robotworld, Landmark
from multiagent.scenario import BaseScenario

np.random.seed(2)

class Scenario(BaseScenario):
    def make_world(self):
        # define scenario properties
        num_agents = 2
        num_objects = 1
        num_goals = 1
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
        world.goals = [Landmark() for i in range(num_goals)]
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
            agent.state.angles = (2 * np.random.rand(world.num_joints) - 1) * np.pi
            # agent.state.angles = (np.array([0.0,0.5])) * math.pi
            agent.state.p_pos = np.array(origins[i][:])
            agent.state.grasp = True

        # set properties for objects
        for i, object in enumerate(world.objects):
            object.color = np.array([0, 0, 1])
            object.state.p_pos = world.agents[0].position_end_effector()


        # set goal properties
        world.goals[0].state.p_pos = np.array([-1.0, 0.0])
        world.goals[0].color = np.array([1, 0, 0])
        # world.touched = False

        # manual adjustments
        # world.agents[0].state.angles = np.array([0, 0])
        # world.agents[1].state.angles = np.array([math.pi/2, 0])

    def reward(self, agent, world):
        reward = np.linalg.norm(world.goals[0].state.p_pos - world.objects[0].state.p_pos)
        if world.objects[0].name is None: reward += 0.5
        return -reward


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
        if len(world.goals) > 0:
            goal_obs = np.ndarray.tolist(world.goals[0].state.p_pos - agent.state.p_pos)

        return state_obs + grasp_obs + object_obs + partner_obs + goal_obs


