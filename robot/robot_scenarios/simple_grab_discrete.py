import numpy as np
from robot.robot_core import Robot, Robotworld, Landmark
from multiagent.scenario import BaseScenario

# np.random.seed(2)

class Scenario(BaseScenario):
    def make_world(self):
        # define scenario properties
        num_agents = 1
        num_objects = 1
        num_joints = 1
        arm_length = 0.35
        world_res = 8

        # create world
        world = Robotworld()
        world.discrete_world = True

        # add agents
        world.agents = [Robot() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i

        # add objects
        world.objects = [Landmark() for i in range(num_objects)]
        for i, object in enumerate(world.objects):
            object.name = 'object %d' % i

        # add goals
        world.goals = [Landmark() for i in range(1)]
        for i, goal in enumerate(world.goals):
            goal.name = 'goal'

        # add world properties
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
            agent.state.p_pos = np.array(origins[i][:])

        # set properties for objects
        for i, object in enumerate(world.objects):
            object.color = np.array([0, 0, 1])
            object.state.angles = np.random.randint(world.resolution, size=world.num_joints)
            object.state.p_pos = world.get_position(object)

        # set properties for goal
        world.goals[0].state.angles = np.random.randint(world.resolution, size=world.num_joints)
        # make sure that object and goal are not the same after initialization
        while world.goals[0].state.angles == world.objects[0].state.angles:
            world.goals[0].state.angles = np.random.randint(world.resolution, size=world.num_joints)
        world.goals[0].state.p_pos = world.get_position(world.goals[0])
        world.goals[0].color = np.array([1, 0, 0])


    def reward(self, agent, world):
        # reward based on agent's distance to object and twice the object's distance to goal
        r_goal = np.linalg.norm(world.goals[0].state.p_pos - world.objects[0].state.p_pos)
        r_object = np.linalg.norm(world.objects[0].state.p_pos - world.get_joint_pos(agent, 1))
        if world.objects[0].state.grabbed: r_goal *= 0.5
        return -2 * r_goal - r_object

    def observation(self, agent, world):





        state_obs, object_obs, goal_obs, picked_obs = [], [], [], [-1.0]
        for i in range(1, world.num_joints + 1):
            # state_obs += agent.get_joint_pos(i).tolist()
            state_obs += world.get_joint_pos(agent, i).tolist()
        grasp_obs = [1.0 if agent.state.grasp else -1.0]
        # if world.objects[0].state.grabbed and object.state.who_grabbed == agent.name:

        for object in world.objects:
            object_obs += np.ndarray.tolist(object.state.p_pos - agent.state.p_pos)
            if object.state.grabbed and object.state.who_grabbed == agent.name:
                picked_obs = [1.0]

        # for i in range(len(world.objects)):
        #     object_obs += np.ndarray.tolist(world.objects[i].state.p_pos - agent.state.p_pos)

        # update goal observation
        goal_obs = np.ndarray.tolist(world.goals[0].state.p_pos - agent.state.p_pos)
        return state_obs + grasp_obs + picked_obs + object_obs + goal_obs




