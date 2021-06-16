from multiagent.core import AgentState, Agent, World, Entity
from multiagent.rendering import make_circle
import math
import numpy as np


class RobotState(AgentState):
    def __init__(self):
        super(RobotState, self).__init__()
        # length of robot arm
        self.lengths = []
        # state angles per joint
        self.angles = []
        # robot is grasping something
        self.grasp = 0.0


class Landmark(Entity):
    def __init__(self):
        super().__init__()
        self.state.grabbed = False
        self.state.who_grabbed = None

    def create_object_points(self, size=0.025):
        pos = self.state.p_pos
        points = size * np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) + pos
        return points.tolist()

    def create_goal_points(self, size=0.015):
        pos = self.state.p_pos
        points = size * np.array([[-2, -1], [0, 2.5], [2, -1]]) + pos
        return points.tolist()

    def create_goal_points2(self, radius=0.075, res=30, filled=True):
        points = []
        for i in range(res):
            ang = 2 * math.pi * i / res
            points.append((math.cos(ang) * radius, math.sin(ang) * radius))
        return points + self.state.p_pos





class Robot(Agent):
    def __init__(self):
        super().__init__()
        # robot state
        self.state = RobotState()

    def create_robot_points(self, shorter_end=False):
        # returns a vector of the joint locations of a multiple joint robot arm
        points = [self.state.p_pos]
        lengths = self.state.lengths
        # cumulate state for defining relative joint positions
        cum_state = self.state.angles.cumsum()
        for i in range(len(self.state.angles)):
            length = lengths[i]
            # remove part of last arm for better rendering with gripper
            if shorter_end and (i == range(len(self.state.angles))[-1]):
                length -= 0.045
            # joint coordinates per segment
            joint_coordinates = [math.cos(cum_state[i]) * length,
                                 math.sin(cum_state[i]) * length]
            # add the joint coordinates to the points vector
            points.append([sum(x) for x in zip(points[i], joint_coordinates)])
        return points

    def create_gripper_points(self, radius=0.05, res=30, gripped=False):
        # return a vector of the gripper points for rendering
        # angle of gripper clearance
        phi = math.pi / 3
        points = list()
        # orientation of the gripper w.r.t. end effector
        orientation = sum(self.state.angles)
        # smaller gripper and clearance when gripped
        if gripped:
            radius *= 0.8
            phi /= 3
        # create the gripper points relative to end effector orientation
        for i in range(res):
            ang = 2 * math.pi * i / res
            if (ang >= 0.5 * phi) and (ang <= 2 * math.pi - 0.5 * phi):
                points.append([math.cos(ang + orientation) * radius,
                               math.sin(ang + orientation) * radius])
        # translate gripped points to end effector location
        points = np.array(points) + [self.position_end_effector()]
        return points

        # TODO: change name to global
    def get_joint_pos(self, joint):  # returns a numpy array of pos
        # only works for continuous
        if joint == 0: return self.state.p_pos
        angle = self.state.angles.cumsum()[joint - 1]  # cumsum because of relative angle definition
        pos = self.get_joint_pos(joint - 1) + self.state.lengths[joint - 1] * np.array([np.cos(angle), np.sin(angle)])
        # print(f" joint: {joint} gives pos: {pos}")
        # print(f" angles are: {self.state.angles}")
        return pos

    def local_joint_pos(self, joint):
        # only works for continuous
        # TODO: place to world
        if joint == 0: return np.array([0.0, 0.0])
        angle = self.state.angles.cumsum()[joint - 1]  # cumsum because of relative angle definition
        pos = self.local_joint_pos(joint - 1) + self.state.lengths[joint - 1] * np.array([np.cos(angle), np.sin(angle)])
        return pos

    def position_end_effector(self):
        # TODO: still used by continuous
        # give the position of the end effector
        return np.array(self.create_robot_points()[-1])

    def within_reach(self, world, object, grasp_range=0.075):
        dist = np.linalg.norm(world.get_position(self) - object.state.p_pos)
        return dist <= grasp_range



class Robotworld(World):
    def __init__(self):
        super(Robotworld, self).__init__()
        # define arm length of robots
        self.arm_length = None
        # joint per robot
        self.num_joints = None
        # list of all world goals
        self.goals = []
        # step when a full unit of torque is applied
        self.step_size = math.pi / 25
        # continuous or discrete world and action space
        self.discrete_world = False
        # state resolution for discrete case
        self.resolution = None
        #
        self.touched = False

    @property
    def entities(self):
        return self.agents + self.objects + self.goals

    def step(self):
        for i, agent in enumerate(self.agents):
            self.update_agent_state(agent, discrete=self.discrete_world)
            for object in self.objects:
                self.update_object_state(agent, object, discrete=self.discrete_world)
            # self.update_world_state()
            # print(f' Robots have kissed = {self.touched}')

    def update_world_state(self, threshold=0.10):
        dist = np.linalg.norm(self.agents[0].position_end_effector() - self.agents[1].position_end_effector())
        if dist < threshold:
            # print(f'Robots kiss!')
            self.touched = True

    def update_agent_state(self, agent, discrete=False):
        if discrete and sum(agent.action.u) > 0.0:
            # make sure agent.action.u is one-hot vector
            action = np.where(agent.action.u == 1)[0][0]
            # TODO: this code is ugly and only works for 1 or 2 joints per agent
            if action == 0:
                agent.state.angles[0] += 1
            elif action == 1:
                agent.state.angles[0] -= 1
            if self.num_joints == 2: # when 2 joints
                if action == 2:
                    agent.state.angles[1] += 1
                elif action == 3:
                    agent.state.angles[1] -= 1
            for i in range(len(agent.state.angles)):
                if agent.state.angles[i] >= self.resolution: agent.state.angles[i] %= self.resolution
                if agent.state.angles[i] < 0: agent.state.angles[i] += self.resolution

        elif not discrete: # then continuous action space
            # change the agent state as influence of a step
            for i in range(len(agent.state.angles)):  # 2 when agent has 2 joints
                agent.state.angles[i] += agent.action.u[i] * self.step_size
                # make sure state stays within resolution
                if agent.state.angles[i] > math.pi:
                    agent.state.angles[i] -= 2 * math.pi
                elif agent.state.angles[i] <= -math.pi:
                    agent.state.angles[i] += 2 * math.pi
            # activate gripper when last action element == 1.0
            if agent.action.u[self.num_joints] > 0:
                agent.state.grasp = 1.0
            else:
                agent.state.grasp = 0.0


    def update_object_state(self, agent, object, discrete=False):
        """"""

        if discrete and sum(agent.action.u) > 0.0: #TODO: only works for one agent

            action = np.where(agent.action.u == 1)[0][0]
            if self.num_joints == 1: action += 2 # to make scenario compatible with 2 joints
            if action == 4:  # close gripper
                if self.object_grabbable(agent, object):
                    object.state.grabbed = True
                    object.state.who_grabbed = agent.name
                    print(f'You should now be grabbing!!')
                agent.state.grasp = True
            elif action == 5:
                agent.state.grasp = False
                object.state.grabbed = False
                object.state.who_grabbed = None
            if object.state.grabbed and object.state.who_grabbed == agent.name:
                object.state.p_pos = self.get_position(agent)

        # continuous
        # adjust the position of the object when manipulated by robot
        elif not discrete:
            if (agent.within_reach(self, object) == True) and (agent.state.grasp == True):
                object.state.p_pos = agent.position_end_effector()
            # object.state.angles = agent.state.angles[0] # This works only for the simple_grab scenario

    # def update_object_state_discrete(self, agent, object):
    #     # only works for one agent
    #     if sum(agent.action.u) > 0.0:
    #         action = np.where(agent.action.u == 1)[0][0]
    #         if action == 4: # close gripper
    #             if self.object_grabbable(agent, object):
    #                 # object.state.p_pos = self.get_position(agent)
    #                 object.state.grabbed = True
    #                 object.state.who_grabbed = agent.name
    #                 # while agent.state.grasp:
    #                 #     object.state.p_pos = self.get_position(agent)
    #                 print(f'You should now be grabbing!!')
    #             agent.state.grasp = True
    #         elif action == 5:
    #             agent.state.grasp = False
    #             object.state.grabbed = False
    #             object.state.who_grabbed = None
    #     if object.state.grabbed and object.state.who_grabbed == agent.name:
    #         object.state.p_pos = self.get_position(agent)

    def object_grabbable(self, agent, object):
        if agent.state.grasp == False and (not object.state.grabbed) \
                and agent.within_reach(self, object):
            return True
        else:
            return False

    def robot_position(self, n, r=0.5):
        # determine robot's origin position for different configurations
        if n == 1:
            return [[0, 0]]
        else:
            phi = 2 * math.pi / n
            position = [[r * math.cos(phi * i + math.pi),
                         r * math.sin(phi * i + math.pi)] for i in range(n)]
            return position

    def object_position(self, agent_pos, radius=0.7):
        """ Sample object position from uniform unit circle centered around agent"""
        length = np.sqrt(np.random.uniform(0, radius**2))
        angle = np.pi * np.random.uniform(0, 2)
        object_pos = length * np.array([np.cos(angle), np.sin(angle)])
        return object_pos + agent_pos

    def random_object_pos(self):
        # sample random object position from uniform distribution
        dist = np.random.random_sample() * self.arm_length * self.num_joints
        angle = np.random.random_sample() * 2 * math.pi
        random_pos = dist * np.array([math.cos(angle), math.sin(angle)])
        return random_pos

    def get_position(self, entity):
        if not self.discrete_world: # temporary fix for continuous
            return entity.position_end_effector()
        pos = np.array([0.0, 0.0]) # only works for one agent
        angles = entity.state.angles.cumsum()  # cumsum because of relative angle definition
        step = 2 * np.pi / self.resolution
        for i in range(self.num_joints):
            # joint_pos = self.arm_length * np.array([np.cos(angles[i] * step), np.sin(angles[i] * step)])
            # print(f'{"test"}')
            # pos += joint_pos
            pos += self.arm_length * np.array([np.cos(angles[i] * step), np.sin(angles[i] * step)])
        return pos


    def get_joint_pos(self, agent, joint):
        if joint == 0: return np.array([0.0, 0.0])
        angle = agent.state.angles.cumsum()[joint - 1]
        if self.discrete_world:
            step = 2 * np.pi / self.resolution
        else: step = 1
        pos = self.get_joint_pos(agent, joint - 1) + agent.state.lengths[joint - 1] * np.array([np.cos(angle * step),
                                                                                                np.sin(angle * step)])
        return pos

    def create_robot_points(self, robot, shorter_end=False, discrete=False):
        # returns a vector of the joint locations of a multiple joint robot arm
        points = [robot.state.p_pos]
        lengths = robot.state.lengths
        # cumulate state for defining relative joint positions
        cum_state = robot.state.angles.cumsum()
        for i in range(len(robot.state.angles)):
            length = lengths[i]
            # remove part of last arm for better rendering with gripper
            if shorter_end and (i == range(len(robot.state.angles))[-1]):
                length -= 0.045
            # joint coordinates per segment

            if discrete:
                step = 2 * np.pi / self.resolution
                joint_coordinates = [math.cos(cum_state[i] * step) * length,
                                     math.sin(cum_state[i] * step) * length]
            else:
                joint_coordinates = [math.cos(cum_state[i]) * length,
                                     math.sin(cum_state[i]) * length]
            # add the joint coordinates to the points vector
            points.append([sum(x) for x in zip(points[i], joint_coordinates)])
        return points

    def create_gripper_points(self, robot, radius=0.05, res=30, gripped=False):
        # return a vector of the gripper points for rendering
        # angle of gripper clearance
        phi = math.pi / 3
        points = list()
        # orientation of the gripper w.r.t. end effector
        orientation = sum(robot.state.angles)
        if self.discrete_world:
            orientation *= 2 * np.pi / self.resolution
        # smaller gripper and clearance when gripped
        if gripped:
            radius *= 0.8
            phi /= 3
        # create the gripper points relative to end effector orientation
        for i in range(res):
            ang = 2 * math.pi * i / res
            if (ang >= 0.5 * phi) and (ang <= 2 * math.pi - 0.5 * phi):
                points.append([math.cos(ang + orientation) * radius,
                               math.sin(ang + orientation) * radius])
        # translate gripped points to end effector location
        points = np.array(points) + [self.get_position(robot)]
        return points