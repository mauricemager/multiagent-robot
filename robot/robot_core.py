from multiagent.core import AgentState, Agent, World, Entity
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

    def create_object_points(self, size=0.025):
        pos = self.state.p_pos
        points = size * np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) + pos
        return points.tolist()

    def create_goal_points(self, size=0.015):
        pos = self.state.p_pos
        points = size * np.array([[-2, -1], [0, 2.5], [2, -1]]) + pos
        return points.tolist()

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


    def position_end_effector(self):
        # give the position of the end effector
        return np.array(self.create_robot_points()[-1])

    def within_reach(self, object, grasp_range=0.06):
        # test whether and object is within grasping range for a robot
        end_pos = np.array(self.position_end_effector())
        obj_pos = np.array(object.state.p_pos)
        dist = np.linalg.norm(obj_pos - end_pos)
        return dist <= grasp_range


class Robotworld(World):
    def __init__(self):
        super(Robotworld, self).__init__()
        # define arm length of robots
        self.arm_length = None
        # joint per robot
        self.num_joints = None
        #
        self.goals = []
        # step when a full unit of torque is applied
        self.step_size = math.pi / 75

    @property
    def entities(self):
        return self.agents + self.objects + self.goals

    def step(self):
        for agent in self.agents:
            self.update_agent_state(agent)
            for object in self.objects: # TODO: limit to only one grabbing a object
                self.update_object_state(agent, object)

    def update_agent_state(self, agent):
        # change the agent state as influence of a step
        for i in range(len(agent.state.angles)):  # 2 when agent has 2 joints
            agent.state.angles[i] += agent.action.u[i] * self.step_size
            # make sure state stays within resolution
            if agent.state.angles[i] > math.pi: agent.state.angles[i] -= 2 * math.pi
            elif agent.state.angles[i] <= -math.pi: agent.state.angles[i] += 2 * math.pi
        # activate gripper when last action element == 1.0
        if agent.action.u[-1] > 0: agent.state.grasp = 1.0
        else: agent.state.grasp = 0.0

    def update_object_state(self, agent, object):
        # adjust the position of the object when manipulated by robot
        if (agent.within_reach(object) == True) and (agent.state.grasp == True):
            object.state.p_pos = agent.position_end_effector()

    def robot_position(self, n, r=0.5):
        # determine robot's origin position for different configurations
        if n == 1: return [[0, 0]]
        else:
            phi = 2 * math.pi / n
            position = [[r * math.cos(phi * i + math.pi),
                         r * math.sin(phi * i + math.pi)] for i in range(n)]
            return position

    def random_object_pos(self):
        # sample random object position from uniform distribution
        dist = np.random.random_sample() * self.arm_length * self.num_joints
        angle = np.random.random_sample() * 2 * math.pi
        random_pos = dist * np.array([math.cos(angle), math.sin(angle)])
        return random_pos