from multiagent.multi_discrete import MultiDiscrete
from multiagent.environment import MultiAgentEnv
import numpy as np
# from robot.robot_core import Hieragent

# environment for all agents in the multiagent world
class RobotEnv(MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        super(RobotEnv, self).__init__(*args, **kwargs)
        self.discrete_action_space = self.world.discrete_world
        self.shared_reward = True


    def _set_action(self, action, agent, action_space, time=None):
        # agent.action.u = np.zeros(self.world.dim_p + 1) # ANDERS
        agent.action.u = np.zeros(self.world.dim_p + 1) # ANDERS
        # agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                # agent.action.u = np.zeros(self.world.dim_p)
                agent.action.u = np.zeros(self.world.num_joints * 2 + 2) #TODO "wat is dit" -Tessa >>>>?
                # process discrete action
                # print(action)
                agent.action.u = action
            else:
                if self.force_discrete_action: # niet nodig
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space: # eigenlijk ook niet nodig
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                    agent.action.u[-1] = action[0][5]
                else: # alleen dit nodig voor continue
                    # agent.action.u = action[0]
                    agent.action.u = action
            # sensitivity = 1.0
            # if agent.accel is not None: # volgensmij ook niet nodig
            #     sensitivity = agent.accel
            # agent.action.u *= sensitivity
            # action = action[1:]


    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent

        # self.done_callback = bool(
        #     np.linalg.norm(self.world.goals[0].state.p_pos - self.world.objects[0].state.p_pos) == 0.0)
        #
        # self.done_callback = bool(np.linalg.norm(self.world.goals[0].state.p_pos -
        #                                          self.world.objects[0].state.p_pos) == 0.0)
        #                      and not

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def _get_done(self, agent):
        return False

        # goal_reached = bool(np.linalg.norm(self.world.goals[0].state.p_pos - self.world.objects[0].state.p_pos) == 0.0)
        # return not agent.state.grasp and goal_reached
        #


    # render environment
    def render(self, mode='human'):

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1.5
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            self.render_geoms = []
            for e, entity in enumerate(self.world.entities):
                if 'agent' in entity.name:
                    # print(f" points test = {self.world.create_robot_points(entity, shorter_end=True)}")
                    geom = rendering.make_polyline(self.world.create_robot_points(entity, shorter_end=True,
                                                                                  discrete=self.world.discrete_world))
                    geom.set_color(*entity.color, alpha=0.5)
                    geom.set_linewidth(5)
                    gripper = rendering.make_polyline(self.world.create_gripper_points(entity,
                                                                                       gripped=entity.state.grasp))
                    # gripper = rendering.make_polyline(entity.create_gripper_points(gripped=False))
                    gripper.set_color(*entity.color, alpha=0.5)
                    gripper.set_linewidth(5)
                    self.render_geoms.append(geom)
                    self.render_geoms.append(gripper)

                elif 'object' in entity.name:
                    geom = rendering.make_polygon(entity.create_object_points())
                    geom.set_color(*entity.color)
                    self.render_geoms.append(geom)

                elif 'goal' in entity.name:
                    # geom = rendering.make_polygon(entity.create_goal_points())
                    geom = rendering.make_polygon(entity.create_goal_points2())
                    # print(f'type test = {type(geom)}')
                    # print(f'test points= {entity.create_goal_points2()}')
                    # geom = rendering.make_circle(radius=0.05)
                    geom.set_color(*entity.color, alpha=0.5)
                    self.render_geoms.append(geom)

                elif 'end_pos' in entity.name:
                    geom = rendering.make_polyline(entity.create_robot_points(shorter_end=True))
                    geom.set_color(*entity.color, alpha=0.5)
                    geom.set_linewidth(5)
                    gripper = rendering.make_polyline(entity.create_gripper_points(gripped=entity.state.grasp))
                    # gripper = rendering.make_polyline(entity.create_gripper_points(gripped=False))
                    gripper.set_color(*entity.color, alpha=0.5)
                    gripper.set_linewidth(5)
                    self.render_geoms.append(geom)
                    self.render_geoms.append(gripper)

            for viewer in self.viewers:
                viewer.geoms = []
                # print(self.render_geoms)
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    def permutate(self, num):
        if num == 0:
            print(f' Not inverted, state is equal to originial state')
        elif num == 1:
            self.world.invert_x()
            print(f'Inverted state over X-axis')
        elif num == 2:
            self.world.invert_y()
            print(f'Inverted over y-axis')
        elif num == 3:
            self.world.invert_x()
            self.world.invert_y()


# class HierEnv(MultiAgentEnv):
#     def __init__(self, *args, **kwargs):
#         super(HierEnv, self).__init__(*args, **kwargs)
#
#     def _set_action(self, action, agent, **kwargs):
#         # assuming action = np.array([0,0,1,0])
#         task = action.argmax() + 1
#         agent.action.u = task
#
#     def step(self, master_action):
#         obs_n = []
#         reward_n = []
#         done_n = []
#         info_n = {'n': []}
#
#         self.agent = Hieragent()
#         self._set_action(master_action, self.agent)
#         self.world.step()
#
#         # self.agents = self.world.policy_agents
#         # # set action for each agent
#         # for i, agent in enumerate(self.agents):
#         #     self._set_action(action_n[i], agent, self.action_space[i])
#         # # advance world state
#         # self.world.step()
#
#         for agent in self.agents:
#             obs_n.append(self._get_obs(agent))
#             reward_n.append(self._get_reward(agent))
#             done_n.append(self._get_done(agent))
#             info_n['n'].append(self._get_info(agent))
#
#         # all agents get total reward in cooperative case
#         reward = np.sum(reward_n)
#         if self.shared_reward:
#             reward_n = [reward] * self.n
#
#         return obs_n, reward_n, done_n, info_n
#
#     def render(self, mode='human'):
#
#         for i in range(len(self.viewers)):
#             # create viewers (if necessary)
#             if self.viewers[i] is None:
#                 # import rendering only if we need it (and don't import for headless machines)
#                 from multiagent import rendering
#                 self.viewers[i] = rendering.Viewer(700, 700)
#
#         results = []
#         for i in range(len(self.viewers)):
#             from multiagent import rendering
#             # update bounds to center around agent
#             cam_range = 1.5
#             if self.shared_viewer:
#                 pos = np.zeros(self.world.dim_p)
#             else:
#                 pos = self.agents[i].state.p_pos
#             self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
#             # update geometry positions
#             self.render_geoms = []
#             for e, entity in enumerate(self.world.roboworld.entities):
#                 if 'agent' in entity.name:
#                     # print(f" points test = {self.world.create_robot_points(entity, shorter_end=True)}")
#                     geom = rendering.make_polyline(self.world.roboworld.create_robot_points(entity, shorter_end=True,
#                                                    discrete=self.world.roboworld.discrete_world))
#                     geom.set_color(*entity.color, alpha=0.5)
#                     geom.set_linewidth(5)
#                     gripper = rendering.make_polyline(self.world.roboworld.create_gripper_points(entity,
#                                                                                        gripped=entity.state.grasp))
#                     # gripper = rendering.make_polyline(entity.create_gripper_points(gripped=False))
#                     gripper.set_color(*entity.color, alpha=0.5)
#                     gripper.set_linewidth(5)
#                     self.render_geoms.append(geom)
#                     self.render_geoms.append(gripper)
#
#                 elif 'object' in entity.name:
#                     geom = rendering.make_polygon(entity.create_object_points())
#                     geom.set_color(*entity.color)
#                     self.render_geoms.append(geom)
#
#                 elif 'goal' in entity.name:
#                     # geom = rendering.make_polygon(entity.create_goal_points())
#                     geom = rendering.make_polygon(entity.create_goal_points2())
#                     # print(f'type test = {type(geom)}')
#                     # print(f'test points= {entity.create_goal_points2()}')
#                     # geom = rendering.make_circle(radius=0.05)
#                     geom.set_color(*entity.color, alpha=0.5)
#                     self.render_geoms.append(geom)
#
#                 elif 'end_pos' in entity.name:
#                     geom = rendering.make_polyline(entity.create_robot_points(shorter_end=True))
#                     geom.set_color(*entity.color, alpha=0.5)
#                     geom.set_linewidth(5)
#                     gripper = rendering.make_polyline(entity.create_gripper_points(gripped=entity.state.grasp))
#                     # gripper = rendering.make_polyline(entity.create_gripper_points(gripped=False))
#                     gripper.set_color(*entity.color, alpha=0.5)
#                     gripper.set_linewidth(5)
#                     self.render_geoms.append(geom)
#                     self.render_geoms.append(gripper)
#
#             for viewer in self.viewers:
#                 viewer.geoms = []
#                 # print(self.render_geoms)
#                 for geom in self.render_geoms:
#                     viewer.add_geom(geom)
#
#             # render to display or array
#             results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))
#
#         return results