import numpy as np
from robot.robot_scenarios.collaborative_tasks import CollScenario


class Scenario(CollScenario):

    def reset_world(self, world):
        """Overwrite collaborative scenario reset method and add task specific initial states"""
        super().reset_world(world)

        # add task specific initial states
        world.agents[1].state.grasp = True
        world.objects[0].state.p_pos = world.agents[1].position_end_effector()

    def reward(self, agent, world):
        """Overwrite collaborative reward function to learn task2 objective"""

        # reward based on distance of agents' end effectors to object
        reward = np.linalg.norm(world.objects[0].state.p_pos - world.agents[0].get_joint_pos(world.num_joints))
        # incentivize to keep holding object, else reward is doubled
        if world.objects[0].state.who_grabbed is None: reward *= 2.0

        # return negative reward
        return -reward
