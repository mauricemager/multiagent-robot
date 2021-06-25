import numpy as np
from robot.robot_scenarios.collaborative_tasks import CollScenario


class Scenario(CollScenario):

    def reset_world(self, world):
        """Overwrite collaborative scenario reset method and add task specific initial states"""
        super().reset_world(world)

        # add task specific initial states
        world.objects[0].state.p_pos = world.agents[0].position_end_effector()
        world.agents[0].state.grasp = True

    def reward(self, agent, world):
        """Overwrite collaborative reward function to learn task4 objective"""

        # reward based on object's distance to goal position
        reward = np.linalg.norm(world.goals[0].state.p_pos - world.objects[0].state.p_pos)
        # add negative reward when object is not picked up
        if world.objects[0].state.who_grabbed is None: reward += 0.5

        # return negative reward
        return -reward
