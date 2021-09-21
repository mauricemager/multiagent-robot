import numpy as np
from robot.robot_scenarios.collaborative_tasks import CollScenario

# np.random.seed(7)

class Scenario(CollScenario):

    def reset_world(self, world):
        """Overwrite collaborative scenario reset method and add task specific initial states"""
        super().reset_world(world)

        # add task specific initial states
        world.agents[1].state.angles[0] = np.pi * np.random.rand(1) + np.pi/2
        world.objects[0].state.p_pos = world.agents[1].position_end_effector()
        world.agents[0].state.angles[0] =  2 * np.random.rand(1) - 1
        world.agents[0].state.angles[1] =  (2 * np.random.rand(1) - 1 ) * np.pi/2

    def reward(self, agent, world):
        """Overwrite collaborative reward function to learn task3 objective"""

        # reward based on left agent's (a_0) distance to object
        reward = np.linalg.norm(world.objects[0].state.p_pos - world.agents[0].get_joint_pos(world.num_joints)) + 0.5
        # give positive reward when left agent has object (termination state)
        if world.objects[0].state.who_grabbed == 'agent 0':
            reward = 0.
        # return negative reward
        return -reward
