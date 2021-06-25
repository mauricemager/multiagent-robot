import numpy as np
from robot.robot_scenarios.collaborative_tasks import CollScenario


class Scenario(CollScenario):

    def reward(self, agent, world):
        """Overwrite collaborative reward function to learn task1 objective"""

        reward = 0.0
        # disregard reward of left agent (a_0)
        if '0' in agent.name:
            pass
        else:
            # reward based on right agent's (a_1) end-effector distance to object
            reward = np.linalg.norm(world.objects[0].state.p_pos - agent.get_joint_pos(world.num_joints)) + 0.5
            # add positive reward when agents grabs object to incentive to keep holding object
            if agent.state.grasp and agent.within_reach(world, world.objects[0]): reward -= 0.5

        # return negative reward
        return -reward
