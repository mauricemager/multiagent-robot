import numpy as np
from pyglet.window import key


# individual agent policy
class Policy(object):
    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError()


# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class RobotPolicy(Policy):
    def __init__(self, env, agent_index):
        # super(RobotInteractivePolicy, self).__init__(env, agent_index)
        super(RobotPolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(6)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[0].window.on_key_press = self.key_press
        env.viewers[0].window.on_key_release = self.key_release

        print(f"test move = {self.move}")

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = np.zeros(6)
            if self.move[0]: u[0] = 1
            if self.move[1]: u[1] = 1
            if self.move[2]: u[2] = 1
            if self.move[3]: u[3] = 1
            if self.move[4]: u[4] = 1
            if self.move[5]: u[5] = 1
        else:
            u = np.zeros(3)  # 6-d because of no-move action
            if self.move[0]: u[0] += 1.0
            if self.move[1]: u[0] -= 1.0
            if self.move[2]: u[1] += 1.0
            if self.move[3]: u[1] -= 1.0
            if self.move[4]: u[2] += 1.0
            # if True not in self.move:
            #     u[0] += 1.0
        # return np.concatenate([u, np.zeros(self.env.world.dim_c)])
        return u

    def key_press(self, k, mod):
        if k == key.LEFT:  self.move[0] = True
        if k == key.RIGHT: self.move[1] = True
        if k == key.UP:    self.move[2] = True
        if k == key.DOWN:  self.move[3] = True
        if k == key.SPACE: self.move[4] = True
        if k == key.LALT:  self.move[5] = True

    def key_release(self, k, mod):
        if k == key.LEFT:  self.move[0] = False
        if k == key.RIGHT: self.move[1] = False
        if k == key.UP:    self.move[2] = False
        if k == key.DOWN:  self.move[3] = False
        if k == key.SPACE: self.move[4] = False
        if k == key.LALT:  self.move[5] = False
