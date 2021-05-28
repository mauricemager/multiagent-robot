import torch
from torch import Tensor, tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
from random import random, randint
from torch.nn.functional import one_hot

# ------------------ changes made to this file -----------------
# imports
from torch.distributions import Normal

# self.step() changed to sample a action from distribution given by network
# self.exploration removed OUNoise and introduced self.var line 45
# added self.variance in self.scale2_noise() line 61

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
            self.variance = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale
            self.variance = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = onehot_from_logits(action, eps=self.exploration)
            else:
                action = onehot_from_logits(action)

            # if explore:
            #     if random() > self.exploration: # do greedy action
            #         action = onehot_from_logits(action)
            #     else: # take random action
            #         action = one_hot(tensor(randint(0, 5)), num_classes=6).unsqueeze(0).type(torch.float64)
            # # old
            # if explore:
            #     action = gumbel_softmax(action, hard=True)
            # else:
            #     action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()), requires_grad=False) # old
                # dist = Normal(action, self.variance)
                # action = dist.sample()
            action = action.clamp(-1, 1)
        return action

        # # print(f'exploration noise scale = {self.exploration.scale}')
        # # print(f"self.exploration.scale = {self.exploration.scale}")
        # network_output = self.policy(obs)
        # dist = Normal(network_output, 0.001)
        # action = dist.sample().clamp(-1, 1)
        # # mu, sigma = split(network_output, 3, dim=1)
        # # print(f"mu = {mu} and sigma = {sigma}")
        # # if explore: sigma *= self.exploration.scale * 10
        # # action = normal(mu, sigma).clamp(-1, 1)
        # # print(action)
        #
        # return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
