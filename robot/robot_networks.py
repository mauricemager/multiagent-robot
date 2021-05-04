import torch.nn as nn
import torch.nn.functional as F


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
        )
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Tanh())
        self.sigma_head = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Softplus())

    def forward(self, x):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        x = self.hidden(x)
        mu = self.mu_head(x)
        sigma = self.sigma_head(x)
        return mu, sigma
