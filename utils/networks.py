import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

        # self.noise_fc1 = torch.randn(self.fc1.weight.size()) * 0.1 + 0
        # self.noise_fc2 = torch.randn(self.fc2.weight.size()) * 0.1 + 0
        # self.noise_fc3 = torch.randn(self.fc3.weight.size()) * 0.1 + 0


    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))

        # x = self.in_fn(X)
        # x = self.fc1(x)
        # self.fc1.weight = add_noise(self.fc1.weight)
        # x = self.nonlin(x)
        # x = self.fc2(x)
        # self.fc2.weight = add_noise(self.fc2.weight)
        # x = self.nonlin(x)
        # x = self.fc3(x)
        # self.fc3.weight = add_noise(self.fc3.weight)
        # out = self.out_fn(x)

        return out

def add_noise(weights):
    noise = torch.randn(nn.Parameter(weights).size()) * 0.1 + 0
    # print(f" noise sample = {noise}")
    noise = noise.to(device=weights.device)
    with torch.no_grad():
        weight_noise = nn.Parameter(weights + noise)
    return weight_noise

# def add_noise(weights):
#     noise = torch.randn(weights.size()) * 0.1 + 0
#     noise = noise.to(device=weights.device)
#     with torch.no_grad():
#         weights.add_(noise)
#     return weights
