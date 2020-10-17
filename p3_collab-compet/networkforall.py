import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(Network, self).__init__()

        #self.input_norm = nn.BatchNorm1d(input_dim)
        #self.input_norm.weight.data.fill_(1)
        #self.input_norm.bias.data.fill_(0)

        self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        self.bn1 = nn.BatchNorm1d(hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.bn2 = nn.BatchNorm1d(hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        #self.nonlin = f.leaky_relu #or relu
        self.actor = actor
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.actor:
            # return a vector of the force
            h1 = f.leaky_relu(self.bn1(self.fc1(x)))
            h2 = f.leaky_relu(self.bn2(self.fc2(h1)))
            #h1 = f.leaky_relu(self.fc1(x))
            #h2 = f.leaky_relu(self.fc2(h1))
            h3 = self.fc3(h2)
            return f.tanh(h3)
            '''
            mean = f.tanh(self.fc3(h2))
            log_std = self.fc3(h2)
            log_std = torch.clamp(log_std, min=-10, max=1)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = f.tanh(x_t)
            #log_prob = normal.log_prob(x_t)
            #log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            #log_prob = log_prob.sum(-1, keepdim=True)
            return action# log_prob, x_t, mean, log_std
            '''

        else:
            # critic network simply outputs a number
            h1 = f.leaky_relu(self.bn1(self.fc1(x)))
            h2 = f.leaky_relu(self.bn2(self.fc2(h1)))
            #h1 = f.leaky_relu(self.fc1(x))
            #h2 = f.leaky_relu(self.fc2(h1))
            h3 = self.fc3(h2)
            return h3