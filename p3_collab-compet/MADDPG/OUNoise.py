import numpy as np
import torch


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale, dtype=torch.float)
    
class RNoise:
    """uniformly distributed random noise process"""

    def __init__(self, shape, amplitude):
        """Initialize parameters and noise process
        Params
        ======
            shape (int): dimension of each action
            buffer_size (int): maximum size of buffer
            amplitude (int): size of each training batch
        """
        
        self.amplitude = amplitude
        self.shape = shape
        self.state = np.zeros(self.shape)
    def reset(self):
        """Reset the internal state (= noise) to zero."""
        self.state = np.zeros(self.shape)

    def noise(self):
        """Return a noise sample."""
        self.state = self.amplitude*(2.*np.random.rand(self.shape) - 1.)
        return torch.tensor(self.state, dtype=torch.float)