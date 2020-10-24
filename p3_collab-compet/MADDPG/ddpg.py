# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise, RNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        #self.noise = OUNoise(out_actor, scale=1.0 )
        self.noise = RNoise(out_actor, 0.5)
        
        self.epsilon = 1.
        self.epsilon_decay_rate = 0.999
        self.epsilon_min = 0.2

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor, weight_decay=0.0)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=0.0)
        
    def epsilon_decay(self):
        self.epsilon = max(self.epsilon_decay_rate*self.epsilon, self.epsilon_min)

    def act(self, obs, rand=0., add_noise=True):
        if np.random.random() < rand:
            action = np.random.randn(2)                    # select an action (for each agent)
            action = np.clip(action, -1, 1)                  # all actions between -1 and 1
            action = torch.tensor(action, dtype=torch.float)
        else:
            obs = obs.to(device)
            
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(obs)
            self.actor.train()
            if add_noise:
                action += self.epsilon * self.noise.noise()
                action = action.squeeze(0)

        return action
    
    def reset(self):
        self.noise.reset()

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action