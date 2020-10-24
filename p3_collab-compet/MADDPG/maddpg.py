# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class MADDPG:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-4, lr_critic=1.0e-5, discount_factor=0.99, tau=1.0e-2):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        agent1 = DDPGAgent(in_actor, hidden_in_actor, hidden_out_actor, out_actor, 
                           in_critic, hidden_in_critic, hidden_out_critic, 
                           lr_actor, lr_critic)
        agent2 = DDPGAgent(in_actor, hidden_in_actor, hidden_out_actor, out_actor, 
                           in_critic, hidden_in_critic, hidden_out_critic,  
                           lr_actor, lr_critic)
        self.maddpg_agent = [agent1, agent2]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [agent.actor for agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [agent.target_actor for agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, rand=0.0, add_noise=True):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, rand, add_noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions
    
    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [agent.target_act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions
    
    def reset(self):
        '''reset the noise class'''
        for agent in self.maddpg_agent:
            agent.reset()

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """
        
        # helper function to re-order inputs from perspective of the agent
        def rotate(inputs):
            return inputs[agent_number:] + inputs[:agent_number]

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, action, reward, next_obs, done = map(transpose_to_tensor, samples)
        
        # prepare next step actions for actor learning process          

        # ensure obs_full sees from perspective of agent by re-ordering obs
        # to make agent_number obs first in the order
        obs_full = torch.cat(rotate(obs), dim=1)
        next_obs_full = torch.cat(rotate(next_obs), dim=1)
        
        agent = self.maddpg_agent[agent_number]
        
        # ---------------------------- update critic ---------------------------- #
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = []
        for i, ob in enumerate(next_obs):
            if i == agent_number:
                target_actions.append(agent.target_actor(ob))
            else:
                target_actions.append(self.maddpg_agent[i].act(ob))
        target_actions = torch.cat(rotate(target_actions), dim=1)
        
        target_critic_input = torch.cat((next_obs_full,target_actions), dim=1).to(device)
        
        #with torch.no_grad():
        q_next = agent.target_critic(target_critic_input)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        ### perhaps don't rotate?????
        action_full = torch.cat(rotate(action), dim=1)
        critic_input = torch.cat((obs_full, action_full), dim=1).to(device)
        q = agent.critic(critic_input)

        loss_func = torch.nn.MSELoss()
        critic_loss = loss_func(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = []
        for i, ob in enumerate(obs):
            if i == agent_number:
                q_input.append(agent.actor(ob))
            else:
                q_input.append(self.maddpg_agent[i].act(ob))
                
        q_input = torch.cat(rotate(q_input), dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)