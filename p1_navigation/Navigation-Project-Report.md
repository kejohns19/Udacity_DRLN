# Learning Algorithm

I started with the DQN model that we created earlier in the course.  This is comprised of ```dqn_agent.py``` and a ```model.py``` file.  I started with the default hyperparameters as below:
```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 2        # how often to update the network
```
The QNetwork model is comprised of three fully connected layers with ReLU activation between both the first and second layers:
```python
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```
I used values of 64 for both the hidden layers.

The DQN agent includes act, step, learn, and soft_update functions.  The learn function includes the key code to compute the loss and update the DQN model weights.
```python
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     
```

I created a DQN function in the Navigation jupyter notebook (copied below) to broker between the learning agent and the environment.  
```python
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)                 # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores
```

The function runs until it achieves an average score of 13 over the previous 100 episodes or reaches the maximum number episodes.  I varied several parameters but in general it took between 400-500 episodes to achieve an average score over 13.

# Plot of Rewards

The below output and plot uses default hyperparameters:

```
Episode 100	Average Score: 1.00
Episode 200	Average Score: 4.52
Episode 300	Average Score: 8.26
Episode 400	Average Score: 9.80
Episode 500	Average Score: 11.98
Episode 572	Average Score: 13.00
Environment solved in 472 episodes!	Average Score: 13.00
```
![](https://github.com/kejohns19/Udacity_DRLN/blob/master/images/learning_rate_mse_dqn.png?raw=true)

# Updates to model hyperparameters and impact on learning rate

Changing batch size from 64 to 32 had a positive affect on episodes to solve - reducing by approximately 100 episodes:
```
Episode 100	Average Score: 0.90
Episode 200	Average Score: 3.45
Episode 300	Average Score: 7.80
Episode 400	Average Score: 10.53
Episode 465	Average Score: 13.00
Environment solved in 365 episodes!	Average Score: 13.00
```
![](https://github.com/kejohns19/Udacity_DRLN/blob/master/images/learning_rate_mse_dqn_bs32.png?raw=true)

# Substituting loss function and impact on learning rate

Using Smoothed L1 loss instead of MSE ```loss = F.smooth_l1_loss(Q_expected, Q_targets)``` but moving back to a batch size of 64 also similarly reduced the episodes needed to solve:
```
Episode 100	Average Score: 0.53
Episode 200	Average Score: 4.72
Episode 300	Average Score: 7.23
Episode 400	Average Score: 10.27
Episode 500	Average Score: 12.95
Episode 502	Average Score: 13.13
Environment solved in 402 episodes!	Average Score: 13.13
```
![](https://github.com/kejohns19/Udacity_DRLN/blob/master/images/learning_rate_mse_dqn_smoothed_L1.png?raw=true)

# Implementing Double DQN and impact on learning rate

I updated the DQN agent to Double DQN Learning by inserting the following code in the learn function for the agent:
```python
        #################Updates for Double DQN learning###########################
        self.qnetwork_local.eval()
        with torch.no_grad():
            actions_q_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1).long()
            Q_targets_next = self.qnetwork_target(next_states).gather(1,actions_q_local)
        self.qnetwork_local.train()
        ############################################################################

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
```
Using batch size of 64 and the original loss function, the episodes to solve again similarly improved:
```
Episode 100	Average Score: 0.61
Episode 200	Average Score: 4.10
Episode 300	Average Score: 7.44
Episode 400	Average Score: 10.10
Episode 481	Average Score: 13.05
Environment solved in 381 episodes!	Average Score: 13.05
```

# Ideas for Future Work

Additional improvement can be acheived by continued iteration on hyperparameters like batch size and exploring additional loss functions.  One could also investigate a more agreesive learning rate for soft update factor (TAU).  Furthermore one could explore updating the model after every frame instead of after every two frames as I chose.  
