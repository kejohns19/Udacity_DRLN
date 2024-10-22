## Policy Gradient for continuous actions project

Before diving into the details of the report below are key resources that I consulted

---

References: 

1\.  [https://people.eecs.berkeley.edu/~pabbeel/nips-tutorial-policy-optimization-Schulman-Abbeel.pdf](https://people.eecs.berkeley.edu/~pabbeel/nips-tutorial-policy-optimization-Schulman-Abbeel.pdf) 

2\.  [https://towardsdatascience.com/rl-train-the-robotic-arm-to-reach-a-ball-part-02-fc8822ace1d8](https://towardsdatascience.com/rl-train-the-robotic-arm-to-reach-a-ball-part-02-fc8822ace1d8) 

3\.  [https://github.com/TomLin/RLND-project/tree/master/p2-continuous-control](https://github.com/TomLin/RLND-project/tree/master/p2-continuous-control) 

4\.  [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter15](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter15) 

5\.  [https://github.com/higgsfield/RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2) 

6\.  [http://proceedings.mlr.press/v70/chou17a/chou17a.pdf](http://proceedings.mlr.press/v70/chou17a/chou17a.pdf)

---

## Learning Algorithm
I explored a few differnt approaches for this project, starting with PPO, using the REINFORCE architecture from a prior Pong learning project.  The major change was to reconfigure the actor network to output a probability distribution (mu, sigma) instead of specific actions.  This allowed the algorithm to explore the continous space while still enabling the calculation of log probabiliites needed to generate a differentiable loss function.  However I could not get my PPO architecture to learn against the more difficult continuous action Reacher environment.  This may be due to poor hyperparameter choices.   I could also have mis-configured a reward or loss calculation.  

I chose to explore an A2C model and began with the excellent writeup at reference #3.  I used the same approach to calculating the action probability distrubtion - using a Normal distribution from an actor network generated mu and sigma.  Actions were then passed through a tanh function to bound them between the action range of -1 to 1.  I varied a static sigma (between 0.5 and 1) for most learning cycles.  At the end of the project I tried to let the network learn the sigma as well but I could not achieve the desired target reward.  

### Advantage Actory-Critic pseudocode
The below figure from refernece #1 outlines the Advantage Actory-Critic algorithm

![](https://github.com/kejohns19/Udacity_DRLN/blob/master/images/A2C_method_overview.png?raw=true)

### Neural network architecture
I used an actor and critic neural network.  Both the actor and critic shared the first two fully connected layers which each contained 128 nodes.  The actor then included a final fully connected layer which output four number which represented the mus for the Normal distribution.  This was combined with a specific sigma which calculated the action from the Normal probability distribution.  (As an aside I tried a beta distribution as well where I tried to learn both the alpha and beta parameters but couldn't get it to learn).  

The critic again used the first two fully connected layers from the actor and then included a final fully connected layer to output a single value.  This value was then passed to the learning function as the baseline to calculate the advantage estimator (differnece with thediscounted future rewards generated from the actor policy).

### Learning function
The learning function calculates two losses - the advantage loss and the critic loss.  The advantage loss is calculated by multiplying the advantage estimator by the log probabilities of the actions used by the actor policy to generate the discounted future rewards.  The critic loss is the mean squared error of the difference between the values generated by the critic and discounted future rewards.  The total loss is then the negative advantage loss (because we want to maximize this value) plus the critic loss (we want to minimize this value.  The losses are then back propigated through the network using an Adam optimization policy.

I also explored introducing an entropy term.  This term howeverd didn't appears to impact learning.  This is likely due to the fact that the sigma for the Normal probablity distribution of actions is sufficiently large to enable good exploration in the environment. If the sigma was lower, entropy may contribut to learning by forcing more exploration.  However it is not clear what the tradeoff may be.  

### Key Hyperparameters
Below is a summary of key hyperparameters used in the model
* Learning rate (lr) = 1.5e-4
* Reward discount rate (gamma) = 0.99
* Step Horizon (n_steps) = 10
* Fully connected hidden size = 128 for both layers 

## Reward Plot
Below is a plot of the rewards over episodes for the A2C network model with a static sigma of exp(-0.5).

![](https://github.com/kejohns19/Udacity_DRLN/blob/master/images/reward_plot_1.png?raw=true)

The model required 159 episodes to achieve an average score above 30 for 100 consecutive episodes.  The last 45 episodes averaged a reward of approximately 38.

## Ideas for Future Work
Below are some ideas for future work.

* Explore different step horizons.   I used 10 steps but would be interesting to explore longer horizons, perhaps up to 50.
* Explore anealing the static sigma for the probabiliyt distribution over time.  This may led to more accurate policies during evaluation.  
* Explore an LSTM neural network for the actor and critic.  given the fact that the Reacther environment has vecolicty vectors that change slowly perhaps an LSTM may generate more robust actions.  LSTMs have been applied to A2C learning modles to more efficiently allocate resources across 5G spectrum (https://ieeexplore.ieee.org/document/9112328)
