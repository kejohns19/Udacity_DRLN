## Collaborative Competition Multi Actor Deep Reinforcement Learning

Before diving into the details of the report below are key resources that I consulted

---

References: 

1\. https://spinningup.openai.com/en/latest/algorithms/sac.html

2\. https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac

3\. https://github.com/adithya-subramanian/Multi_Agent_Soft_Actor_Critic/

4\. https://github.com/ChenglongChen/pytorch-DRL

5\. https://github.com/1576012404/multi-agent-ppo

6\. https://medium.com/@amulyareddyk97/coding-multi-agent-reinforcement-learning-algorithms-683394556645

---

## Learning Algorithm - Soft Actor Critic (SAC)

Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches. It isn’t a direct successor to TD3 (having been published roughly concurrently), but it incorporates the clipped double-Q trick, and due to the inherent stochasticity of the policy in SAC, it also winds up benefiting from something like target policy smoothing.

A central feature of SAC is entropy regularization. The policy is trained to maximize a trade-off between expected return and entropy, a measure of randomness in the policy. This has a close connection to the exploration-exploitation trade-off: increasing entropy results in more exploration, which can accelerate learning later on. It can also prevent the policy from prematurely converging to a bad local optimum.

Below is the SAC algorithm pseudocode.  Reference #1 is an excellent overview of the algorithm.

![](https://raw.githubusercontent.com/kejohns19/Udacity_DRLN/master/images/SAC%20algo%20pseudocode.svg)

I updated the Spinning Up codebase to make it compatible with Pytorch 0.4.0 (for example I had to replace nn.Identity with nn.Sequential).  I then modified the code to initiate multiple agents.

### Neural network architecture

I used an actor and two critic neural networks (per the SAC algorithm). All networks share the the same initial network strucure, Linear containing 256 nodes followed by a ReLU activation function. The actor then included two final fully connected layers one which output an average (mu) and the other which output a log_std devitation - each output diminsions corresponded to the action dim for the actor, in this case the action dim was two.  The average and log_std were used to sample specific actions from a Normal distribution. 

The critic again used the first two fully connected layers from the actor and then included a final fully connected layer to output a single value. This value was then passed to the learning function as the baseline to calculate the advantage estimator (differnece with thediscounted future rewards generated from the actor policy).

#### Key Hyperparameters

Key hyperparameters include:

- xmy
- dld

### Reward Plot

Below is a plot of rewards over time to reach the goal of +0.5 reward over 100 consecutive episodes:

![](https://github.com/kejohns19/Udacity_DRLN/raw/master/images/p3_plot_0.5_target.png)
