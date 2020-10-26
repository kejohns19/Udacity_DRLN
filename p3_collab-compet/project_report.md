## Collaborative Competition Multi Actor Deep Reinforcement Learning

Before diving into the details of the report below are key resources that I consulted:

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

Below is the SAC algorithm pseudocode.  The Spinning Up SAC overview (reference #1) is an excellent review of the algorithm.  I copied the above two paragrapsh from this review.

![](https://raw.githubusercontent.com/kejohns19/Udacity_DRLN/master/images/SAC%20algo%20pseudocode.svg)

I updated the Spinning Up codebase to make it compatible with Pytorch 0.4.0 (for example I had to replace nn.Identity with nn.Sequential).  I then modified the code to initiate multiple agents. A key (late) learning during this exercise is to `.clone()` the observation inputs for re-use as inputs for multiple models (in this case critic networks for multiple models).  Before learning about cloning the inputs my model would start to learn but then stop and reverse.  Cloning the inputs creates a copy of the original variable that doesn’t forget the history of ops so to allow gradient flow and avoid errors with inlace ops.

### Neural network architecture

I used an actor and two critic neural networks (per the SAC algorithm). All networks share the the same initial network strucure, two Linear layers containing 512 & 256 nodes each followed by a ReLU activation function. The actor input dim is the observation size for one agent which in this case is 24.  The actor then included two final fully connected layers one which output an average (mu) and the other which output a log_std devitation - each output diminsions corresponded to the action dim for the actor, in this case the action dim was two.  The average and log_std were used to sample specific actions from a Normal distribution.  The final output is squashed by a Tanh function to ensure actions are within the boundaries (-1, 1).

Both the critics use the first two fully connected layer architecutre as in the actor and then include a final fully connected layer to output a single value.  However the input dim for both critics include the full observation and action space - in this particular case that is 24 observations per agent (2) plus 2 actions per agent (2) which is 52.

The loss functions for the two critic networks are:

![](https://github.com/kejohns19/Udacity_DRLN/blob/master/images/eq1.svg)

where the target is given by:

![](https://github.com/kejohns19/Udacity_DRLN/blob/master/images/eq2.svg)

To learn the optimal policy the action model is optimized according to:

![](https://github.com/kejohns19/Udacity_DRLN/blob/master/images/eq3.svg)

#### Key Hyperparameters

Key hyperparameters include:

- Learning rate - 1e-3
- Reward discount rate (gamma) - 0.995
- Soft update value (tau) - 0.995
- Entropy trade-off coeff (alpha) - 0.2
- Sample batchsize - 128 samples
- Update after every - 2 episodes
- Updates per episode - once per step

### Reward Plot

Below is a plot of rewards over time to reach the goal of +0.5 reward over 100 consecutive episodes (taking the maximum over both agents). The algorithm acheived the goal reward ater 1671 episodes.

![](https://github.com/kejohns19/Udacity_DRLN/raw/master/images/p3_plot_0.5_target.png)

The final model weights are saved in the following file `/model_dir/episode-solved-0.5.pt`.  I further tested the algorithm on ten episodes with a deterministic action space (instead of a stochastic action space sampled over a Normal distribution).  I accomplished this by simplying passing back the mu instead of the sampled distribution using mu and log_prob.  Below are the scores for both agents for ten consequtive episdoes:

```
Episode 0	Score 0: 2.60	Score 1: 2.60	Length: 1001
Episode 1	Score 0: 2.60	Score 1: 2.70	Length: 1001
Episode 2	Score 0: 2.70	Score 1: 2.60	Length: 1001
Episode 3	Score 0: 2.60	Score 1: 2.60	Length: 1001
Episode 4	Score 0: 2.60	Score 1: 2.70	Length: 1001
Episode 5	Score 0: 2.70	Score 1: 2.60	Length: 1001
Episode 6	Score 0: 2.60	Score 1: 2.60	Length: 1001
Episode 7	Score 0: 2.60	Score 1: 2.70	Length: 1001
Episode 8	Score 0: 2.70	Score 1: 2.60	Length: 1001
Episode 9	Score 0: 2.60	Score 1: 2.60	Length: 1001
```

I continued training the algorithms to attempt to acheive a goal of +2.0 reward over 100 consecuttive episodes.  The algorithm acheived the goal reward after a further 121 episodes.  Below is a plot of the full rewards history to acheive the higher level reward.

![](https://github.com/kejohns19/Udacity_DRLN/raw/master/images/p3_plot_2.0_target.png)

The final model weights are saved in the following file `/model_dir/episode-solved-2.0.pt`.

## Ideas for Future Work

The multi-agent SAC algorithm peformed fairly well however it did require some hyperparameter tuning.  Calling the update/learn function too early or too often resulted in model instability.  Understanding the trade-offs and sweet spot for how often to train the model would be beneficial.  It appears this is very much a black art.  For instance I tried to train the algorithm for five times per step instead of one time per step but this slowed down training whereas I thought training gains per episode would accelerate.

Another idea is to implement a prioritized experience replay buffer.  This may lead to more efficient training and less instability.  I explored this option but in the end acheived good training through trying different hyperparater combinations.  

I also would be interested in implementing a multi-agent PPO approach which should be more robust to hyperparameters tuning (perhaps however by sacrificing some training efficiency).  Understand the trade-offs in training efficienty vs training stability between MASAC and MAPPO would be beneficial.  
