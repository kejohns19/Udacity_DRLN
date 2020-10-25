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
