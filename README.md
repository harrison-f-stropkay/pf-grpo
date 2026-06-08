# Particle Filtering Rollouts for GRPO

## TL;DR

This repository sheds light on the following question: **What happens if we use particle filtering to generate GRPO rollouts?**

## Some background

Red Hat’s AI Innovation team introduced Particle Filtering (PF) as an inference-time scaling method in [Rollout Roulette](https://arxiv.org/abs/2502.01618). Instead of independently sampling $n$ full trajectories, PF maintains a population of $n$ candidates throughout generation. At each reasoning step, it extends each candidate sample, scores the resulting partial trajectories with a process reward model, converts those scores into a probability distribution with softmax, and resamples $n$ candidates from that distribution. It repeats this generate-score-resample cycle until every particle generates a complete answer. Then, the method returns the complete answer that receives the highest score from the reward model. Note that **instead of using only the highest-scoring trajectory, we will include all complete trajectories in the GRPO rollout**. Relative to an alteranative where we run PF $n$ times, our choice of using all of the complete trajectories will keep the increase in training time modest.

The team also explored PF outputs as synthetic training data in a series of blog posts. [One post](https://ai-innovation.team/blog/r1-like-reasoning-update-2) even mentions "Generat[ing] synthetic data inside GRPO itself", but the public writeups don't mention the specific ablation I'm wondering about: vanilla GRPO rollouts versus PF GRPO rollouts.

Outside of the team's work, [TSR](https://arxiv.org/pdf/2602.11767) uses search-generated trajectories for GRPO. TSR uses trajectory search methods such as best-of-N and beam search, whereas this project focuses specifically on PF. TSR claims "up to 15% performance gains" across its various sampling strategies and benchmarks. There is also related work on tree-search-based RL for LLMs, including [Tree-GRPO](https://arxiv.org/abs/2509.21240) and [Tree-OPO](https://arxiv.org/abs/2509.09284). Both report performance gains.

<!-- On one hand, using particle filtering (PF) for GRPO could improve training efficieny due to better candidate quality. On the other hand, the candidates may be less faithful to the policy’s natural sampling distribution, affecting the validity of the GRPO update. -->
