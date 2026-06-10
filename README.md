# Particle Filtering Rollouts for GRPO

## TL;DR

This repository attempts to shed light on the following question:

> What happens if we use particle filtering to generate GRPO rollouts?

This experiment is motivated by the hypothesis that improved GRPO trajectory quality will lead to improved training signal. It also approaches the more general challenge of amortizing inference-time compute into training.

On one hand, using particle filtering (PF) for GRPO could improve training efficiency due to better candidate quality. On the other hand, the candidates may be less faithful to the policy’s natural sampling distribution, affecting the validity of the GRPO update. In principle, importance sampling corrections (like PPO's clipped probability ratio) can account for this off-policy bias--but without them, PF-GRPO introduces a second independent source of error on top of the reward collapse described below.

In addition to the PF GRPO experiments, I contribute to `training_hub`. I introduce a `trajectory_group_fn` parameter to `lora_grpo` that complements the `rollout_fn` parameter. It is used to inject whole trajectory groups; in my case, I use it to supply LoRA GRPO with particles. My diff is [here](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/compare/main...harrison-f-stropkay:training_hub:main). If you'd like, I'd be happy to open a PR.

## Some background

In [Rollout Roulette](https://arxiv.org/abs/2502.01618), Red Hat’s AI Innovation team introduced Particle Filtering (PF) as an inference-time scaling method. Instead of independently sampling $n$ full trajectories, PF maintains a population of $n$ candidates throughout generation. At each reasoning step, it extends each candidate sample, scores the resulting partial trajectories with a process reward model, converts those scores into a probability distribution with softmax, and resamples $n$ candidates from that distribution. It repeats this generate-score-resample cycle until every particle generates a complete answer. Then, the method returns the complete answer that receives the highest score from the reward model.

Instead of using only the highest-scoring trajectory, I include all complete trajectories in the GRPO rollout. Relative to an alternative where I run PF $n$ times, this choice keeps the increase in training time modest.

The team also explored PF outputs as synthetic training data in a series of blog posts. [One post](https://ai-innovation.team/blog/r1-like-reasoning-update-2) even mentions "Generat[ing] synthetic data inside GRPO itself", but the writeups don't mention the specific ablation I'm wondering about: vanilla GRPO rollouts versus PF GRPO rollouts. Outside of the team's work, [TSR](https://arxiv.org/pdf/2602.11767) uses search-generated trajectories for GRPO. Beam search yields the best results: up to 15% performance gains on WebShop, an agentic web navigation benchmark. TSR does not explore PF as a search strategy or address mathematical reasoning as an evaluation domain. Additionally, [Tree-GRPO](https://arxiv.org/abs/2509.21240) and [Tree-OPO](https://arxiv.org/abs/2509.09284) use tree-search-based RL for LLMs, and both report performance gains.

## Implementation details

I use `training_hub`'s LoRA GRPO instead of full GRPO due to time and compute constraints. I hold policy compute fixed across runs, and I use the following:

- **Base model:** `Qwen/Qwen2.5-Math-1.5B-Instruct`
- **Process Reward Model:** `Qwen/Qwen2.5-Math-PRM-7B`
- **Training data:** `rasbt/math_full_minus_math500`
- **Temperature:** `0.7`
- **Number of GRPO updates:** `50`
- **Prompts/update:** `1`
- **Rollouts/prompt:** `16`
- **Reasoning Step generation:** `step_token="\n\n"`, `max_steps=50`, `stop_token="\\boxed"`

## Post-mortem

I hadn't accounted for two aspects of my experiments: (i) my reward model was binary (1 if the model answered correctly, 0 otherwise), and (ii) PF is more prone to collapse than other ITS strategies. For both reasons, PF + GRPO provided no training signal--it left `Qwen2.5-Math-1.5B-Instruct` untouched, so I never evaluated the models. I still think PF can help amortize ITS into train-time compute, but not without addressing these two obstacles.

For (ii), PF produced as few as one distinct answer per question, offering no signal to GRPO. To see why, consider a rollout of size 16. Even if the PRM is useless, assigning equal weight to every candidate at each step, we expect only about 10 distinct particles:

$$\mathbb{E}[n_{\text{particles}}]=16\left(1-\left(\frac{15}{16}\right)^{16}\right)\approx 10.29$$

Of course, each particle will likely generate a distinct candidate at the next reasoning step, but this stands in opposition to the fact that the PRM is likely better than random sampling, meaning a strong PRM concentrates weight on a small number of particles, driving the effective sample size toward 1. This effect was compounded as PF continued, yielding trajectory groups like the following:

```
To solve the problem, we need to determine the equations...by the four points is:\n\\[\n\\boxed{\\frac{192\\sqrt{14}}{25}}.\n\\]
To solve the problem, we need to determine the equations...by the four points is:\n\\[\n\\boxed{\\frac{192\\sqrt{14}}{25}}.\n\\]
We need to find the area of the quadrilateral formed...by the four points is:\n\\[\n\\boxed{\\frac{192\\sqrt{14}}{25}}.\n\\]
We need to find the area of the quadrilateral formed...by the four points is:\n\\[\n\\boxed{\\frac{192\\sqrt{14}}{25}}.\n\\]
```

Combined with (i): even with some diversity of answers, all answers were either correct or incorrect. So, the reward was uniform for all trajectories, leading to no GRPO update. The reason is that GRPO computes advantages by normalizing rewards within each group--subtracting the mean and dividing by the standard deviation. When all rewards are the same, the standard deviation is zero and every advantage is zero, so the gradient vanishes.

To counteract these factors, I considered raising the temperature of the policy, but neglected that idea when I realized that it would only exacerbate the effects of (ii) by pruning the frequent bad steps. I raised the size of the rollout groups to 32, hoping that PF would yield better diversity. However, the rewards were always the same across all trajectories.

Thus, this initial attempt at integrating PF into GRPO was unsuccessful. In the future, I'd like to experiment with a compute-for-diversity tradeoff: instead of using all particles, we could run PF multiple times, counteracting (ii). This aligns with the TSR paper, which ran beam search and BoN multiple times for each task.

Another promising approach would be to use a non-binary reward function, such as using the PRM as the RM, yielding a float instead of a boolean. This would guarantee a distribution of rewards and therefore training signal for GRPO gradient updates--and collapse alone could no longer zero the gradient.

## AI Usage

For ideation, I used ChatGPT to answer questions about GRPO and other ML concepts that I touched on in this project. The vast majority of this phase was spent reading papers without AI assistance.

For development, I used OpenAI's Codex VS Code extension. For `core.py`, I defined the function signatures, and I iterated with Codex for implementation, bouncing ideas and using it like a pair programmer. This is generally how I program these days. I always review AI-generated code, and I prefer to work in small chunks--I have a rough idea of what I want, and I let Codex handle the details.

Resolving dependencies and environments were a major pain point, and Codex helped with this greatly; `compat.py` was entirely written by Codex. I don't think that AI ever slowed me down.

Codex also helped me with my `training_hub` diff. I had a half-baked idea for how I'd inject PF into GRPO, so Codex greatly decreased the amount of time I spent on the diff. I bet it took me less than 2 hours. In particular, Codex found a few instances of docs that needed to be updated that I probably would have missed.

As for best practices that I'd recommend for a team adopting AI-assisted development, my first suggestion is to use AI for code reviews. I often find that AI discovers bugs that I overlook. Second, I'd suggest using AI _after_ thinking about the problem yourself. In the past, I've wasted a lot of time when neither the AI nor I really know what I want. Lastly, I'd encourage experimenting with the models and reasoning settings. Sometimes (e.g., when I want to rename a few variables at the same time), I'll use a faster model, and other times (e.g., when I want AI to implement an entire function or module that I've described), I'll use a slower, more performant model. The key is matching the tool to the task.

<!-- During Your Onsite

We'll have a casual conversation about your project, no presentation needed. We'll walk through your code together and discuss:

- The ML questions your project explored and what you found
- Design decisions you made and trade-offs you weighed
- What you learned about the libraries and what surprised you
- How you approached learning unfamiliar codebases
- our experience with AI-assisted development on this project -->

## How to reproduce this work

1.  **Get your hands on a GPU.** I used a Run:ai workspace based on `vllm/vllm-openai:v0.13.0` with one 80GB H100. Note that I haven't tested this code in other GPU environments.

2.  Clone this repo and my fork of `training_hub` (I use the `trajectory_group_fn` parameter from my fork):

    ```bash
    git clone https://github.com/harrison-f-stropkay/pf-grpo.git
    git clone https://github.com/harrison-f-stropkay/training_hub.git
    ```

3.  Test that the CPU-only code is in working order:

    ```bash
    uv run pytest
    ```

4.  Install the heavy ML dependencies. The order does matter. This was a major pain point, so it's probably best to just copy and paste the following code block.

    ```bash
    python3 -m venv --system-site-packages .venv-final
    source .venv-final/bin/activate
    python -m pip install -U pip
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TRAINING_HUB=0.1.dev120
    python -m pip install -e ../training_hub -e . --no-deps
    python -m pip install datasets math-verify litellm==1.74.1 openpipe-art==0.5.18 nest_asyncio wandb==0.22.1 matplotlib instructlab-training rhai-innovation-mini-trainer
    python -m pip install --force-reinstall --no-deps trl==0.20.0 unsloth==2025.12.9 unsloth-zoo==2025.12.7 accelerate==1.7.0 peft==0.19.1
    python -m pip install megatron-core "its-hub[experimental]" "reward-hub[prm]==0.1.10" --no-deps
    ```

5.  Within the activated virtual environment, kick off the training runs:

    ```bash
    python cli.py \
    --method vanilla \
    --run-dir runs/vanilla-1x32 \
    --num-iterations 50 \
    --prompts-per-update 1 \
    --rollouts-per-prompt 32 \
    --max-tokens 2048 \
    --seed 0

    python cli.py \
    --method pf \
    --run-dir runs/pf-1x32 \
    --num-iterations 50 \
    --prompts-per-update 1 \
    --rollouts-per-prompt 32 \
    --max-tokens 2048 \
    --seed 0
    ```
