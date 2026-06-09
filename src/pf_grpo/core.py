from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, cast

import math_verify
from art import Trajectory, TrajectoryGroup
from datasets import load_dataset
from its_hub.core.algorithms.particle_gibbs import ParticleFiltering
from its_hub.core.lms.step_generation import StepGeneration
from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel
from reward_hub.base import AggregationMethod


@dataclass(frozen=True)
class Task:
    problem: str
    answer: str


@dataclass(frozen=True)
class RolloutTask(Task):
    # Putting config on the pickled task makes each rollout self-contained for ART's spawn backend.
    config: Any


def _task(row: Mapping[str, Any]) -> Task:
    return Task(
        problem=str(row.get("problem") or row.get("question") or row["prompt"]),
        answer=str(row.get("answer") or row.get("final_answer") or row["target"]),
    )


def _message(message: Any) -> dict[str, str]:
    if isinstance(message, dict):
        return {"role": message["role"], "content": message["content"]}
    return {"role": message.role, "content": message.extract_text_content()}


def load_math_tasks(name: str, seed: int, split: str = "train") -> list[Task]:
    rows = cast(list[Mapping[str, Any]], list(load_dataset(name, split=split)))
    random.Random(seed).shuffle(rows)
    return [_task(row) for row in rows]


def rollout_tasks(tasks: list[Task], config: Any) -> list[RolloutTask]:
    return [RolloutTask(task.problem, task.answer, config) for task in tasks]


def make_messages(task: Task) -> list[dict[str, str]]:
    prompt = (
        "Solve the following math problem. Show your work, and put the final answer in \\boxed{}."
    )
    return [{"role": "user", "content": f"{prompt}\n\n{task.problem}"}]


def reward_math(response: str, task: Task) -> float:
    return float(math_verify.verify(math_verify.parse(task.answer), math_verify.parse(response)))


class ItsHubPolicyAdapter:
    def __init__(self, model: Any, config: Any):
        self.model = model
        self.config = config

    async def agenerate(
        self, messages: list[Any] | list[list[Any]], stop: str | None = None, **_: Any
    ):
        if messages and isinstance(messages[0], list):
            return [await self.agenerate(batch, stop=stop) for batch in messages]
        response = await self.model.openai_client().chat.completions.create(
            model=self.model.get_inference_name(),
            messages=[_message(message) for message in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stop=stop,
        )
        return {"role": "assistant", "content": response.choices[0].message.content or ""}


async def vanilla_rollout(model: Any, task: Task, config: Any) -> Trajectory:
    messages = make_messages(task)
    response = await model.openai_client().chat.completions.create(
        model=model.get_inference_name(),
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    choice = response.choices[0]
    trajectory = Trajectory(messages_and_choices=[*messages, choice])
    trajectory.reward = reward_math(choice.message.content or "", task)
    return trajectory


async def pf_trajectory_group(
    model: Any, task: Task, group_size: int, config: Any
) -> TrajectoryGroup:
    messages = make_messages(task)
    result = await ParticleFiltering(
        StepGeneration(max_steps=config.max_steps, step_token="\n\n", stop_token=r"\boxed"),
        LocalVllmProcessRewardModel(config.prm_model, "cuda:0", AggregationMethod.MODEL),
    ).ainfer(
        ItsHubPolicyAdapter(model, config), messages, budget=group_size, return_response_only=False
    )

    trajectories = [
        Trajectory(messages_and_choices=[*messages, response]) for response in result.responses
    ]
    for trajectory, response in zip(trajectories, result.responses):
        trajectory.reward = reward_math(str(response.get("content") or ""), task)
    return TrajectoryGroup(trajectories)


async def rollout_fn(model: Any, task: RolloutTask) -> Trajectory:
    return await vanilla_rollout(model, task, task.config)


async def trajectory_group_fn(model: Any, task: RolloutTask, group_size: int) -> TrajectoryGroup:
    return await pf_trajectory_group(model, task, group_size, task.config)
