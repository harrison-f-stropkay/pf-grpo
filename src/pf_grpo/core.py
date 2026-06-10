from __future__ import annotations

import asyncio
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import Mapping, cast

import math_verify
from datasets import load_dataset


@dataclass(frozen=True)
class Task:
    problem: str
    answer: str


@dataclass(frozen=True)
class RolloutTask(Task):
    # Putting config on the pickled task makes each rollout self-contained for ART's spawn backend.
    config: Namespace


def _task(row: Mapping[str, object]) -> Task:
    answer = next(row[key] for key in ("answer", "final_answer", "target") if key in row)
    return Task(
        problem=str(row.get("problem") or row.get("question") or row["prompt"]),
        answer=str(answer),
    )


def _message(message) -> dict[str, str]:
    if isinstance(message, dict):
        return {"role": message["role"], "content": message["content"]}
    return {"role": message.role, "content": message.extract_text_content()}


def load_math_tasks(name: str, seed: int, split: str = "train") -> list[Task]:
    rows = cast(list[Mapping[str, object]], list(load_dataset(name, split=split)))
    random.Random(seed).shuffle(rows)
    return [_task(row) for row in rows]


def rollout_tasks(tasks: list[Task], config: Namespace) -> list[RolloutTask]:
    return [RolloutTask(task.problem, task.answer, config) for task in tasks]


def make_messages(task: Task) -> list[dict[str, str]]:
    prompt = (
        "Solve the following math problem. Show your work, and put the final answer in \\boxed{}."
    )
    return [{"role": "user", "content": f"{prompt}\n\n{task.problem}"}]


def reward_math(response: str, task: Task) -> float:
    return float(math_verify.verify(math_verify.parse(task.answer), math_verify.parse(response)))


class ItsHubPolicyAdapter:
    def __init__(self, model, config: Namespace):
        self.model = model
        self.config = config

    async def agenerate(self, messages, stop: str | None = None, **_):
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


class AsyncProcessRewardModel:
    def __init__(self, model):
        self.model = model

    async def ascore(self, prompt: str, responses: str | list[str]):
        response_list = [responses] if isinstance(responses, str) else responses
        messages = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for response in response_list
        ]
        scores = await asyncio.to_thread(self.model.score, messages)
        return scores[0] if isinstance(responses, str) else scores


async def vanilla_rollout(model, task: Task, config: Namespace):
    from art import Trajectory

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


async def pf_trajectory_group(model, task: Task, group_size: int, config: Namespace):
    from art import Trajectory, TrajectoryGroup
    from its_hub.core.algorithms.particle_gibbs import ParticleFiltering
    from its_hub.core.lms.step_generation import StepGeneration
    from its_hub.api import ChatMessage
    from reward_hub.hf.reward import HuggingFaceProcessRewardModel

    messages = make_messages(task)
    prm = AsyncProcessRewardModel(HuggingFaceProcessRewardModel(config.prm_model))
    pf_messages = [ChatMessage.from_dict(message) for message in messages]
    result = await ParticleFiltering(
        StepGeneration(max_steps=config.max_steps, step_token="\n\n", stop_token=r"\boxed"),
        prm,
    ).ainfer(
        ItsHubPolicyAdapter(model, config),
        pf_messages,
        budget=group_size,
        return_response_only=False,
    )

    trajectories = [
        Trajectory(messages_and_choices=[*messages, response]) for response in result.responses
    ]
    for trajectory, response in zip(trajectories, result.responses):
        trajectory.reward = reward_math(str(response.get("content") or ""), task)
    return TrajectoryGroup(trajectories)


async def rollout_fn(model, task: RolloutTask):
    return await vanilla_rollout(model, task, task.config)


async def trajectory_group_fn(model, task: RolloutTask, group_size: int):
    return await pf_trajectory_group(model, task, group_size, task.config)
