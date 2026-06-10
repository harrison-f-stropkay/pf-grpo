from __future__ import annotations

import argparse
import json
from pathlib import Path

from pf_grpo.compat import patch_runtime_deps
from pf_grpo.core import load_math_tasks, rollout_fn, rollout_tasks, trajectory_group_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["vanilla", "pf"], required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--prm-model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--dataset", default="rasbt/math_full_minus_math500")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--prompts-per-update", type=int, default=4)
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    hook = {"rollout_fn": rollout_fn} if args.method == "vanilla" else {"trajectory_group_fn": trajectory_group_fn}

    patch_runtime_deps()
    from training_hub.algorithms.lora_grpo import lora_grpo

    result = lora_grpo(
        model_path=args.base_model,
        ckpt_output_dir=str(run_dir / "checkpoints"),
        tasks=rollout_tasks(load_math_tasks(args.dataset, args.seed), args),
        backend="art",
        num_iterations=args.num_iterations,
        group_size=args.rollouts_per_prompt,
        prompt_batch_size=args.prompts_per_update,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=1 if args.method == "pf" else args.prompts_per_update,
        **hook,
    )
    (run_dir / "training_results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
