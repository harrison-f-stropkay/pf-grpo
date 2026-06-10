# Below are compatability shims for running in JHU APL's Run:ai cluster. Note that I haven't tested these shims elsewhere.
# Before arriving at these shims, I was in vLLM/torch/ART/CUDA compatability hell...until I was rescued by OpenAI's Codex.
# The coding agent produced all of these fixes. Three cheers and a tip of the (red) hat for Codex.

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from collections.abc import Callable
from types import ModuleType


class ImportPatch(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, module_name: str, patch: Callable[[ModuleType], None]):
        self.module_name = module_name
        self.patch = patch
        self.loader = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.module_name:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is self:
            return None
        self.loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.patch(module)


def patch_runtime_deps() -> None:
    """Install required compatibility patches before Training Hub imports ART/TRL."""
    patch_multiprocessing_context()
    patch_qwen35_handler_import()
    patch_on_import("vllm.sampling_params", patch_vllm_guided_decoding)
    patch_on_import("art.megatron.model_support.registry", patch_art_model_support)
    patch_on_import("art.model", patch_art_model_train)


def patch_multiprocessing_context() -> None:
    """Force fork so ART/vLLM subprocess startup works reliably on Run:ai."""
    import multiprocessing as mp

    if getattr(mp.get_context, "_pf_grpo_patched", False):
        return
    original_get_context = mp.get_context

    def get_context(method=None):
        return original_get_context("fork" if method == "spawn" else method)

    get_context._pf_grpo_patched = True
    mp.get_context = get_context


def patch_on_import(module_name: str, patch: Callable[[ModuleType], None]) -> None:
    """Apply required patches at first import without triggering heavy imports early."""
    if module := sys.modules.get(module_name):
        patch(module)
    elif not any(getattr(finder, "module_name", None) == module_name for finder in sys.meta_path):
        sys.meta_path.insert(0, ImportPatch(module_name, patch))


def patch_qwen35_handler_import() -> None:
    """Prevent ART's unused Qwen3.5 handlers from requiring Megatron Bridge."""
    if "art.megatron.model_support.handlers.qwen3_5" in sys.modules:
        return
    qwen35 = types.ModuleType("art.megatron.model_support.handlers.qwen3_5")
    handler = types.SimpleNamespace(
        key="qwen3_5_disabled",
        native_vllm_lora_status="unsupported",
        is_moe=False,
    )
    qwen35.QWEN3_5_DENSE_HANDLER = handler
    qwen35.QWEN3_5_MOE_HANDLER = handler
    qwen35.Qwen35DenseHandler = object
    qwen35.Qwen35MoeHandler = object
    sys.modules[qwen35.__name__] = qwen35


def patch_vllm_guided_decoding(module: ModuleType) -> None:
    """Bridge TRL's expected guided-decoding API to vLLM 0.13's renamed class."""
    if not hasattr(module, "GuidedDecodingParams") and hasattr(module, "StructuredOutputsParams"):
        module.GuidedDecodingParams = module.StructuredOutputsParams


def patch_art_model_support(module: ModuleType) -> None:
    """Allow ART to register Qwen2.5 through its default dense model path."""
    for name in (
        "get_model_support_spec",
        "get_model_support_handler",
        "default_target_modules_for_model",
        "native_vllm_lora_status_for_model",
        "model_requires_merged_rollout",
        "model_uses_expert_parallel",
    ):
        fn = getattr(module, name, None)
        if fn is not None:
            fn.__kwdefaults__ = {**(fn.__kwdefaults__ or {}), "allow_unvalidated_arch": True}


def patch_art_model_train(module: ModuleType) -> None:
    """Restore the model.train API required by this Training Hub branch."""
    trainable_model = getattr(module, "TrainableModel", None)
    if trainable_model is None or hasattr(trainable_model, "train"):
        return

    async def train(self, trajectory_groups, config=None, **kwargs):
        params = config.model_dump(exclude_none=True) if config is not None else {}
        return await self.backend().train(self, trajectory_groups, **params, **kwargs)

    trainable_model.train = train
