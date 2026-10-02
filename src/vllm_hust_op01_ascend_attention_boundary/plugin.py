"""Plugin entry point for the independent Ascend boundary package."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

from . import boundary

ENABLE_ENV = "VLLM_HUST_OP01_ASCEND_ATTENTION_BOUNDARY_ENABLE"
KILL_SWITCH_ENV = "VLLM_HUST_OP01_ASCEND_ATTENTION_BOUNDARY_KILL_SWITCH"
EVIDENCE_ENV = "VLLM_HUST_OP01_ASCEND_ATTENTION_BOUNDARY_EVIDENCE"
PATCH_MARKER = "__vllm_hust_op01_ascend_attention_boundary__"
_seen: set[str] = set()
_lock = threading.Lock()


def _enabled(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def _evidence(event: str, mechanism: str) -> None:
    if not _enabled(os.getenv(EVIDENCE_ENV)):
        return
    text = f"{event} mechanism={mechanism}"
    with _lock:
        if text in _seen:
            return
        _seen.add(text)
    print(f"LEGACY017_EVIDENCE {text}", file=sys.stderr, flush=True)


def _replace(old: Callable[..., Any], new: Callable[..., Any]) -> None:
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        try:
            namespace = vars(module)
        except TypeError:
            continue
        for name, value in tuple(namespace.items()):
            if value is old:
                try:
                    setattr(module, name, new)
                except (AttributeError, TypeError):
                    pass


def _check_host(module: Any) -> None:
    function = getattr(module, "split_decodes_and_prefills", None)
    if not callable(function):
        raise RuntimeError("OP01 Ascend boundary host function is missing")
    if getattr(function, PATCH_MARKER, False):
        return
    source = inspect.getsource(function)
    if "is_prefill" not in source or (
        "argmax" not in source and "_find_first_true_boundary" not in source
    ):
        raise RuntimeError("OP01 Ascend boundary host source does not match")


def register() -> None:
    if _enabled(os.getenv(KILL_SWITCH_ENV)) or not _enabled(os.getenv(ENABLE_ENV)):
        return
    version = importlib.metadata.version("vllm")
    if not version.startswith("0.23."):
        raise RuntimeError(f"OP01 Ascend boundary requires vLLM 0.23.x, got {version}")
    ascend = importlib.import_module("vllm_ascend.attention.utils")
    _check_host(ascend)
    boundary.install(ascend, _replace, _evidence, PATCH_MARKER)
    _evidence("installed", "ascend_boundary_first_true_search")
