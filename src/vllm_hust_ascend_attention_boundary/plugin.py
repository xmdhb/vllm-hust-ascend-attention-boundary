"""Plugin entry point for the independent Ascend boundary package."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import Any

from . import boundary

ENABLE_ENV = "VLLM_HUST_ASCEND_ATTENTION_BOUNDARY_ENABLE"
KILL_SWITCH_ENV = "VLLM_HUST_ASCEND_ATTENTION_BOUNDARY_KILL_SWITCH"
EVIDENCE_ENV = "VLLM_HUST_ASCEND_ATTENTION_BOUNDARY_EVIDENCE"
PATCH_MARKER = "__vllm_hust_ascend_attention_boundary__"
SUPPORTED_VLLM_PREFIX = "0.23."
SUPPORTED_VLLM_ASCEND_PREFIX = "0.23."
SUPPORTED_ASCEND_COMMITS = frozenset(
    {"1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8"}
)
HOST_FUNCTION = "split_decodes_and_prefills"
HOST_PARAMETER_NAMES = (
    "common_attn_metadata",
    "decode_threshold",
    "require_uniform",
    "treat_short_extends_as_decodes",
)
HOST_PARAMETER_DEFAULTS = (inspect.Parameter.empty, 1, False, True)
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


def _resolve_host_commit(module: Any) -> str | None:
    declared = getattr(module, "__vllm_hust_host_commit__", None)
    if declared:
        return str(declared)

    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    path = Path(module_file).resolve()
    for parent in (path.parent, *path.parents):
        if not (parent / ".git").exists():
            continue
        result = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            if commit:
                return commit
        break
    return None


def _check_host(module: Any, *, host_commit: str | None = None) -> None:
    function = getattr(module, HOST_FUNCTION, None)
    if not callable(function):
        raise RuntimeError("Ascend boundary host function is missing")
    if getattr(function, PATCH_MARKER, False):
        return

    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    if tuple(parameter.name for parameter in parameters) != HOST_PARAMETER_NAMES:
        raise RuntimeError(
            "Ascend boundary host ABI mismatch: unexpected function parameters"
        )
    for parameter, expected_default in zip(parameters, HOST_PARAMETER_DEFAULTS):
        if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise RuntimeError(
                "Ascend boundary host ABI mismatch: unsupported parameter kind"
            )
        if parameter.default != expected_default:
            raise RuntimeError(
                "Ascend boundary host ABI mismatch: unexpected parameter default"
            )

    source = inspect.getsource(function)
    if "query_start_loc" not in source or "is_prefill" not in source:
        raise RuntimeError("Ascend boundary host semantics do not match")

    actual_commit = host_commit or _resolve_host_commit(module)
    if actual_commit is None:
        raise RuntimeError(
            "Ascend boundary host commit cannot be verified; refusing to install"
        )
    if actual_commit not in SUPPORTED_ASCEND_COMMITS:
        raise RuntimeError(
            "Ascend boundary host commit is unsupported: "
            f"{actual_commit}; supported={sorted(SUPPORTED_ASCEND_COMMITS)}"
        )


def register() -> None:
    if _enabled(os.getenv(KILL_SWITCH_ENV)) or not _enabled(os.getenv(ENABLE_ENV)):
        return
    version = importlib.metadata.version("vllm")
    if not version.startswith(SUPPORTED_VLLM_PREFIX):
        raise RuntimeError(f"Ascend boundary requires vLLM 0.23.x, got {version}")
    ascend_version = importlib.metadata.version("vllm-ascend")
    if not ascend_version.startswith(SUPPORTED_VLLM_ASCEND_PREFIX):
        raise RuntimeError(
            "Ascend boundary requires vllm-ascend 0.23.x, "
            f"got {ascend_version}"
        )
    ascend = importlib.import_module("vllm_ascend.attention.utils")
    _check_host(ascend)
    boundary.install(ascend, _replace, _evidence, PATCH_MARKER)
    _evidence("installed", "ascend_boundary_first_true_search")
