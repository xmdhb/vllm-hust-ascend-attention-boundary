from __future__ import annotations

import importlib
import types

import pytest

from vllm_hust_ascend_attention_boundary import plugin


CURRENT_ASCEND_COMMIT = next(iter(plugin.SUPPORTED_ASCEND_COMMITS))
UNSUPPORTED_ASCEND_COMMIT = "0" * 40


def compatible_host(
    common_attn_metadata,
    decode_threshold=1,
    require_uniform=False,
    treat_short_extends_as_decodes=True,
):
    query_start_loc = common_attn_metadata.query_start_loc_cpu
    is_prefill = query_start_loc > decode_threshold
    return (0, 0, 0, 0) if is_prefill else (0, 0, 0, 0)


def incompatible_abi_host(common_attn_metadata, decode_threshold=1):
    return (0, 0, 0, 0)


def incompatible_semantics_host(
    common_attn_metadata,
    decode_threshold=1,
    require_uniform=False,
    treat_short_extends_as_decodes=True,
):
    return (0, 0, 0, 0)


def make_host(function, commit=CURRENT_ASCEND_COMMIT):
    return types.SimpleNamespace(
        split_decodes_and_prefills=function,
        __vllm_hust_host_commit__=commit,
    )


def test_missing_host_function_is_rejected():
    with pytest.raises(RuntimeError, match="host function is missing"):
        plugin._check_host(types.SimpleNamespace(), host_commit=CURRENT_ASCEND_COMMIT)


def test_host_abi_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="host ABI mismatch"):
        plugin._check_host(
            make_host(incompatible_abi_host), host_commit=CURRENT_ASCEND_COMMIT
        )


def test_host_semantics_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="host semantics do not match"):
        plugin._check_host(
            make_host(incompatible_semantics_host),
            host_commit=CURRENT_ASCEND_COMMIT,
        )


def test_current_host_signature_and_commit_are_accepted():
    plugin._check_host(make_host(compatible_host), host_commit=CURRENT_ASCEND_COMMIT)


def test_installed_current_host_is_accepted_when_available():
    try:
        host = importlib.import_module("vllm_ascend.attention.utils")
    except ModuleNotFoundError:
        pytest.skip("vllm-ascend is not installed in this source-test environment")
    plugin._check_host(host)


def test_rejected_host_is_not_replaced(monkeypatch):
    host = make_host(compatible_host, UNSUPPORTED_ASCEND_COMMIT)
    original = host.split_decodes_and_prefills
    real_import_module = importlib.import_module
    monkeypatch.setenv(plugin.ENABLE_ENV, "1")
    monkeypatch.delenv(plugin.KILL_SWITCH_ENV, raising=False)
    monkeypatch.setattr(plugin.importlib.metadata, "version", lambda name: "0.23.0")
    monkeypatch.setattr(
        plugin.importlib,
        "import_module",
        lambda name: host
        if name == "vllm_ascend.attention.utils"
        else real_import_module(name),
    )

    with pytest.raises(RuntimeError, match="host commit is unsupported"):
        plugin.register()

    assert host.split_decodes_and_prefills is original


def test_host_version_mismatch_is_rejected_before_replacement(monkeypatch):
    host = make_host(compatible_host)
    original = host.split_decodes_and_prefills
    monkeypatch.setenv(plugin.ENABLE_ENV, "1")
    monkeypatch.delenv(plugin.KILL_SWITCH_ENV, raising=False)
    monkeypatch.setattr(
        plugin.importlib.metadata,
        "version",
        lambda name: "0.24.0" if name == "vllm" else "0.23.0",
    )
    monkeypatch.setattr(plugin.importlib, "import_module", lambda name: host)

    with pytest.raises(RuntimeError, match="requires vLLM 0.23.x"):
        plugin.register()

    assert host.split_decodes_and_prefills is original


def test_duplicate_registration_does_not_stack_patch(monkeypatch):
    host = make_host(compatible_host)
    real_import_module = importlib.import_module
    monkeypatch.setenv(plugin.ENABLE_ENV, "1")
    monkeypatch.delenv(plugin.KILL_SWITCH_ENV, raising=False)
    monkeypatch.setattr(plugin.importlib.metadata, "version", lambda name: "0.23.0")
    monkeypatch.setattr(
        plugin.importlib,
        "import_module",
        lambda name: host
        if name == "vllm_ascend.attention.utils"
        else real_import_module(name),
    )

    plugin.register()
    patched = host.split_decodes_and_prefills
    plugin.register()

    assert host.split_decodes_and_prefills is patched
    assert getattr(patched, plugin.PATCH_MARKER) == "ascend_boundary_first_true_search"
