import sys

from vllm_hust_ascend_attention_boundary.plugin import register


def test_default_disabled_does_not_import_vllm(monkeypatch):
    monkeypatch.delenv("VLLM_HUST_ASCEND_ATTENTION_BOUNDARY_ENABLE", raising=False)
    monkeypatch.delenv("VLLM_HUST_ASCEND_ATTENTION_BOUNDARY_KILL_SWITCH", raising=False)
    register()
    assert "vllm" not in sys.modules
