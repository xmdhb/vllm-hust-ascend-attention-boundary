# OP01 — Ascend 注意力边界查找

独立包名：`vllm-hust-op01-ascend-attention-boundary`

GitHub 仓库：`xmdhb/vllm-hust-ascend-attention-boundary`

对应 Ascend #9：在 `split_decodes_and_prefills` 中用 `searchsorted` 查找首个 prefill 边界。

启用变量：

```bash
export VLLM_HUST_OP01_ASCEND_ATTENTION_BOUNDARY_ENABLE=1
export VLLM_HUST_OP01_ASCEND_ATTENTION_BOUNDARY_EVIDENCE=1
```

默认关闭，`VLLM_HUST_OP01_ASCEND_ATTENTION_BOUNDARY_KILL_SWITCH=1` 优先级最高。

打包：在本目录执行 `python3 -m build --wheel`。测试：`PYTHONPATH=src python3 -m pytest -q tests`。
