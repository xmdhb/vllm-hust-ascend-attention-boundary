"""Ascend #9 boundary implementation."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F


def find_first_true_boundary(sorted_mask: torch.Tensor) -> int:
    if sorted_mask.numel() == 0:
        return 0
    return int(torch.searchsorted(sorted_mask.to(dtype=torch.int32), 1).item())


def compute_boundary(
    ascend: ModuleType,
    metadata: Any,
    evidence: Callable[[str, str], None],
    decode_threshold: int = 1,
    require_uniform: bool = False,
    treat_short_extends_as_decodes: bool = True,
) -> tuple[int, int, int, int]:
    long_seq_metadata = metadata.prefill_context_parallel_metadata
    query_lens_pcp_full = (
        long_seq_metadata.query_lens_pcp_full_cpu if long_seq_metadata else None
    )
    max_query_len_pcp_full = (
        long_seq_metadata.max_query_len_pcp_full if long_seq_metadata else 0
    )
    max_query_len = (
        metadata.max_query_len
        if max_query_len_pcp_full == 0
        else max_query_len_pcp_full
    )
    num_reqs = metadata.num_reqs
    if num_reqs == 0:
        return 0, 0, 0, 0

    num_tokens = metadata.num_actual_tokens
    query_start_loc = metadata.query_start_loc_cpu
    if ascend.is_pd_decode_recompute_scheduler_enabled():
        treat_short_extends_as_decodes = True

    if (
        max_query_len <= decode_threshold
        and (not require_uniform or decode_threshold <= 1)
        and treat_short_extends_as_decodes
    ):
        evidence("runtime_effective", "ascend_boundary_first_true_search")
        return num_reqs, 0, num_tokens, 0

    query_lens_sharded = query_start_loc[1:] - query_start_loc[:-1]
    query_lens = (
        query_lens_sharded
        if query_lens_pcp_full is None
        else query_lens_pcp_full
    )
    if query_lens[0].item() > decode_threshold:
        evidence("runtime_effective", "ascend_boundary_first_true_search")
        return 0, num_reqs, 0, num_tokens

    if require_uniform:
        if torch.all((query_lens == query_lens[0]) | (query_lens == 0)):
            evidence("runtime_effective", "ascend_boundary_first_true_search")
            return num_reqs, 0, num_tokens, 0
        is_prefill = query_lens != query_lens[0]
    else:
        is_prefill = query_lens > decode_threshold

    if not treat_short_extends_as_decodes:
        assert metadata.is_prefilling is not None
        is_prefilling = metadata.is_prefilling[: query_lens.shape[0]]
        if is_prefilling.shape[0] < query_lens.shape[0]:
            is_prefilling = F.pad(
                is_prefilling,
                (0, query_lens.shape[0] - is_prefilling.shape[0]),
                value=False,
            )
        is_prefill |= is_prefilling

    if not torch.any(is_prefill):
        evidence("runtime_effective", "ascend_boundary_first_true_search")
        return num_reqs, 0, num_tokens, 0

    first_prefill = find_first_true_boundary(is_prefill)
    result = (
        first_prefill,
        num_reqs - first_prefill,
        query_start_loc[first_prefill].item(),
        num_tokens - query_start_loc[first_prefill].item(),
    )
    evidence("runtime_effective", "ascend_boundary_first_true_search")
    return result


def install(
    ascend: ModuleType,
    replace_imported_references: Callable,
    evidence: Callable[[str, str], None],
    marker: str,
) -> None:
    old = ascend.split_decodes_and_prefills
    if getattr(old, marker, False):
        return

    def patched(
        metadata: Any,
        decode_threshold: int = 1,
        require_uniform: bool = False,
        treat_short_extends_as_decodes: bool = True,
    ) -> tuple[int, int, int, int]:
        return compute_boundary(
            ascend,
            metadata,
            evidence,
            decode_threshold,
            require_uniform,
            treat_short_extends_as_decodes,
        )

    setattr(patched, marker, "ascend_boundary_first_true_search")
    ascend.split_decodes_and_prefills = patched
    replace_imported_references(old, patched)


__all__ = ["find_first_true_boundary", "compute_boundary", "install"]
