import torch

from vllm_hust_ascend_attention_boundary.boundary import find_first_true_boundary


def test_find_first_true_boundary():
    assert find_first_true_boundary(torch.tensor([], dtype=torch.bool)) == 0
    assert find_first_true_boundary(torch.tensor([False, False, True])) == 2
    assert find_first_true_boundary(torch.tensor([False, False])) == 2
