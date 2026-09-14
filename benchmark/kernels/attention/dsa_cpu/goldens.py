"""Independent FP32 correctness oracle for CPU DSA."""

from __future__ import annotations

import torch


def indexer_topk_fp32(
    q: torch.Tensor,
    keys: torch.Tensor,
    gates: torch.Tensor,
    positions: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    scores = torch.einsum("thd,sd->ths", q.float(), keys.float()).relu_()
    scores = (scores * gates.float().unsqueeze(-1)).sum(dim=1)
    output = torch.full((q.shape[0], topk), -1, dtype=torch.int32)
    for token in range(q.shape[0]):
        valid = min(keys.shape[0], int(positions[token]) + 1)
        count = min(topk, valid)
        if count == valid:
            selected = torch.arange(valid)
        else:
            selected = torch.topk(scores[token, :valid], count, sorted=False).indices
        output[token, :count] = selected.to(torch.int32)
    return output


def sparse_mla_latent_fp32(
    q_abs: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    output = torch.empty(q_abs.shape, dtype=torch.float32)
    for token in range(q_abs.shape[0]):
        indices = topk_indices[token]
        indices = indices[indices >= 0].to(torch.int64)
        selected_c = c_kv[indices].float()
        selected_pe = k_pe[indices].float()
        scores = torch.mm(q_abs[token].float(), selected_c.transpose(0, 1))
        scores.add_(torch.mm(q_pe[token].float(), selected_pe.transpose(0, 1)))
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[token] = torch.mm(probabilities, selected_c)
    return output


def minimum_row_overlap(actual: torch.Tensor, expected: torch.Tensor) -> float:
    minimum = 1.0
    for actual_row, expected_row in zip(actual, expected):
        actual_set = set(actual_row[actual_row >= 0].tolist())
        expected_set = set(expected_row[expected_row >= 0].tolist())
        overlap = len(actual_set & expected_set) / max(1, len(expected_set))
        minimum = min(minimum, overlap)
    return minimum


def output_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    maximum_absolute = float((actual_fp32 - expected_fp32).abs().max())
    relative_l2 = float(
        (actual_fp32 - expected_fp32).norm() / expected_fp32.norm().clamp_min(1e-12)
    )
    return maximum_absolute, relative_l2
