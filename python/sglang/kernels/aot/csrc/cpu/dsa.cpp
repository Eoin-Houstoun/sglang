/* Copyright 2026 SGLang Team. */

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace {

void check_cpu(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
}

}  // namespace

at::Tensor dsa_indexer_topk_cpu(
    const at::Tensor& q,
    const at::Tensor& keys,
    const at::Tensor& gates,
    const at::Tensor& positions,
    int64_t topk) {
  check_cpu(q, "q");
  check_cpu(keys, "keys");
  check_cpu(gates, "gates");
  check_cpu(positions, "positions");
  TORCH_CHECK(q.dim() == 3, "q must have shape [tokens, heads, head_dim]");
  TORCH_CHECK(keys.dim() == 2, "keys must have shape [kv_tokens, head_dim]");
  TORCH_CHECK(gates.dim() == 2, "gates must have shape [tokens, heads]");
  TORCH_CHECK(positions.dim() == 1, "positions must have shape [tokens]");
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(q.is_floating_point() && keys.is_floating_point(), "q and keys must be floating point");
  TORCH_CHECK(gates.is_floating_point(), "gates must be floating point");
  TORCH_CHECK(
      positions.scalar_type() == at::kLong || positions.scalar_type() == at::kInt,
      "positions must have int32 or int64 dtype");
  TORCH_CHECK(q.size(0) == gates.size(0) && q.size(0) == positions.size(0), "token dimensions must match");
  TORCH_CHECK(q.size(1) == gates.size(1), "head dimensions must match");
  TORCH_CHECK(q.size(2) == keys.size(1), "q and keys head dimensions must match");

  const int64_t tokens = q.size(0);
  const int64_t kv_tokens = keys.size(0);
  auto result = at::full({tokens, topk}, -1, q.options().dtype(at::kInt));
  if (tokens == 0 || kv_tokens == 0) {
    return result;
  }

  const at::Tensor q_fp32 = q.to(at::kFloat);
  const at::Tensor keys_t_fp32 = keys.to(at::kFloat).transpose(0, 1).contiguous();
  const at::Tensor gates_fp32 = gates.to(at::kFloat);
  const at::Tensor positions_i64 = positions.to(at::kLong).contiguous();
  const auto* position_data = positions_i64.const_data_ptr<int64_t>();

  for (int64_t token = 0; token < tokens; ++token) {
    const int64_t valid = std::max<int64_t>(0, std::min<int64_t>(kv_tokens, position_data[token] + 1));
    const int64_t count = std::min(topk, valid);
    if (count == 0) {
      continue;
    }
    if (count == valid) {
      result[token].narrow(0, 0, count).copy_(at::arange(count, result.options()));
      continue;
    }

    const at::Tensor dots = at::mm(q_fp32[token], keys_t_fp32.narrow(1, 0, valid));
    const at::Tensor scores = (at::relu(dots) * gates_fp32[token].unsqueeze(1)).sum(0);
    const at::Tensor selected = std::get<1>(at::topk(scores, count, 0, true, false));
    result[token].narrow(0, 0, count).copy_(selected.to(at::kInt));
  }
  return result;
}

at::Tensor sparse_mla_attention_cpu(
    const at::Tensor& q_abs,
    const at::Tensor& q_pe,
    const at::Tensor& c_kv,
    const at::Tensor& k_pe,
    const at::Tensor& topk_indices,
    double softmax_scale) {
  check_cpu(q_abs, "q_abs");
  check_cpu(q_pe, "q_pe");
  check_cpu(c_kv, "c_kv");
  check_cpu(k_pe, "k_pe");
  check_cpu(topk_indices, "topk_indices");
  TORCH_CHECK(q_abs.dim() == 3, "q_abs must have shape [tokens, heads, latent_dim]");
  TORCH_CHECK(q_pe.dim() == 3, "q_pe must have shape [tokens, heads, rope_dim]");
  TORCH_CHECK(c_kv.dim() == 2, "c_kv must have shape [kv_tokens, latent_dim]");
  TORCH_CHECK(k_pe.dim() == 2, "k_pe must have shape [kv_tokens, rope_dim]");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must have shape [tokens, topk]");
  TORCH_CHECK(softmax_scale > 0.0, "softmax_scale must be positive");
  TORCH_CHECK(
      q_abs.is_floating_point() && q_pe.is_floating_point() && c_kv.is_floating_point() && k_pe.is_floating_point(),
      "MLA inputs must be floating point");
  TORCH_CHECK(
      topk_indices.scalar_type() == at::kInt || topk_indices.scalar_type() == at::kLong,
      "topk_indices must have int32 or int64 dtype");
  TORCH_CHECK(
      q_abs.size(0) == q_pe.size(0) && q_abs.size(0) == topk_indices.size(0), "token dimensions must match");
  TORCH_CHECK(q_abs.size(1) == q_pe.size(1), "head dimensions must match");
  TORCH_CHECK(q_abs.size(2) == c_kv.size(1), "latent dimensions must match");
  TORCH_CHECK(q_pe.size(2) == k_pe.size(1), "rope dimensions must match");
  TORCH_CHECK(c_kv.size(0) == k_pe.size(0), "KV token dimensions must match");

  const int64_t tokens = q_abs.size(0);
  const int64_t kv_tokens = c_kv.size(0);
  at::Tensor output_fp32 = at::zeros(q_abs.sizes(), q_abs.options().dtype(at::kFloat));
  if (tokens == 0) {
    return output_fp32.to(q_abs.scalar_type());
  }

  const at::Tensor q_abs_fp32 = q_abs.to(at::kFloat);
  const at::Tensor q_pe_fp32 = q_pe.to(at::kFloat);
  const at::Tensor c_kv_fp32 = c_kv.to(at::kFloat);
  const at::Tensor k_pe_fp32 = k_pe.to(at::kFloat);
  const at::Tensor indices_i64 = topk_indices.to(at::kLong);

  auto compute_token = [&](int64_t token) {
    const at::Tensor row = indices_i64[token];
    const at::Tensor valid_indices = row.masked_select(row.ge(0));
    if (valid_indices.numel() == 0) {
      return;
    }
    TORCH_CHECK(valid_indices.max().item<int64_t>() < kv_tokens, "topk_indices contains an out-of-range index");

    const at::Tensor selected_c = c_kv_fp32.index_select(0, valid_indices);
    const at::Tensor selected_pe = k_pe_fp32.index_select(0, valid_indices);
    at::Tensor scores = at::mm(q_abs_fp32[token], selected_c.transpose(0, 1));
    scores.add_(at::mm(q_pe_fp32[token], selected_pe.transpose(0, 1)));
    const at::Tensor probabilities = at::softmax(scores * softmax_scale, -1);
    output_fp32[token].copy_(at::mm(probabilities, selected_c));
  };
  if (tokens < 64) {
    for (int64_t token = 0; token < tokens; ++token) {
      compute_token(token);
    }
  } else {
    at::parallel_for(0, tokens, 1, [&](int64_t begin, int64_t end) {
      for (int64_t token = begin; token < end; ++token) {
        compute_token(token);
      }
    });
  }
  return output_fp32.to(q_abs.scalar_type());
}
