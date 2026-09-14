/* Copyright 2026 SGLang Team.
 *
 * CPU DeepSeek Sparse Attention operation boundaries. Discovery replaces the
 * explicit stubs below with optimized AVX-512/AMX implementations while the
 * schemas and benchmark contract remain fixed.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor dsa_indexer_topk_cpu(
    const at::Tensor& q,
    const at::Tensor& keys,
    const at::Tensor& gates,
    const at::Tensor& positions,
    int64_t topk) {
  TORCH_CHECK(q.device().is_cpu(), "q must be a CPU tensor");
  TORCH_CHECK(q.dim() == 3, "q must have shape [tokens, heads, head_dim]");
  TORCH_CHECK(keys.dim() == 2, "keys must have shape [kv_tokens, head_dim]");
  TORCH_CHECK(gates.dim() == 2, "gates must have shape [tokens, heads]");
  TORCH_CHECK(positions.dim() == 1, "positions must have shape [tokens]");
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(false, "dsa_indexer_topk_cpu is not implemented");
}

at::Tensor sparse_mla_attention_cpu(
    const at::Tensor& q_abs,
    const at::Tensor& q_pe,
    const at::Tensor& c_kv,
    const at::Tensor& k_pe,
    const at::Tensor& topk_indices,
    double softmax_scale) {
  TORCH_CHECK(q_abs.device().is_cpu(), "q_abs must be a CPU tensor");
  TORCH_CHECK(q_abs.dim() == 3, "q_abs must have shape [tokens, heads, latent_dim]");
  TORCH_CHECK(q_pe.dim() == 3, "q_pe must have shape [tokens, heads, rope_dim]");
  TORCH_CHECK(c_kv.dim() == 2, "c_kv must have shape [kv_tokens, latent_dim]");
  TORCH_CHECK(k_pe.dim() == 2, "k_pe must have shape [kv_tokens, rope_dim]");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must have shape [tokens, topk]");
  TORCH_CHECK(softmax_scale > 0.0, "softmax_scale must be positive");
  TORCH_CHECK(false, "sparse_mla_attention_cpu is not implemented");
}
