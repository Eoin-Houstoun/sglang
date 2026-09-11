"""Seeded weights and activations for the DSA CPU benchmark at GLM-5.2's attention shape.

The Indexer is SGLang's own (`dsa_indexer.Indexer`), built on CPU with the shape below and
its weights loaded from a seeded generator, so the maths is the module's. Only the hidden
width and q_lora_rank are reduced from GLM-5.2 (6144 and 2048); head counts, head dims,
index_topk and rope are the model's.
"""

import msgspec
import torch

from sglang.srt.layers.attention.dsa.dsa_cpu import DSACPUBatch, index_k_cpu
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer


class DSAConfig(msgspec.Struct, frozen=True):
    hidden_size: int = 1024
    q_lora_rank: int = 512
    num_attention_heads: int = 64
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    rope_theta: float = 8_000_000.0
    max_position_embeddings: int = 16384
    dtype: torch.dtype = torch.bfloat16

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim**-0.5


GLM52_SMALL = DSAConfig()


class DSAWeights(msgspec.Struct):
    wq_b: torch.Tensor  # [Hi*Di, q_lora_rank]
    wk: torch.Tensor  # [Di, hidden]
    weights_proj: torch.Tensor  # [Hi, hidden]
    k_norm_weight: torch.Tensor  # [Di]
    k_norm_bias: torch.Tensor  # [Di]
    W_UK: torch.Tensor  # [H, qk_nope, kv_lora]
    W_UV: torch.Tensor  # [H, kv_lora, v_head]


class DSABatch(msgspec.Struct):
    name: str
    x: torch.Tensor  # [T, hidden] layer input of the query tokens
    q_lora: torch.Tensor  # [T, q_lora_rank]
    positions: torch.Tensor  # [T] int64
    q_nope: torch.Tensor  # [T, H, qk_nope]
    q_pe: torch.Tensor  # [T, H, rope], rotary applied
    c_kv: torch.Tensor  # [S, kv_lora] full KV cache, query tokens included
    k_pe: torch.Tensor  # [S, rope], rotary applied
    index_k_cache: torch.Tensor  # [S - T, Di] indexer keys of the context before this step
    causal_offset: int  # query t may attend to s <= t + causal_offset

    def cpu_batch(self) -> DSACPUBatch:
        return DSACPUBatch(index_k_cache=self.index_k_cache)


# name -> (query tokens, kv length, sequences)
OPERATING_POINTS = {
    "prefill_1k": (1024, 1024, 1),
    "prefill_2k": (2048, 2048, 1),
    "prefill_4k": (4096, 4096, 1),
    "prefill_8k": (8192, 8192, 1),
    "decode_8k": (1, 8192, 4),
}
WEIGHT_SEED = 20260910


def rope_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    return freqs.cos(), freqs.sin()


def apply_rope_interleaved(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, theta: float) -> torch.Tensor:
    """Rotate the first `rotary_dim` features of x's last axis using interleaved pairs (2i, 2i+1)."""
    cos, sin = rope_cos_sin(positions, rotary_dim, theta)
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [rotary_dim // 2]
    cos, sin = cos.view(shape), sin.view(shape)
    rot = x[..., :rotary_dim].float()
    x1, x2 = rot[..., 0::2], rot[..., 1::2]
    out = torch.empty_like(rot)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return torch.cat([out.to(x.dtype), x[..., rotary_dim:]], dim=-1)


def make_weights(config: DSAConfig = GLM52_SMALL, seed: int = WEIGHT_SEED) -> DSAWeights:
    g = torch.Generator().manual_seed(seed)
    Hi, Di, H = config.index_n_heads, config.index_head_dim, config.num_attention_heads

    def lin(out_f, in_f):
        return (torch.randn(out_f, in_f, generator=g) * in_f**-0.5).to(config.dtype)

    return DSAWeights(
        wq_b=lin(Hi * Di, config.q_lora_rank),
        wk=lin(Di, config.hidden_size),
        weights_proj=lin(Hi, config.hidden_size),
        k_norm_weight=torch.ones(Di, dtype=config.dtype),
        k_norm_bias=torch.zeros(Di, dtype=config.dtype),
        W_UK=(torch.randn(H, config.qk_nope_head_dim, config.kv_lora_rank, generator=g) * config.qk_nope_head_dim**-0.5).to(config.dtype),
        W_UV=(torch.randn(H, config.kv_lora_rank, config.v_head_dim, generator=g) * config.kv_lora_rank**-0.5).to(config.dtype),
    )


def build_indexer(config: DSAConfig = GLM52_SMALL, weights: DSAWeights | None = None) -> Indexer:
    """SGLang's Indexer at this shape on CPU, weights loaded from `weights` (seeded by default)."""
    weights = weights or make_weights(config)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(config.dtype)
    try:
        indexer = Indexer(
            hidden_size=config.hidden_size,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            index_topk=config.index_topk,
            q_lora_rank=config.q_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            layer_id=0,
            scale_fmt=None,
            block_size=128,
            rope_scaling=None,
            is_neox_style=False,  # GLM-5.2: indexer_rope_interleave = true
            prefix="indexer",
            quant_config=None,
            alt_stream=None,
            config=None,
        )
    finally:
        torch.set_default_dtype(prev)
    with torch.no_grad():
        indexer.wq_b.weight.data.copy_(weights.wq_b)
        indexer.wk.weight.data.copy_(weights.wk)
        indexer.weights_proj.weight.data.copy_(weights.weights_proj)
        indexer.k_norm.weight.data.copy_(weights.k_norm_weight.to(indexer.k_norm.weight.dtype))
        indexer.k_norm.bias.data.copy_(weights.k_norm_bias.to(indexer.k_norm.bias.dtype))
    indexer.eval()
    return indexer


def full_context_tensors(name: str, config: DSAConfig, seed: int, seq: int):
    """Activations for the whole context of one sequence: (x, q_lora, positions, q_nope, q_pe, c_kv, k_pe)."""
    _, S, _ = OPERATING_POINTS[name]
    point_id = list(OPERATING_POINTS).index(name)
    g = torch.Generator().manual_seed(seed * 100_003 + point_id * 101 + seq)
    positions = torch.arange(S)
    x = torch.randn(S, config.hidden_size, generator=g).to(config.dtype)
    q_lora = torch.randn(S, config.q_lora_rank, generator=g).to(config.dtype)
    q_nope = torch.randn(S, config.num_attention_heads, config.qk_nope_head_dim, generator=g).to(config.dtype)
    q_pe = apply_rope_interleaved(torch.randn(S, config.num_attention_heads, config.qk_rope_head_dim, generator=g).to(config.dtype), positions, config.qk_rope_head_dim, config.rope_theta)
    c_kv = torch.randn(S, config.kv_lora_rank, generator=g).to(config.dtype)
    k_pe = apply_rope_interleaved(torch.randn(S, config.qk_rope_head_dim, generator=g).to(config.dtype), positions, config.qk_rope_head_dim, config.rope_theta)
    return x, q_lora, positions, q_nope, q_pe, c_kv, k_pe


def make_batches(name: str, config: DSAConfig, indexer: Indexer, seed: int = 0) -> list[DSABatch]:
    """Deterministic activations for one operating point, one DSABatch per sequence."""
    T, S, n_seq = OPERATING_POINTS[name]
    batches = []
    for i in range(n_seq):
        x, q_lora, positions, q_nope, q_pe, c_kv, k_pe = full_context_tensors(name, config, seed, i)
        q = slice(S - T, S)
        with torch.no_grad():
            cache = index_k_cpu(indexer, x[: S - T], positions[: S - T])
        batches.append(
            DSABatch(
                name=name,
                x=x[q],
                q_lora=q_lora[q],
                positions=positions[q],
                q_nope=q_nope[q],
                q_pe=q_pe[q],
                c_kv=c_kv,
                k_pe=k_pe,
                index_k_cache=cache,
                causal_offset=S - T,
            )
        )
    return batches


def dense_mla_attention(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, scale, causal_offset=0, chunk=256):
    """Absorbed MLA over every valid position: the benchmark's stand-in for the dense fallback."""
    T = q_nope.shape[0]
    S = c_kv.shape[0]
    q_abs = torch.einsum("thd,hdr->thr", q_nope, W_UK)
    out = torch.empty(T, W_UV.shape[0], W_UV.shape[-1], dtype=q_nope.dtype)
    s_idx = torch.arange(S)
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        scores = torch.einsum("thr,sr->ths", q_abs[t0:t1], c_kv) + torch.einsum("thd,sd->ths", q_pe[t0:t1], k_pe)
        scores = scores.float() * scale
        limit = (torch.arange(t0, t1) + causal_offset)[:, None, None]
        scores.masked_fill_(s_idx[None, None, :] > limit, float("-inf"))
        p = torch.softmax(scores, dim=-1).to(c_kv.dtype)
        o_lat = torch.einsum("ths,sr->thr", p, c_kv)
        out[t0:t1] = torch.einsum("thr,hrv->thv", o_lat, W_UV)
    return out
