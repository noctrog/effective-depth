from typing import List, Optional, Tuple
from einops import rearrange
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaConfig,
    LlamaRMSNorm,
    Cache,
    Unpack,
    FlashAttentionKwargs,
    apply_rotary_pos_emb,
)


class TPLlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        assert config.intermediate_size % self.world_size == 0

        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size // self.world_size,
            bias=config.mlp_bias,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size // self.world_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size // self.world_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )

    @torch.no_grad()
    def init_from_layer(self, layer: LlamaMLP) -> None:
        local_h = self.intermediate_size // self.world_size
        start = self.rank * local_h
        end = (self.rank + 1) * local_h

        self.gate_proj.weight.copy_(layer.gate_proj.weight[start:end])
        self.up_proj.weight.copy_(layer.up_proj.weight[start:end])
        self.down_proj.weight.copy_(layer.down_proj.weight[:, start:end])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        h = torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(
            hidden_states
        )
        out = self.down_proj(h)
        dist.all_reduce(out)
        return out


class TPLlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
    ):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.enable_gqa = config.num_attention_heads != config.num_key_value_heads
        gqa_ratio = config.num_attention_heads // config.num_key_value_heads

        self.attention_dropout = config.attention_dropout

        # Process group
        self.world_size = dist.get_world_size()
        self.rank = int(os.environ["LOCAL_RANK"])

        # Split the heads for each GPU
        assert config.num_attention_heads % self.world_size == 0

        self.n_heads_per_gpu = config.num_attention_heads // self.world_size
        self.n_kv_heads_per_gpu = (
            config.num_attention_heads // (self.world_size * gqa_ratio)
            if self.enable_gqa
            else self.n_heads_per_gpu
        )

        self.head_size = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.head_size * self.n_heads_per_gpu,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.head_size * self.n_kv_heads_per_gpu,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.head_size * self.n_kv_heads_per_gpu,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.head_size * self.n_heads_per_gpu,
            config.hidden_size,
            bias=config.attention_bias,
        )

    @torch.no_grad()
    def init_from_layer(self, layer: LlamaAttention) -> None:
        local_h = self.n_heads_per_gpu * self.head_size
        local_kv_h = self.n_kv_heads_per_gpu * self.head_size
        start = self.rank * local_h
        end = (self.rank + 1) * local_h
        start_kv = self.rank * local_kv_h
        end_kv = (self.rank + 1) * local_kv_h

        self.q_proj.weight.copy_(layer.q_proj.weight[start:end])
        self.k_proj.weight.copy_(layer.k_proj.weight[start_kv:end_kv])
        self.v_proj.weight.copy_(layer.v_proj.weight[start_kv:end_kv])
        self.o_proj.weight.copy_(layer.o_proj.weight[:, start:end])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        """
        Args:
            * x: (batch_size, seq_len, dim)
        """
        B, T = hidden_states.shape[0], hidden_states.shape[1]

        q, k, v = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )  # (B, T, dim // n_ranks)

        q = rearrange(
            q,
            "b t (nhg h) -> b nhg t h",
            b=B,
            t=T,
            nhg=self.n_heads_per_gpu,
            h=self.head_size,
        )
        k = rearrange(
            k,
            "b t (nhg h) -> b nhg t h",
            b=B,
            t=T,
            nhg=self.n_kv_heads_per_gpu,
            h=self.head_size,
        )
        v = rearrange(
            v,
            "b t (nhg h) -> b nhg t h",
            b=B,
            t=T,
            nhg=self.n_kv_heads_per_gpu,
            h=self.head_size,
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            # attn_mask=attention_mask,
            is_causal=True,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            enable_gqa=self.enable_gqa,
        )  # (B, n_heads, T, h_dim)

        attn_output = attn_output.transpose(1, 2).reshape(
            B, T, -1
        )  # (B, T, dim // n_ranks)

        attn_output = self.o_proj(attn_output)  # (B, T, dim)
        dist.all_reduce(attn_output, op=dist.ReduceOp.SUM)
        # HF expects attn_weights as the second output, we could not care less
        return attn_output, None


class TPParallelLlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig, num_blocks: int):
        super().__init__()

        self.num_blocks = num_blocks
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        assert config.intermediate_size * num_blocks % self.world_size == 0
        assert config.intermediate_size % self.world_size == 0

        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size * num_blocks // self.world_size,
            bias=config.mlp_bias,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size * num_blocks // self.world_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.ModuleList(
            [
                nn.Linear(
                    config.intermediate_size // self.world_size,
                    config.hidden_size,
                    bias=config.mlp_bias,
                )
                for _ in range(num_blocks)
            ]
        )

    @torch.no_grad()
    def init_from_layers(self, layers: List[LlamaMLP]) -> None:
        assert len(layers) == self.num_blocks
        slice_size = self.intermediate_size // self.world_size
        start_idx = self.rank * slice_size
        end_idx = (self.rank + 1) * slice_size

        for i, block in enumerate(layers):
            # Output slicing indices for this GPU's portion
            local_start = i * slice_size
            local_end = (i + 1) * slice_size

            self.gate_proj.weight.data[local_start:local_end] = (
                block.gate_proj.weight.data[start_idx:end_idx]
            )
            self.up_proj.weight.data[local_start:local_end] = block.up_proj.weight.data[
                start_idx:end_idx
            ]

            # For l3, each GPU gets a horizontal slice
            self.down_proj[i].weight.data = block.down_proj.weight.data[
                :, start_idx:end_idx
            ]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        h = torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(
            hidden_states
        )
        h = h.reshape(B, T, self.num_blocks, self.intermediate_size // self.world_size)
        outs = torch.unbind(h, dim=2)

        outs = [self.down_proj[i](out_i) for i, out_i in enumerate(outs)]
        final_states = sum(outs)

        dist.all_reduce(final_states)
        return final_states


class TPParallelLlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_ids: Tuple[int],
        num_blocks=2,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.n_gpus = dist.get_world_size()
        self.rank = dist.get_rank()

        assert config.hidden_size % self.n_gpus == 0
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % self.n_gpus == 0
        assert self.n_gpus % num_blocks == 0

        assert config.hidden_size % config.num_attention_heads == 0

        self.hidden_size = config.hidden_size
        self.layer_ids = layer_ids
        self.head_size = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.attention_dropout = config.attention_dropout

        # Process group
        self.world_size = dist.get_world_size()
        self.rank = int(os.environ["LOCAL_RANK"])

        # Split the heads for each GPU
        assert config.num_attention_heads % self.world_size == 0

        self.n_heads_per_gpu = config.num_attention_heads * num_blocks // self.n_gpus
        self.enable_gqa = config.num_attention_heads != config.num_key_value_heads
        gqa_ratio = config.num_attention_heads // config.num_key_value_heads
        if self.enable_gqa:
            assert self.n_heads_per_gpu % gqa_ratio == 0
        self.n_kv_heads_per_gpu = (
            config.num_key_value_heads * num_blocks // self.world_size
            if self.enable_gqa
            else self.n_heads_per_gpu
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.head_size * self.n_heads_per_gpu,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.head_size * self.n_kv_heads_per_gpu,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.head_size * self.n_kv_heads_per_gpu,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.head_size * self.n_heads_per_gpu,
            config.hidden_size,
            bias=config.attention_bias,
        )

        gpus_per_block = dist.get_world_size() // len(layer_ids)
        self.b_id = dist.get_rank() // gpus_per_block

    @torch.no_grad()
    def init_from_layers(self, layers: List[LlamaAttention]) -> None:
        assert len(layers) == self.num_blocks

        local_h = self.n_heads_per_gpu * self.head_size
        local_kv_h = self.n_kv_heads_per_gpu * self.head_size
        gpus_per_block = self.n_gpus // self.num_blocks
        i = self.rank % gpus_per_block
        b_id = self.rank // gpus_per_block

        start = i * local_h
        end = (i + 1) * local_h

        start_kv = i * local_kv_h
        end_kv = (i + 1) * local_kv_h

        self.q_proj.weight.copy_(layers[b_id].q_proj.weight[start:end])
        self.k_proj.weight.copy_(layers[b_id].k_proj.weight[start_kv:end_kv])
        self.v_proj.weight.copy_(layers[b_id].v_proj.weight[start_kv:end_kv])
        self.o_proj.weight.copy_(layers[b_id].o_proj.weight[:, start:end])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        """
        Args:
            * x: (batch_size, seq_len, dim)
        """
        B, T = hidden_states.shape[0], hidden_states.shape[1]

        q, k, v = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )  # (B, T, dim // n_ranks)

        q = rearrange(
            q,
            "b t (nhg h) -> b nhg t h",
            b=B,
            t=T,
            nhg=self.n_heads_per_gpu,
            h=self.head_size,
        )
        k = rearrange(
            k,
            "b t (nhg h) -> b nhg t h",
            b=B,
            t=T,
            nhg=self.n_kv_heads_per_gpu,
            h=self.head_size,
        )
        v = rearrange(
            v,
            "b t (nhg h) -> b nhg t h",
            b=B,
            t=T,
            nhg=self.n_kv_heads_per_gpu,
            h=self.head_size,
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_ids[self.b_id], cache_kwargs)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            # attn_mask=attention_mask,
            is_causal=True,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            enable_gqa=self.enable_gqa,
        )  # (B, n_heads, T, h_dim)

        attn_output = attn_output.transpose(1, 2).reshape(
            B, T, -1
        )  # (B, T, dim // n_ranks)

        attn_output = self.o_proj(attn_output)  # (B, T, dim)

        # The sum is done over the blocks and over the TP for each block automatically!!
        dist.all_reduce(attn_output, op=dist.ReduceOp.SUM)
        # HF expects attn_weights as the second output, we could not care less
        return attn_output, None


class TPParallelLlamaDecoderLayer(nn.Module):
    def __init__(
        self, config: LlamaConfig, layers: Tuple[LlamaDecoderLayer, LlamaDecoderLayer]
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        layer_ids = [layer.self_attn.layer_idx for layer in layers]
        self.self_attn = TPParallelLlamaAttention(
            config, num_blocks=len(layers), layer_ids=layer_ids
        )
        self.self_attn.init_from_layers([layer.self_attn for layer in layers])
        self.mlp = TPParallelLlamaMLP(config, num_blocks=len(layers))
        self.mlp.init_from_layers([layer.mlp for layer in layers])

        # The pre-attention layernorm can be sent to the correct blocks!
        gpus_per_block = dist.get_world_size() // len(layers)
        b_id = dist.get_rank() // gpus_per_block
        self.input_layernorm = layers[b_id].input_layernorm

        # The post-attention layernorm needs to be reduced.
        # For now we just average TODO investigate
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # assert len(layers) == 2
        # reduced_ln_weight = slerp_with_magnitude(
        #     layers[0].post_attention_layernorm.weight.data,
        #     layers[1].post_attention_layernorm.weight.data,
        # )
        # self.post_attention_layernorm.weight.data = reduced_ln_weight
        self.post_attention_layernorm.weight.data = sum(
            layer.post_attention_layernorm.weight.data for layer in layers
        ) / len(layers)
        self.post_attention_layernorm.variance_epsilon = min(
            layer.post_attention_layernorm.variance_epsilon for layer in layers
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class TPLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size

        self.hidden_size = config.hidden_size

        self.self_attn = TPLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = TPLlamaMLP(config=config)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def init_from_llama_decoder_layer(self, other: LlamaDecoderLayer):
        self.self_attn.init_from_layer(other.self_attn)
        self.mlp.init_from_layer(other.mlp)

        self.input_layernorm = other.input_layernorm
        self.post_attention_layernorm = other.post_attention_layernorm
