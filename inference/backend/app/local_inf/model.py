# -*- coding: utf-8 -*-
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn


class Mamba3Config:
    def __init__(
        self,
        d_model=768,
        d_state=64,
        d_head=64,
        n_groups=1,
        mimo_rank=4,
        expand=4,
        num_layers=15,
        use_kmoe=True,
        kmoe_num_experts=8,
        kmoe_top_k=2,
        kmoe_r1=4,
        kmoe_r2=1024,
        kmoe_r3=256,
        ffn_expand=6,
        num_kv_heads=4,
        rms_norm_eps=1e-5,
        layer_scale_init=1e-2,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.expand = expand
        self.num_layers = num_layers
        self.d_inner = int(expand * d_model)
        self.n_heads = self.d_inner // d_head
        self.n_groups = n_groups
        self.mimo_rank = mimo_rank
        self.rms_norm_eps = rms_norm_eps

        self.use_kmoe = use_kmoe
        self.kmoe_num_experts = kmoe_num_experts
        self.kmoe_top_k = kmoe_top_k
        self.kmoe_r1 = kmoe_r1
        self.kmoe_r2 = kmoe_r2
        self.kmoe_r3 = kmoe_r3
        self.ffn_expand = ffn_expand

        self.num_kv_heads = num_kv_heads
        self.kv_groups = self.n_heads // num_kv_heads
        self.layer_scale_init = layer_scale_init


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-2):
        super().__init__()
        self.gamma = mx.array(np.ones(dim) * init_value, dtype=mx.float32)

    def __call__(self, x):
        return x * self.gamma


def fast_scaled_tanh(x, scale=10.0):
    return mx.tanh(x / scale) * scale


def silu(x):
    return x * mx.sigmoid(x)


def fast_silu_gating(gate, feat):
    return silu(gate) * feat


def apply_rope(x, angles):
    n_half = angles.shape[-1]
    x_reshaped = x.reshape(*x.shape[:-2], n_half, 2, x.shape[-1])
    x1, x2 = x_reshaped[..., 0, :], x_reshaped[..., 1, :]
    sin_a = mx.expand_dims(mx.sin(angles), -1)
    cos_a = mx.expand_dims(mx.cos(angles), -1)
    out = mx.stack([x1 * cos_a - x2 * sin_a, x2 * cos_a + x1 * sin_a], axis=-2)
    return out.reshape(x.shape)


class TuckerMoE(nn.Module):
    def __init__(self, dim_in, dim_out, num_experts=8, top_k=2, r1=4, r2=1024, r3=256):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.router = nn.Linear(dim_in, num_experts, bias=False)
        self.U_expert = mx.random.normal((num_experts, r1)) * 0.02
        self.U_in = mx.random.normal((dim_in, r3)) * 0.02
        self.U_out = mx.random.normal((r2, dim_out)) * 0.02
        self.core = mx.random.normal((r1, r3, r2)) * 0.02
        self.bias = mx.zeros((dim_out,))
        self.inner_norm = nn.RMSNorm(r3)

    def _get_G(self):
        if not hasattr(self, "_G_cache") or self._G_cache is None:
            self._G_cache = mx.einsum("er, rst -> est", self.U_expert, self.core)
            mx.eval(self._G_cache)
        return self._G_cache

    def __call__(self, x, temperature=0.5):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        bl = x_flat.shape[0]
        raw_logits = self.router(x_flat)
        capped = fast_scaled_tanh(raw_logits, 10.0)
        router_logits = capped / temperature

        if self.top_k < self.num_experts:
            top_k_indices = mx.argpartition(-router_logits, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        else:
            top_k_indices = mx.arange(self.num_experts).reshape(1, -1).broadcast_to(bl, self.num_experts)
        top_k_probs = mx.take_along_axis(router_logits, top_k_indices, axis=-1)
        top_k_probs = mx.softmax(top_k_probs, axis=-1)

        x_shared = self.inner_norm(mx.matmul(x_flat, self.U_in))
        g_experts = self._get_G()
        g_token = g_experts[top_k_indices]
        expert_outs = mx.einsum("br, bkrt -> bkt", x_shared, g_token)
        out_core = mx.sum(expert_outs * mx.expand_dims(top_k_probs, -1), axis=1)
        out = mx.matmul(out_core, self.U_out).reshape(*orig_shape[:-1], -1)
        return out + self.bias


class MixtralMoEFeedForward(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        d_ff = int(math.ceil(config.ffn_expand * config.d_model / 256) * 256)
        kw = dict(
            num_experts=config.kmoe_num_experts,
            top_k=config.kmoe_top_k,
            r1=config.kmoe_r1,
            r2=config.kmoe_r2,
            r3=config.kmoe_r3,
        )
        self.gate_proj = TuckerMoE(config.d_model, d_ff, **kw)
        self.up_proj = TuckerMoE(config.d_model, d_ff, **kw)
        self.down_proj = TuckerMoE(d_ff, config.d_model, **kw)

    def __call__(self, x):
        gate = self.gate_proj(x)
        feat = self.up_proj(x)
        return self.down_proj(fast_silu_gating(gate, feat))


class Mamba3Block(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        d_in, h, g, p, n, r = (
            config.d_model,
            config.n_heads,
            config.n_groups,
            config.d_head,
            config.d_state,
            config.mimo_rank,
        )
        self.ratio, self.dim_z, self.dim_x = h // g, h * p, h * p
        self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda = g * n * r, g * n * r, g, g, g

        total_in_dim = self.dim_z + self.dim_x + self.dim_B + self.dim_C + self.dim_dt + self.dim_A + self.dim_lambda
        self.in_proj = nn.Linear(d_in, total_in_dim, bias=True)

        if config.use_kmoe:
            kw = dict(
                num_experts=config.kmoe_num_experts,
                top_k=config.kmoe_top_k,
                r1=config.kmoe_r1,
                r2=config.kmoe_r2,
                r3=config.kmoe_r3,
            )
            self.x_up_proj = TuckerMoE(h * p, h * p * r, **kw)
            self.out_proj = TuckerMoE(d_in, d_in, **kw)
        else:
            self.x_up_proj = nn.Linear(p, p * r, bias=False)
            self.out_proj = nn.Linear(d_in, d_in, bias=False)

        self.y_down_proj = nn.Linear(p * r, p, bias=False)
        self.theta_log = mx.random.normal((g, n // 2))
        self.D = mx.ones((h,))
        self.norm_B = nn.RMSNorm(n * r, eps=config.rms_norm_eps)
        self.norm_C = nn.RMSNorm(n * r, eps=config.rms_norm_eps)
        self.bias_B = mx.zeros((g, n, r))
        self.bias_C = mx.zeros((g, n, r))
        self.mamba_dense_proj = nn.Linear(config.d_inner, d_in, bias=False)
        self.pre_gate_norm = nn.RMSNorm(h * p)
        self.norm_mamba = nn.RMSNorm(config.d_model)
        self.norm_out_proj = nn.RMSNorm(config.d_model)
        self.ls_mamba = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_out_proj = LayerScale(config.d_model, init_value=config.layer_scale_init)

        splits = [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda]
        import numpy as _np

        self._split_indices = _np.cumsum(splits)[:-1].tolist()
        self._theta_rep_cache = None
        self._D_rep_cache = None

    def __call__(self, x, cache=None):
        b_sz, l, _ = x.shape
        h, g, p, n, r, ratio = (
            self.config.n_heads,
            self.config.n_groups,
            self.config.d_head,
            self.config.d_state,
            self.config.mimo_rank,
            self.ratio,
        )

        residual_mamba = x
        u = self.norm_mamba(x)
        proj_out = self.in_proj(u)
        z, x_prime, b_param, c_param, dt, a_param, lambda_param = mx.split(proj_out, self._split_indices, axis=-1)
        x_prime = x_prime.reshape(b_sz, l, h, p)

        dt = mx.logaddexp(mx.array(0.0), dt)
        A = -mx.exp(a_param)

        def bg(t):
            return mx.repeat(t, ratio, axis=2)

        dt_b = bg(mx.expand_dims(dt, -1)).squeeze(-1)
        A_b = bg(mx.expand_dims(A, -1)).squeeze(-1)

        if self._theta_rep_cache is None:
            theta = mx.exp(self.theta_log)
            self._theta_rep_cache = mx.repeat(theta, ratio, axis=0)
            mx.eval(self._theta_rep_cache)
        theta_rep = self._theta_rep_cache

        current_angle_step = mx.einsum("blh, hn -> blhn", dt_b, theta_rep)
        if cache is not None:
            prev_h, prev_input, prev_angle_sum = cache
            angles = prev_angle_sum + mx.cumsum(current_angle_step, axis=1)
        else:
            angles = mx.cumsum(current_angle_step, axis=1)
        new_angle_sum = angles[:, -1:]

        b_reshaped = self.norm_B(b_param.reshape(b_sz, l, g, n * r)).reshape(b_sz, l, g, n, r)
        c_reshaped = self.norm_C(c_param.reshape(b_sz, l, g, n * r)).reshape(b_sz, l, g, n, r)

        b_rotated = apply_rope(bg(b_reshaped) + self.bias_B, angles)
        c_rotated = apply_rope(bg(c_reshaped) + self.bias_C, angles)

        if self.config.use_kmoe:
            x_up = self.x_up_proj(x_prime.reshape(b_sz, l, -1))
            x_ssm = x_up.reshape(b_sz, l, h, p, r)
        else:
            x_ssm = self.x_up_proj(x_prime).reshape(b_sz, l, h, p, r)

        input_signal = mx.einsum("blhnr, blhpr -> blhnp", b_rotated, x_ssm)
        lv = mx.sigmoid(bg(mx.expand_dims(lambda_param, -1)).squeeze(-1)).reshape(b_sz, l, h, 1, 1)
        dv = dt_b.reshape(b_sz, l, h, 1, 1)
        av = mx.exp(dt_b * A_b).reshape(b_sz, l, h, 1, 1)

        if cache is not None:
            ip = prev_input
        else:
            ip = mx.concatenate([mx.zeros_like(input_signal[:, :1]), input_signal[:, :-1]], axis=1)
        u_ssm = lv * dv * input_signal + (1 - lv) * dv * av * ip

        if cache is not None:
            h_s = prev_h
        else:
            h_s = mx.zeros((b_sz, h, n, p), dtype=x.dtype)

        if l == 1:
            h_s = h_s * av[:, 0] + u_ssm[:, 0]
            y_stack = mx.einsum("bhnp, bhnr -> bhpr", h_s, c_rotated[:, 0])[:, None, ...]
        else:
            S = mx.cumsum(dt_b * A_b, axis=1)
            S_trans = S.transpose(0, 2, 1)
            S_t = mx.expand_dims(S_trans, 3)
            S_i = mx.expand_dims(S_trans, 2)
            M = mx.exp(S_t - S_i)
            indices = mx.arange(l)
            mask = (mx.expand_dims(indices, 0) > mx.expand_dims(indices, 1)).reshape(1, 1, l, l)
            M = mx.where(mask, 0.0, M)
            u_flat = u_ssm.transpose(0, 2, 1, 3, 4).reshape(b_sz, h, l, n * p)
            h_seq = mx.matmul(M, u_flat).reshape(b_sz, h, l, n, p).transpose(0, 2, 1, 3, 4)
            decay = mx.exp(S).reshape(b_sz, l, h, 1, 1)
            h_seq = h_seq + decay * mx.expand_dims(h_s, 1)
            h_s = h_seq[:, -1, ...]
            y_stack = mx.einsum("blhnp, blhnr -> blhpr", h_seq, c_rotated)

        new_cache = (h_s, input_signal[:, -1:], new_angle_sum)
        y = self.y_down_proj(y_stack.reshape(b_sz, l, h, p * r)).reshape(b_sz, l, h * p)
        if self._D_rep_cache is None:
            self._D_rep_cache = mx.repeat(self.D, p, axis=0)
            mx.eval(self._D_rep_cache)
        D_rep = self._D_rep_cache
        y = y + x_prime.reshape(b_sz, l, h * p) * D_rep

        mamba_out = self.mamba_dense_proj(self.pre_gate_norm(y) * silu(z))
        mid_x = residual_mamba + self.ls_mamba(mamba_out)
        normed_mid = self.norm_out_proj(mid_x)
        proj_out = self.out_proj(normed_mid)
        return mid_x + self.ls_out_proj(proj_out), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.head_dim = 64
        self.num_heads = config.d_model // 64
        self.num_kv_heads = config.num_kv_heads
        self.kv_groups = self.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.num_heads * 64, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.num_kv_heads * 64, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.num_kv_heads * 64, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=True)

        self.norm_attn = nn.RMSNorm(config.d_model)
        self.use_kmoe = config.use_kmoe
        if config.use_kmoe:
            self.ffn = MixtralMoEFeedForward(config)
        else:
            d_ff = int(math.ceil(8 * config.d_model / 3 / 256) * 256)
            self.ffn_gate = nn.Linear(config.d_model, d_ff, bias=False)
            self.ffn_up = nn.Linear(config.d_model, d_ff, bias=False)
            self.ffn_down = nn.Linear(d_ff, config.d_model, bias=False)

        self.norm_ffn = nn.RMSNorm(config.d_model)
        self.ls_attn = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_ffn = LayerScale(config.d_model, init_value=config.layer_scale_init)

    def __call__(self, x, cache=None, seq_pos=None):
        B, L, D = x.shape
        residual = x
        nx = self.norm_attn(x)

        q = self.q_proj(nx).reshape(B, L, self.num_heads, 64).transpose(0, 2, 1, 3)
        k = self.k_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
        v = self.v_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)

        k = mx.repeat(k, self.kv_groups, axis=1)
        v = mx.repeat(v, self.kv_groups, axis=1)

        if cache is not None:
            k_cache, v_cache = cache
            if seq_pos is not None:
                indices = mx.arange(L) + seq_pos
                k_cache[:, :, indices, :] = k
                v_cache[:, :, indices, :] = v
                k = k_cache
                v = v_cache
            else:
                k = mx.concatenate([k_cache, k], axis=2)
                v = mx.concatenate([v_cache, v], axis=2)
        new_cache = (k, v) if seq_pos is None else (k_cache, v_cache)

        # Keep KV cache storage dtype independent from model dtype.
        # For decode L=1 this casts only query/output vectors, not full cache.
        if q.dtype != k.dtype:
            q = q.astype(k.dtype)

        if seq_pos is not None:
            max_L = k.shape[2]
            mask = mx.arange(max_L).reshape(1, 1, 1, max_L) > (seq_pos + L - 1)
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(64), mask=mask)
        else:
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(64))
        attn_out = attn.transpose(0, 2, 1, 3).reshape(B, L, D)
        if attn_out.dtype != residual.dtype:
            attn_out = attn_out.astype(residual.dtype)
        x = residual + self.ls_attn(self.o_proj(attn_out))
        residual = x
        h = self.norm_ffn(x)
        if self.use_kmoe:
            ffn_out = self.ffn(h)
        else:
            ffn_out = self.ffn_down(fast_silu_gating(self.ffn_gate(h), self.ffn_up(h)))
        return residual + self.ls_ffn(ffn_out), new_cache


class TrueHybridMamba(nn.Module):
    def __init__(self, config: Mamba3Config, mamba_ratio=4):
        super().__init__()
        self.layers = []
        for _ in range(config.num_layers):
            for _ in range(mamba_ratio):
                blk = Mamba3Block(config)
                blk.l_type = "mamba"
                self.layers.append(blk)
            blk = TransformerBlock(config)
            blk.l_type = "transformer"
            self.layers.append(blk)

    def __call__(self, x, caches=None, seq_pos=None):
        if caches is None:
            caches = [None] * len(self.layers)
        new_caches = []
        L = x.shape[1]
        for layer, cache in zip(self.layers, caches):
            if getattr(layer, "l_type", None) == "transformer":
                if L == 1 and cache is not None and seq_pos is not None and getattr(layer, "_compiled_decode", None) is not None:
                    k_cache, v_cache = cache
                    x, nk, nv = layer._compiled_decode(x, k_cache, v_cache, mx.array(seq_pos, dtype=mx.int32))
                    new_cache = (nk, nv)
                else:
                    x, new_cache = layer(x, cache=cache, seq_pos=seq_pos)
            elif L == 1 and cache is not None and getattr(layer, "_compiled_decode", None) is not None:
                h_s, prev_in, prev_ang = cache
                x, nh, npi, nas = layer._compiled_decode(x, h_s, prev_in, prev_ang)
                new_cache = (nh, npi, nas)
            else:
                x, new_cache = layer(x, cache=cache)
            new_caches.append(new_cache)
        return x, new_caches


class Mamba3LanguageModel(nn.Module):
    def __init__(self, config: Mamba3Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.backbone = TrueHybridMamba(config)
        self.norm = nn.RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)

    def __call__(self, input_ids, caches=None, seq_pos=None):
        self.head.weight = self.embed.weight
        x = self.embed(input_ids)
        hidden, new_caches = self.backbone(x, caches, seq_pos=seq_pos)
        hidden = self.norm(hidden)
        logits = fast_scaled_tanh(self.head(hidden / math.sqrt(self.config.d_model)), 30.0)
        return logits, new_caches
