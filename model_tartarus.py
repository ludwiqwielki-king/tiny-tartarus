#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_tartarus.py — Architektura Transformera (RoPE + RMSNorm + SwiGLU + Tied Embeddings)
dla projektu „Tartarus: The Fall of the Ancients”.

Konwencje matematyczne zaprojektowane w ścisłej zgodności 1:1 z silnikiem TypeScript (Etap 4/5):
    - RMSNorm (eps=1e-5): wariancja i rsqrt w float32, parametr skali `g`, pre-norm + rms_final.
    - RoPE (theta=10000.0): konwencja half-split (x1 = x[..., :hd//2], x2 = x[..., hd//2:]).
    - Causal Self-Attention: wq, wk, wv, wo z bias=False; w treningu SDPA is_causal=True.
    - SwiGLU FFN: w_down(silu(w_gate(x)) * w_up(x)), wszystkie warstwy z bias=False.
    - Tied embeddings: logits = F.linear(rms_final(x), tok_emb.weight).
    - Kanoniczne nazwy state_dict pod eksporter TNY3.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURACJA MODELU I PROFILE (S / M)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TartarusConfig:
    vocab_size: int = 2048
    ctx_len: int = 256
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    d_ff: int = 512
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    tied_embeddings: bool = True
    profile: str = "S"

    def __post_init__(self) -> None:
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size musi być > 0, otrzymano {self.vocab_size}")
        if self.ctx_len <= 0:
            raise ValueError(f"ctx_len musi być > 0, otrzymano {self.ctx_len}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers musi być > 0, otrzymano {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads musi być > 0, otrzymano {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"Wymagane d_model % n_heads == 0 ({self.d_model} % {self.n_heads})")
        hd = self.d_model // self.n_heads
        if hd % 2 != 0:
            raise ValueError(f"Wymagane head_dim % 2 == 0 dla RoPE half-split (head_dim={hd})")
        if self.d_model % 4 != 0:
            raise ValueError(f"Wymagane d_model % 4 == 0 pod alignment TNY3 (d_model={self.d_model})")
        if self.d_ff % 4 != 0:
            raise ValueError(f"Wymagane d_ff % 4 == 0 pod alignment TNY3 (d_ff={self.d_ff})")
        if self.rms_eps <= 0:
            raise ValueError(f"rms_eps musi być > 0, otrzymano {self.rms_eps}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta musi być > 0, otrzymano {self.rope_theta}")
        if not self.tied_embeddings:
            raise ValueError("Architektura Tartarus wymaga tied_embeddings=True")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TartarusConfig":
        return cls(**d)

    @classmethod
    def from_profile(cls, profile: str, **overrides: Any) -> "TartarusConfig":
        key = profile.upper()
        if key == "S":
            base: Dict[str, Any] = dict(
                vocab_size=2048,
                ctx_len=256,
                d_model=192,
                n_layers=4,
                n_heads=6,
                d_ff=512,
                rms_eps=1e-5,
                rope_theta=10000.0,
                tied_embeddings=True,
                profile="S",
            )
        elif key == "M":
            base = dict(
                vocab_size=2048,
                ctx_len=256,
                d_model=256,
                n_layers=6,
                n_heads=8,
                d_ff=640,
                rms_eps=1e-5,
                rope_theta=10000.0,
                tied_embeddings=True,
                profile="M",
            )
        else:
            raise ValueError(f"Nieznany profil {profile!r} (obsługiwane: 'S', 'M')")

        for k, v in overrides.items():
            if v is not None:
                base[k] = v
        return cls(**base)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKI SKŁADOWE (RMSNorm, RoPE half-split, Causal Attention, SwiGLU)
# ═══════════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """RMSNorm z obliczeniami wariancji w float32 i parametrem skali `g`."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = (x_fp32 * inv_rms).to(dtype=x.dtype)
        return normed * self.g.to(dtype=x.dtype)


def precompute_rope_tables(
    ctx_len: int,
    head_dim: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prekomputuje tablice cos i sin o kształcie [ctx_len, head_dim // 2] w float32
    zgodnie z konwencją half-split.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim musi być parzyste, otrzymano {head_dim}")
    half = head_dim // 2
    pair_idx = torch.arange(0, head_dim, 2, dtype=torch.float32)  # [half]
    inv_freq = 1.0 / (theta ** (pair_idx / float(head_dim)))       # [half]
    positions = torch.arange(ctx_len, dtype=torch.float32)         # [ctx_len]
    freqs = torch.outer(positions, inv_freq)                       # [ctx_len, half]
    assert freqs.shape == (ctx_len, half)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope_half_split(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Obraca tensor [B, n_heads, T, head_dim] przy użyciu tablic cos, sin [T, head_dim // 2].
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)  # [1, 1, T, half]
    s = sin.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)  # [1, 1, T, half]
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return torch.cat((out1, out2), dim=-1)


class KVCache:
    """Prealokowany bufor KV-Cache [n_layers, B, n_heads, ctx_len, head_dim] dla szybkiej inferencji."""

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        ctx_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.k = torch.zeros(
            (n_layers, batch_size, n_heads, ctx_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v = torch.zeros(
            (n_layers, batch_size, n_heads, ctx_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.seq_len = 0

    def reset(self) -> None:
        self.k.zero_()
        self.v.zero_()
        self.seq_len = 0


class TartarusBlock(nn.Module):
    """
    Blok Transformera wystawiający kanoniczne nazwy wag pod eksporter TNY3:
        blocks.{i}.rms_att.g
        blocks.{i}.wq.weight, wk.weight, wv.weight, wo.weight
        blocks.{i}.rms_ffn.g
        blocks.{i}.w_gate.weight, w_up.weight, w_down.weight
    """

    def __init__(self, cfg: TartarusConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model

        self.rms_att = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.rms_ffn = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def _attention(
        self,
        h: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        layer_idx: int,
        kv_cache: Optional[KVCache],
        start_pos: int,
    ) -> torch.Tensor:
        bsz, seq_len, _ = h.shape
        q = self.wq(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        cos_slice = rope_cos[start_pos : start_pos + seq_len]
        sin_slice = rope_sin[start_pos : start_pos + seq_len]
        q = apply_rope_half_split(q, cos_slice, sin_slice)
        k = apply_rope_half_split(k, cos_slice, sin_slice)

        if kv_cache is None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            end_pos = start_pos + seq_len
            kv_cache.k[layer_idx, :bsz, :, start_pos:end_pos, :] = k
            kv_cache.v[layer_idx, :bsz, :, start_pos:end_pos, :] = v
            k_all = kv_cache.k[layer_idx, :bsz, :, :end_pos, :]
            v_all = kv_cache.v[layer_idx, :bsz, :, :end_pos, :]

            if seq_len == 1:
                y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=None, dropout_p=0.0, is_causal=False)
            elif start_pos == 0:
                y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=None, dropout_p=0.0, is_causal=True)
            else:
                q_pos = torch.arange(start_pos, end_pos, device=h.device).unsqueeze(1)
                k_pos = torch.arange(0, end_pos, device=h.device).unsqueeze(0)
                attn_mask = k_pos <= q_pos  # bool [seq_len, end_pos]
                y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.wo(y)

    def _ffn(self, h: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        layer_idx: int = 0,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self._attention(
            self.rms_att(x),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            layer_idx=layer_idx,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        x = x + self._ffn(self.rms_ffn(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# GŁÓWNY MODEL TARTARUS
# ═══════════════════════════════════════════════════════════════════════════════


class TartarusModel(nn.Module):
    """
    Dekoderowy Transformer RoPE + RMSNorm + SwiGLU z wiązanymi wagami osadzeń (tied_embeddings).
    Nazwy parametrów w state_dict zachowują kanoniczny schemat pod eksporter TNY3:
        tok_emb.weight
        blocks.{i}.rms_att.g
        blocks.{i}.wq.weight / wk / wv / wo
        blocks.{i}.rms_ffn.g
        blocks.{i}.w_gate.weight / w_up / w_down
        rms_final.g
    """

    def __init__(self, cfg: TartarusConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TartarusBlock(cfg) for _ in range(cfg.n_layers)])
        self.rms_final = RMSNorm(cfg.d_model, eps=cfg.rms_eps)

        rope_cos, rope_sin = precompute_rope_tables(cfg.ctx_len, cfg.head_dim, theta=cfg.rope_theta)
        # persistent=False -> tablice RoPE nie zaśmiecają state_dict()
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std_base = 0.02
        std_res = 0.02 / math.sqrt(2.0 * self.cfg.n_layers)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std_base)
        for blk in self.blocks:
            nn.init.ones_(blk.rms_att.g)
            nn.init.ones_(blk.rms_ffn.g)
            for proj in (blk.wq, blk.wk, blk.wv, blk.w_gate, blk.w_up):
                nn.init.normal_(proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(blk.wo.weight, mean=0.0, std=std_res)
            nn.init.normal_(blk.w_down.weight, mean=0.0, std=std_res)
        nn.init.ones_(self.rms_final.g)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        _, seq_len = idx.shape
        if start_pos + seq_len > self.cfg.ctx_len:
            raise ValueError(
                f"Długość sekwencji ({start_pos} + {seq_len}) przekracza ctx_len={self.cfg.ctx_len}"
            )

        x = self.tok_emb(idx)
        for layer_idx, blk in enumerate(self.blocks):
            x = blk(
                x,
                rope_cos=self.rope_cos,
                rope_sin=self.rope_sin,
                layer_idx=layer_idx,
                kv_cache=kv_cache,
                start_pos=start_pos,
            )
        x = self.rms_final(x)
        # Tied embeddings: używamy tok_emb.weight (w tym samym dtype co x przy AMP)
        logits = F.linear(x, self.tok_emb.weight.to(dtype=x.dtype))
        return logits

    def init_kv_cache(self, batch_size: int = 1, dtype: Optional[torch.dtype] = None) -> KVCache:
        dev = self.tok_emb.weight.device
        dt = dtype if dtype is not None else self.tok_emb.weight.dtype
        return KVCache(
            n_layers=self.cfg.n_layers,
            batch_size=batch_size,
            n_heads=self.cfg.n_heads,
            ctx_len=self.cfg.ctx_len,
            head_dim=self.cfg.head_dim,
            device=dev,
            dtype=dt,
        )

    @torch.inference_mode()
    def forward_step(
        self,
        token_ids: torch.Tensor,
        start_pos: int,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """
        Referencyjny krok z KV-Cache: przyjmuje `token_ids` o kształcie [B, T_new]
        od pozycji `start_pos` i zwraca logity [B, T_new, vocab_size].
        """
        logits = self.forward(token_ids, kv_cache=kv_cache, start_pos=start_pos)
        kv_cache.seq_len = start_pos + token_ids.shape[1]
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Sequence[int] | torch.Tensor,
        max_new_tokens: int = 128,
        eos_token_id: int = 1,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> List[int]:
        """
        Autoregresyjna generacja zachłanna (temperature=0) lub próbkująca z KV-Cache.
        Zwraca wyłącznie wygenerowane nowe tokeny (włącznie z <|eos|>, jeśli wystąpił).
        """
        was_training = self.training
        self.eval()
        try:
            dev = self.tok_emb.weight.device
            if isinstance(prompt_ids, torch.Tensor):
                prompt_t = prompt_ids.to(device=dev, dtype=torch.long)
                if prompt_t.ndim == 1:
                    prompt_t = prompt_t.unsqueeze(0)
            else:
                prompt_t = torch.tensor([list(prompt_ids)], device=dev, dtype=torch.long)

            if prompt_t.shape[0] != 1:
                raise ValueError("generate() obsługuje batch_size=1")
            prompt_len = int(prompt_t.shape[1])
            if prompt_len == 0 or prompt_len >= self.cfg.ctx_len:
                return []

            cache = self.init_kv_cache(batch_size=1, dtype=self.tok_emb.weight.dtype)
            logits = self.forward_step(prompt_t, start_pos=0, kv_cache=cache)
            next_logits = logits[:, -1, :]  # [1, V]

            generated: List[int] = []
            max_steps = min(max_new_tokens, self.cfg.ctx_len - prompt_len)

            for step in range(max_steps):
                if temperature <= 0.0:
                    next_id = int(torch.argmax(next_logits, dim=-1).item())
                else:
                    scaled = next_logits.float() / temperature
                    if top_k is not None and top_k > 0:
                        k_val = min(top_k, scaled.shape[-1])
                        top_vals, _ = torch.topk(scaled, k_val, dim=-1)
                        scaled = scaled.masked_fill(scaled < top_vals[:, [-1]], float("-inf"))
                    probs = F.softmax(scaled, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

                generated.append(next_id)
                if next_id == eos_token_id:
                    break

                cur_pos = prompt_len + step
                if cur_pos + 1 >= self.cfg.ctx_len:
                    break
                step_in = torch.tensor([[next_id]], device=dev, dtype=torch.long)
                step_logits = self.forward_step(step_in, start_pos=cur_pos, kv_cache=cache)
                next_logits = step_logits[:, -1, :]

            return generated
        finally:
            if was_training:
                self.train()


def expected_state_dict_keys(n_layers: int) -> List[str]:
    keys = ["tok_emb.weight"]
    for i in range(n_layers):
        keys.extend(
            [
                f"blocks.{i}.rms_att.g",
                f"blocks.{i}.wq.weight",
                f"blocks.{i}.wk.weight",
                f"blocks.{i}.wv.weight",
                f"blocks.{i}.wo.weight",
                f"blocks.{i}.rms_ffn.g",
                f"blocks.{i}.w_gate.weight",
                f"blocks.{i}.w_up.weight",
                f"blocks.{i}.w_down.weight",
            ]
        )
    keys.append("rms_final.g")
    return keys


def self_test() -> None:
    """Weryfikacja wymiarów, nazw state_dict oraz równoważności KV-Cache == Full Forward."""
    torch.manual_seed(42)

    for prof, approx in (("S", 2_164_416), ("M", 5_049_600)):
        cfg = TartarusConfig.from_profile(prof)
        m = TartarusModel(cfg).eval()
        n_params = m.num_parameters()
        assert n_params == approx, f"Profil {prof}: {n_params} != {approx}"
        sd_keys = list(m.state_dict().keys())
        exp_keys = expected_state_dict_keys(cfg.n_layers)
        assert sd_keys == exp_keys, f"Niezgodność kluczy state_dict!\ngot={sd_keys}\nexp={exp_keys}"
        print(f"[PASS] Profil {prof}: params={n_params:,} ({n_params/1e6:.2f}M), state_dict={len(sd_keys)} tensorów")

    # Test asercji wymiarów
    for bad_kwargs in (
        dict(d_model=190, n_heads=6),    # d_model % n_heads != 0
        dict(d_model=198, n_heads=6),    # head_dim=33 nieparzyste
        dict(d_model=194, n_heads=2),    # d_model % 4 != 0
        dict(d_ff=510),                  # d_ff % 4 != 0
    ):
        try:
            TartarusConfig(**bad_kwargs)
            raise AssertionError(f"Oczekiwano ValueError dla {bad_kwargs}")
        except ValueError:
            pass
    print("[PASS] Asercje wymiarów konstruktora działają poprawnie")

    # Test zgodności KV-Cache == Full Forward
    cfg = TartarusConfig.from_profile("S")
    model = TartarusModel(cfg).eval()
    seq = torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.long)

    with torch.inference_mode():
        full_logits = model(seq)  # [1, 32, V]

        # 1) Krok po kroku od t=0..31
        cache_step = model.init_kv_cache(batch_size=1)
        step_logits_list = []
        for t in range(seq.shape[1]):
            out_t = model.forward_step(seq[:, t : t + 1], start_pos=t, kv_cache=cache_step)
            step_logits_list.append(out_t)
        step_logits = torch.cat(step_logits_list, dim=1)
        max_diff_step = float((full_logits - step_logits).abs().max().item())
        assert max_diff_step < 1e-5, f"KV-cache step mismatch: {max_diff_step}"

        # 2) Prefill [:20] + krok po kroku [20..31]
        cache_prefill = model.init_kv_cache(batch_size=1)
        prefill_logits = model.forward_step(seq[:, :20], start_pos=0, kv_cache=cache_prefill)
        tail_list = [prefill_logits]
        for t in range(20, 32):
            tail_list.append(model.forward_step(seq[:, t : t + 1], start_pos=t, kv_cache=cache_prefill))
        mixed_logits = torch.cat(tail_list, dim=1)
        max_diff_mixed = float((full_logits - mixed_logits).abs().max().item())
        assert max_diff_mixed < 1e-5, f"KV-cache prefill+step mismatch: {max_diff_mixed}"

    print(f"[PASS] KV-Cache == Full Forward (max_diff_step={max_diff_step:.2e}, max_diff_prefill={max_diff_mixed:.2e})")


if __name__ == "__main__":
    self_test()
