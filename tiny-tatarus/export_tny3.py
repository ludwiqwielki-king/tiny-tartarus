#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_tny3.py — Eksporter wag modelu Tartarus (INT8 per-row) oraz tokenizera do
jednego pliku binarnego `model.tny3` (Little-Endian, stały nagłówek 64 B, wyrównanie 4 B,
CRC-32) wraz z pełną weryfikacją round-trip i generacją `golden_test.json`.

Struktura binarna TNY3 (format_version = 1):
    [0x00 .. 0x40)  64-bajtowy nagłówek (<4s2H6I2f4I2I):
        0x00 (4B) : magic = "TNY3"
        0x04 (2B) : format_version = 1 (uint16)
        0x06 (2B) : flags (uint16):
                      bit 0 (1): tied_embeddings
                      bit 1 (2): rope_interleaved (0 = half-split, 1 = interleaved)
                      bit 2 (4): swiglu
        0x08 (4B) : vocab_size (uint32)
        0x0C (4B) : ctx_len (uint32)
        0x10 (4B) : d_model (uint32)
        0x14 (4B) : n_layers (uint32)
        0x18 (4B) : n_heads (uint32)
        0x1C (4B) : d_ff (uint32)
        0x20 (4B) : rope_theta (float32)
        0x24 (4B) : norm_eps (float32)
        0x28 (4B) : tok_offset (uint32, zawsze 64)
        0x2C (4B) : tok_len (uint32, dokładna długość UTF-8 JSON sekcji tokenizera)
        0x30 (4B) : weights_offset (uint32, wyrównany do 4 B przez padding 0x00 po tok_len)
        0x34 (4B) : weights_len (uint32, dokładna długość sekcji wag w bajtach)
        0x38 (4B) : payload_crc32 (uint32, zlib.crc32 over raw[64:])
        0x3C (4B) : reserved = 0 (uint32)

    [tok_offset .. tok_offset + tok_len):
        UTF-8 JSON tokenizera (format, vocab_size, special_tokens, byte_offset,
        pretokenize_pattern, pretokenize_flags, merges, vocab) + padding zerami do 4 B.

    [weights_offset .. weights_offset + weights_len):
        1. tok_emb.weight            : INT8 per-row [V, D]
        2. Dla i = 0 .. n_layers - 1 :
             blocks.{i}.rms_att.g    : float32[D]
             blocks.{i}.wq.weight    : INT8 per-row [D, D]
             blocks.{i}.wk.weight    : INT8 per-row [D, D]
             blocks.{i}.wv.weight    : INT8 per-row [D, D]
             blocks.{i}.wo.weight    : INT8 per-row [D, D]
             blocks.{i}.rms_ffn.g    : float32[D]
             blocks.{i}.w_gate.weight: INT8 per-row [H_ff, D]
             blocks.{i}.w_up.weight  : INT8 per-row [H_ff, D]
             blocks.{i}.w_down.weight: INT8 per-row [D, H_ff]
        3. rms_final.g               : float32[D]
        4. (opcjonalnie lm_head.weight, jeśli tied_embeddings == 0)

    Format macierzy INT8 per-row [rows, cols]:
        Dla każdego wiersza r = 0 .. rows - 1:
            float32 scale (4B LE) + cols * int8 (cols B)
        Przy D % 4 == 0 i H_ff % 4 == 0 każdy wiersz ma długość (4 + cols) % 4 == 0.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import struct
import sys
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from model_tartarus import TartarusConfig, TartarusModel, expected_state_dict_keys
from tokenizer_tartarus import TartarusTokenizer, sha256_file
from train_tartarus import quantized_copy


# ═══════════════════════════════════════════════════════════════════════════════
# STAŁE FORMATU BINARNEGO TNY3
# ═══════════════════════════════════════════════════════════════════════════════

TNY3_MAGIC = b"TNY3"
TNY3_VERSION = 1
HEADER_STRUCT = struct.Struct("<4s2H6I2f4I2I")
assert HEADER_STRUCT.size == 64

FLAG_TIED_EMBEDDINGS = 1 << 0
FLAG_ROPE_INTERLEAVED = 1 << 1  # 0 = half-split, 1 = interleaved
FLAG_SWIGLU = 1 << 2


def align4(n: int) -> int:
    return (n + 3) & ~3


# ═══════════════════════════════════════════════════════════════════════════════
# KWANTYZACJA I PAKOWANIE / ROZPAKOWYWANIE TENSORÓW
# ═══════════════════════════════════════════════════════════════════════════════


def quantize_per_row_int8(tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symetryczna kwantyzacja INT8 per-row [-127, 127], 1:1 zgodna z `quantized_copy`:
        scale[r] = max(max(|W[r]|) / 127.0, 1e-8)  (float32)
        q[r, c]  = clamp(round(W[r, c] / scale[r]), -127, 127)  (int8)
    """
    if tensor.ndim != 2:
        raise ValueError(f"Oczekiwano tensora 2D, otrzymano shape={tuple(tensor.shape)}")
    w = tensor.detach().cpu().float()
    max_abs = w.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / 127.0).clamp_min(1e-8)
    q = torch.clamp(torch.round(w / scale), -127.0, 127.0).to(torch.int8)

    scale_np = scale.squeeze(-1).numpy().astype("<f4", copy=False)
    q_np = q.numpy().astype("i1", copy=False)
    return scale_np, q_np


def pack_matrix_int8(tensor: torch.Tensor) -> bytes:
    rows, cols = tensor.shape
    if cols % 4 != 0:
        raise ValueError(f"Liczba kolumn ({cols}) musi być podzielna przez 4 dla wyrównania TNY3")
    scale_np, q_np = quantize_per_row_int8(tensor)
    row_dtype = np.dtype([("scale", "<f4"), ("data", "i1", (cols,))])
    packed = np.empty(rows, dtype=row_dtype)
    packed["scale"] = scale_np
    packed["data"] = q_np
    raw = packed.tobytes()
    assert len(raw) == rows * (4 + cols)
    assert len(raw) % 4 == 0
    return raw


def unpack_matrix_int8(buf: memoryview, offset: int, rows: int, cols: int) -> Tuple[torch.Tensor, int]:
    if cols % 4 != 0:
        raise ValueError(f"cols={cols} nie jest podzielne przez 4")
    if offset % 4 != 0:
        raise ValueError(f"Niewyrównany offset macierzy: {offset}")
    row_stride = 4 + cols
    byte_len = rows * row_stride
    end = offset + byte_len
    if end > len(buf):
        raise ValueError(f"Poza zakresem bufora TNY3 przy odczycie macierzy [{rows}, {cols}]: {end} > {len(buf)}")

    row_dtype = np.dtype([("scale", "<f4"), ("data", "i1", (cols,))])
    arr = np.frombuffer(buf[offset:end], dtype=row_dtype, count=rows)
    scale = torch.from_numpy(arr["scale"].astype(np.float32, copy=True)).unsqueeze(1)  # [rows, 1]
    q = torch.from_numpy(arr["data"].astype(np.float32, copy=True))                    # [rows, cols]
    return q * scale, end


def pack_vector_f32(tensor: torch.Tensor) -> bytes:
    if tensor.ndim != 1:
        raise ValueError(f"Oczekiwano wektora 1D, otrzymano shape={tuple(tensor.shape)}")
    raw = tensor.detach().cpu().float().numpy().astype("<f4", copy=False).tobytes()
    assert len(raw) == tensor.numel() * 4
    return raw


def unpack_vector_f32(buf: memoryview, offset: int, length: int) -> Tuple[torch.Tensor, int]:
    if offset % 4 != 0:
        raise ValueError(f"Niewyrównany offset wektora: {offset}")
    end = offset + length * 4
    if end > len(buf):
        raise ValueError(f"Poza zakresem bufora TNY3 przy odczycie wektora [{length}]: {end} > {len(buf)}")
    arr = np.frombuffer(buf[offset:end], dtype=np.dtype("<f4"), count=length).astype(np.float32, copy=True)
    return torch.from_numpy(arr), end


def canonical_weight_specs(cfg: TartarusConfig) -> List[Tuple[str, Tuple[int, ...], str]]:
    """Zwraca jawną, niezmienną kolejność tensorów w sekcji wag TNY3."""
    specs: List[Tuple[str, Tuple[int, ...], str]] = [
        ("tok_emb.weight", (cfg.vocab_size, cfg.d_model), "int8"),
    ]
    for i in range(cfg.n_layers):
        specs.extend(
            [
                (f"blocks.{i}.rms_att.g", (cfg.d_model,), "f32"),
                (f"blocks.{i}.wq.weight", (cfg.d_model, cfg.d_model), "int8"),
                (f"blocks.{i}.wk.weight", (cfg.d_model, cfg.d_model), "int8"),
                (f"blocks.{i}.wv.weight", (cfg.d_model, cfg.d_model), "int8"),
                (f"blocks.{i}.wo.weight", (cfg.d_model, cfg.d_model), "int8"),
                (f"blocks.{i}.rms_ffn.g", (cfg.d_model,), "f32"),
                (f"blocks.{i}.w_gate.weight", (cfg.d_ff, cfg.d_model), "int8"),
                (f"blocks.{i}.w_up.weight", (cfg.d_ff, cfg.d_model), "int8"),
                (f"blocks.{i}.w_down.weight", (cfg.d_model, cfg.d_ff), "int8"),
            ]
        )
    specs.append(("rms_final.g", (cfg.d_model,), "f32"))
    if not cfg.tied_embeddings:
        specs.append(("lm_head.weight", (cfg.vocab_size, cfg.d_model), "int8"))
    return specs


# ═══════════════════════════════════════════════════════════════════════════════
# EKSPORT ORAZ ODCZYT MODEL.TNY3
# ═══════════════════════════════════════════════════════════════════════════════


def export_model_tny3(
    model: TartarusModel,
    tokenizer: TartarusTokenizer,
    out_path: str,
    include_vocab_table: bool = True,
) -> Dict[str, Any]:
    cfg = model.cfg
    cfg.validate()
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError(
            f"Niezgodność vocab_size między modelem ({cfg.vocab_size}) a tokenizerem ({tokenizer.vocab_size})"
        )

    model_cpu = model.detach() if hasattr(model, "detach") else model
    sd = {k: v.detach().cpu().float() for k, v in model_cpu.state_dict().items()}
    expected_keys = expected_state_dict_keys(cfg.n_layers)
    if list(sd.keys()) != expected_keys:
        raise ValueError(f"Niespodziewane klucze state_dict: {list(sd.keys())} != {expected_keys}")

    tok_dict = tokenizer.to_dict()
    if not include_vocab_table:
        tok_dict = {k: v for k, v in tok_dict.items() if k != "vocab"}
    tok_bytes = json.dumps(tok_dict, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    tok_offset = HEADER_STRUCT.size  # 64
    tok_len = len(tok_bytes)
    weights_offset = align4(tok_offset + tok_len)
    tok_pad_len = weights_offset - (tok_offset + tok_len)

    weight_chunks: List[bytes] = []
    for name, expected_shape, kind in canonical_weight_specs(cfg):
        tensor = sd[name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Zły kształt {name}: {tuple(tensor.shape)} != {expected_shape}")
        if kind == "f32":
            weight_chunks.append(pack_vector_f32(tensor))
        elif kind == "int8":
            weight_chunks.append(pack_matrix_int8(tensor))
        else:
            raise ValueError(f"Nieznany typ pakowania: {kind}")

    weights_bytes = b"".join(weight_chunks)
    weights_len = len(weights_bytes)
    assert weights_offset % 4 == 0
    assert weights_len % 4 == 0

    payload = tok_bytes + (b"\x00" * tok_pad_len) + weights_bytes
    payload_crc32 = zlib.crc32(payload) & 0xFFFFFFFF

    flags = (
        (FLAG_TIED_EMBEDDINGS if cfg.tied_embeddings else 0)
        | (0)  # bit 1 = 0 (half-split RoPE)
        | (FLAG_SWIGLU)
    )

    header = HEADER_STRUCT.pack(
        TNY3_MAGIC,
        TNY3_VERSION,
        flags,
        cfg.vocab_size,
        cfg.ctx_len,
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.d_ff,
        float(cfg.rope_theta),
        float(cfg.rms_eps),
        tok_offset,
        tok_len,
        weights_offset,
        weights_len,
        payload_crc32,
        0,  # reserved
    )
    assert len(header) == 64

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(header)
        fh.write(payload)

    return {
        "file": out_path,
        "total_bytes": len(header) + len(payload),
        "header_bytes": len(header),
        "tok_offset": tok_offset,
        "tok_len": tok_len,
        "tok_pad_bytes": tok_pad_len,
        "weights_offset": weights_offset,
        "weights_len": weights_len,
        "flags": flags,
        "payload_crc32": payload_crc32,
        "sha256": sha256_file(out_path),
    }


def load_tny3(path: str) -> Tuple[TartarusModel, TartarusTokenizer, Dict[str, Any]]:
    """
    Odczytuje plik `model.tny3` bezpośrednio z dysku, weryfikuje nagłówek, alignment,
    CRC-32 i rekonstruuje TartarusModel (z wagami dekwantyzowanymi do float32)
    oraz TartarusTokenizer.
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    if len(raw) < HEADER_STRUCT.size:
        raise ValueError(f"Plik TNY3 jest zbyt mały ({len(raw)} B)")

    (
        magic,
        version,
        flags,
        vocab_size,
        ctx_len,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        rope_theta,
        norm_eps,
        tok_offset,
        tok_len,
        weights_offset,
        weights_len,
        payload_crc32,
        reserved,
    ) = HEADER_STRUCT.unpack_from(raw, 0)

    if magic != TNY3_MAGIC:
        raise ValueError(f"Nieprawidłowy magic TNY3: {magic!r}")
    if version != TNY3_VERSION:
        raise ValueError(f"Nieobsługiwana wersja formatu TNY3: {version}")
    if tok_offset != HEADER_STRUCT.size or tok_offset % 4 != 0:
        raise ValueError(f"Nieprawidłowy tok_offset={tok_offset}")
    if weights_offset % 4 != 0 or weights_offset < tok_offset + tok_len:
        raise ValueError(f"Nieprawidłowy weights_offset={weights_offset} (tok_end={tok_offset + tok_len})")
    if weights_offset + weights_len != len(raw):
        raise ValueError(
            f"Rozbieżność rozmiaru pliku TNY3: {weights_offset + weights_len} != {len(raw)}"
        )

    pad_slice = raw[tok_offset + tok_len : weights_offset]
    if any(b != 0 for b in pad_slice):
        raise ValueError("Niezerowy bajt w obszarze paddingu po sekcji tokenizera")

    actual_crc = zlib.crc32(memoryview(raw)[HEADER_STRUCT.size :]) & 0xFFFFFFFF
    if actual_crc != payload_crc32:
        raise ValueError(f"Błąd CRC-32 payloadu TNY3: 0x{actual_crc:08x} != 0x{payload_crc32:08x}")

    tied_embeddings = bool(flags & FLAG_TIED_EMBEDDINGS)
    rope_interleaved = bool(flags & FLAG_ROPE_INTERLEAVED)
    swiglu = bool(flags & FLAG_SWIGLU)
    if rope_interleaved:
        raise ValueError("Ten loader obsługuje wyłącznie konwencję RoPE half-split (bit 1 == 0)")
    if not swiglu:
        raise ValueError("Wymagana flaga SwiGLU (bit 2 == 1)")

    tok_json = json.loads(raw[tok_offset : tok_offset + tok_len].decode("utf-8"))
    tokenizer = TartarusTokenizer.from_dict(tok_json)

    # Ustal profil na podstawie wymiarów (informacyjnie)
    if d_model == 192 and n_layers == 4 and n_heads == 6 and d_ff == 512:
        profile = "S"
    elif d_model == 256 and n_layers == 6 and n_heads == 8 and d_ff == 640:
        profile = "M"
    else:
        profile = "custom"

    cfg = TartarusConfig(
        vocab_size=vocab_size,
        ctx_len=ctx_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        rms_eps=float(norm_eps),
        rope_theta=float(rope_theta),
        tied_embeddings=tied_embeddings,
        profile=profile,
    )

    mv = memoryview(raw)
    cursor = weights_offset
    sd: Dict[str, torch.Tensor] = {}
    for name, expected_shape, kind in canonical_weight_specs(cfg):
        if kind == "f32":
            tensor, cursor = unpack_vector_f32(mv, cursor, expected_shape[0])
        elif kind == "int8":
            tensor, cursor = unpack_matrix_int8(mv, cursor, expected_shape[0], expected_shape[1])
        else:
            raise ValueError(f"Nieznany typ: {kind}")
        sd[name] = tensor

    if cursor != weights_offset + weights_len:
        raise ValueError(f"Niepełny odczyt sekcji wag TNY3: cursor={cursor} != {weights_offset + weights_len}")

    model = TartarusModel(cfg)
    model.load_state_dict(sd, strict=True)
    model.eval()

    header_info = {
        "magic": magic.decode("ascii"),
        "format_version": version,
        "flags": flags,
        "tied_embeddings": tied_embeddings,
        "rope_convention": "interleaved" if rope_interleaved else "half-split",
        "swiglu": swiglu,
        "tok_offset": tok_offset,
        "tok_len": tok_len,
        "weights_offset": weights_offset,
        "weights_len": weights_len,
        "payload_crc32": f"0x{payload_crc32:08x}",
        "reserved": reserved,
    }
    return model, tokenizer, header_info


# ═══════════════════════════════════════════════════════════════════════════════
# WERYFIKACJA ROUND-TRIP TNY3 == QUANTIZED_COPY ORAZ ZŁOTY TEST (golden_test.json)
# ═══════════════════════════════════════════════════════════════════════════════


DEFAULT_GOLDEN_PROMPTS: List[Tuple[str, str]] = [
    (
        "golden_pl_state_query",
        "<|state|> lokacja=Gaj Świętogora; typ=zamek; teren=las; tura=14; frakcja=Plemiona Słowiańskie; "
        "dowódca=Wojewoda Siemomysł; hp=42/75; mana=9/16; złoto=390; ruch=5; jednostki=Leszyj Puszczyx2,Wołkodłakx3; "
        "wrogowie=Królestwo Egiptu(4p) <|user|> Ile punktów życia i złota ma nasza armia? <|npc|>",
    ),
    (
        "golden_pl_diacritics_lore",
        "<|state|> lokacja=Twierdza Tartar; hp=18/70; złoto=-40; jednostki=Jötunn Mroźnyx1,Żmij Gorynyczx2; wrogowie=brak "
        "<|user|> Krótko: „Zażółć gęślą jaźń” — czym wyróżnia się Żywiołak Kamienny oraz Völva Wieszczka? <|npc|>",
    ),
    (
        "golden_pl_refusal_short",
        "<|state|> lokacja=Żelazny Most; hp=0/75; ruch=0; jednostki=Czarny Mnichx1; wrogowie=brak "
        "<|user|> Raportuj: rzuć Grom Peruna i wyślij wodza przez góry! <|npc|>",
    ),
]


@torch.inference_mode()
def verify_tny3_parity_and_generate_golden(
    ref_q_model: TartarusModel,
    tny3_model: TartarusModel,
    tny3_tokenizer: TartarusTokenizer,
    export_meta: Dict[str, Any],
    header_info: Dict[str, Any],
    golden_out_path: str,
    prompts: Sequence[Tuple[str, str]] = DEFAULT_GOLDEN_PROMPTS,
) -> Dict[str, Any]:
    # 1. Weryfikacja identyczności wag (co do bita w float32)
    ref_sd = ref_q_model.state_dict()
    tny_sd = tny3_model.state_dict()
    max_weight_diff = 0.0
    for k in ref_sd:
        diff = float((ref_sd[k].float() - tny_sd[k].float()).abs().max().item())
        max_weight_diff = max(max_weight_diff, diff)
        assert diff == 0.0, f"Rozbieżność wag dla {k}: max_diff={diff}"

    # 2. Weryfikacja identyczności logitów na wszystkich sekwencjach testowych
    golden_cases: List[Dict[str, Any]] = []
    max_logit_diff = 0.0

    for name, text in prompts:
        ids = tny3_tokenizer.encode(text)
        if not ids:
            raise ValueError(f"Puste token_ids dla przypadku {name}")
        if len(ids) > tny3_model.cfg.ctx_len:
            raise ValueError(f"Sekwencja {name} ma długość {len(ids)} > ctx_len={tny3_model.cfg.ctx_len}")

        inp = torch.tensor([ids], dtype=torch.long)
        ref_logits = ref_q_model(inp).float()  # [1, T, V]
        tny_logits = tny3_model(inp).float()   # [1, T, V]

        diff = float((ref_logits - tny_logits).abs().max().item())
        max_logit_diff = max(max_logit_diff, diff)
        assert diff < 1e-6, f"Niezgodność logitów TNY3 vs quantized_copy dla {name}: {diff}"

        last_vec = tny_logits[0, -1, :].tolist()
        last_t = tny_logits[0, -1, :]
        top_vals, top_idxs = torch.topk(last_t, k=5)
        top5 = [
            {
                "token_id": int(tid),
                "logit": float(val),
                "piece": repr(tny3_tokenizer.decode([int(tid)], skip_pad=False)),
            }
            for val, tid in zip(top_vals.tolist(), top_idxs.tolist())
        ]

        golden_cases.append(
            {
                "name": name,
                "text": text,
                "token_ids": ids,
                "seq_len": len(ids),
                "greedy_next_token_id": int(torch.argmax(last_t).item()),
                "top5": top5,
                "last_step_logits": last_vec,
            }
        )

    golden_payload = {
        "format": "tartarus-golden-test-v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tny3_file": os.path.basename(export_meta["file"]),
        "tny3_sha256": export_meta["sha256"],
        "tny3_header": header_info,
        "cfg": tny3_model.cfg.to_dict(),
        "verification": {
            "max_weight_diff": max_weight_diff,
            "max_logit_diff": max_logit_diff,
        },
        "num_cases": len(golden_cases),
        "cases": golden_cases,
    }

    os.makedirs(os.path.dirname(os.path.abspath(golden_out_path)), exist_ok=True)
    with open(golden_out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(golden_payload, ensure_ascii=False, indent=2) + "\n")

    return golden_payload


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eksport checkpointu Tartarus do model.tny3 + golden_test.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Ścieżka do checkpointu (.pt)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Ścieżka do tokenizer.json")
    parser.add_argument("--out-tny3", type=str, default="model.tny3", help="Wyjściowy plik binarny TNY3")
    parser.add_argument("--out-golden", type=str, default="golden_test.json", help="Wyjściowy plik złotego testu")
    parser.add_argument("--omit-vocab-table", action="store_true", help="Pomiń pomocniczą tablicę vocab w sekcji tokenizera (zostawia merges)")
    parser.add_argument("--skip-sha-check", action="store_true", help="Pomiń sprawdzenie tokenizer_sha256 względem checkpointu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Brak checkpointu: {args.checkpoint}")
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Brak tokenizer.json: {args.tokenizer}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TartarusConfig.from_dict(ckpt["cfg"])

    tok_sha = sha256_file(args.tokenizer)
    ckpt_sha = ckpt.get("tokenizer_sha256")
    if not args.skip_sha_check and ckpt_sha != tok_sha:
        raise ValueError(
            f"Niezgodność tokenizer_sha256!\n  checkpoint: {ckpt_sha}\n  tokenizer : {tok_sha}"
        )

    tokenizer = TartarusTokenizer.load(args.tokenizer)
    base_model = TartarusModel(cfg).float()
    base_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    base_model.eval()

    ref_q_model = quantized_copy(base_model).float()
    export_meta = export_model_tny3(
        model=base_model,
        tokenizer=tokenizer,
        out_path=args.out_tny3,
        include_vocab_table=not args.omit_vocab_table,
    )

    # Natychmiastowe przeładowanie model.tny3 z dysku i weryfikacja 1:1
    tny3_model, tny3_tokenizer, header_info = load_tny3(args.out_tny3)
    golden = verify_tny3_parity_and_generate_golden(
        ref_q_model=ref_q_model,
        tny3_model=tny3_model,
        tny3_tokenizer=tny3_tokenizer,
        export_meta=export_meta,
        header_info=header_info,
        golden_out_path=args.out_golden,
    )

    print("=" * 88)
    print("TARTARUS ETAP 4 — EKSPORT TNY3 I ZŁOTY TEST ZAKOŃCZONE POMYŚLNIE")
    print("=" * 88)
    print(f"Checkpoint      : {args.checkpoint} (profil={cfg.profile}, params={base_model.num_parameters():,})")
    print(f"Wyjście TNY3    : {args.out_tny3} ({export_meta['total_bytes']:,} B = {export_meta['total_bytes'] / (1024*1024):.2f} MiB)")
    print(f"  Header        : 64 B | version={header_info['format_version']} | flags=0b{header_info['flags']:03b} (tied={header_info['tied_embeddings']}, rope={header_info['rope_convention']}, swiglu={header_info['swiglu']})")
    print(f"  Tokenizer     : offset={export_meta['tok_offset']} | len={export_meta['tok_len']:,} B | pad={export_meta['tok_pad_bytes']} B")
    print(f"  Weights INT8  : offset={export_meta['weights_offset']} (mod 4 = {export_meta['weights_offset'] % 4}) | len={export_meta['weights_len']:,} B (mod 4 = {export_meta['weights_len'] % 4})")
    print(f"  Payload CRC32 : {header_info['payload_crc32']} | SHA256={export_meta['sha256'][:16]}...")
    print()
    print("[WERYFIKACJA TNY3 == quantized_copy]")
    print(f"  max_weight_diff : {golden['verification']['max_weight_diff']:.2e}")
    print(f"  max_logit_diff  : {golden['verification']['max_logit_diff']:.2e}")
    print()
    print(f"[ZŁOTY TEST: {args.out_golden}] ({golden['num_cases']} przypadków)")
    for c in golden["cases"]:
        print(f"  - {c['name']} (len={c['seq_len']} tok, argmax={c['greedy_next_token_id']}, top1={c['top5'][0]['piece']} @ {c['top5'][0]['logit']:.4f})")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
