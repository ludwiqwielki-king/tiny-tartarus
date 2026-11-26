#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_tartarus.py — Pipeline treningu GPU (CSR memmap + Bucket Batching + AMP +
dwutorowa ewaluacja float vs INT8 + próbki jakościowe) dla modelu Tartarus.

Użycie:
    # Trening na GPU (Profil S, domyślnie):
    python3 train_tartarus.py --profile S --data-dir . --out-dir checkpoints

    # Trening Profilu M:
    python3 train_tartarus.py --profile M --batch-size 128 --epochs 20

    # Ewaluacja samego checkpointu (val float vs INT8 + 16 próbek jakościowych):
    python3 train_tartarus.py --eval-only --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model_tartarus import TartarusConfig, TartarusModel
from tokenizer_tartarus import EOS_ID, NPC_TOKEN, PAD_ID, TartarusTokenizer, load_jsonl, sha256_file


# ═══════════════════════════════════════════════════════════════════════════════
# WCZYTYWANIE I WALIDACJA PACZEK CSR (np.memmap)
# ═══════════════════════════════════════════════════════════════════════════════


class CSRSplit:
    """Reprezentuje jeden split (train/val/test) w formacie CSR podpięty przez np.memmap."""

    def __init__(
        self,
        split_name: str,
        data_dir: str,
        meta_split: Dict[str, Any],
        vocab_size: int,
        ctx_len: int,
    ) -> None:
        self.name = split_name
        files = meta_split["files"]
        tok_path = os.path.join(data_dir, files["tokens"]["filename"])
        msk_path = os.path.join(data_dir, files["mask"]["filename"])
        off_path = os.path.join(data_dir, files["offsets"]["filename"])

        for p in (tok_path, msk_path, off_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Brak pliku CSR: {p}")

        self.tokens = np.memmap(tok_path, dtype=np.dtype("<u2"), mode="r")
        self.mask = np.memmap(msk_path, dtype=np.dtype("u1"), mode="r")
        self.offsets = np.memmap(off_path, dtype=np.dtype("<u8"), mode="r")

        n_samples = int(meta_split["num_samples"])
        t_total = int(meta_split["total_tokens"])
        if self.tokens.shape != (t_total,) or self.mask.shape != (t_total,):
            raise ValueError(f"{split_name}: niezgodna długość tokens/mask ({self.tokens.shape} / {self.mask.shape} vs {t_total})")
        if self.offsets.shape != (n_samples + 1,):
            raise ValueError(f"{split_name}: niezgodny kształt offsets ({self.offsets.shape} vs {n_samples + 1})")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != t_total:
            raise ValueError(f"{split_name}: uszkodzone granice offsets ({self.offsets[0]} .. {self.offsets[-1]})")

        self.lengths = (self.offsets[1:] - self.offsets[:-1]).astype(np.int64)
        self.num_samples = n_samples
        self.total_tokens = t_total
        self.max_seq_len = int(self.lengths.max())
        self.min_seq_len = int(self.lengths.min())
        self.mean_seq_len = float(self.lengths.mean())

        if self.max_seq_len > ctx_len:
            raise ValueError(
                f"{split_name}: maksymalna długość próbki ({self.max_seq_len}) przekracza cfg.ctx_len ({ctx_len})!"
            )
        if int(self.tokens.max()) >= vocab_size:
            raise ValueError(
                f"{split_name}: wykryto token ID >= vocab_size ({int(self.tokens.max())} >= {vocab_size})"
            )


def load_and_validate_csr(
    data_dir: str,
    tokenizer_path: str,
    cfg: TartarusConfig,
    splits: Tuple[str, ...] = ("train", "val"),
) -> Tuple[Dict[str, Any], Dict[str, CSRSplit]]:
    meta_path = os.path.join(data_dir, "dataset_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Brak dataset_meta.json w {data_dir}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Brak tokenizer.json: {tokenizer_path}")

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    if int(meta["vocab_size"]) != cfg.vocab_size:
        raise ValueError(
            f"Niezgodność vocab_size: dataset_meta.json={meta['vocab_size']} != cfg.vocab_size={cfg.vocab_size}"
        )

    actual_sha = sha256_file(tokenizer_path)
    if meta["tokenizer_sha256"] != actual_sha:
        raise ValueError(
            f"Niezgodność tokenizer_sha256!\n  meta: {meta['tokenizer_sha256']}\n  file: {actual_sha}"
        )

    loaded: Dict[str, CSRSplit] = {}
    for sp in splits:
        if sp not in meta["splits"]:
            raise KeyError(f"Brak splitu {sp!r} w dataset_meta.json")
        loaded[sp] = CSRSplit(sp, data_dir, meta["splits"][sp], cfg.vocab_size, cfg.ctx_len)

    return meta, loaded


# ═══════════════════════════════════════════════════════════════════════════════
# SZYBKI BUCKET BATCHER POD GPU (ZERO TOKENIZERA W PĘTLI)
# ═══════════════════════════════════════════════════════════════════════════════


def build_bucket_batches(
    lengths: np.ndarray,
    batch_size: int,
    rng: Optional[random.Random] = None,
    bucket_mult: int = 8,
    shuffle: bool = True,
) -> List[List[int]]:
    """
    Grupuje próbki w kubełki o zbliżonej długości (sort + tasowanie wewnątrz kubełka),
    tworzy batche i tasuje kolejność batchy w epoce. Minimalizuje narzut <|pad|>.
    """
    sorted_idx = np.argsort(lengths, kind="stable").tolist()
    if not shuffle or rng is None:
        return [sorted_idx[i : i + batch_size] for i in range(0, len(sorted_idx), batch_size)]

    pool_size = max(batch_size, batch_size * bucket_mult)
    batches: List[List[int]] = []
    for start in range(0, len(sorted_idx), pool_size):
        pool = sorted_idx[start : start + pool_size]
        rng.shuffle(pool)
        for b_start in range(0, len(pool), batch_size):
            batches.append(pool[b_start : b_start + batch_size])

    rng.shuffle(batches)
    return batches


def iter_csr_batches(
    split: CSRSplit,
    batch_size: int,
    device: torch.device,
    rng: Optional[random.Random] = None,
    shuffle: bool = True,
    pin_memory: bool = False,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Składa batche z right-paddingiem wyłącznie do najdłuższej sekwencji w batchu,
    opcjonalnie z pin_memory=True i nieblokującym transferem na GPU.
    """
    batches = build_bucket_batches(split.lengths, batch_size=batch_size, rng=rng, shuffle=shuffle)
    offsets = split.offsets
    tokens_mm = split.tokens
    mask_mm = split.mask
    lengths = split.lengths
    use_non_blocking = device.type == "cuda"

    for b_indices in batches:
        bsz = len(b_indices)
        max_len = max(int(lengths[idx]) for idx in b_indices)

        x_np = np.full((bsz, max_len), PAD_ID, dtype=np.int64)
        m_np = np.zeros((bsz, max_len), dtype=np.float32)

        for row, idx in enumerate(b_indices):
            s = int(offsets[idx])
            e = int(offsets[idx + 1])
            seq_len = e - s
            x_np[row, :seq_len] = tokens_mm[s:e]
            m_np[row, :seq_len] = mask_mm[s:e]

        x_cpu = torch.from_numpy(x_np)
        m_cpu = torch.from_numpy(m_np)
        if pin_memory and use_non_blocking:
            x_cpu = x_cpu.pin_memory()
            m_cpu = m_cpu.pin_memory()

        x_dev = x_cpu.to(device, non_blocking=use_non_blocking)
        m_dev = m_cpu.to(device, non_blocking=use_non_blocking)
        yield x_dev, m_dev


# ═══════════════════════════════════════════════════════════════════════════════
# SYMULACJA KWANTYZACJI INT8 PER-ROW ORAZ EWALUACJA DWUTOROWA
# ═══════════════════════════════════════════════════════════════════════════════


def quantized_copy(model: TartarusModel) -> TartarusModel:
    """
    Tworzy kopię modelu z symulowaną kwantyzacją INT8 per-row [-127, 127]
    dla wszystkich macierzy 2D (dim == 2: tok_emb.weight oraz wagi warstw Linear).
    Wektory norm (dim < 2) pozostają w pełnej precyzji float32.
    """
    q_model = copy.deepcopy(model)
    q_model.eval()
    with torch.no_grad():
        for p in q_model.parameters():
            p.requires_grad_(False)
            if p.dim() == 2:
                w = p.detach().float()
                max_abs = w.abs().amax(dim=-1, keepdim=True)
                scale = (max_abs / 127.0).clamp_min(1e-8)
                q = torch.clamp(torch.round(w / scale), -127.0, 127.0)
                p.copy_((q * scale).to(dtype=p.dtype))
    return q_model


@torch.inference_mode()
def evaluate_split(
    model: TartarusModel,
    split: CSRSplit,
    batch_size: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    pin_memory: bool = False,
) -> Tuple[float, float]:
    """
    Liczy dokładną średnią stratę ważoną maską (mask == 1) na całym splicie oraz perplexity.
    Brak synchronizacji na każdym batchu — pojedyncza redukcja na końcu splitu.
    """
    was_training = model.training
    model.eval()

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_weight = torch.zeros((), device=device, dtype=torch.float32)

    amp_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
        if (device.type == "cuda" and amp_dtype is not None)
        else contextlib.nullcontext()
    )

    try:
        for x, m in iter_csr_batches(split, batch_size=batch_size, device=device, shuffle=False, pin_memory=pin_memory):
            inp = x[:, :-1]
            targets = x[:, 1:]
            weights = m[:, 1:]

            with amp_ctx:
                logits = model(inp)
            per_tok = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view_as(targets)

            total_loss += (per_tok * weights).sum()
            total_weight += weights.sum()

        avg_loss = float((total_loss / total_weight.clamp_min(1.0)).item())
        ppl = math.exp(min(avg_loss, 20.0))
        return avg_loss, ppl
    finally:
        if was_training:
            model.train()


# ═══════════════════════════════════════════════════════════════════════════════
# HARMONOGRAM LR I OPTYMALIZATOR
# ═══════════════════════════════════════════════════════════════════════════════


def compute_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= total_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def build_optimizer(model: TartarusModel, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay_params: List[torch.nn.Parameter] = []
    nodecay_params: List[torch.nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    kwargs: Dict[str, Any] = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8)
    if next(model.parameters()).device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# EWALUACJA JAKOŚCIOWA NA VAL (MODEL INT8)
# ═══════════════════════════════════════════════════════════════════════════════


REFUSAL_KEYWORDS = (
    "nie ", "brak", "odmow", "odmaw", "odrzuc", "niemoż", "za mało",
    "niewystarcz", "błąd", "przekracza", "limit", "poległ", "nie żyje",
    "ujemnym", "nie istnieje", "nie znam", "nie ma", "nie odnajduj",
)


def select_qualitative_val_samples(val_path: str, per_cat: int = 4) -> List[Dict[str, Any]]:
    rows = load_jsonl(val_path)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    chosen: List[Dict[str, Any]] = []
    for cat in ("dialog", "state_query", "lore_entity", "refusal"):
        pool = by_cat.get(cat, [])
        if not pool:
            continue
        step = max(1, len(pool) // per_cat)
        picked = [pool[i * step] for i in range(min(per_cat, len(pool)))]
        chosen.extend(picked)
    return chosen


def evaluate_qualitative_int8(
    model: TartarusModel,
    tokenizer: TartarusTokenizer,
    val_jsonl_path: str,
    per_cat: int = 4,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    q_model = quantized_copy(model)
    samples = select_qualitative_val_samples(val_jsonl_path, per_cat=per_cat)

    print("=" * 96)
    print(f"EWALUACJA JAKOŚCIOWA (INT8 quantized_copy, temperature=0, {len(samples)} próbek z val.jsonl)")
    print("=" * 96)

    eos_hits = 0
    state_num_checked = 0
    state_num_ok = 0
    refusal_checked = 0
    refusal_ok = 0

    for idx, row in enumerate(samples, start=1):
        full_text = row["text"]
        cat = row.get("category", "?")
        split_pos = full_text.index(NPC_TOKEN) + len(NPC_TOKEN)
        prompt_text = full_text[:split_pos]
        ref_npc = full_text[split_pos:].strip()

        prompt_ids = tokenizer.encode(prompt_text)
        gen_ids = q_model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=EOS_ID,
            temperature=0.0,
        )
        ended_eos = len(gen_ids) > 0 and gen_ids[-1] == EOS_ID
        if ended_eos:
            eos_hits += 1

        gen_text = tokenizer.decode(gen_ids, skip_pad=True, skip_special=False).strip()

        # Kontrola zgodności liczbowej ze <|state|> w state_query
        state_part = prompt_text.split("<|user|>")[0]
        ref_nums = re.findall(r"-?\d+", ref_npc)
        state_nums = set(re.findall(r"-?\d+", state_part))
        expected_state_nums = [n for n in ref_nums if n in state_nums]
        num_note = ""
        if cat == "state_query" and expected_state_nums:
            state_num_checked += 1
            gen_nums = set(re.findall(r"-?\d+", gen_text))
            matched = all(n in gen_nums for n in expected_state_nums)
            if matched:
                state_num_ok += 1
            num_note = f" | liczby_state={'OK' if matched else 'MISMATCH'}({expected_state_nums})"

        refusal_note = ""
        if cat == "refusal":
            refusal_checked += 1
            low = gen_text.lower()
            is_ref = any(kw in low for kw in REFUSAL_KEYWORDS)
            if is_ref:
                refusal_ok += 1
            refusal_note = f" | odmowa={'OK' if is_ref else 'BRAK'}"

        print(f"[{idx:02d}] ({cat}) EOS={'TAK' if ended_eos else 'NIE'}{num_note}{refusal_note}")
        print(f"  PROMPT: {prompt_text}")
        print(f"  REF   : {ref_npc}")
        print(f"  INT8  : {gen_text}")
        print()

    summary = {
        "num_samples": len(samples),
        "eos_rate": eos_hits / max(1, len(samples)),
        "state_num_ok": state_num_ok,
        "state_num_checked": state_num_checked,
        "refusal_ok": refusal_ok,
        "refusal_checked": refusal_checked,
    }
    print(
        f"[PODSUMOWANIE JAKOŚCIOWE INT8] "
        f"EOS={eos_hits}/{len(samples)} ({100*summary['eos_rate']:.0f}%) | "
        f"Liczby state_query={state_num_ok}/{state_num_checked} | "
        f"Odmowy refusal={refusal_ok}/{refusal_checked}"
    )
    print("=" * 96)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# GŁÓWNA PĘTLA TRENINGOWA GPU / CPU
# ═══════════════════════════════════════════════════════════════════════════════


def save_checkpoint(
    path: str,
    model: TartarusModel,
    optimizer: Optional[torch.optim.Optimizer],
    tokenizer_sha256: str,
    epoch: int,
    global_step: int,
    metrics: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "cfg": model.cfg.to_dict(),
        "tokenizer_sha256": tokenizer_sha256,
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def resolve_device_and_amp(device_arg: str) -> Tuple[torch.device, Optional[torch.dtype], bool]:
    if device_arg == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device_arg)

    if dev.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return dev, torch.bfloat16, False
        return dev, torch.float16, True
    return dev, None, False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening GPU oraz ewaluacja modelu Tartarus (RoPE + RMSNorm + SwiGLU).")
    parser.add_argument("--profile", type=str, default="S", choices=["S", "M", "s", "m"], help="Profil modelu (S lub M)")
    parser.add_argument("--data-dir", type=str, default=".", help="Katalog z plikami CSR oraz val.jsonl")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Ścieżka do tokenizer.json")
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="Katalog na checkpointy best.pt / last.pt")

    parser.add_argument("--epochs", type=int, default=20, help="Liczba epok treningowych (domyślnie: 20)")
    parser.add_argument("--batch-size", type=int, default=128, help="Rozmiar mikro-batcha (domyślnie: 128)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Kroki akumulacji gradientu (efektywny batch = bs * grad_accum)")
    parser.add_argument("--max-steps", type=int, default=0, help="Opcjonalny limit kroków optymalizatora (0 = pełne epoki)")
    parser.add_argument("--lr", type=float, default=6e-4, help="Maksymalna stopa uczenia (domyślnie: 6e-4)")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Końcowa stopa uczenia cosine decay")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay dla macierzy 2D (domyślnie: 0.05)")
    parser.add_argument("--warmup-steps", type=int, default=150, help="Liczba kroków linear warmup")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Limit normy gradientu (domyślnie: 1.0)")

    parser.add_argument("--log-every", type=int, default=25, help="Co ile kroków synchronizować i logować train_loss")
    parser.add_argument("--eval-every", type=int, default=250, help="Co ile kroków liczyć val float + INT8")
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowości")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Urządzenie obliczeniowe")
    parser.add_argument("--no-pin-memory", action="store_true", help="Wyłącz pin_memory dla transferu H2D")

    parser.add_argument("--eval-only", action="store_true", help="Tylko ewaluacja (float vs INT8 + próbki jakościowe)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Checkpoint dla trybu --eval-only")
    parser.add_argument("--qual-per-cat", type=int, default=4, help="Liczba próbek na kategorię do ewaluacji jakościowej")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device, amp_dtype, use_grad_scaler = resolve_device_and_amp(args.device)
    pin_memory = (device.type == "cuda") and (not args.no_pin_memory)

    tokenizer = TartarusTokenizer.load(args.tokenizer)
    tok_sha = sha256_file(args.tokenizer)

    if args.eval_only:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = TartarusConfig.from_dict(ckpt["cfg"])
        if ckpt.get("tokenizer_sha256") != tok_sha:
            raise ValueError("Niezgodność tokenizer_sha256 między checkpointem a tokenizer.json!")
        _, splits = load_and_validate_csr(args.data_dir, args.tokenizer, cfg, splits=("val",))
        model = TartarusModel(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        val_loss_f, val_ppl_f = evaluate_split(model, splits["val"], args.batch_size, device, amp_dtype, pin_memory)
        q_model = quantized_copy(model)
        val_loss_q, val_ppl_q = evaluate_split(q_model, splits["val"], args.batch_size, device, amp_dtype, pin_memory)

        print(f"[EVAL-ONLY] Checkpoint: {args.checkpoint} (profil={cfg.profile}, params={model.num_parameters():,})")
        print(f"  FLOAT: val_loss={val_loss_f:.4f} | val_ppl={val_ppl_f:.2f}")
        print(f"  INT8 : val_loss={val_loss_q:.4f} | val_ppl={val_ppl_q:.2f} | Δ={val_loss_q - val_loss_f:+.4f}")

        val_jsonl = os.path.join(args.data_dir, "val.jsonl")
        if os.path.exists(val_jsonl):
            evaluate_qualitative_int8(model, tokenizer, val_jsonl, per_cat=args.qual_per_cat)
        return 0

    cfg = TartarusConfig.from_profile(args.profile, vocab_size=tokenizer.vocab_size)
    meta, splits = load_and_validate_csr(args.data_dir, args.tokenizer, cfg, splits=("train", "val"))
    train_split = splits["train"]
    val_split = splits["val"]

    model = TartarusModel(cfg).to(device)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

    grad_accum = max(1, args.grad_accum)
    micro_batches_per_epoch = len(build_bucket_batches(train_split.lengths, args.batch_size, shuffle=False))
    steps_per_epoch = math.ceil(micro_batches_per_epoch / grad_accum)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))

    amp_label = (
        "none (fp32 CPU)"
        if amp_dtype is None
        else ("bf16" if amp_dtype == torch.bfloat16 else "fp16 + GradScaler")
    )
    print("=" * 96)
    print(f"TARTARUS ETAP 3 — TRENING MODELU (Profil {cfg.profile}, {model.num_parameters():,} param.)")
    print("=" * 96)
    print(f"Urządzenie      : {device} | AMP: {amp_label} | pin_memory={pin_memory}")
    print(f"Konfiguracja    : vocab={cfg.vocab_size}, ctx={cfg.ctx_len}, d={cfg.d_model}, L={cfg.n_layers}, H={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"Dane CSR        : train={train_split.num_samples:,} (max_len={train_split.max_seq_len}), val={val_split.num_samples:,} (max_len={val_split.max_seq_len})")
    print(
        f"Harmonogram     : epochs={args.epochs}, bs={args.batch_size}, grad_accum={grad_accum} (eff_bs={args.batch_size * grad_accum}), "
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup={warmup_steps}"
    )
    print(f"Optymalizator   : AdamW(lr={args.lr} -> {args.min_lr}, wd={args.weight_decay}, betas=(0.9, 0.95), clip={args.grad_clip})")
    print("=" * 96)

    amp_ctx_factory = (
        (lambda: torch.amp.autocast(device_type="cuda", dtype=amp_dtype))
        if (device.type == "cuda" and amp_dtype is not None)
        else contextlib.nullcontext
    )

    best_val_loss_int8 = float("inf")
    best_val_loss_float = float("inf")
    global_step = 0
    last_eval_step = -1
    history: List[Dict[str, Any]] = []

    loss_acc = torch.zeros((), device=device, dtype=torch.float32)
    weight_acc = torch.zeros((), device=device, dtype=torch.float32)
    steps_since_log = 0
    t_log_start = time.perf_counter()

    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    def run_eval_and_checkpoint(epoch_idx: int, step_idx: int) -> None:
        nonlocal best_val_loss_int8, best_val_loss_float, last_eval_step
        if step_idx == last_eval_step:
            return
        last_eval_step = step_idx
        t_eval = time.perf_counter()
        val_loss_f, val_ppl_f = evaluate_split(model, val_split, args.batch_size, device, amp_dtype, pin_memory)
        q_model = quantized_copy(model)
        val_loss_q, val_ppl_q = evaluate_split(q_model, val_split, args.batch_size, device, amp_dtype, pin_memory)
        eval_dt = time.perf_counter() - t_eval

        best_val_loss_float = min(best_val_loss_float, val_loss_f)
        improved = val_loss_q < best_val_loss_int8
        if improved:
            best_val_loss_int8 = val_loss_q

        metrics = {
            "val_loss_float": val_loss_f,
            "val_ppl_float": val_ppl_f,
            "val_loss_int8": val_loss_q,
            "val_ppl_int8": val_ppl_q,
            "int8_gap": val_loss_q - val_loss_f,
            "best_val_loss_int8": best_val_loss_int8,
            "best_val_loss_float": best_val_loss_float,
        }
        history.append({"epoch": epoch_idx, "step": step_idx, **metrics})

        mark = " ★ BEST INT8" if improved else ""
        print(
            f"[EVAL ep={epoch_idx} step={step_idx:>5}] "
            f"FLOAT val={val_loss_f:.4f} (ppl={val_ppl_f:.2f}) | "
            f"INT8 val={val_loss_q:.4f} (ppl={val_ppl_q:.2f}, Δ={val_loss_q - val_loss_f:+.4f}) "
            f"({eval_dt:.2f}s){mark}"
        )

        if improved:
            save_checkpoint(best_path, model, optimizer, tok_sha, epoch_idx, step_idx, metrics)
        save_checkpoint(last_path, model, optimizer, tok_sha, epoch_idx, step_idx, metrics)

    model.train()
    stop_early = False

    for epoch in range(1, args.epochs + 1):
        epoch_rng = random.Random(args.seed + epoch * 1009)
        optimizer.zero_grad(set_to_none=True)
        micro_in_accum = 0
        batch_iter = iter_csr_batches(
            train_split,
            batch_size=args.batch_size,
            device=device,
            rng=epoch_rng,
            shuffle=True,
            pin_memory=pin_memory,
        )

        for mb_idx, (x, m) in enumerate(batch_iter, start=1):
            inp = x[:, :-1]
            targets = x[:, 1:]
            weights = m[:, 1:]

            with amp_ctx_factory():
                logits = model(inp)
                per_tok = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                ).view_as(targets)
                w_sum = weights.sum().clamp_min(1.0)
                loss = (per_tok * weights).sum() / w_sum
                loss_scaled = loss / float(grad_accum) if grad_accum > 1 else loss

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            loss_acc += loss.detach()
            weight_acc += w_sum.detach()
            steps_since_log += 1
            micro_in_accum += 1

            is_step_boundary = (micro_in_accum == grad_accum) or (mb_idx == micro_batches_per_epoch)
            if not is_step_boundary:
                continue

            lr_now = compute_lr(global_step, total_steps, warmup_steps, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            micro_in_accum = 0
            global_step += 1

            if global_step % args.log_every == 0 or global_step == total_steps:
                mean_train_loss = float((loss_acc / float(max(1, steps_since_log))).item())
                loss_tok_count = int(weight_acc.item())
                dt = max(1e-6, time.perf_counter() - t_log_start)
                print(
                    f"  ep={epoch:>2}/{args.epochs} step={global_step:>5}/{total_steps} | "
                    f"lr={lr_now:.2e} | train_loss={mean_train_loss:.4f} | "
                    f"loss_tok/s={loss_tok_count / dt:,.0f}"
                )
                loss_acc.zero_()
                weight_acc.zero_()
                steps_since_log = 0
                t_log_start = time.perf_counter()

            if args.eval_every > 0 and (global_step % args.eval_every == 0) and global_step < total_steps:
                run_eval_and_checkpoint(epoch, global_step)

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_early = True
                break

        # Ewaluacja na koniec epoki lub przy wcześniejszym zatrzymaniu
        run_eval_and_checkpoint(epoch, global_step)
        if stop_early:
            break

    os.makedirs(args.out_dir, exist_ok=True)
    hist_path = os.path.join(args.out_dir, "train_history.json")
    with open(hist_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"cfg": cfg.to_dict(), "history": history}, ensure_ascii=False, indent=2) + "\n")

    # Załaduj najlepszy checkpoint (INT8) do ewaluacji jakościowej
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    best_model = TartarusModel(cfg).to(device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])

    val_jsonl = os.path.join(args.data_dir, "val.jsonl")
    if os.path.exists(val_jsonl):
        evaluate_qualitative_int8(best_model, tokenizer, val_jsonl, per_cat=args.qual_per_cat)

    print(
        f"[KONIEC] best_val_loss_int8={best_val_loss_int8:.4f} | "
        f"best_val_loss_float={best_val_loss_float:.4f} | checkpointy w: {os.path.abspath(args.out_dir)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
