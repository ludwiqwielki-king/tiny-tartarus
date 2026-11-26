#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokenizer_tartarus.py — Deterministyczny Byte-Level BPE Tokenizer dla projektu
„Tartarus: The Fall of the Ancients” (czysty Python, 1:1 zgodność z TypeScriptem).

Specyfikacja słownika:
    0 : <|pad|>
    1 : <|eos|>
    2 : <|state|>
    3 : <|user|>
    4 : <|npc|>
    5..260 : 256 surowych bajtów UTF-8 (0..255 -> ID = 5 + bajt)
    261..vocab_size-1 : wyuczone scalenia BPE (merges) wyłącznie na train.jsonl

Kluczowa zasada zgodności Python <-> TypeScript:
    - Tokeny specjalne są wydzielane PRZED pre-tokenizacją i NIGDY nie trafiają do BPE.
    - Pre-tokenizacja NIE używa klas \\w, \\d ani \\s (ponieważ \\w w Pythonie re łapie
      polskie znaki, a w JS /\\w/ wyłącznie ASCII). Stosuje wyłącznie jawne klasy znaków
      identyczne w Pythonie `re` oraz JS `RegExp(..., "gu")`.
    - Dekodowanie buforuje bajty kolejnych tokenów nie-specjalnych i dopiero na końcu
      (lub przed tokenem specjalnym) wywołuje `.decode("utf-8", errors="replace")`.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# STAŁE FORMATU I PRE-TOKENIZACJI (1:1 PYTHON <-> TYPESCRIPT)
# ═══════════════════════════════════════════════════════════════════════════════

TOKENIZER_FORMAT_VERSION = "tartarus-bpe-v1"
DATASET_FORMAT_VERSION = "tartarus-csr-v1"

PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|eos|>"
STATE_TOKEN = "<|state|>"
USER_TOKEN = "<|user|>"
NPC_TOKEN = "<|npc|>"

SPECIAL_TOKENS: Tuple[str, ...] = (
    PAD_TOKEN,    # 0
    EOS_TOKEN,    # 1
    STATE_TOKEN,  # 2
    USER_TOKEN,   # 3
    NPC_TOKEN,    # 4
)
SPECIAL_TOKEN_TO_ID: Dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}

PAD_ID = 0
EOS_ID = 1
STATE_ID = 2
USER_ID = 3
NPC_ID = 4

BYTE_OFFSET = len(SPECIAL_TOKENS)   # 5
BASE_VOCAB_SIZE = BYTE_OFFSET + 256   # 261

# Jawna klasa liter (ASCII + polskie znaki diakrytyczne + Latin-1 w tym ö/Ö z nazw Jötunn/Völva).
# Zero użycia \w / \d / \s!
LETTERS_CLASS = "A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźżÀ-ÖØ-öø-ÿ"
PRETOKENIZE_PATTERN = (
    f" ?[{LETTERS_CLASS}]+"
    f"| ?[0-9]{{1,3}}"
    f"| ?[^{LETTERS_CLASS}0-9 \\t\\r\\n]"
    f"|[ \\t\\r\\n]+"
)
PRETOKENIZE_FLAGS = "gu"


def _build_special_regex(special_tokens: Dict[str, int]) -> Optional[re.Pattern[str]]:
    if not special_tokens:
        return None
    # Najdłuższe tokeny najpierw (longest-match-first)
    ordered = sorted(special_tokens.keys(), key=lambda s: (-len(s), s))
    return re.compile("(" + "|".join(re.escape(tok) for tok in ordered) + ")")


# ═══════════════════════════════════════════════════════════════════════════════
# KLASA TOKENIZERA BYTE-LEVEL BPE
# ═══════════════════════════════════════════════════════════════════════════════


class TartarusTokenizer:
    """
    Deterministyczny Byte-Level BPE Tokenizer bez ciężkich zależności.
    Struktura wewnętrzna odzwierciedla wprost `tartarus_tokenizer.ts`.
    """

    def __init__(
        self,
        special_tokens: Dict[str, int],
        byte_offset: int,
        pretokenize_pattern: str,
        merges: Sequence[Sequence[int]],
        requested_vocab_size: Optional[int] = None,
    ) -> None:
        self.special_tokens: Dict[str, int] = dict(special_tokens)
        self.id_to_special: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self.byte_offset: int = int(byte_offset)
        self.pretokenize_pattern: str = pretokenize_pattern

        if self.byte_offset != len(self.special_tokens):
            raise ValueError(
                f"byte_offset ({self.byte_offset}) musi równać się liczbie tokenów specjalnych ({len(self.special_tokens)})"
            )
        for tok, expected_id in SPECIAL_TOKEN_TO_ID.items():
            if self.special_tokens.get(tok) != expected_id:
                raise ValueError(f"Oczekiwano {tok}={expected_id}, otrzymano {self.special_tokens.get(tok)}")

        self.pad_id: int = self.special_tokens["<|pad|>"]
        self.eos_id: int = self.special_tokens["<|eos|>"]
        self.state_id: int = self.special_tokens["<|state|>"]
        self.user_id: int = self.special_tokens["<|user|>"]
        self.npc_id: int = self.special_tokens["<|npc|>"]

        self.special_re: Optional[re.Pattern[str]] = _build_special_regex(self.special_tokens)
        self.pretokenize_re: re.Pattern[str] = re.compile(self.pretokenize_pattern)

        # Tablica surowych bajtów dla każdego ID >= byte_offset
        self.id_to_bytes: List[bytes] = [b""] * self.byte_offset + [bytes([b]) for b in range(256)]
        self.merges: List[Tuple[int, int, int]] = []
        self.merges_rank: Dict[Tuple[int, int], Tuple[int, int]] = {}

        base_end = self.byte_offset + 256
        for rank, entry in enumerate(merges):
            if len(entry) == 2:
                left, right = int(entry[0]), int(entry[1])
                new_id = base_end + rank
            elif len(entry) == 3:
                left, right, new_id = int(entry[0]), int(entry[1]), int(entry[2])
            else:
                raise ValueError(f"Nieprawidłowy wpis merge w pozycji {rank}: {entry}")

            expected_new_id = base_end + rank
            if new_id != expected_new_id:
                raise ValueError(f"Niespójne ID scalenia w pozycji {rank}: {new_id} != {expected_new_id}")
            if not (self.byte_offset <= left < new_id and self.byte_offset <= right < new_id):
                raise ValueError(f"Nieprawidłowi rodzice merge ({left}, {right}) dla ID={new_id}")

            self.merges.append((left, right, new_id))
            self.merges_rank[(left, right)] = (rank, new_id)
            self.id_to_bytes.append(self.id_to_bytes[left] + self.id_to_bytes[right])

        self.vocab_size: int = len(self.id_to_bytes)
        self.requested_vocab_size: int = (
            int(requested_vocab_size) if requested_vocab_size is not None else self.vocab_size
        )
        self._chunk_cache: Dict[str, Tuple[int, ...]] = {}

    @classmethod
    def train(
        cls,
        train_texts: Iterable[str],
        vocab_size: int = 2048,
        verbose: bool = False,
    ) -> Tuple["TartarusTokenizer", Dict[str, Any]]:
        """Trenuje tokenizer BPE wyłącznie na podanych tekstach treningowych."""
        merges, stats = train_bpe_merges(train_texts, max_vocab_size=vocab_size, verbose=verbose)
        tok = cls(
            special_tokens=SPECIAL_TOKEN_TO_ID,
            byte_offset=BYTE_OFFSET,
            pretokenize_pattern=PRETOKENIZE_PATTERN,
            merges=merges,
            requested_vocab_size=vocab_size,
        )
        return tok, stats

    def split_special(self, text: str) -> List[str]:
        if not text:
            return []
        if self.special_re is None:
            return [text]
        return [seg for seg in self.special_re.split(text) if seg]

    def pretokenize_segment(self, segment: str) -> List[str]:
        if not segment:
            return []
        return self.pretokenize_re.findall(segment)

    def _encode_chunk(self, chunk: str) -> Tuple[int, ...]:
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            return cached

        ids = [self.byte_offset + b for b in chunk.encode("utf-8")]
        merges_rank = self.merges_rank

        while len(ids) >= 2:
            best_rank: Optional[int] = None
            best_left = 0
            best_right = 0
            best_new_id = 0

            for i in range(len(ids) - 1):
                info = merges_rank.get((ids[i], ids[i + 1]))
                if info is not None:
                    rank, new_id = info
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_left = ids[i]
                        best_right = ids[i + 1]
                        best_new_id = new_id

            if best_rank is None:
                break

            next_ids: List[int] = []
            i = 0
            n = len(ids)
            while i < n:
                if i + 1 < n and ids[i] == best_left and ids[i + 1] == best_right:
                    next_ids.append(best_new_id)
                    i += 2
                else:
                    next_ids.append(ids[i])
                    i += 1
            ids = next_ids

        res = tuple(ids)
        if len(self._chunk_cache) < 50000:
            self._chunk_cache[chunk] = res
        return res

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        out: List[int] = []
        for seg in self.split_special(text):
            spec_id = self.special_tokens.get(seg)
            if spec_id is not None:
                out.append(spec_id)
                continue
            for chunk in self.pretokenize_re.findall(seg):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(
        self,
        ids: Sequence[int],
        skip_pad: bool = True,
        skip_special: bool = False,
    ) -> str:
        out_parts: List[str] = []
        byte_buf = bytearray()

        def flush_bytes() -> None:
            if byte_buf:
                out_parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for raw_tid in ids:
            tid = int(raw_tid)
            if tid < 0 or tid >= self.vocab_size:
                raise ValueError(f"Token ID poza zakresem słownika [0, {self.vocab_size}): {tid}")
            if tid < self.byte_offset:
                flush_bytes()
                if tid == self.pad_id and skip_pad:
                    continue
                if not skip_special:
                    out_parts.append(self.id_to_special[tid])
            else:
                byte_buf.extend(self.id_to_bytes[tid])

        flush_bytes()
        return "".join(out_parts)

    def inspect_tokens(self, text: str) -> List[str]:
        """Zwraca czytelną listę kawałków (tylko do diagnostyki — pojedynczy bajt daje hex lub U+FFFD)."""
        ids = self.encode(text)
        pieces: List[str] = []
        for tid in ids:
            if tid < self.byte_offset:
                pieces.append(self.id_to_special[tid])
            else:
                b = self.id_to_bytes[tid]
                try:
                    pieces.append(b.decode("utf-8", errors="strict"))
                except UnicodeDecodeError:
                    pieces.append(f"<0x{b.hex()}>")
        return pieces

    def to_dict(self) -> Dict[str, Any]:
        vocab_entries: List[Dict[str, Any]] = []
        for tid in range(self.vocab_size):
            if tid < self.byte_offset:
                vocab_entries.append(
                    {
                        "id": tid,
                        "type": "special",
                        "token": self.id_to_special[tid],
                    }
                )
            elif tid < self.byte_offset + 256:
                raw = self.id_to_bytes[tid]
                vocab_entries.append(
                    {
                        "id": tid,
                        "type": "byte",
                        "hex": raw.hex(),
                        "text": raw.decode("utf-8", errors="replace"),
                    }
                )
            else:
                rank = tid - (self.byte_offset + 256)
                left, right, _ = self.merges[rank]
                raw = self.id_to_bytes[tid]
                vocab_entries.append(
                    {
                        "id": tid,
                        "type": "merge",
                        "pair": [left, right],
                        "hex": raw.hex(),
                        "text": raw.decode("utf-8", errors="replace"),
                    }
                )

        return {
            "format": TOKENIZER_FORMAT_VERSION,
            "vocab_size": self.vocab_size,
            "requested_vocab_size": self.requested_vocab_size,
            "special_tokens": dict(self.special_tokens),
            "byte_offset": self.byte_offset,
            "pretokenize_pattern": self.pretokenize_pattern,
            "pretokenize_flags": PRETOKENIZE_FLAGS,
            "merges": [[left, right, new_id] for left, right, new_id in self.merges],
            "vocab": vocab_entries,
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
        return sha256_file(path)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TartarusTokenizer":
        fmt = cfg.get("format")
        if fmt != TOKENIZER_FORMAT_VERSION:
            raise ValueError(f"Nieobsługiwany format tokenizera: {fmt!r} (oczekiwano {TOKENIZER_FORMAT_VERSION!r})")
        return cls(
            special_tokens=cfg["special_tokens"],
            byte_offset=int(cfg["byte_offset"]),
            pretokenize_pattern=cfg["pretokenize_pattern"],
            merges=cfg["merges"],
            requested_vocab_size=cfg.get("requested_vocab_size"),
        )

    @classmethod
    def load(cls, path: str) -> "TartarusTokenizer":
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cls.from_dict(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# SZYBKI, DETERMINISTYCZNY TRENING BPE (DOKŁADNY INDEKS ODWRÓCONY PAR)
# ═══════════════════════════════════════════════════════════════════════════════


def train_bpe_merges(
    train_texts: Iterable[str],
    max_vocab_size: int,
    verbose: bool = False,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, Any]]:
    """
    Trenuje uporządkowaną listę merges BPE na podanym zbiorze tekstów.
    Używa dokładnej aktualizacji liczników par przez indeks odwrócony (100% identycznej
    z pełnym skanem brute-force na każdym kroku, z deterministycznym rozstrzyganiem remisów).
    """
    if max_vocab_size < BASE_VOCAB_SIZE:
        raise ValueError(f"vocab_size musi wynosić co najmniej {BASE_VOCAB_SIZE}, podano {max_vocab_size}")

    special_re = _build_special_regex(SPECIAL_TOKEN_TO_ID)
    pretokenize_re = re.compile(PRETOKENIZE_PATTERN)

    word_freqs: collections.Counter[Tuple[int, ...]] = collections.Counter()
    total_records = 0
    total_chars = 0
    total_pretokens = 0
    total_base_bytes = 0

    for text in train_texts:
        total_records += 1
        total_chars += len(text)
        segments = [seg for seg in special_re.split(text) if seg] if special_re else [text]
        for seg in segments:
            if seg in SPECIAL_TOKEN_TO_ID:
                continue
            for chunk in pretokenize_re.findall(seg):
                ids_tuple = tuple(BYTE_OFFSET + b for b in chunk.encode("utf-8"))
                word_freqs[ids_tuple] += 1
                total_pretokens += 1
                total_base_bytes += len(ids_tuple)

    words: List[List[int]] = [list(w) for w in word_freqs.keys()]
    counts: List[int] = list(word_freqs.values())

    pair_counts: collections.Counter[Tuple[int, int]] = collections.Counter()
    pair_words: Dict[Tuple[int, int], set[int]] = collections.defaultdict(set)

    for w_idx, (w, freq) in enumerate(zip(words, counts)):
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            pair_counts[p] += freq
            pair_words[p].add(w_idx)

    target_merges = max_vocab_size - BASE_VOCAB_SIZE
    merges: List[Tuple[int, int, int]] = []
    merge_frequencies: List[int] = []
    saturated_at: Optional[int] = None

    t0 = time.perf_counter()
    for step in range(target_merges):
        if not pair_counts:
            saturated_at = BASE_VOCAB_SIZE + step
            break

        # Deterministyczny wybór najczęstszej pary (remisy rozstrzygane leksykograficznie po (left, right))
        best_pair, best_freq = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if best_freq <= 0:
            saturated_at = BASE_VOCAB_SIZE + step
            break

        new_id = BASE_VOCAB_SIZE + step
        left, right = best_pair
        merges.append((left, right, new_id))
        merge_frequencies.append(best_freq)

        affected_indices = list(pair_words.pop(best_pair, ()))
        for w_idx in affected_indices:
            old_w = words[w_idx]
            freq = counts[w_idx]

            new_w: List[int] = []
            i = 0
            n = len(old_w)
            while i < n:
                if i + 1 < n and old_w[i] == left and old_w[i + 1] == right:
                    new_w.append(new_id)
                    i += 2
                else:
                    new_w.append(old_w[i])
                    i += 1

            if new_w == old_w:
                continue

            for j in range(len(old_w) - 1):
                p = (old_w[j], old_w[j + 1])
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                if p != best_pair and p in pair_words:
                    pair_words[p].discard(w_idx)
                    if not pair_words[p]:
                        del pair_words[p]

            words[w_idx] = new_w
            for j in range(len(new_w) - 1):
                p = (new_w[j], new_w[j + 1])
                pair_counts[p] += freq
                pair_words[p].add(w_idx)

        if verbose and ((step + 1) % 500 == 0 or step + 1 == target_merges):
            elapsed = time.perf_counter() - t0
            print(f"  [BPE] step {step + 1}/{target_merges} | last_freq={best_freq} | {elapsed:.2f}s")

    train_time = time.perf_counter() - t0
    stats = {
        "train_records": total_records,
        "train_chars": total_chars,
        "total_pretokens": total_pretokens,
        "unique_pretokens": len(word_freqs),
        "total_base_bytes": total_base_bytes,
        "requested_vocab_size": max_vocab_size,
        "effective_vocab_size": BASE_VOCAB_SIZE + len(merges),
        "num_merges": len(merges),
        "saturated_at": saturated_at,
        "merge_frequencies": merge_frequencies,
        "train_seconds": train_time,
    }
    return merges, stats


# ═══════════════════════════════════════════════════════════════════════════════
# POMOCNICZE I/O, EKSPERYMENT PORÓWNAWCZY ORAZ PRZYKŁADY PODZIAŁU
# ═══════════════════════════════════════════════════════════════════════════════


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Błąd JSON w {path}:{line_no}: {exc}") from exc
            if "text" not in obj:
                raise ValueError(f"Brak pola 'text' w {path}:{line_no}")
            rows.append(obj)
    return rows


def compute_seq_stats(lengths: Sequence[int], total_chars: int) -> Dict[str, Any]:
    if not lengths:
        raise ValueError("Pusta lista długości sekwencji")
    s_lens = sorted(lengths)
    total_tokens = sum(s_lens)
    n = len(s_lens)
    p95_idx = int(0.95 * (n - 1))
    fit_256 = sum(1 for x in s_lens if x <= 256)
    fit_512 = sum(1 for x in s_lens if x <= 512)
    return {
        "num_samples": n,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "chars_per_token": total_chars / total_tokens,
        "min": s_lens[0],
        "mean": statistics.mean(s_lens),
        "median": statistics.median(s_lens),
        "p95": s_lens[p95_idx],
        "max": s_lens[-1],
        "pct_le_256": 100.0 * fit_256 / n,
        "pct_le_512": 100.0 * fit_512 / n,
    }


def evaluate_tokenizer_on_texts(tokenizer: TartarusTokenizer, texts: Sequence[str]) -> Dict[str, Any]:
    lengths: List[int] = []
    total_chars = 0
    for txt in texts:
        total_chars += len(txt)
        ids = tokenizer.encode(txt)
        lengths.append(len(ids))
    return compute_seq_stats(lengths, total_chars)


def run_vocab_experiment(
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    sizes: Sequence[int] = (1024, 1536, 2048, 3072, 4096),
    selected_vocab: int = 2048,
) -> Dict[str, Any]:
    """
    Przeprowadza eksperyment porównawczy na jednym, deterministycznym przebiegu BPE
    wytrenowanym wyłącznie na train_texts (ponieważ merges BPE są prefiksowo monotoniczne).
    """
    sorted_sizes = sorted(set(int(s) for s in sizes))
    max_size = max(max(sorted_sizes), int(selected_vocab))

    all_merges, train_stats = train_bpe_merges(train_texts, max_vocab_size=max_size, verbose=False)
    merge_freqs = train_stats["merge_frequencies"]

    print("=" * 98)
    print("TARTARUS — EKSPERYMENT PORÓWNAWCZY ROZMIARU SŁOWNIKA BYTE-LEVEL BPE")
    print("=" * 98)
    print(
        f"Korpus treningowy  : {train_stats['train_records']:,} próbek | "
        f"{train_stats['train_chars']:,} znaków | "
        f"{train_stats['total_pretokens']:,} pre-tokenów ({train_stats['unique_pretokens']:,} unikalnych)"
    )
    print(
        f"Sufit słownika BPE : {train_stats['effective_vocab_size']} tokenów "
        f"({train_stats['num_merges']} merges; czas treningu: {train_stats['train_seconds']:.2f}s)"
    )
    if train_stats["saturated_at"] is not None:
        print(
            f"Uwaga saturacji    : Wszystkie pary wewnątrz pre-tokenów w train wyczerpały się przy "
            f"vocab={train_stats['saturated_at']}!"
        )
    print()

    results: List[Dict[str, Any]] = []
    header = (
        f"{'Target':>6} {'EffV':>6} {'LastFreq':>8} | "
        f"{'Train c/t':>9} {'mean/p95/max':>15} | "
        f"{'Val c/t':>7} {'mean/p95/max':>15} | "
        f"{'Gap Δc/t':>10} | {'Emb D=192':>9} {'Emb D=256':>9}"
    )
    print(header)
    print("-" * len(header))

    for target_v in sorted_sizes:
        n_merges = min(len(all_merges), max(0, target_v - BASE_VOCAB_SIZE))
        tok = TartarusTokenizer(
            special_tokens=SPECIAL_TOKEN_TO_ID,
            byte_offset=BYTE_OFFSET,
            pretokenize_pattern=PRETOKENIZE_PATTERN,
            merges=all_merges[:n_merges],
            requested_vocab_size=target_v,
        )
        eff_v = tok.vocab_size
        last_freq = merge_freqs[n_merges - 1] if n_merges > 0 else 0

        tr = evaluate_tokenizer_on_texts(tok, train_texts)
        va = evaluate_tokenizer_on_texts(tok, val_texts)
        gap_abs = tr["chars_per_token"] - va["chars_per_token"]
        gap_pct = 100.0 * gap_abs / tr["chars_per_token"]
        emb_192 = eff_v * 192
        emb_256 = eff_v * 256

        tr_len_str = f"{tr['mean']:.1f}/{tr['p95']}/{tr['max']}"
        va_len_str = f"{va['mean']:.1f}/{va['p95']}/{va['max']}"
        gap_str = f"{gap_abs:+.3f}({gap_pct:4.1f}%)"

        marker = " *" if target_v == selected_vocab else ""
        print(
            f"{target_v:>6} {eff_v:>6} {last_freq:>8} | "
            f"{tr['chars_per_token']:>9.3f} {tr_len_str:>15} | "
            f"{va['chars_per_token']:>7.3f} {va_len_str:>15} | "
            f"{gap_str:>10} | {emb_192/1000:>7.1f}k {emb_256/1000:>8.1f}k{marker}"
        )

        results.append(
            {
                "requested_vocab_size": target_v,
                "effective_vocab_size": eff_v,
                "num_merges": n_merges,
                "last_merge_freq": last_freq,
                "train": tr,
                "val": va,
                "compression_gap_abs": gap_abs,
                "compression_gap_pct": gap_pct,
                "embedding_params_d192": emb_192,
                "embedding_params_d256": emb_256,
            }
        )

    print("-" * len(header))
    print("(* = domyślny/wybrany rozmiar słownika do eksportu)")
    print()

    # Szczegółowe rozkłady dla każdego progu
    print("[SZCZEGÓŁOWY ROZKŁAD DŁUGOŚCI I KOMPRESJI (TRAIN vs VAL)]")
    for row in results:
        tr = row["train"]
        va = row["val"]
        sat_note = (
            f" [SATURACJA @ {row['effective_vocab_size']}]"
            if row["effective_vocab_size"] < row["requested_vocab_size"]
            else ""
        )
        print(
            f"  vocab={row['requested_vocab_size']} (eff={row['effective_vocab_size']}{sat_note}): "
            f"TRAIN[min={tr['min']}, mean={tr['mean']:.1f}, med={tr['median']:.0f}, p95={tr['p95']}, max={tr['max']}, <=256:{tr['pct_le_256']:.1f}%] | "
            f"VAL[min={va['min']}, mean={va['mean']:.1f}, med={va['median']:.0f}, p95={va['p95']}, max={va['max']}, <=256:{va['pct_le_256']:.1f}%]"
        )
    print()

    # Zbuduj tokenizer dla wybranego progu i wypisz ok. 20 przykładów podziału
    sel_merges = all_merges[: min(len(all_merges), max(0, selected_vocab - BASE_VOCAB_SIZE))]
    selected_tok = TartarusTokenizer(
        special_tokens=SPECIAL_TOKEN_TO_ID,
        byte_offset=BYTE_OFFSET,
        pretokenize_pattern=PRETOKENIZE_PATTERN,
        merges=sel_merges,
        requested_vocab_size=selected_vocab,
    )
    print_split_examples(selected_tok)

    return {
        "train_corpus_stats": {k: v for k, v in train_stats.items() if k != "merge_frequencies"},
        "results": results,
        "selected_vocab_size": selected_tok.vocab_size,
        "selected_tokenizer": selected_tok,
    }


def print_split_examples(tokenizer: TartarusTokenizer) -> None:
    examples: List[Tuple[str, str]] = [
        # Polskie słowa i końcówki fleksyjne
        ("polskie fleksje", "niezwyciężonymi oddziałami wojowników"),
        ("polskie ogonki", "Zażółć gęślą jaźń — żółty źrebięć"),
        ("polskie wielkie", "ĄĆĘŁŃÓŚŹŻ ąćęłńóśźż Jötunn Völva"),
        ("polska deklinacja", "Berserker, Berserkera, Berserkerowi, Berserkerem, Berserkerze"),
        ("polska czasownik", "wskrzesić wskrzesimy wskazuje wytrzymałość"),
        # Terminy Tartarus (train)
        ("Tartarus lore 1", "Gaj Świętogora i Nekropolia Siedmiu Klątw"),
        ("Tartarus lore 2", "Żmij Gorynycz oraz Bagienna Dziwożaba"),
        ("Tartarus lore 3", "Kapłan Swaroga rzuca Grom Peruna (30–45 obrażeń)"),
        ("Tartarus lore 4", "Egida Ateny i Święty Kielich Żywota"),
        ("Tartarus lore 5", "Chram Płomiennego Swaroga — koszt 120 złota"),
        # Fragmenty <|state|>
        ("state pełny", "<|state|> lokacja=Czarny Kruk; typ=zamek; teren=wzgórza; tura=25"),
        ("state zasoby", "hp=15/75; mana=16/20; złoto=390; ruch=5"),
        ("state jednostki", "jednostki=Miotacz Ognia Greckiegox3,Katafrakt Varangianinx1"),
        ("state wrogowie", "wrogowie=Królestwo Egiptu(starcie); dowódca=poległ"),
        ("state ujemny", "złoto=-250; wrogowie=Hellada(4p)"),
        # Odłożone nazwy z VAL (holdout)
        ("VAL frakcja", "Cesarstwo Bizantyńskie pod wodzą Basileusa Konstantyna XI"),
        ("VAL jednostka 1", "Wrony Odyna i Strzelec Samosieczki"),
        ("VAL jednostka 2", "Żywiołak Kamienny oraz Balsamista Anubisa"),
        ("VAL jednostka 3", "Czarny Mnich, Cyklop Brązowy i Dromona Cesarska"),
        ("VAL lokacja", "Kazamaty Czarnego Bazaltu i Krypta Pradawnych Architektów"),
        ("VAL czar/artefakt", "Grzęzawisko Mokradeł oraz Topór Krwawego Jarla"),
        # Odłożone nazwy z TEST (holdout)
        ("TEST nazwy", "Völva Wieszczka, Kołdun Czarnoksiężnik i Krwawa Harpia"),
    ]

    print(f"[PRZYKŁADY PODZIAŁU NA TOKENY (vocab={tokenizer.vocab_size}, {len(examples)} przykładów)]")
    for label, text in examples:
        ids = tokenizer.encode(text)
        pieces = tokenizer.inspect_tokens(text)
        decoded = tokenizer.decode(ids, skip_pad=False)
        assert decoded == text, f"Błąd round-trip dla przykładu {label}: {decoded!r} != {text!r}"
        print(f"  [{label:<18}] ({len(ids):>2} tok) {text!r}")
        print(f"      -> {pieces}")
    print("=" * 98)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PAKOWANIE DO FORMATU BINARNEGO CSR DLA GPU + PEŁNA WERYFIKACJA INTEGRALNOŚCI
# ═══════════════════════════════════════════════════════════════════════════════


def encode_record_with_mask(
    tokenizer: TartarusTokenizer,
    text: str,
) -> Tuple[List[int], List[int]]:
    """
    Koduje pojedynczy rekord `<|state|> ... <|user|> ... <|npc|> ... <|eos|>`
    oraz wyznacza dokładną maskę straty:
        0: dla wszystkich tokenów od <|state|> do <|npc|> włącznie
        1: dla wszystkich tokenów odpowiedzi od pierwszego po <|npc|> do <|eos|> włącznie
    """
    ids = tokenizer.encode(text)

    if not ids:
        raise ValueError("Pusty rekord po tokenizacji")
    if PAD_ID in ids:
        raise ValueError(f"Wykryto zakazany token <|pad|> (ID=0) w rekordzie: {text!r}")
    if ids[0] != STATE_ID:
        raise ValueError(f"Rekord nie zaczyna się od <|state|> (ID={STATE_ID}): {text!r}")
    if ids[-1] != EOS_ID:
        raise ValueError(f"Rekord nie kończy się na <|eos|> (ID={EOS_ID}): {text!r}")

    state_positions = [i for i, t in enumerate(ids) if t == STATE_ID]
    user_positions = [i for i, t in enumerate(ids) if t == USER_ID]
    npc_positions = [i for i, t in enumerate(ids) if t == NPC_ID]
    eos_positions = [i for i, t in enumerate(ids) if t == EOS_ID]

    if len(state_positions) != 1 or len(user_positions) != 1 or len(npc_positions) != 1 or len(eos_positions) != 1:
        raise ValueError(
            f"Nieprawidłowa krotność tagów specjalnych "
            f"(state={len(state_positions)}, user={len(user_positions)}, "
            f"npc={len(npc_positions)}, eos={len(eos_positions)}): {text!r}"
        )

    pos_s, pos_u, pos_n, pos_e = state_positions[0], user_positions[0], npc_positions[0], eos_positions[0]
    if not (0 == pos_s < pos_u < pos_n < pos_e == len(ids) - 1):
        raise ValueError(f"Nieprawidłowa kolejność tagów ({pos_s}, {pos_u}, {pos_n}, {pos_e}): {text!r}")
    if pos_n + 1 >= pos_e:
        raise ValueError(f"Pusta odpowiedź pomiędzy <|npc|> a <|eos|>: {text!r}")

    # 0 od <|state|> do <|npc|> włącznie; 1 od pierwszego tokenu po <|npc|> do <|eos|> włącznie
    mask = [0] * (pos_n + 1) + [1] * (len(ids) - (pos_n + 1))
    assert len(mask) == len(ids)
    return ids, mask


def encode_dataset_csr(
    tokenizer: TartarusTokenizer,
    data_dir: str,
    out_dir: str,
    tokenizer_path: str,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, Any]:
    """
    Pre-tokenizuje pliki {split}.jsonl do surowego formatu CSR:
        {split}_tokens.bin  (<u2, uint16 little-endian, brak <|pad|>)
        {split}_mask.bin    (u1, uint8)
        {split}_offsets.bin (<u8, uint64 little-endian, długość N+1)
    Oraz zapisuje `dataset_meta.json` i wykonuje pełną weryfikację po odczycie z dysku.
    """
    if tokenizer.vocab_size > 65535:
        raise ValueError(f"vocab_size={tokenizer.vocab_size} przekracza zakres uint16 (65535)")

    os.makedirs(out_dir, exist_ok=True)
    tok_sha256 = sha256_file(tokenizer_path)

    split_meta: Dict[str, Any] = {}
    loaded_texts_by_split: Dict[str, List[str]] = {}

    for split in splits:
        in_path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Brak pliku wejściowego Etapu 1: {in_path}")

        rows = load_jsonl(in_path)
        texts = [r["text"] for r in rows]
        loaded_texts_by_split[split] = texts

        all_tokens: List[int] = []
        all_masks: List[int] = []
        offsets: List[int] = [0]
        seq_lengths: List[int] = []
        total_chars = 0

        for txt in texts:
            total_chars += len(txt)
            ids, mask = encode_record_with_mask(tokenizer, txt)
            all_tokens.extend(ids)
            all_masks.extend(mask)
            offsets.append(len(all_tokens))
            seq_lengths.append(len(ids))

        tokens_arr = np.asarray(all_tokens, dtype=np.dtype("<u2"))
        mask_arr = np.asarray(all_masks, dtype=np.dtype("u1"))
        offsets_arr = np.asarray(offsets, dtype=np.dtype("<u8"))

        tokens_path = os.path.join(out_dir, f"{split}_tokens.bin")
        mask_path = os.path.join(out_dir, f"{split}_mask.bin")
        offsets_path = os.path.join(out_dir, f"{split}_offsets.bin")

        tokens_arr.tofile(tokens_path)
        mask_arr.tofile(mask_path)
        offsets_arr.tofile(offsets_path)

        t_total = int(tokens_arr.size)
        n_samples = len(texts)
        loss_tokens = int(np.sum(mask_arr, dtype=np.int64))
        prompt_tokens = t_total - loss_tokens
        seq_stats = compute_seq_stats(seq_lengths, total_chars)

        split_meta[split] = {
            "source_file": os.path.basename(in_path),
            "num_samples": n_samples,
            "total_tokens": t_total,
            "prompt_tokens_mask0": prompt_tokens,
            "loss_tokens_mask1": loss_tokens,
            "loss_token_ratio": loss_tokens / t_total if t_total else 0.0,
            "seq_len_stats": seq_stats,
            "files": {
                "tokens": {
                    "filename": os.path.basename(tokens_path),
                    "dtype": "<u2",
                    "bytes": os.path.getsize(tokens_path),
                    "sha256": sha256_file(tokens_path),
                },
                "mask": {
                    "filename": os.path.basename(mask_path),
                    "dtype": "u1",
                    "bytes": os.path.getsize(mask_path),
                    "sha256": sha256_file(mask_path),
                },
                "offsets": {
                    "filename": os.path.basename(offsets_path),
                    "dtype": "<u8",
                    "bytes": os.path.getsize(offsets_path),
                    "sha256": sha256_file(offsets_path),
                },
            },
        }

    meta = {
        "format": DATASET_FORMAT_VERSION,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tokenizer_file": os.path.basename(tokenizer_path),
        "tokenizer_sha256": tok_sha256,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": dict(tokenizer.special_tokens),
        "dtypes": {
            "tokens": "<u2",
            "mask": "u1",
            "offsets": "<u8",
        },
        "endianness": "little",
        "loss_mask_rule": "0 for tokens from <|state|> through <|npc|> inclusive; 1 for tokens after <|npc|> through <|eos|> inclusive",
        "splits": split_meta,
    }

    meta_path = os.path.join(out_dir, "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    verify_csr_dataset(
        tokenizer=tokenizer,
        out_dir=out_dir,
        meta=meta,
        expected_texts_by_split=loaded_texts_by_split,
    )
    return meta


def verify_csr_dataset(
    tokenizer: TartarusTokenizer,
    out_dir: str,
    meta: Optional[Dict[str, Any]] = None,
    expected_texts_by_split: Optional[Dict[str, List[str]]] = None,
) -> None:
    """
    Odczytuje zapisane pliki .bin bezpośrednio przez np.memmap i wykonuje pełny zestaw asercji.
    """
    if meta is None:
        with open(os.path.join(out_dir, "dataset_meta.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    vocab_size = int(meta["vocab_size"])
    assert vocab_size == tokenizer.vocab_size, f"Niezgodność vocab_size: {vocab_size} != {tokenizer.vocab_size}"

    for split, s_info in meta["splits"].items():
        n_samples = int(s_info["num_samples"])
        t_total = int(s_info["total_tokens"])

        tok_file = os.path.join(out_dir, s_info["files"]["tokens"]["filename"])
        msk_file = os.path.join(out_dir, s_info["files"]["mask"]["filename"])
        off_file = os.path.join(out_dir, s_info["files"]["offsets"]["filename"])

        assert os.path.getsize(tok_file) == 2 * t_total, f"{tok_file}: zły rozmiar pliku!"
        assert os.path.getsize(msk_file) == 1 * t_total, f"{msk_file}: zły rozmiar pliku!"
        assert os.path.getsize(off_file) == 8 * (n_samples + 1), f"{off_file}: zły rozmiar pliku!"

        tokens = np.memmap(tok_file, dtype=np.dtype("<u2"), mode="r")
        mask = np.memmap(msk_file, dtype=np.dtype("u1"), mode="r")
        offsets = np.memmap(off_file, dtype=np.dtype("<u8"), mode="r")

        assert tokens.shape == (t_total,)
        assert mask.shape == (t_total,)
        assert offsets.shape == (n_samples + 1,)

        assert int(offsets[0]) == 0, f"{split}: offsets[0] != 0"
        assert int(offsets[-1]) == t_total, f"{split}: offsets[-1] != T_total"
        assert bool(np.all(offsets[1:] > offsets[:-1])), f"{split}: offsets nie są ściśle rosnące"

        # Zakaz <|pad|> w plikach CSR, zakres [1, vocab_size - 1]
        assert not bool(np.any(tokens == PAD_ID)), f"{split}: wykryto <|pad|> w tokens.bin!"
        assert bool(np.all(tokens < vocab_size)), f"{split}: ID poza zakresem < {vocab_size}!"
        assert bool(np.all((mask == 0) | (mask == 1))), f"{split}: nieprawidłowe wartości w mask.bin!"

        # Pełna weryfikacja granic <|state|> .. <|npc|> .. <|eos|> oraz round-trip UTF-8 dla każdej próbki
        expected_texts = expected_texts_by_split.get(split) if expected_texts_by_split else None
        for i in range(n_samples):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            seq = tokens[start:end]
            m_seq = mask[start:end]

            assert int(seq[0]) == STATE_ID, f"{split}[{i}]: brak <|state|> na początku"
            assert int(seq[-1]) == EOS_ID, f"{split}[{i}]: brak <|eos|> na końcu"

            npc_idxs = np.where(seq == NPC_ID)[0]
            assert npc_idxs.size == 1, f"{split}[{i}]: oczekiwano dokładnie 1x <|npc|>"
            npc_idx = int(npc_idxs[0])

            assert bool(np.all(m_seq[: npc_idx + 1] == 0)), f"{split}[{i}]: mask!=0 przed/na <|npc|>"
            assert bool(np.all(m_seq[npc_idx + 1 :] == 1)), f"{split}[{i}]: mask!=1 po <|npc|>"

            if expected_texts is not None:
                decoded = tokenizer.decode(seq.tolist(), skip_pad=False)
                assert decoded == expected_texts[i], f"{split}[{i}]: błąd round-trip UTF-8!"


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR PRZYPADKÓW TESTÓW ZGODNOŚCI PYTHON <-> TYPESCRIPT (tokenizer_parity.json)
# ═══════════════════════════════════════════════════════════════════════════════


def generate_parity_cases(
    tokenizer: TartarusTokenizer,
    data_dir: str,
    out_path: str,
) -> Dict[str, Any]:
    """
    Generuje co najmniej 100 zróżnicowanych przypadków testowych pokrywających wszystkie
    pułapki między Pythonem a JS/TS (polskie znaki, wielkie/małe litery, pojedyncze
    litery, interpunkcja —/–/„”/=/;, liczby, nazwy holdout z val/test, przypadki brzegowe
    UTF-8 oraz pełne rekordy <|state|>...<|eos|>).
    """
    raw_cases: List[Tuple[str, str]] = [
        # 1. Podstawowe i brzegowe
        ("empty", ""),
        ("single_space", " "),
        ("multi_space", "   "),
        ("whitespace_mix", "a\tb\nc\r\nd  e"),
        # 2. Pojedyncze polskie litery z ogonkami + wielkie/małe + nordyckie ö
        ("pl_a_ogonek", "ą"),
        ("pl_c_acute", "ć"),
        ("pl_e_ogonek", "ę"),
        ("pl_l_stroke", "ł"),
        ("pl_n_acute", "ń"),
        ("pl_o_acute", "ó"),
        ("pl_s_acute", "ś"),
        ("pl_z_acute", "ź"),
        ("pl_z_dot", "ż"),
        ("pl_upper_A", "Ą"),
        ("pl_upper_C", "Ć"),
        ("pl_upper_E", "Ę"),
        ("pl_upper_L", "Ł"),
        ("pl_upper_N", "Ń"),
        ("pl_upper_O", "Ó"),
        ("pl_upper_S", "Ś"),
        ("pl_upper_Z_acute", "Ź"),
        ("pl_upper_Z_dot", "Ż"),
        ("nordic_o_umlaut", "ö"),
        ("nordic_upper_O_umlaut", "Ö"),
        # 3. Ciągi polskich znaków, pangramy, wielkie/małe
        ("pl_all_lower", "ąćęłńóśźż"),
        ("pl_all_upper", "ĄĆĘŁŃÓŚŹŻ"),
        ("pl_pangram_lower", "zażółć gęślą jaźń"),
        ("pl_pangram_upper", "ZAŻÓŁĆ GĘŚLĄ JAŹŃ"),
        ("pl_pangram_title", "Zażółć Gęślą Jaźń"),
        ("pl_tongue_twister", "Chrząszcz brzmi w trzcinie w Szczebrzeszynie."),
        ("pl_inflection_1", "niezwyciężonymi oddziałami łuczników i berserkerów"),
        ("pl_inflection_2", "Świętogórze, Świętogora, Świętogorowi, Świętogorem"),
        ("pl_leading_trailing_space", " łąka źródło ćma "),
        # 4. Znaki specjalne typograficzne i interpunkcja Tartarus
        ("em_dash", " — "),
        ("en_dash_range", "30–45 obrażeń (6–10 w 1. rundzie)"),
        ("polish_quotes", "„Sanktuarium Światła i Ładu” — wpis z kroniki."),
        ("punct_combo", "= ; / ( ) + - % ! ? : , . „ ” — –"),
        ("negative_numbers", "złoto=-40; hp=0/75; bonus=(-3 Obrony); atak=+6"),
        ("large_numbers", "0 1 7 15 100 990 1024 2048 1234567890"),
        ("hp_mana_ratios", "hp=15/75; mana=16/20; ruch=8"),
        # 5. Tokeny specjalne (parsowane PRZED BPE)
        ("only_pad", "<|pad|>"),
        ("only_eos", "<|eos|>"),
        ("only_state", "<|state|>"),
        ("only_user", "<|user|>"),
        ("only_npc", "<|npc|>"),
        ("adjacent_specials", "<|pad|><|state|><|user|><|npc|><|eos|>"),
        ("padded_record_prefix", "<|pad|><|pad|><|state|> hp=10/70 <|user|> Ile? <|npc|> 10. <|eos|>"),
        ("fake_special_lookalike", "<|state|> lokacja=<|nieznany|> <|user|> test <|stat|> <|npc|> ok <|eos|>"),
        # 6. Złożone ciągi UTF-8 spoza alfabetu polskiego (weryfikacja 4-bajtowego UTF-8 i symboli)
        ("unicode_symbols", "Koszt: 150 zł… lub 20 €!"),
        ("unicode_emoji_4byte", "Szturm ⚔️ na warownię 🛡️! 😊"),
        # 7. Nazwy encji holdout z VAL
        ("val_faction", "Cesarstwo Bizantyńskie"),
        ("val_leader", "Basileus Konstantyn XI"),
        ("val_unit_1", "Wrony Odyna"),
        ("val_unit_2", "Strzelec Samosieczki"),
        ("val_unit_3", "Żywiołak Kamienny"),
        ("val_unit_4", "Balsamista Anubisa"),
        ("val_unit_5", "Czarny Mnich"),
        ("val_unit_6", "Cyklop Brązowy"),
        ("val_unit_7", "Dromona Cesarska"),
        ("val_loc_1", "Twierdza Tartar"),
        ("val_loc_2", "Wieża Wichrów"),
        ("val_loc_3", "Żelazny Most"),
        ("val_loc_4", "Sanktuarium Bogów Olimpu"),
        ("val_loc_5", "Krypta Pradawnych Architektów"),
        ("val_loc_6", "Kazamaty Czarnego Bazaltu"),
        ("val_spell_1", "Aura Natchnienia"),
        ("val_spell_2", "Grzęzawisko Mokradeł"),
        ("val_art_1", "Topór Krwawego Jarla"),
        ("val_art_2", "Kryształowe Oko Tartaru"),
        ("val_unknown_1", "Zeppelin Cesarski"),
        ("val_unknown_2", "Golem Laserowy"),
        ("val_unknown_3", "Księga Cybernetyki"),
        # 8. Nazwy encji holdout z TEST
        ("test_faction", "Hellada — Król Leonidas Niezłomny"),
        ("test_unit_1", "Völva Wieszczka"),
        ("test_unit_2", "Kołdun Czarnoksiężnik"),
        ("test_unit_3", "Strzyga Żelazna"),
        ("test_unit_4", "Awiz Krwawy"),
        ("test_unit_5", "Gargulec Hipodromu"),
        ("test_unit_6", "Święty Likwidator"),
        ("test_unit_7", "Krwawa Harpia"),
        ("test_unit_8", "Mantyka Wyrocznia"),
        ("test_unit_9", "Trirema Hellady"),
        ("test_loc_1", "Burzowa Twierdza"),
        ("test_loc_2", "Srebrna Przystań"),
        ("test_loc_3", "Wzgórze Cieni"),
        ("test_loc_4", "Kaplica Mroźnego Boreasza"),
        ("test_loc_5", "Czeluść Strąconych Bóstw"),
        ("test_loc_6", "Zgliszcza Podniebnej Cytadeli"),
        ("test_spell_1", "Grad Ognisty"),
        ("test_spell_2", "Całun Grozy"),
        ("test_art_1", "Włócznia Słoneczna Ra"),
        ("test_art_2", "Skarabeusz Nieśmiertelności"),
        ("test_unknown_1", "Paladyn Atomowy"),
        ("test_unknown_2", "Zaklęcie Teleportacji Masowej"),
        # 9. Złożone frazy mieszane (state + dialog)
        ("mix_state_compact", "<|state|> lokacja=Jötunn Mroźny; złoto=-500; wrogowie=brak"),
        ("mix_state_units", "jednostki=Völva Wieszczkax2,Żmij Gorynyczx1; wrogowie=Hellada(8p)"),
        ("mix_prompt_inference", "<|state|> lokacja=Twierdza Tartar; hp=12/75; mana=2/16; złoto=35; ruch=0; jednostki=Cyklop Brązowyx2; wrogowie=brak <|user|> Raportuj: czy stać nas na Żywiołaka Kamiennego? <|npc|>"),
    ]

    # Dodaj po 4 pełne rekordy (po 1 z każdej kategorii) z train, val oraz test -> +12 rekordów
    for split in ("train", "val", "test"):
        split_path = os.path.join(data_dir, f"{split}.jsonl")
        if os.path.exists(split_path):
            rows = load_jsonl(split_path)
            seen_cats = set()
            for r in rows:
                cat = r.get("category", "unknown")
                if cat not in seen_cats:
                    seen_cats.add(cat)
                    raw_cases.append((f"full_{split}_{cat}", r["text"]))
                if len(seen_cats) == 4:
                    break
            # Plus kilka dodatkowych rekordów z val i test
            if split in ("val", "test"):
                for extra_idx in (10, 50, 100, 250, 500):
                    if extra_idx < len(rows):
                        raw_cases.append((f"full_{split}_idx{extra_idx}", rows[extra_idx]["text"]))

    cases_out: List[Dict[str, Any]] = []
    for name, text in raw_cases:
        ids = tokenizer.encode(text)
        dec_keep_pad = tokenizer.decode(ids, skip_pad=False)
        dec_skip_pad = tokenizer.decode(ids, skip_pad=True)
        assert dec_keep_pad == text, f"Błąd round-trip w przypadku testowym {name}: {dec_keep_pad!r} != {text!r}"
        cases_out.append(
            {
                "name": name,
                "text": text,
                "token_ids": ids,
                "expected_decode_keep_pad": dec_keep_pad,
                "expected_decode_skip_pad": dec_skip_pad,
            }
        )

    payload = {
        "format": TOKENIZER_FORMAT_VERSION,
        "vocab_size": tokenizer.vocab_size,
        "num_cases": len(cases_out),
        "cases": cases_out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trening, eksperyment porównawczy i pakowanie CSR dla tokenizera Tartarus Byte-Level BPE."
    )
    parser.add_argument("--data-dir", type=str, default=".", help="Katalog z train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--out-dir", type=str, default=".", help="Katalog wyjściowy dla tokenizer.json i plików .bin")
    parser.add_argument("--vocab-size", type=int, default=2048, help="Docelowy rozmiar słownika (domyślnie: 2048)")
    parser.add_argument(
        "--experiment-sizes",
        type=int,
        nargs="+",
        default=[1024, 1536, 2048, 3072, 4096],
        help="Rozmiary słownika do eksperymentu porównawczego (domyślnie: 1024 1536 2048 3072 4096)",
    )
    parser.add_argument("--tokenizer-out", type=str, default="tokenizer.json", help="Nazwa pliku wyjściowego tokenizera")
    parser.add_argument("--parity-out", type=str, default="tokenizer_parity.json", help="Nazwa pliku testowego parity TS/Python")
    parser.add_argument("--skip-experiment", action="store_true", help="Pomiń eksperyment porównawczy i trenuj tylko --vocab-size")
    parser.add_argument("--skip-csr", action="store_true", help="Pomiń zapis paczek binarnych CSR (.bin)")
    parser.add_argument("--skip-parity", action="store_true", help="Pomiń generację tokenizer_parity.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Brak pliku treningowego: {train_path}")

    train_rows = load_jsonl(train_path)
    train_texts = [r["text"] for r in train_rows]

    if not args.skip_experiment:
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Brak pliku walidacyjnego do eksperymentu: {val_path}")
        val_texts = [r["text"] for r in load_jsonl(val_path)]
        exp = run_vocab_experiment(
            train_texts=train_texts,
            val_texts=val_texts,
            sizes=args.experiment_sizes,
            selected_vocab=args.vocab_size,
        )
        tokenizer: TartarusTokenizer = exp["selected_tokenizer"]
    else:
        tokenizer, stats = TartarusTokenizer.train(train_texts, vocab_size=args.vocab_size, verbose=True)
        print(
            f"[TRAIN] Wygenerowano słownik vocab_size={tokenizer.vocab_size} "
            f"(requested={args.vocab_size}, merges={len(tokenizer.merges)}) w {stats['train_seconds']:.2f}s"
        )
        print_split_examples(tokenizer)

    tok_path = os.path.join(args.out_dir, args.tokenizer_out)
    tok_sha = tokenizer.save(tok_path)
    # Weryfikacja przeładowania z dysku
    reloaded = TartarusTokenizer.load(tok_path)
    assert reloaded.merges == tokenizer.merges
    assert reloaded.vocab_size == tokenizer.vocab_size
    print(f"[EXPORT] Zapisano {tok_path} (vocab_size={reloaded.vocab_size}, sha256={tok_sha[:16]}...)")

    if not args.skip_parity:
        parity_path = os.path.join(args.out_dir, args.parity_out)
        parity = generate_parity_cases(reloaded, args.data_dir, parity_path)
        print(f"[PARITY] Zapisano {parity_path} ({parity['num_cases']} przypadków testowych)")

    if not args.skip_csr:
        meta = encode_dataset_csr(
            tokenizer=reloaded,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            tokenizer_path=tok_path,
        )
        print("[CSR] Zapisano i zweryfikowano binarne paczki CSR dla GPU:")
        for sp, info in meta["splits"].items():
            st = info["seq_len_stats"]
            print(
                f"  [{sp.upper():<5}] N={info['num_samples']:>5} | T_total={info['total_tokens']:>7} | "
                f"mask=1: {info['loss_tokens_mask1']:>7} ({100*info['loss_token_ratio']:.1f}%) | "
                f"len mean={st['mean']:.1f}, p95={st['p95']}, max={st['max']} | "
                f"tokens={info['files']['tokens']['bytes']:,}B"
            )
        print(f"  [META ] {os.path.join(args.out_dir, 'dataset_meta.json')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
