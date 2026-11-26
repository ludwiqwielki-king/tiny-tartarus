#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_dante_dataset.py — Edukacyjny podgląd zbioru Dantego (Etap 2, katalog DANTE/stage2_data).

Wypisuje:
  1. Statystyki z dataset_meta.json (T_total, min/śr./max długości na split, kontrola max <= 256),
  2. Rozpad przykładowych słów/fraz dantejskich na tokeny (ile tokenów i jakie),
  3. Jeden pełny rekord rozpisany token-po-token z wartością loss_mask (0 vs 1).

Uruchomienie (z katalogu DANTE/stage2_data):
    PYTHONPATH=/workspace python inspect_dante_dataset.py
"""

from __future__ import annotations

import json
import os
import sys

from tokenizer_tartarus import TartarusTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(HERE, "dante_tokenizer.json")
META_PATH = os.path.join(HERE, "dataset_meta.json")

WORDS = [
    "Wergiliusz",
    "Beatrycze",
    "Malebolge",
    "Francesca da Rimini",
    "contrapasso",
    "Lucyfer",
]


def hr(title: str) -> None:
    print()
    print("═" * 96)
    print(f" {title}")
    print("═" * 96)


def section_stats() -> None:
    hr("1) STATYSTYKI ZBIORU (dataset_meta.json)")
    with open(META_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    print(f"format            : {meta['format']}")
    print(f"tokenizer_file    : {meta['tokenizer_file']}")
    print(f"tokenizer_sha256  : {meta['tokenizer_sha256'][:32]}...")
    print(f"vocab_size        : {meta['vocab_size']}")
    print(f"dtypes            : {meta['dtypes']} | endianness={meta['endianness']}")
    print(f"loss_mask_rule    : {meta['loss_mask_rule']}")
    print()
    header = f"{'split':<6} {'N próbek':>9} {'T_total':>10} {'mask=1 (loss)':>14} {'min':>5} {'mean':>7} {'p95':>5} {'max':>5} {'max<=256?':>10}"
    print(header)
    print("-" * len(header))
    all_ok = True
    for split, info in meta["splits"].items():
        st = info["seq_len_stats"]
        ok = st["max"] <= 256
        all_ok &= ok
        print(
            f"{split:<6} {info['num_samples']:>9,} {info['total_tokens']:>10,} "
            f"{info['loss_tokens_mask1']:>10,} ({100*info['loss_token_ratio']:4.1f}%) "
            f"{st['min']:>5} {st['mean']:>7.1f} {st['p95']:>5} {st['max']:>5} {'TAK ✔' if ok else 'NIE ✘':>9}"
        )
    grand = sum(info["total_tokens"] for info in meta["splits"].values())
    print("-" * len(header))
    print(f"SUMA tokenów (wszystkie splity): {grand:,}")
    print(f"WERYFIKACJA OKNA 256: {'WSZYSTKIE SPLITY MIESZCZĄ SIĘ W max <= 256 ✔' if all_ok else 'UWAGA: przekroczenie 256!'}")


def section_words(tok: TartarusTokenizer) -> None:
    hr("2) ROZPAD PRZYKŁADOWYCH SŁÓW/FAZ NA TOKENY (słownik dantecki, vocab=2048)")
    for w in WORDS:
        pieces = tok.inspect_tokens(w)
        ids = tok.encode(w)
        print(f"\n  {w!r}")
        print(f"    liczba tokenów : {len(ids)}  (bajty UTF-8: {len(w.encode('utf-8'))})")
        print(f"    kawałki        : {pieces}")
        # weryfikacja round-trip
        assert tok.decode(ids, skip_pad=False) == w, f"round-trip nieudany dla {w!r}"
    print("\n  (decode(ids) == oryginał dla każdej frazy — round-trip UTF-8 OK)")


def section_record(tok: TartarusTokenizer) -> None:
    hr("3) PEŁNY REKORD TOKEN-PO-TOKEN Z LOSS_MASK (0 = prompt, 1 = liczone w stracie)")
    with open(os.path.join(HERE, "train.jsonl"), "r", encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    text = row["text"]
    ids = tok.encode(text)

    # Odtworzenie maski dokładnie wg reguły z encode_record_with_mask / dataset_meta.json
    NPC_ID = tok.special_tokens["<|npc|>"]
    first_npc = ids.index(NPC_ID)
    mask = [0] * (first_npc + 1) + [1] * (len(ids) - (first_npc + 1))

    pieces = tok.inspect_tokens(text)
    print(f"  kategoria rekordu : {row.get('category')}, split={row.get('split')}")
    print(f"  surowy tekst      : {text}\n")
    hdr = f"  {'idx':>4} {'id':>5} {'piece':<14} {'mask':>4}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, (tid, piece, m) in enumerate(zip(ids, pieces, mask)):
        flag = "◀ koniec PROMPTU (<|npc|>)" if tid == NPC_ID else ("◀ <|eos|> (mask=1)" if i == len(ids) - 1 else "")
        print(f"  {i:>4} {tid:>5} {piece:<14} {m:>4}  {flag}")
    print(f"\n  Podsumowanie: {len(ids)} tokenów | mask=0: {sum(1 for m in mask if m == 0)} "
          f"(state+user+npc) | mask=1: {sum(1 for m in mask if m == 1)} (odpowiedź npc + eos)")


def main() -> int:
    tok = TartarusTokenizer.load(TOKENIZER_PATH)
    assert tok.format_version_check() if hasattr(tok, "format_version_check") else True
    print(f"Załadowano tokenizer: {TOKENIZER_PATH} (vocab_size={tok.vocab_size})")
    section_stats()
    section_words(tok)
    section_record(tok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
