#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_tartarus_dataset.py — Skrypt pre-tokenizujący dane z Etapu 1 (`train.jsonl`,
`val.jsonl`, `test.jsonl`) do płaskich plików binarnych CSR dla GPU:
    - {split}_tokens.bin  (<u2 / uint16 little-endian, bez <|pad|>)
    - {split}_mask.bin    (u1 / uint8, 0 od <|state|> do <|npc|> włącznie, 1 do <|eos|>)
    - {split}_offsets.bin (<u8 / uint64 little-endian, długość N + 1)
    - dataset_meta.json   (tokenizer_sha256, rozmiary, statystyki długości i asercje)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from tokenizer_tartarus import TartarusTokenizer, encode_dataset_csr


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenizuje zbiory Etapu 1 do formatu binarnego CSR dla GPU."
    )
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Ścieżka do tokenizer.json")
    parser.add_argument("--data-dir", type=str, default=".", help="Katalog z train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--out-dir", type=str, default=".", help="Katalog wyjściowy dla plików .bin i dataset_meta.json")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splity do spakowania (domyślnie: train val test)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tokenizer = TartarusTokenizer.load(args.tokenizer)
    meta = encode_dataset_csr(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        tokenizer_path=args.tokenizer,
        splits=tuple(args.splits),
    )

    print("=" * 76)
    print("TARTARUS — PRE-TOKENIZACJA CSR DLA GPU ZAKOŃCZONA POMYŚLNIE")
    print("=" * 76)
    print(f"Tokenizer       : {os.path.abspath(args.tokenizer)} (vocab_size={meta['vocab_size']})")
    print(f"Tokenizer SHA256: {meta['tokenizer_sha256']}")
    print()
    for split, info in meta["splits"].items():
        st = info["seq_len_stats"]
        files = info["files"]
        print(f"[{split.upper()}] N={info['num_samples']:,} próbek | T_total={info['total_tokens']:,} tokenów")
        print(
            f"  mask=0 (prompt): {info['prompt_tokens_mask0']:,} | "
            f"mask=1 (loss)  : {info['loss_tokens_mask1']:,} ({100.0 * info['loss_token_ratio']:.1f}%)"
        )
        print(
            f"  Długości (tok) : min={st['min']}, mean={st['mean']:.1f}, "
            f"p95={st['p95']}, max={st['max']} (<=256: {st['pct_le_256']:.1f}%)"
        )
        print(
            f"  Pliki          : {files['tokens']['filename']} ({files['tokens']['bytes']:,} B), "
            f"{files['mask']['filename']} ({files['mask']['bytes']:,} B), "
            f"{files['offsets']['filename']} ({files['offsets']['bytes']:,} B)"
        )
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
