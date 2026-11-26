/**
 * verify_tokenizer.ts — Ścisły test zgodności 1:1 (TypeScript <-> Python)
 * dla modułu `tartarus_tokenizer.ts`.
 */

import fs from "node:fs";
import path from "node:path";
import { TartarusTokenizer, type TartarusTokenizerConfig } from "./tartarus_tokenizer.js";

interface ParityCase {
  name: string;
  text: string;
  token_ids: number[];
  expected_decode_keep_pad: string;
  expected_decode_skip_pad: string;
}

interface ParityPayload {
  format: string;
  vocab_size: number;
  num_cases: number;
  cases: ParityCase[];
}

function assertArraysEqual(a: readonly number[], b: readonly number[], label: string): void {
  if (a.length !== b.length) {
    throw new Error(`[${label}] Różna długość tablic: TS=${a.length} !== PY=${b.length}\nTS=${JSON.stringify(a)}\nPY=${JSON.stringify(b)}`);
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      throw new Error(
        `[${label}] Niezgodność tokenu na indeksie ${i}: TS=${a[i]} !== PY=${b[i]}\nTS=${JSON.stringify(a)}\nPY=${JSON.stringify(b)}`
      );
    }
  }
}

export function runParitySuite(baseDir: string = "."): void {
  const tokPath = path.join(baseDir, "tokenizer.json");
  const parityPath = path.join(baseDir, "tokenizer_parity.json");

  const config = JSON.parse(fs.readFileSync(tokPath, "utf-8")) as TartarusTokenizerConfig;
  const parity = JSON.parse(fs.readFileSync(parityPath, "utf-8")) as ParityPayload;

  const tsTok = new TartarusTokenizer(config);
  if (tsTok.vocabSize !== parity.vocab_size) {
    throw new Error(`Niezgodność vocab_size: TS=${tsTok.vocabSize} !== PY=${parity.vocab_size}`);
  }

  for (const c of parity.cases) {
    const tsIds = tsTok.encode(c.text);
    assertArraysEqual(tsIds, c.token_ids, `encode:${c.name}`);

    const decKeep = tsTok.decode(c.token_ids, { skipPad: false });
    if (decKeep !== c.text) {
      throw new Error(`[decodeKeepPad:${c.name}] TS=${JSON.stringify(decKeep)} !== TEXT=${JSON.stringify(c.text)}`);
    }

    const decSkip = tsTok.decode(c.token_ids, { skipPad: true });
    if (decSkip !== c.expected_decode_skip_pad) {
      throw new Error(
        `[decodeSkipPad:${c.name}] TS=${JSON.stringify(decSkip)} !== PY=${JSON.stringify(c.expected_decode_skip_pad)}`
      );
    }
  }

  // Dodatkowy test konfiguracji minimalnej (jak z TNY3 w Etapie 5 — bez opcjonalnego pola `vocab`)
  const minimalConfig: TartarusTokenizerConfig = {
    format: config.format,
    vocab_size: config.vocab_size,
    special_tokens: config.special_tokens,
    byte_offset: config.byte_offset,
    pretokenize_pattern: config.pretokenize_pattern,
    pretokenize_flags: config.pretokenize_flags,
    merges: config.merges,
  };
  const minimalTok = new TartarusTokenizer(minimalConfig);
  const sample = parity.cases[parity.cases.length - 1];
  assertArraysEqual(minimalTok.encode(sample.text), sample.token_ids, "minimal_tny3_config");

  console.log(`PASS [TS Parity]: ${parity.cases.length}/${parity.cases.length} przypadków 100% zgodnych z Pythonem!`);

  // Weryfikacja pełnych splitów val i test wprost z binarnych plików CSR (jeśli istnieją)
  for (const split of ["val", "test"] as const) {
    const jsonlPath = path.join(baseDir, `${split}.jsonl`);
    const tokBinPath = path.join(baseDir, `${split}_tokens.bin`);
    const offBinPath = path.join(baseDir, `${split}_offsets.bin`);

    if (!fs.existsSync(jsonlPath) || !fs.existsSync(tokBinPath) || !fs.existsSync(offBinPath)) {
      continue;
    }

    const lines = fs
      .readFileSync(jsonlPath, "utf-8")
      .split("\n")
      .filter((l: string) => l.trim().length > 0)
      .map((l: string) => (JSON.parse(l) as { text: string }).text);

    const tokBuf = fs.readFileSync(tokBinPath);
    const offBuf = fs.readFileSync(offBinPath);
    const tokensU16 = new Uint16Array(tokBuf.buffer, tokBuf.byteOffset, tokBuf.byteLength / 2);
    const offsetsU64 = new BigUint64Array(offBuf.buffer, offBuf.byteOffset, offBuf.byteLength / 8);

    if (offsetsU64.length !== lines.length + 1) {
      throw new Error(`[CSR:${split}] offsets.length=${offsetsU64.length} !== ${lines.length + 1}`);
    }

    for (let i = 0; i < lines.length; i++) {
      const start = Number(offsetsU64[i]);
      const end = Number(offsetsU64[i + 1]);
      const expectedSlice = Array.from(tokensU16.subarray(start, end));
      const tsIds = tsTok.encode(lines[i]);
      assertArraysEqual(tsIds, expectedSlice, `csr:${split}[${i}]`);
      const dec = tsTok.decode(expectedSlice, { skipPad: false });
      if (dec !== lines[i]) {
        throw new Error(`[csr_decode:${split}[${i}]] Błąd round-trip UTF-8`);
      }
    }
    console.log(`PASS [TS vs CSR ${split}.bin]: ${lines.length}/${lines.length} próbek 100% zgodnych co do bajtu!`);
  }
}

runParitySuite(".");
