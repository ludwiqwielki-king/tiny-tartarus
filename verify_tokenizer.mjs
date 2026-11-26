#!/usr/bin/env node
/**
 * verify_tokenizer.mjs — Test zgodności 1:1 Python <-> TypeScript dla `tartarus_tokenizer.ts`.
 * Działa w czystym Node.js (>= 18) bez zewnętrznych zależności.
 * Jeśli w środowisku dostępny jest pakiet `typescript`, automatycznie kompiluje w locie
 * `tartarus_tokenizer.ts`; w przeciwnym razie ładuje skompilowany `tartarus_tokenizer.mjs`.
 */

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

async function loadTartarusTokenizerClass(baseDir) {
  const tsPath = path.join(baseDir, "tartarus_tokenizer.ts");
  const mjsPath = path.join(baseDir, "tartarus_tokenizer.mjs");

  try {
    const tsMod = await import("typescript");
    const ts = tsMod.default ?? tsMod;
    const source = fs.readFileSync(tsPath, "utf-8");
    const transpiled = ts.transpileModule(source, {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.ES2022,
        strict: true,
      },
    });
    const dataUrl = "data:text/javascript;base64," + Buffer.from(transpiled.outputText, "utf-8").toString("base64");
    const mod = await import(dataUrl);
    return { TartarusTokenizer: mod.TartarusTokenizer, sourceMode: "live-transpiled tartarus_tokenizer.ts" };
  } catch {
    if (!fs.existsSync(mjsPath)) {
      throw new Error(
        `Nie znaleziono pakietu 'typescript' ani skompilowanego pliku ${mjsPath}. ` +
          `Zainstaluj typescript lub skompiluj tartarus_tokenizer.ts.`
      );
    }
    const mod = await import(pathToFileURL(mjsPath).href);
    return { TartarusTokenizer: mod.TartarusTokenizer, sourceMode: "compiled tartarus_tokenizer.mjs" };
  }
}

function assertArraysEqual(a, b, label) {
  if (a.length !== b.length) {
    throw new Error(
      `[${label}] Różna długość tablic: JS/TS=${a.length} !== PY=${b.length}\n` +
        `JS=${JSON.stringify(a)}\nPY=${JSON.stringify(b)}`
    );
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      throw new Error(
        `[${label}] Niezgodność tokenu na indeksie ${i}: JS/TS=${a[i]} !== PY=${b[i]}\n` +
          `JS=${JSON.stringify(a)}\nPY=${JSON.stringify(b)}`
      );
    }
  }
}

async function main() {
  const baseDir = process.argv[2] ?? ".";
  const { TartarusTokenizer, sourceMode } = await loadTartarusTokenizerClass(baseDir);

  const tokPath = path.join(baseDir, "tokenizer.json");
  const parityPath = path.join(baseDir, "tokenizer_parity.json");

  const config = JSON.parse(fs.readFileSync(tokPath, "utf-8"));
  const parity = JSON.parse(fs.readFileSync(parityPath, "utf-8"));

  const tsTok = new TartarusTokenizer(config);
  if (tsTok.vocabSize !== parity.vocab_size) {
    throw new Error(`Niezgodność vocab_size: TS=${tsTok.vocabSize} !== PY=${parity.vocab_size}`);
  }

  console.log(`Źródło modułu : ${sourceMode}`);
  console.log(`Vocab size    : ${tsTok.vocabSize}`);

  for (const c of parity.cases) {
    const tsIds = tsTok.encode(c.text);
    assertArraysEqual(tsIds, c.token_ids, `encode:${c.name}`);

    const decKeep = tsTok.decode(c.token_ids, { skipPad: false });
    if (decKeep !== c.text) {
      throw new Error(
        `[decodeKeepPad:${c.name}] TS=${JSON.stringify(decKeep)} !== TEXT=${JSON.stringify(c.text)}`
      );
    }

    const decSkip = tsTok.decode(c.token_ids, { skipPad: true });
    if (decSkip !== c.expected_decode_skip_pad) {
      throw new Error(
        `[decodeSkipPad:${c.name}] TS=${JSON.stringify(decSkip)} !== PY=${JSON.stringify(c.expected_decode_skip_pad)}`
      );
    }
  }

  // Test minimalnego obiektu konfiguracji (symulacja zasilenia z TNY3 w Etapie 5 bez pola `vocab`)
  const minimalTok = new TartarusTokenizer({
    format: config.format,
    vocab_size: config.vocab_size,
    special_tokens: config.special_tokens,
    byte_offset: config.byte_offset,
    pretokenize_pattern: config.pretokenize_pattern,
    pretokenize_flags: config.pretokenize_flags,
    merges: config.merges,
  });
  const lastCase = parity.cases[parity.cases.length - 1];
  assertArraysEqual(minimalTok.encode(lastCase.text), lastCase.token_ids, "minimal_tny3_config");
  if (minimalTok.decode(lastCase.token_ids, { skipPad: false }) !== lastCase.text) {
    throw new Error("Błąd decode dla minimalnej konfiguracji TNY3");
  }

  console.log(`PASS [Parity JSON]: ${parity.cases.length}/${parity.cases.length} przypadków testowych 100% zgodnych!`);

  // Pełna weryfikacja splitów holdout (val + test) przeciwko binarnym plikom CSR na dysku
  for (const split of ["val", "test"]) {
    const jsonlPath = path.join(baseDir, `${split}.jsonl`);
    const tokBinPath = path.join(baseDir, `${split}_tokens.bin`);
    const offBinPath = path.join(baseDir, `${split}_offsets.bin`);

    if (!fs.existsSync(jsonlPath) || !fs.existsSync(tokBinPath) || !fs.existsSync(offBinPath)) {
      continue;
    }

    const texts = fs
      .readFileSync(jsonlPath, "utf-8")
      .split("\n")
      .filter((l) => l.trim().length > 0)
      .map((l) => JSON.parse(l).text);

    const tokBuf = fs.readFileSync(tokBinPath);
    const offBuf = fs.readFileSync(offBinPath);
    const tokensU16 = new Uint16Array(tokBuf.buffer, tokBuf.byteOffset, tokBuf.byteLength / 2);
    const offsetsU64 = new BigUint64Array(offBuf.buffer, offBuf.byteOffset, offBuf.byteLength / 8);

    if (offsetsU64.length !== texts.length + 1) {
      throw new Error(`[CSR:${split}] offsets.length=${offsetsU64.length} !== ${texts.length + 1}`);
    }

    for (let i = 0; i < texts.length; i++) {
      const start = Number(offsetsU64[i]);
      const end = Number(offsetsU64[i + 1]);
      const expectedIds = Array.from(tokensU16.subarray(start, end));
      const tsIds = tsTok.encode(texts[i]);
      assertArraysEqual(tsIds, expectedIds, `csr:${split}[${i}]`);
      const decoded = tsTok.decode(expectedIds, { skipPad: false });
      if (decoded !== texts[i]) {
        throw new Error(`[csr_decode:${split}[${i}]] Błąd round-trip UTF-8!`);
      }
    }
    console.log(`PASS [CSR ${split}_tokens.bin]: ${texts.length}/${texts.length} pełnych rekordów 100% zgodnych!`);
  }
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
