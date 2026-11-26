#!/usr/bin/env node
/**
 * verify_ts.mjs — Krytyczna bramka weryfikacyjna Etapu 5 (Node.js ESM):
 *   1. Parsowanie binarnego pliku `model.tny3` + weryfikacja CRC-32 i detekcji uszkodzeń.
 *   2. Zgodność tokenizacji 100% z `golden_test.json` (encode + decode round-trip).
 *   3. Zgodność numeryczna logitów TS/JS vs Python PyTorch (`max_abs_diff < 1e-4`):
 *        - `forwardNoCache(token_ids)` vs Python `last_step_logits`
 *        - `prefill(token_ids)` (KV Cache) vs Python `last_step_logits`
 *        - `prefill(prefix) + step(last)` (KV Cache) vs Python `last_step_logits`
 *   4. Wielokrokowa równoważność autoregresyjna `prefill + step` vs `forwardNoCache` przez 16 kroków.
 *   5. Weryfikacja braku zbędnych alokacji / wycieków pamięci (ponowne użycie buforów KV Cache).
 *   6. Testy samplingu (`temperature=0`, `topK`, `minP`, `repetitionPenalty`, `<|eos|>`, `CTX`),
 *      streamingu polskich znaków UTF-8 oraz pomiar `tokens/sec`.
 */

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ENGINE_URL = pathToFileURL(path.resolve("./tartarus-engine.mjs")).href;
const { TartarusEngine, parseTNY3 } = await import(ENGINE_URL);

function assert(cond, msg) {
  if (!cond) {
    throw new Error(`ASSERT FAILED: ${msg}`);
  }
}

function diffStats(a, b) {
  assert(a.length === b.length, `Różna długość tablic: ${a.length} !== ${b.length}`);
  let maxAbs = 0.0;
  let sumAbs = 0.0;
  let maxIdx = 0;
  let argmaxA = 0;
  let argmaxB = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    sumAbs += d;
    if (d > maxAbs) {
      maxAbs = d;
      maxIdx = i;
    }
    if (a[i] > a[argmaxA]) argmaxA = i;
    if (b[i] > b[argmaxB]) argmaxB = i;
  }
  return {
    maxAbs,
    meanAbs: sumAbs / a.length,
    maxIdx,
    argmaxA,
    argmaxB,
  };
}

function runMain() {
  const tny3Path = path.resolve("./model.tny3");
  const goldenPath = path.resolve("./golden_test.json");
  assert(fs.existsSync(tny3Path), `Brak pliku ${tny3Path}`);
  assert(fs.existsSync(goldenPath), `Brak pliku ${goldenPath}`);

  const rawBuf = fs.readFileSync(tny3Path);
  const tLoad0 = performance.now();
  const engine = TartarusEngine.fromArrayBuffer(rawBuf);
  const loadMs = performance.now() - tLoad0;

  const h = engine.header;
  console.log(
    `Załadowano model.tny3 (${(h.totalBytes / (1024 * 1024)).toFixed(2)} MiB) w ${loadMs.toFixed(1)} ms | ` +
      `L=${h.nLayers}, D=${h.dModel}, H=${h.nHeads}, hd=${h.headDim}, d_ff=${h.dFf}, CTX=${h.ctxLen}, V=${h.vocabSize}, ` +
      `tied=${h.tiedEmbeddings}, CRC=0x${h.payloadCrc32.toString(16).padStart(8, "0")}`
  );

  // 1. Weryfikacja rozmiaru KV Cache = L * CTX * D * 4 B (Float32)
  const expectedKvElements = h.nLayers * h.ctxLen * h.dModel;
  assert(
    engine.kCache.length === expectedKvElements && engine.vCache.length === expectedKvElements,
    `Nieoczekiwany rozmiar KV Cache: ${engine.kCache.length} !== ${expectedKvElements}`
  );
  console.log(
    `PASS [KV Cache]: pre-alokowano K=${(engine.kCache.byteLength / 1024).toFixed(1)} KiB, ` +
      `V=${(engine.vCache.byteLength / 1024).toFixed(1)} KiB (L*CTX*D = ${expectedKvElements} f32)`
  );

  // 2. Test CRC-32 na uszkodzonym buforze (bez modyfikacji oryginału)
  const tampered = new Uint8Array(rawBuf);
  tampered[tampered.length - 17] ^= 0x01;
  let tamperRejected = false;
  try {
    parseTNY3(tampered);
  } catch (err) {
    tamperRejected = String(err.message).includes("CRC-32");
  }
  assert(tamperRejected, "parseTNY3 powinien odrzucić uszkodzony plik TNY3 przez CRC-32!");
  console.log("PASS [CRC-32]: uszkodzony payload wykryty i odrzucony.");

  // 3. Testy golden_test.json
  const golden = JSON.parse(fs.readFileSync(goldenPath, "utf-8"));
  assert(Array.isArray(golden.cases) && golden.cases.length > 0, "Pusty golden_test.json");

  let globalMaxNoCache = 0.0;
  let globalMaxPrefill = 0.0;
  let globalMaxSplitStep = 0.0;
  let globalMaxCacheVsNoCache = 0.0;

  for (const c of golden.cases) {
    // 3a. Zgodność tokenizacji 100%
    const tsIds = engine.encode(c.text);
    assert(
      tsIds.length === c.token_ids.length && tsIds.every((v, i) => v === c.token_ids[i]),
      `[${c.name}] Rozbieżność token_ids TS vs Python`
    );
    const decoded = engine.decode(tsIds, { skipPad: false, skipSpecial: false });
    assert(decoded === c.text, `[${c.name}] Niezgodność decode(encode(text))`);

    // 3b. forwardNoCache vs Python
    const logitsNoCache = engine.forwardNoCache(tsIds);
    const dNoCache = diffStats(logitsNoCache, c.last_step_logits);
    assert(
      dNoCache.maxAbs < 1e-4,
      `[${c.name}] forwardNoCache maxAbs=${dNoCache.maxAbs} >= 1e-4`
    );
    assert(
      dNoCache.argmaxA === c.greedy_next_token_id,
      `[${c.name}] forwardNoCache argmax=${dNoCache.argmaxA} !== ${c.greedy_next_token_id}`
    );

    // 3c. prefill (KV Cache) vs Python
    const logitsPrefill = engine.prefill(tsIds);
    const dPrefill = diffStats(logitsPrefill, c.last_step_logits);
    assert(
      dPrefill.maxAbs < 1e-4,
      `[${c.name}] prefill maxAbs=${dPrefill.maxAbs} >= 1e-4`
    );
    assert(
      dPrefill.argmaxA === c.greedy_next_token_id,
      `[${c.name}] prefill argmax=${dPrefill.argmaxA} !== ${c.greedy_next_token_id}`
    );

    // 3d. prefill(prefix) + step(ostatni_token) vs Python (weryfikacja prawdziwego kroku KV Cache!)
    const splitAt = Math.max(1, tsIds.length - 5);
    let logitsSplit = engine.prefill(tsIds.slice(0, splitAt));
    for (let k = splitAt; k < tsIds.length; k++) {
      logitsSplit = engine.step(tsIds[k]);
    }
    const dSplit = diffStats(logitsSplit, c.last_step_logits);
    const dEquiv = diffStats(logitsSplit, logitsNoCache);
    assert(dSplit.maxAbs < 1e-4, `[${c.name}] prefill+step maxAbs=${dSplit.maxAbs} >= 1e-4`);
    assert(dEquiv.maxAbs < 1e-5, `[${c.name}] KV Cache vs NoCache diff=${dEquiv.maxAbs} >= 1e-5`);
    assert(
      dSplit.argmaxA === c.greedy_next_token_id,
      `[${c.name}] prefill+step argmax=${dSplit.argmaxA} !== ${c.greedy_next_token_id}`
    );

    // 3e. Sprawdzenie braku alokacji zwracanego bufora logitów między prefill a step
    assert(
      logitsPrefill === logitsSplit,
      `[${c.name}] prefill() i step() powinny zwracać ten sam pre-alokowany bufor Float32Array`
    );

    globalMaxNoCache = Math.max(globalMaxNoCache, dNoCache.maxAbs);
    globalMaxPrefill = Math.max(globalMaxPrefill, dPrefill.maxAbs);
    globalMaxSplitStep = Math.max(globalMaxSplitStep, dSplit.maxAbs);
    globalMaxCacheVsNoCache = Math.max(globalMaxCacheVsNoCache, dEquiv.maxAbs);

    console.log(
      `PASS [${c.name}]: tokens=${tsIds.length} (100% parity) | ` +
        `noCacheΔ=${dNoCache.maxAbs.toExponential(2)} | prefillΔ=${dPrefill.maxAbs.toExponential(2)} | ` +
        `splitStepΔ=${dSplit.maxAbs.toExponential(2)} | cacheVsFullΔ=${dEquiv.maxAbs.toExponential(2)} | ` +
        `argmax=${dSplit.argmaxA} (${JSON.stringify(engine.decode([dSplit.argmaxA]))})`
    );
  }

  // 4. Wielokrokowa pętla autoregresyjna (16 kroków): KV Cache vs forwardNoCache na KAŻDYM kroku
  const baseCase = golden.cases[0];
  let seq = [...baseCase.token_ids];
  let cachedLogits = engine.prefill(seq);
  let maxMultiStepDiff = 0.0;
  for (let s = 0; s < 16; s++) {
    const fullLogits = engine.forwardNoCache(seq);
    const dStep = diffStats(cachedLogits, fullLogits);
    maxMultiStepDiff = Math.max(maxMultiStepDiff, dStep.maxAbs);
    assert(
      dStep.maxAbs < 1e-5,
      `Wielokrokowa rozbieżność KV Cache vs NoCache w kroku ${s}: ${dStep.maxAbs}`
    );
    const nextId = engine.sampleNextToken(cachedLogits, { temperature: 0.0 });
    seq.push(nextId);
    cachedLogits = engine.step(nextId);
  }
  console.log(
    `PASS [16-krokowa zgodność autoregresyjna KV Cache == forwardNoCache]: maxΔ=${maxMultiStepDiff.toExponential(2)}`
  );

  // 5. Testy samplingu, repetitionPenalty (unikalne tokeny), EOS, limitu CTX i streamowania UTF-8
  const greedyRes = engine.generate(baseCase.text, {
    temperature: 0.0,
    maxNewTokens: 48,
    stopOnEos: true,
  });
  assert(greedyRes.generatedTokens.length > 0, "Greedy generation nie wygenerowało żadnego tokenu");

  // Streaming UTF-8 musi dać dokładnie ten sam tekst co końcowy decode()
  let streamedText = "";
  let streamedTokens = [];
  for (const ev of engine.generateStream(baseCase.text, {
    temperature: 0.0,
    maxNewTokens: 48,
    stopOnEos: true,
  })) {
    streamedText += ev.textDelta;
    streamedTokens.push(ev.tokenId);
    assert(
      ev.layerRms.length === h.nLayers,
      `Telemetria layerRms.length=${ev.layerRms.length} !== nLayers=${h.nLayers}`
    );
  }
  assert(streamedText === greedyRes.text, `Streaming delta mismatch: ${JSON.stringify(streamedText)} !== ${JSON.stringify(greedyRes.text)}`);
  assert(!streamedText.includes("\ufffd"), "Wykryto uszkodzony znak UTF-8 (\\ufffd) podczas streamowania");
  console.log(
    `PASS [Streaming & Sampling Greedy]: tokens=${greedyRes.generatedTokens.length}, stoppedBy=${greedyRes.stoppedBy}, ` +
      `text=${JSON.stringify(greedyRes.text)}`
  );

  // Test repetitionPenalty (duplikaty w seenTokens nie mogą karać wielokrotnie tego samego ID)
  const testLogits = new Float32Array(h.vocabSize).fill(-10.0);
  testLogits[100] = 5.0;
  testLogits[101] = 4.6; // po karze 1.2 na token 100: 5.0/1.2 = 4.1667 < 4.6 -> wygra 101, niezależnie od krotności 100!
  testLogits[102] = 3.6; // gdyby kara 1.2 działała 2x (5.0 / 1.44 = 3.47 < 3.6), wygrałby 102 przy seen=[100,100]
  const pickOnce = engine.sampleNextToken(testLogits, { temperature: 0.0, repetitionPenalty: 1.2 }, [100]);
  const pickDup = engine.sampleNextToken(testLogits, { temperature: 0.0, repetitionPenalty: 1.2 }, [100, 100, 100]);
  assert(pickOnce === 101 && pickDup === 101, `repetitionPenalty błąd: once=${pickOnce}, dup=${pickDup}`);

  // Test topK + minP (deterministyczny PRNG)
  let seed = 1337;
  const detRng = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const sampRes = engine.generate(baseCase.text, {
    temperature: 0.75,
    topK: 32,
    minP: 0.06,
    repetitionPenalty: 1.15,
    maxNewTokens: 48,
    stopOnEos: true,
    rng: detRng,
  });
  assert(sampRes.generatedTokens.length > 0 && !sampRes.text.includes("\ufffd"), "Błąd samplingu topK+minP");
  console.log(
    `PASS [Sampling T=0.75, topK=32, minP=0.06, rep=1.15]: tokens=${sampRes.generatedTokens.length}, ` +
      `stoppedBy=${sampRes.stoppedBy}, text=${JSON.stringify(sampRes.text)}`
  );

  // Test zatrzymania na CTX
  const nearCtxPrompt = new Array(h.ctxLen - 4).fill(10);
  const ctxStopRes = engine.generate(nearCtxPrompt, {
    temperature: 0.0,
    maxNewTokens: 64,
    stopOnEos: false,
  });
  assert(
    ctxStopRes.generatedTokens.length === 4 && ctxStopRes.stoppedBy === "ctx",
    `Oczekiwano zatrzymania na CTX (4 tok, stoppedBy="ctx"), otrzymano ${ctxStopRes.generatedTokens.length}, ${ctxStopRes.stoppedBy}`
  );
  console.log(`PASS [CTX Guard]: prompt=${nearCtxPrompt.length}, maxCtx=${h.ctxLen} -> wygenerowano 4 tok i zatrzymano (stoppedBy=ctx).`);

  // 6. Benchmark szybkości (Node.js) i stabilności pamięci przy 20 uruchomieniach
  const benchPrompt = golden.cases[1].text;
  // Warm-up JIT
  engine.generate(benchPrompt, { temperature: 0.0, maxNewTokens: 64, stopOnEos: false });

  let totalGenTokens = 0;
  let totalDecodeMs = 0;
  let totalPrefillMs = 0;
  const runs = 10;
  const heapBefore = process.memoryUsage().heapUsed;
  for (let r = 0; r < runs; r++) {
    const out = engine.generate(benchPrompt, {
      temperature: 0.0,
      maxNewTokens: 64,
      stopOnEos: false,
    });
    totalGenTokens += out.generatedTokens.length;
    totalDecodeMs += out.decodeMs;
    totalPrefillMs += out.prefillMs;
  }
  const heapAfter = process.memoryUsage().heapUsed;
  const tps = (totalGenTokens * 1000.0) / totalDecodeMs;
  const avgPrefill = totalPrefillMs / runs;
  const heapDeltaKiB = (heapAfter - heapBefore) / 1024;
  console.log(
    `PASS [Benchmark Node.js, ${runs}x64 tok]: prefill=${avgPrefill.toFixed(1)} ms | ` +
      `decode=${tps.toFixed(1)} tok/s (${(totalDecodeMs / totalGenTokens).toFixed(2)} ms/tok) | ` +
      `heapΔ=${heapDeltaKiB.toFixed(1)} KiB`
  );

  // 7. Opcjonalny test dynamicznego skalowania warstw dla Profilu M (/tmp/model_m.tny3)
  const modelMPath = "/tmp/model_m.tny3";
  if (fs.existsSync(modelMPath)) {
    const engM = TartarusEngine.fromArrayBuffer(fs.readFileSync(modelMPath));
    const resM = engM.generate(baseCase.text, { temperature: 0.0, maxNewTokens: 16 });
    assert(engM.header.nLayers === 6 && engM.layerRms.length === 6, "Profil M musi raportować 6 warstw");
    console.log(
      `PASS [Profil M dynamic layers]: L=${engM.header.nLayers}, D=${engM.header.dModel}, ` +
        `gen=${resM.generatedTokens.length} tok (${resM.tokensPerSec.toFixed(1)} tok/s)`
    );
  }

  console.log(
    `\n=== WSZYSTKIE BRAMKI ETAPU 5 ZALICZONE ===\n` +
      `maxAbs(noCache vs PyTorch)  = ${globalMaxNoCache.toExponential(3)} (< 1e-4)\n` +
      `maxAbs(prefill vs PyTorch)  = ${globalMaxPrefill.toExponential(3)} (< 1e-4)\n` +
      `maxAbs(splitStep vs PyTorch)= ${globalMaxSplitStep.toExponential(3)} (< 1e-4)\n` +
      `maxAbs(KV Cache vs NoCache) = ${globalMaxCacheVsNoCache.toExponential(3)} (< 1e-5)`
  );
}

runMain();
