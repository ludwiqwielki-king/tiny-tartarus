/**
 * tartarus-engine.ts — Silnik inferencji W8A32 (INT8 na dysku -> Float32Array w RAM)
 * z pre-alokowanym KV Cache dla formatu binarnego TNY3 (0 zależności od DOM).
 *
 * Zaprojektowany w ścisłej zgodności 1:1 z `model_tartarus.py` i `export_tny3.py`:
 *   - RMSNorm (eps=1e-5, pre-norm + rms_final)
 *   - RoPE (theta=10000.0, konwencja half-split)
 *   - Causal Self-Attention z pre-alokowanym KV Cache [nLayers, ctxLen, dModel]
 *   - SwiGLU FFN: w_down(silu(w_gate(x)) * w_up(x))
 *   - Tied embeddings (lub jawny lm_head)
 */
import { TartarusTokenizer, } from "./tartarus_tokenizer.js";
export const TNY3_MAGIC = "TNY3";
export const TNY3_HEADER_BYTES = 64;
export const TNY3_FORMAT_VERSION = 1;
export const FLAG_TIED_EMBEDDINGS = 1 << 0;
export const FLAG_ROPE_INTERLEAVED = 1 << 1;
export const FLAG_SWIGLU = 1 << 2;
const EngineTextDecoderCtor = globalThis.TextDecoder;
function nowMs() {
    const perf = globalThis.performance;
    return perf !== undefined ? perf.now() : Date.now();
}
// Jednorazowa tablica CRC-32 (IEEE 802.3)
const CRC32_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
        let c = i;
        for (let k = 0; k < 8; k++) {
            c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        }
        table[i] = c >>> 0;
    }
    return table;
})();
export function computeCrc32(bytes) {
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) {
        crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
}
function readVectorF32(dv, offset, length) {
    if (offset % 4 !== 0) {
        throw new Error(`Niewyrównany offset wektora float32: ${offset}`);
    }
    const byteLen = length * 4;
    if (offset + byteLen > dv.byteLength) {
        throw new Error(`Poza zakresem bufora TNY3 przy odczycie wektora[${length}]`);
    }
    const vec = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        vec[i] = dv.getFloat32(offset + i * 4, true);
    }
    return { vec, nextOffset: offset + byteLen };
}
function readMatrixInt8Dequant(buffer, baseByteOffset, dv, offset, rows, cols) {
    if (cols % 4 !== 0) {
        throw new Error(`Liczba kolumn (${cols}) nie jest podzielna przez 4`);
    }
    if (offset % 4 !== 0) {
        throw new Error(`Niewyrównany offset macierzy INT8: ${offset}`);
    }
    const rowStride = 4 + cols;
    const totalBytes = rows * rowStride;
    if (offset + totalBytes > dv.byteLength) {
        throw new Error(`Poza zakresem bufora TNY3 przy odczycie macierzy [${rows}, ${cols}]`);
    }
    const mat = new Float32Array(rows * cols);
    let cursor = offset;
    for (let r = 0; r < rows; r++) {
        const scale = dv.getFloat32(cursor, true);
        const qRow = new Int8Array(buffer, baseByteOffset + cursor + 4, cols);
        const dstBase = r * cols;
        for (let c = 0; c < cols; c++) {
            mat[dstBase + c] = qRow[c] * scale;
        }
        cursor += rowStride;
    }
    return { mat, nextOffset: cursor };
}
export function parseTNY3(input) {
    const buffer = input instanceof ArrayBuffer ? input : input.buffer;
    const baseByteOffset = input instanceof ArrayBuffer ? 0 : input.byteOffset;
    const totalByteLength = input instanceof ArrayBuffer ? input.byteLength : input.byteLength;
    if (totalByteLength < TNY3_HEADER_BYTES) {
        throw new Error(`Plik TNY3 jest zbyt mały (${totalByteLength} B < 64 B)`);
    }
    const dv = new DataView(buffer, baseByteOffset, totalByteLength);
    const raw = new Uint8Array(buffer, baseByteOffset, totalByteLength);
    const magic = String.fromCharCode(raw[0], raw[1], raw[2], raw[3]);
    if (magic !== TNY3_MAGIC) {
        throw new Error(`Nieprawidłowy magic TNY3: ${JSON.stringify(magic)}`);
    }
    const formatVersion = dv.getUint16(4, true);
    if (formatVersion !== TNY3_FORMAT_VERSION) {
        throw new Error(`Nieobsługiwana wersja formatu TNY3: ${formatVersion}`);
    }
    const flags = dv.getUint16(6, true);
    const tiedEmbeddings = (flags & FLAG_TIED_EMBEDDINGS) !== 0;
    const ropeInterleaved = (flags & FLAG_ROPE_INTERLEAVED) !== 0;
    const swiglu = (flags & FLAG_SWIGLU) !== 0;
    if (ropeInterleaved) {
        throw new Error("Silnik obsługuje wyłącznie konwencję RoPE half-split (bit 1 flagi == 0)");
    }
    if (!swiglu) {
        throw new Error("Oczekiwano aktywnej flagi SwiGLU (bit 2 flagi == 1)");
    }
    const vocabSize = dv.getUint32(8, true);
    const ctxLen = dv.getUint32(12, true);
    const dModel = dv.getUint32(16, true);
    const nLayers = dv.getUint32(20, true);
    const nHeads = dv.getUint32(24, true);
    const dFf = dv.getUint32(28, true);
    const ropeTheta = dv.getFloat32(32, true);
    const normEps = dv.getFloat32(36, true);
    const tokOffset = dv.getUint32(40, true);
    const tokLen = dv.getUint32(44, true);
    const weightsOffset = dv.getUint32(48, true);
    const weightsLen = dv.getUint32(52, true);
    const payloadCrc32 = dv.getUint32(56, true);
    const reserved = dv.getUint32(60, true);
    if (vocabSize <= 0 || ctxLen <= 0 || dModel <= 0 || nLayers <= 0 || nHeads <= 0 || dFf <= 0) {
        throw new Error("Nieprawidłowe (zerowe) wymiary modelu w nagłówku TNY3");
    }
    if (dModel % nHeads !== 0) {
        throw new Error(`dModel (${dModel}) nie dzieli się przez nHeads (${nHeads})`);
    }
    const headDim = dModel / nHeads;
    if (headDim % 2 !== 0) {
        throw new Error(`headDim (${headDim}) musi być parzysty dla RoPE half-split`);
    }
    if (dModel % 4 !== 0 || dFf % 4 !== 0) {
        throw new Error(`Wymagane wyrównanie dModel (${dModel}) i dFf (${dFf}) do 4`);
    }
    if (tokOffset !== TNY3_HEADER_BYTES || tokOffset % 4 !== 0) {
        throw new Error(`Nieprawidłowy tok_offset=${tokOffset}`);
    }
    if (weightsOffset % 4 !== 0 || weightsOffset < tokOffset + tokLen) {
        throw new Error(`Nieprawidłowy weights_offset=${weightsOffset} (tok_end=${tokOffset + tokLen})`);
    }
    if (weightsOffset + weightsLen !== totalByteLength) {
        throw new Error(`Niespójny rozmiar pliku TNY3: ${weightsOffset + weightsLen} !== ${totalByteLength}`);
    }
    for (let p = tokOffset + tokLen; p < weightsOffset; p++) {
        if (raw[p] !== 0) {
            throw new Error(`Niezerowy bajt paddingu pod offsetem ${p}`);
        }
    }
    const actualCrc = computeCrc32(raw.subarray(TNY3_HEADER_BYTES));
    if (actualCrc !== payloadCrc32) {
        throw new Error(`Błąd sumy kontrolnej CRC-32 payloadu TNY3: 0x${actualCrc.toString(16)} !== 0x${payloadCrc32.toString(16)}`);
    }
    const tokJsonStr = new EngineTextDecoderCtor("utf-8", { fatal: true }).decode(raw.subarray(tokOffset, tokOffset + tokLen));
    const tokenizerConfig = JSON.parse(tokJsonStr);
    const tokenizer = new TartarusTokenizer(tokenizerConfig);
    if (tokenizer.vocabSize !== vocabSize) {
        throw new Error(`Rozbieżność vocabSize: TNY3=${vocabSize} !== tokenizer=${tokenizer.vocabSize}`);
    }
    let cursor = weightsOffset;
    const embRes = readMatrixInt8Dequant(buffer, baseByteOffset, dv, cursor, vocabSize, dModel);
    const tokEmb = embRes.mat;
    cursor = embRes.nextOffset;
    const layers = new Array(nLayers);
    for (let l = 0; l < nLayers; l++) {
        const rAtt = readVectorF32(dv, cursor, dModel);
        cursor = rAtt.nextOffset;
        const wq = readMatrixInt8Dequant(buffer, baseByteOffset, dv, cursor, dModel, dModel);
        cursor = wq.nextOffset;
        const wk = readMatrixInt8Dequant(buffer, baseByteOffset, dv, cursor, dModel, dModel);
        cursor = wk.nextOffset;
        const wv = readMatrixInt8Dequant(buffer, baseByteOffset, dv, cursor, dModel, dModel);
        cursor = wv.nextOffset;
        const wo = readMatrixInt8Dequant(buffer, baseByteOffset, dv, cursor, dModel, dModel);
        cursor = wo.nextOffset;
        const rFfn = readVectorF32(dv, cursor, dModel);
        cursor = rFfn.nextOffset;
        const wGate = readMatrixInt8Dequant(buffer, baseByteOffset, dv, cursor, dFf, dModel);
        cursor = wGate.nextOffset;
        const wUp = readMatrixInt8Dequant(buffer, baseByteOffset, dv, cursor, dFf, dModel);
        cursor = wUp.nextOffset;
        const wDown = readMatrixInt8Dequant(buffer, baseByteOffset, dv, cursor, dModel, dFf);
        cursor = wDown.nextOffset;
        layers[l] = {
            rmsAtt: rAtt.vec,
            wq: wq.mat,
            wk: wk.mat,
            wv: wv.mat,
            wo: wo.mat,
            rmsFfn: rFfn.vec,
            wGate: wGate.mat,
            wUp: wUp.mat,
            wDown: wDown.mat,
        };
    }
    const rFinal = readVectorF32(dv, cursor, dModel);
    const rmsFinal = rFinal.vec;
    cursor = rFinal.nextOffset;
    let lmHead;
    if (tiedEmbeddings) {
        lmHead = tokEmb;
    }
    else {
        const headRes = readMatrixInt8Dequant(buffer, baseByteOffset, dv, cursor, vocabSize, dModel);
        lmHead = headRes.mat;
        cursor = headRes.nextOffset;
    }
    if (cursor !== weightsOffset + weightsLen) {
        throw new Error(`Niepełny odczyt sekcji wag TNY3: ${cursor} !== ${weightsOffset + weightsLen}`);
    }
    const header = {
        magic,
        formatVersion,
        flags,
        tiedEmbeddings,
        ropeInterleaved,
        swiglu,
        vocabSize,
        ctxLen,
        dModel,
        nLayers,
        nHeads,
        headDim,
        dFf,
        ropeTheta,
        normEps,
        tokOffset,
        tokLen,
        weightsOffset,
        weightsLen,
        payloadCrc32,
        reserved,
        totalBytes: totalByteLength,
    };
    return { header, tokenizerConfig, tokenizer, weights: { tokEmb, layers, rmsFinal, lmHead } };
}
function rmsNormInto(out, x, xOffset, g, dim, eps) {
    let sumSq = 0.0;
    for (let i = 0; i < dim; i++) {
        const v = x[xOffset + i];
        sumSq += v * v;
    }
    const invRms = 1.0 / Math.sqrt(sumSq / dim + eps);
    for (let i = 0; i < dim; i++) {
        out[i] = x[xOffset + i] * invRms * g[i];
    }
}
function matVecInto(out, outOffset, w, x, xOffset, rows, cols) {
    for (let r = 0; r < rows; r++) {
        let acc = 0.0;
        const base = r * cols;
        for (let c = 0; c < cols; c += 4) {
            acc +=
                w[base + c] * x[xOffset + c] +
                    w[base + c + 1] * x[xOffset + c + 1] +
                    w[base + c + 2] * x[xOffset + c + 2] +
                    w[base + c + 3] * x[xOffset + c + 3];
        }
        out[outOffset + r] = acc;
    }
}
function applyRopeHalfSplitInPlace(vec, vecOffset, pos, nHeads, headDim, cosTable, sinTable) {
    const half = headDim >> 1;
    const ropeBase = pos * half;
    for (let h = 0; h < nHeads; h++) {
        const headBase = vecOffset + h * headDim;
        for (let i = 0; i < half; i++) {
            const x1 = vec[headBase + i];
            const x2 = vec[headBase + half + i];
            const c = cosTable[ropeBase + i];
            const s = sinTable[ropeBase + i];
            vec[headBase + i] = x1 * c - x2 * s;
            vec[headBase + half + i] = x2 * c + x1 * s;
        }
    }
}
function silu(z) {
    if (z >= 0) {
        return z / (1.0 + Math.exp(-z));
    }
    const ez = Math.exp(z);
    return (z * ez) / (1.0 + ez);
}
export class TartarusEngine {
    header;
    tokenizer;
    weights;
    padId;
    eosId;
    stateId;
    userId;
    npcId;
    // Prekomputowane tablice RoPE [ctxLen, headDim/2]
    ropeCos;
    ropeSin;
    // Pre-alokowany KV Cache [nLayers, ctxLen, dModel] (zero alokacji w pętli!)
    kCache;
    vCache;
    kvPos = 0;
    // Telemetria warstw (RMS strumienia rezydualnego po każdej warstwie)
    layerRms;
    // Pre-alokowane bufory robocze dla kroku pojedynczego tokenu
    x;
    hNorm;
    q;
    k;
    v;
    attnOut;
    woOut;
    ffGate;
    ffUp;
    ffOut;
    attnScores;
    logits;
    // Bufory pomocnicze do samplingu (bez alokacji na token)
    workLogits;
    workProbs;
    topKIndices;
    topKValues;
    repSeen;
    constructor(parsed) {
        this.header = parsed.header;
        this.tokenizer = parsed.tokenizer;
        this.weights = parsed.weights;
        this.padId = this.tokenizer.getSpecialId("<|pad|>") ?? 0;
        this.eosId = this.tokenizer.getSpecialId("<|eos|>") ?? 1;
        this.stateId = this.tokenizer.getSpecialId("<|state|>") ?? 2;
        this.userId = this.tokenizer.getSpecialId("<|user|>") ?? 3;
        this.npcId = this.tokenizer.getSpecialId("<|npc|>") ?? 4;
        const { ctxLen, dModel, nLayers, headDim, dFf, vocabSize, ropeTheta } = this.header;
        const half = headDim >> 1;
        this.ropeCos = new Float32Array(ctxLen * half);
        this.ropeSin = new Float32Array(ctxLen * half);
        const invFreq = new Float32Array(half);
        for (let i = 0; i < half; i++) {
            invFreq[i] = 1.0 / Math.pow(ropeTheta, (2 * i) / headDim);
        }
        for (let pos = 0; pos < ctxLen; pos++) {
            const base = pos * half;
            for (let i = 0; i < half; i++) {
                const angle = pos * invFreq[i];
                this.ropeCos[base + i] = Math.cos(angle);
                this.ropeSin[base + i] = Math.sin(angle);
            }
        }
        this.kCache = new Float32Array(nLayers * ctxLen * dModel);
        this.vCache = new Float32Array(nLayers * ctxLen * dModel);
        this.layerRms = new Float32Array(nLayers);
        this.x = new Float32Array(dModel);
        this.hNorm = new Float32Array(dModel);
        this.q = new Float32Array(dModel);
        this.k = new Float32Array(dModel);
        this.v = new Float32Array(dModel);
        this.attnOut = new Float32Array(dModel);
        this.woOut = new Float32Array(dModel);
        this.ffGate = new Float32Array(dFf);
        this.ffUp = new Float32Array(dFf);
        this.ffOut = new Float32Array(dModel);
        this.attnScores = new Float32Array(ctxLen);
        this.logits = new Float32Array(vocabSize);
        this.workLogits = new Float32Array(vocabSize);
        this.workProbs = new Float32Array(vocabSize);
        this.topKIndices = new Int32Array(vocabSize);
        this.topKValues = new Float32Array(vocabSize);
        this.repSeen = new Uint8Array(vocabSize);
    }
    static fromArrayBuffer(input) {
        return new TartarusEngine(parseTNY3(input));
    }
    encode(text) {
        return this.tokenizer.encode(text);
    }
    decode(ids, options) {
        return this.tokenizer.decode(ids, options);
    }
    resetCache() {
        this.kvPos = 0;
    }
    runTransformerTokenAtPos(tokenId, pos, computeLogits) {
        const { vocabSize, ctxLen, dModel, nLayers, nHeads, headDim, dFf, normEps } = this.header;
        if (tokenId < 0 || tokenId >= vocabSize) {
            throw new RangeError(`Token ID poza zakresem [0, ${vocabSize}): ${tokenId}`);
        }
        if (pos < 0 || pos >= ctxLen) {
            throw new RangeError(`Pozycja KV-Cache poza zakresem [0, ${ctxLen}): ${pos}`);
        }
        const embBase = tokenId * dModel;
        const tokEmb = this.weights.tokEmb;
        for (let i = 0; i < dModel; i++) {
            this.x[i] = tokEmb[embBase + i];
        }
        const invSqrtHd = 1.0 / Math.sqrt(headDim);
        const layerStride = ctxLen * dModel;
        for (let l = 0; l < nLayers; l++) {
            const lw = this.weights.layers[l];
            const cacheLayerBase = l * layerStride;
            // 1. Pre-norm przed Attention
            rmsNormInto(this.hNorm, this.x, 0, lw.rmsAtt, dModel, normEps);
            // 2. Projekcje Q, K, V
            matVecInto(this.q, 0, lw.wq, this.hNorm, 0, dModel, dModel);
            matVecInto(this.k, 0, lw.wk, this.hNorm, 0, dModel, dModel);
            matVecInto(this.v, 0, lw.wv, this.hNorm, 0, dModel, dModel);
            // 3. RoPE half-split
            applyRopeHalfSplitInPlace(this.q, 0, pos, nHeads, headDim, this.ropeCos, this.ropeSin);
            applyRopeHalfSplitInPlace(this.k, 0, pos, nHeads, headDim, this.ropeCos, this.ropeSin);
            // 4. Zapis K i V do pre-alokowanego bufora KV-Cache
            const cachePosBase = cacheLayerBase + pos * dModel;
            this.kCache.set(this.k, cachePosBase);
            this.vCache.set(this.v, cachePosBase);
            // 5. Causal Multi-Head Self-Attention po [0 .. pos]
            for (let h = 0; h < nHeads; h++) {
                const hOff = h * headDim;
                let maxScore = -Infinity;
                for (let t = 0; t <= pos; t++) {
                    const kOff = cacheLayerBase + t * dModel + hOff;
                    let dot = 0.0;
                    for (let d = 0; d < headDim; d += 4) {
                        dot +=
                            this.q[hOff + d] * this.kCache[kOff + d] +
                                this.q[hOff + d + 1] * this.kCache[kOff + d + 1] +
                                this.q[hOff + d + 2] * this.kCache[kOff + d + 2] +
                                this.q[hOff + d + 3] * this.kCache[kOff + d + 3];
                    }
                    const score = dot * invSqrtHd;
                    this.attnScores[t] = score;
                    if (score > maxScore) {
                        maxScore = score;
                    }
                }
                let sumExp = 0.0;
                for (let t = 0; t <= pos; t++) {
                    const w = Math.exp(this.attnScores[t] - maxScore);
                    this.attnScores[t] = w;
                    sumExp += w;
                }
                const invSum = 1.0 / sumExp;
                for (let d = 0; d < headDim; d++) {
                    this.attnOut[hOff + d] = 0.0;
                }
                for (let t = 0; t <= pos; t++) {
                    const prob = this.attnScores[t] * invSum;
                    const vOff = cacheLayerBase + t * dModel + hOff;
                    for (let d = 0; d < headDim; d++) {
                        this.attnOut[hOff + d] += prob * this.vCache[vOff + d];
                    }
                }
            }
            // 6. Projekcja wyjściowa Wo + Residual
            matVecInto(this.woOut, 0, lw.wo, this.attnOut, 0, dModel, dModel);
            for (let i = 0; i < dModel; i++) {
                this.x[i] += this.woOut[i];
            }
            // 7. Pre-norm przed SwiGLU FFN
            rmsNormInto(this.hNorm, this.x, 0, lw.rmsFfn, dModel, normEps);
            // 8. SwiGLU: w_down(silu(w_gate(h)) * w_up(h)) + Residual
            matVecInto(this.ffGate, 0, lw.wGate, this.hNorm, 0, dFf, dModel);
            matVecInto(this.ffUp, 0, lw.wUp, this.hNorm, 0, dFf, dModel);
            for (let i = 0; i < dFf; i++) {
                this.ffGate[i] = silu(this.ffGate[i]) * this.ffUp[i];
            }
            matVecInto(this.ffOut, 0, lw.wDown, this.ffGate, 0, dModel, dFf);
            let sumSq = 0.0;
            for (let i = 0; i < dModel; i++) {
                const val = this.x[i] + this.ffOut[i];
                this.x[i] = val;
                sumSq += val * val;
            }
            this.layerRms[l] = Math.sqrt(sumSq / dModel);
        }
        if (computeLogits) {
            rmsNormInto(this.hNorm, this.x, 0, this.weights.rmsFinal, dModel, normEps);
            matVecInto(this.logits, 0, this.weights.lmHead, this.hNorm, 0, vocabSize, dModel);
        }
    }
    /**
     * Przetwarza cały prompt od pozycji 0, wypełniając KV-Cache.
     * Zwraca logity dla ostatniego tokenu promptu (ten sam bufor wewnętrzny Float32Array).
     */
    prefill(tokenIds) {
        const n = tokenIds.length;
        if (n <= 0 || n > this.header.ctxLen) {
            throw new RangeError(`Długość promptu (${n}) musi mieścić się w [1, ${this.header.ctxLen}]`);
        }
        this.resetCache();
        for (let i = 0; i < n; i++) {
            this.runTransformerTokenAtPos(tokenIds[i], i, i === n - 1);
        }
        this.kvPos = n;
        return this.logits;
    }
    /**
     * Wykonuje pojedynczy krok autoregresji na pozycji `this.kvPos` z użyciem KV-Cache.
     */
    step(tokenId) {
        const pos = this.kvPos;
        this.runTransformerTokenAtPos(tokenId, pos, true);
        this.kvPos = pos + 1;
        return this.logits;
    }
    /**
     * Pełny przebieg bez użycia trwałego KV-Cache (do testów referencyjnych zgodności z PyTorch).
     * Nie modyfikuje `this.kCache`, `this.vCache` ani `this.kvPos`.
     */
    forwardNoCache(tokenIds) {
        const T = tokenIds.length;
        const { vocabSize, ctxLen, dModel, nLayers, nHeads, headDim, dFf, normEps } = this.header;
        if (T <= 0 || T > ctxLen) {
            throw new RangeError(`Długość sekwencji (${T}) musi mieścić się w [1, ${ctxLen}]`);
        }
        const xSeq = new Float32Array(T * dModel);
        const qSeq = new Float32Array(T * dModel);
        const kSeq = new Float32Array(T * dModel);
        const vSeq = new Float32Array(T * dModel);
        const attnSeq = new Float32Array(T * dModel);
        const scores = new Float32Array(T);
        const hBuf = new Float32Array(dModel);
        const woBuf = new Float32Array(dModel);
        const gateBuf = new Float32Array(dFf);
        const upBuf = new Float32Array(dFf);
        const outBuf = new Float32Array(dModel);
        const logitsOut = new Float32Array(vocabSize);
        for (let t = 0; t < T; t++) {
            const tid = tokenIds[t];
            if (tid < 0 || tid >= vocabSize) {
                throw new RangeError(`Token ID poza zakresem [0, ${vocabSize}): ${tid}`);
            }
            const embBase = tid * dModel;
            xSeq.set(this.weights.tokEmb.subarray(embBase, embBase + dModel), t * dModel);
        }
        const invSqrtHd = 1.0 / Math.sqrt(headDim);
        for (let l = 0; l < nLayers; l++) {
            const lw = this.weights.layers[l];
            for (let t = 0; t < T; t++) {
                const off = t * dModel;
                rmsNormInto(hBuf, xSeq, off, lw.rmsAtt, dModel, normEps);
                matVecInto(qSeq, off, lw.wq, hBuf, 0, dModel, dModel);
                matVecInto(kSeq, off, lw.wk, hBuf, 0, dModel, dModel);
                matVecInto(vSeq, off, lw.wv, hBuf, 0, dModel, dModel);
                applyRopeHalfSplitInPlace(qSeq, off, t, nHeads, headDim, this.ropeCos, this.ropeSin);
                applyRopeHalfSplitInPlace(kSeq, off, t, nHeads, headDim, this.ropeCos, this.ropeSin);
            }
            for (let t = 0; t < T; t++) {
                const qBase = t * dModel;
                const aBase = t * dModel;
                for (let h = 0; h < nHeads; h++) {
                    const hOff = h * headDim;
                    let maxScore = -Infinity;
                    for (let j = 0; j <= t; j++) {
                        const kOff = j * dModel + hOff;
                        let dot = 0.0;
                        for (let d = 0; d < headDim; d += 4) {
                            dot +=
                                qSeq[qBase + hOff + d] * kSeq[kOff + d] +
                                    qSeq[qBase + hOff + d + 1] * kSeq[kOff + d + 1] +
                                    qSeq[qBase + hOff + d + 2] * kSeq[kOff + d + 2] +
                                    qSeq[qBase + hOff + d + 3] * kSeq[kOff + d + 3];
                        }
                        const sc = dot * invSqrtHd;
                        scores[j] = sc;
                        if (sc > maxScore)
                            maxScore = sc;
                    }
                    let sumExp = 0.0;
                    for (let j = 0; j <= t; j++) {
                        const w = Math.exp(scores[j] - maxScore);
                        scores[j] = w;
                        sumExp += w;
                    }
                    const invSum = 1.0 / sumExp;
                    for (let d = 0; d < headDim; d++) {
                        attnSeq[aBase + hOff + d] = 0.0;
                    }
                    for (let j = 0; j <= t; j++) {
                        const prob = scores[j] * invSum;
                        const vOff = j * dModel + hOff;
                        for (let d = 0; d < headDim; d++) {
                            attnSeq[aBase + hOff + d] += prob * vSeq[vOff + d];
                        }
                    }
                }
                matVecInto(woBuf, 0, lw.wo, attnSeq, aBase, dModel, dModel);
                for (let i = 0; i < dModel; i++) {
                    xSeq[aBase + i] += woBuf[i];
                }
                rmsNormInto(hBuf, xSeq, aBase, lw.rmsFfn, dModel, normEps);
                matVecInto(gateBuf, 0, lw.wGate, hBuf, 0, dFf, dModel);
                matVecInto(upBuf, 0, lw.wUp, hBuf, 0, dFf, dModel);
                for (let i = 0; i < dFf; i++) {
                    gateBuf[i] = silu(gateBuf[i]) * upBuf[i];
                }
                matVecInto(outBuf, 0, lw.wDown, gateBuf, 0, dModel, dFf);
                for (let i = 0; i < dModel; i++) {
                    xSeq[aBase + i] += outBuf[i];
                }
            }
        }
        rmsNormInto(hBuf, xSeq, (T - 1) * dModel, this.weights.rmsFinal, dModel, normEps);
        matVecInto(logitsOut, 0, this.weights.lmHead, hBuf, 0, vocabSize, dModel);
        return logitsOut;
    }
    sampleNextToken(logits, options, seenTokens, rng = Math.random) {
        const V = this.header.vocabSize;
        const temperature = options?.temperature ?? 0.0;
        const topK = options?.topK ?? 0;
        const minP = options?.minP ?? 0.0;
        const repPenalty = options?.repetitionPenalty ?? 1.0;
        this.workLogits.set(logits);
        if (repPenalty > 1.0 && seenTokens !== undefined && seenTokens.length > 0) {
            for (let i = 0; i < seenTokens.length; i++) {
                const tid = seenTokens[i];
                if (tid >= 0 && tid < V && this.repSeen[tid] === 0) {
                    this.repSeen[tid] = 1;
                    const val = this.workLogits[tid];
                    this.workLogits[tid] = val > 0 ? val / repPenalty : val * repPenalty;
                }
            }
            for (let i = 0; i < seenTokens.length; i++) {
                const tid = seenTokens[i];
                if (tid >= 0 && tid < V) {
                    this.repSeen[tid] = 0;
                }
            }
        }
        if (temperature <= 0.0) {
            let bestId = 0;
            let bestVal = this.workLogits[0];
            for (let i = 1; i < V; i++) {
                if (this.workLogits[i] > bestVal) {
                    bestVal = this.workLogits[i];
                    bestId = i;
                }
            }
            return bestId;
        }
        const invTemp = 1.0 / temperature;
        let count = V;
        if (topK > 0 && topK < V) {
            // Utrzymujemy posortowaną listę K najlepszych (zero alokacji)
            let filled = 0;
            for (let i = 0; i < V; i++) {
                const v = this.workLogits[i] * invTemp;
                if (filled < topK) {
                    let p = filled;
                    while (p > 0 && this.topKValues[p - 1] < v) {
                        this.topKValues[p] = this.topKValues[p - 1];
                        this.topKIndices[p] = this.topKIndices[p - 1];
                        p--;
                    }
                    this.topKValues[p] = v;
                    this.topKIndices[p] = i;
                    filled++;
                }
                else if (v > this.topKValues[topK - 1]) {
                    let p = topK - 1;
                    while (p > 0 && this.topKValues[p - 1] < v) {
                        this.topKValues[p] = this.topKValues[p - 1];
                        this.topKIndices[p] = this.topKIndices[p - 1];
                        p--;
                    }
                    this.topKValues[p] = v;
                    this.topKIndices[p] = i;
                }
            }
            count = topK;
        }
        else {
            for (let i = 0; i < V; i++) {
                this.topKIndices[i] = i;
                this.topKValues[i] = this.workLogits[i] * invTemp;
            }
        }
        let maxLogit = -Infinity;
        for (let i = 0; i < count; i++) {
            if (this.topKValues[i] > maxLogit) {
                maxLogit = this.topKValues[i];
            }
        }
        let sumExp = 0.0;
        let maxProb = 0.0;
        for (let i = 0; i < count; i++) {
            const e = Math.exp(this.topKValues[i] - maxLogit);
            this.workProbs[i] = e;
            sumExp += e;
            if (e > maxProb)
                maxProb = e;
        }
        if (minP > 0.0 && minP < 1.0) {
            const thresh = minP * maxProb;
            let filteredSum = 0.0;
            for (let i = 0; i < count; i++) {
                if (this.workProbs[i] < thresh) {
                    this.workProbs[i] = 0.0;
                }
                else {
                    filteredSum += this.workProbs[i];
                }
            }
            sumExp = filteredSum > 0.0 ? filteredSum : sumExp;
        }
        const r = rng() * sumExp;
        let cdf = 0.0;
        for (let i = 0; i < count; i++) {
            cdf += this.workProbs[i];
            if (cdf >= r) {
                return this.topKIndices[i];
            }
        }
        return this.topKIndices[count - 1];
    }
    *generateStream(prompt, options) {
        const promptIds = typeof prompt === "string" ? this.encode(prompt) : Array.from(prompt);
        const ctxLen = this.header.ctxLen;
        if (promptIds.length === 0) {
            throw new Error("Prompt po tokenizacji jest pusty");
        }
        if (promptIds.length >= ctxLen) {
            throw new RangeError(`Prompt (${promptIds.length} tok) osiąga lub przekracza limit kontekstu CTX=${ctxLen}`);
        }
        const stopOnEos = options?.stopOnEos ?? true;
        const requestedMaxNew = options?.maxNewTokens ?? 128;
        const ctxRemaining = ctxLen - promptIds.length;
        const maxNew = Math.min(requestedMaxNew, ctxRemaining);
        const rng = options?.rng ?? Math.random;
        const t0 = nowMs();
        let currentLogits = this.prefill(promptIds);
        const prefillMs = nowMs() - t0;
        const generatedTokens = [];
        const streamDecoder = this.tokenizer.createStreamDecoder({ skipPad: true, skipSpecial: true });
        let stoppedBy = requestedMaxNew > ctxRemaining ? "ctx" : "max_tokens";
        const tDecodeStart = nowMs();
        for (let stepIdx = 0; stepIdx < maxNew; stepIdx++) {
            const nextId = this.sampleNextToken(currentLogits, options, generatedTokens, rng);
            generatedTokens.push(nextId);
            const isEos = nextId === this.eosId;
            const hitEosStop = stopOnEos && isEos;
            const isLastStep = stepIdx + 1 >= maxNew || this.kvPos >= ctxLen;
            let textDelta = isEos ? streamDecoder.flush() : streamDecoder.push(nextId);
            if (!isEos && isLastStep) {
                textDelta += streamDecoder.flush();
            }
            yield {
                step: stepIdx,
                tokenId: nextId,
                textDelta,
                pos: this.kvPos,
                isEos,
                layerRms: this.layerRms,
            };
            if (hitEosStop) {
                stoppedBy = "eos";
                break;
            }
            if (this.kvPos >= ctxLen) {
                stoppedBy = "ctx";
                break;
            }
            if (stepIdx + 1 < maxNew) {
                currentLogits = this.step(nextId);
            }
        }
        const decodeMs = nowMs() - tDecodeStart;
        const text = this.decode(generatedTokens, { skipPad: true, skipSpecial: true });
        const tokensPerSec = decodeMs > 0 ? (generatedTokens.length * 1000.0) / decodeMs : 0.0;
        return {
            promptTokens: promptIds,
            generatedTokens,
            text,
            stoppedBy,
            prefillMs,
            decodeMs,
            tokensPerSec,
            totalPos: this.kvPos,
        };
    }
    generate(prompt, options) {
        const gen = this.generateStream(prompt, options);
        while (true) {
            const stepRes = gen.next();
            if (stepRes.done) {
                return stepRes.value;
            }
        }
    }
}
