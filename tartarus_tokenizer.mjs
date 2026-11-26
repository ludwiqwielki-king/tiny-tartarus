/**
 * tartarus_tokenizer.ts — Minimalny, deterministyczny tokenizer Byte-Level BPE (0 zewnętrznych zależności)
 * do projektu „Tartarus: The Fall of the Ancients”.
 *
 * Przeznaczony do bezpośredniego użycia w Node.js oraz przeglądarce (Etap 5 / TNY3):
 * konstruktor przyjmuje wyłącznie sparsowany obiekt JSON z `tokenizer.json`.
 */
const PAIR_STRIDE = 1048576; // 2^20 — jednoznaczny klucz (left * 2^20 + right) << Number.MAX_SAFE_INTEGER
const TextEncoderCtor = globalThis.TextEncoder;
const TextDecoderCtor = globalThis.TextDecoder;
function escapeRegex(literal) {
    return literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
export class TartarusTokenizer {
    vocabSize;
    byteOffset;
    padId;
    specialToId;
    idToSpecial;
    specialRe;
    pretokenizeRe;
    idToBytes;
    mergesRank;
    chunkCache;
    textEncoder;
    textDecoder;
    constructor(config) {
        if (config.format !== undefined && config.format !== "tartarus-bpe-v1") {
            throw new Error(`Nieobsługiwany format tokenizera: ${String(config.format)}`);
        }
        this.specialToId = new Map();
        this.idToSpecial = new Map();
        for (const [tok, id] of Object.entries(config.special_tokens)) {
            if (!Number.isInteger(id) || id < 0) {
                throw new Error(`Nieprawidłowe ID tokenu specjalnego ${tok}: ${String(id)}`);
            }
            this.specialToId.set(tok, id);
            this.idToSpecial.set(id, tok);
        }
        this.byteOffset = config.byte_offset;
        if (this.byteOffset !== this.specialToId.size) {
            throw new Error(`byte_offset (${this.byteOffset}) musi równać się liczbie tokenów specjalnych (${this.specialToId.size})`);
        }
        const padId = this.specialToId.get("<|pad|>");
        if (padId === undefined) {
            throw new Error("Brak wymaganego tokenu specjalnego <|pad|> w konfiguracji");
        }
        this.padId = padId;
        const orderedSpecials = Array.from(this.specialToId.keys()).sort((a, b) => b.length - a.length || (a < b ? -1 : a > b ? 1 : 0));
        this.specialRe =
            orderedSpecials.length > 0
                ? new RegExp("(" + orderedSpecials.map(escapeRegex).join("|") + ")", "g")
                : null;
        let flags = config.pretokenize_flags ?? "gu";
        if (!flags.includes("g"))
            flags += "g";
        if (!flags.includes("u"))
            flags += "u";
        this.pretokenizeRe = new RegExp(config.pretokenize_pattern, flags);
        this.idToBytes = new Array(this.byteOffset + 256);
        for (let i = 0; i < this.byteOffset; i++) {
            this.idToBytes[i] = new Uint8Array(0);
        }
        for (let b = 0; b < 256; b++) {
            this.idToBytes[this.byteOffset + b] = new Uint8Array([b]);
        }
        this.mergesRank = new Map();
        const baseEnd = this.byteOffset + 256;
        for (let rank = 0; rank < config.merges.length; rank++) {
            const entry = config.merges[rank];
            const left = entry[0];
            const right = entry[1];
            const newId = entry.length >= 3 ? entry[2] : baseEnd + rank;
            const expectedNewId = baseEnd + rank;
            if (newId !== expectedNewId) {
                throw new Error(`Niespójne ID scalenia w pozycji ${rank}: ${newId} !== ${expectedNewId}`);
            }
            if (left < this.byteOffset ||
                left >= newId ||
                right < this.byteOffset ||
                right >= newId) {
                throw new Error(`Nieprawidłowi rodzice merge (${left}, ${right}) dla ID=${newId}`);
            }
            const leftBytes = this.idToBytes[left];
            const rightBytes = this.idToBytes[right];
            const merged = new Uint8Array(leftBytes.length + rightBytes.length);
            merged.set(leftBytes, 0);
            merged.set(rightBytes, leftBytes.length);
            this.idToBytes.push(merged);
            this.mergesRank.set(left * PAIR_STRIDE + right, { rank, newId });
        }
        this.vocabSize = this.idToBytes.length;
        if (config.vocab_size !== undefined && config.vocab_size !== this.vocabSize) {
            throw new Error(`Niezgodność vocab_size (${config.vocab_size}) z długością tablicy tokenów (${this.vocabSize})`);
        }
        this.chunkCache = new Map();
        this.textEncoder = new TextEncoderCtor();
        this.textDecoder = new TextDecoderCtor("utf-8", { fatal: false });
    }
    getSpecialId(token) {
        return this.specialToId.get(token);
    }
    getSpecialToken(id) {
        return this.idToSpecial.get(id);
    }
    getTokenBytes(id) {
        if (!Number.isInteger(id) || id < 0 || id >= this.vocabSize) {
            throw new RangeError(`Token ID poza zakresem słownika [0, ${this.vocabSize}): ${String(id)}`);
        }
        return this.idToBytes[id];
    }
    createStreamDecoder(options) {
        const skipPad = options?.skipPad ?? true;
        const skipSpecial = options?.skipSpecial ?? false;
        const dec = new TextDecoderCtor("utf-8", { fatal: false });
        return {
            push: (tid) => {
                if (!Number.isInteger(tid) || tid < 0 || tid >= this.vocabSize) {
                    throw new RangeError(`Token ID poza zakresem słownika [0, ${this.vocabSize}): ${String(tid)}`);
                }
                if (tid < this.byteOffset) {
                    const flushed = dec.decode(new Uint8Array(0), { stream: false });
                    if (tid === this.padId && skipPad)
                        return flushed;
                    if (skipSpecial)
                        return flushed;
                    return flushed + (this.idToSpecial.get(tid) ?? "");
                }
                return dec.decode(this.idToBytes[tid], { stream: true });
            },
            flush: () => dec.decode(new Uint8Array(0), { stream: false }),
        };
    }
    encodeChunk(chunk) {
        const cached = this.chunkCache.get(chunk);
        if (cached !== undefined) {
            return cached;
        }
        const utf8 = this.textEncoder.encode(chunk);
        let ids = new Array(utf8.length);
        for (let i = 0; i < utf8.length; i++) {
            ids[i] = this.byteOffset + utf8[i];
        }
        while (ids.length >= 2) {
            let bestRank = Infinity;
            let bestLeft = 0;
            let bestRight = 0;
            let bestNewId = 0;
            for (let i = 0; i < ids.length - 1; i++) {
                const a = ids[i];
                const b = ids[i + 1];
                const info = this.mergesRank.get(a * PAIR_STRIDE + b);
                if (info !== undefined && info.rank < bestRank) {
                    bestRank = info.rank;
                    bestLeft = a;
                    bestRight = b;
                    bestNewId = info.newId;
                }
            }
            if (bestRank === Infinity) {
                break;
            }
            const nextIds = [];
            let i = 0;
            const n = ids.length;
            while (i < n) {
                if (i + 1 < n && ids[i] === bestLeft && ids[i + 1] === bestRight) {
                    nextIds.push(bestNewId);
                    i += 2;
                }
                else {
                    nextIds.push(ids[i]);
                    i += 1;
                }
            }
            ids = nextIds;
        }
        if (this.chunkCache.size < 50000) {
            this.chunkCache.set(chunk, ids);
        }
        return ids;
    }
    encode(text) {
        if (text.length === 0) {
            return [];
        }
        const segments = this.specialRe !== null ? text.split(this.specialRe) : [text];
        const out = [];
        for (let s = 0; s < segments.length; s++) {
            const seg = segments[s];
            if (seg.length === 0) {
                continue;
            }
            const specId = this.specialToId.get(seg);
            if (specId !== undefined) {
                out.push(specId);
                continue;
            }
            this.pretokenizeRe.lastIndex = 0;
            let match;
            while ((match = this.pretokenizeRe.exec(seg)) !== null) {
                const chunk = match[0];
                if (chunk.length === 0) {
                    this.pretokenizeRe.lastIndex++;
                    continue;
                }
                const chunkIds = this.encodeChunk(chunk);
                for (let k = 0; k < chunkIds.length; k++) {
                    out.push(chunkIds[k]);
                }
            }
        }
        return out;
    }
    decode(ids, options) {
        const skipPad = options?.skipPad ?? true;
        const skipSpecial = options?.skipSpecial ?? false;
        const parts = [];
        const byteBuf = [];
        const flushBytes = () => {
            if (byteBuf.length > 0) {
                parts.push(this.textDecoder.decode(new Uint8Array(byteBuf)));
                byteBuf.length = 0;
            }
        };
        for (const rawTid of ids) {
            const tid = Number(rawTid);
            if (!Number.isInteger(tid) || tid < 0 || tid >= this.vocabSize) {
                throw new RangeError(`Token ID poza zakresem słownika [0, ${this.vocabSize}): ${String(rawTid)}`);
            }
            if (tid < this.byteOffset) {
                flushBytes();
                if (tid === this.padId && skipPad) {
                    continue;
                }
                if (!skipSpecial) {
                    const tokStr = this.idToSpecial.get(tid);
                    if (tokStr === undefined) {
                        throw new RangeError(`Nieznany token specjalny o ID=${tid}`);
                    }
                    parts.push(tokStr);
                }
            }
            else {
                const bytes = this.idToBytes[tid];
                for (let i = 0; i < bytes.length; i++) {
                    byteBuf.push(bytes[i]);
                }
            }
        }
        flushBytes();
        return parts.join("");
    }
}
