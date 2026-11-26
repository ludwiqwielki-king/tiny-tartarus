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
import { TartarusTokenizer, type DecodeOptions, type TartarusTokenizerConfig } from "./tartarus_tokenizer.js";
export declare const TNY3_MAGIC = "TNY3";
export declare const TNY3_HEADER_BYTES = 64;
export declare const TNY3_FORMAT_VERSION = 1;
export declare const FLAG_TIED_EMBEDDINGS: number;
export declare const FLAG_ROPE_INTERLEAVED: number;
export declare const FLAG_SWIGLU: number;
export interface TNY3Header {
    readonly magic: string;
    readonly formatVersion: number;
    readonly flags: number;
    readonly tiedEmbeddings: boolean;
    readonly ropeInterleaved: boolean;
    readonly swiglu: boolean;
    readonly vocabSize: number;
    readonly ctxLen: number;
    readonly dModel: number;
    readonly nLayers: number;
    readonly nHeads: number;
    readonly headDim: number;
    readonly dFf: number;
    readonly ropeTheta: number;
    readonly normEps: number;
    readonly tokOffset: number;
    readonly tokLen: number;
    readonly weightsOffset: number;
    readonly weightsLen: number;
    readonly payloadCrc32: number;
    readonly reserved: number;
    readonly totalBytes: number;
}
export interface LayerWeightsF32 {
    readonly rmsAtt: Float32Array;
    readonly wq: Float32Array;
    readonly wk: Float32Array;
    readonly wv: Float32Array;
    readonly wo: Float32Array;
    readonly rmsFfn: Float32Array;
    readonly wGate: Float32Array;
    readonly wUp: Float32Array;
    readonly wDown: Float32Array;
}
export interface ModelWeightsF32 {
    readonly tokEmb: Float32Array;
    readonly layers: readonly LayerWeightsF32[];
    readonly rmsFinal: Float32Array;
    readonly lmHead: Float32Array;
}
export interface ParsedTNY3 {
    readonly header: TNY3Header;
    readonly tokenizerConfig: TartarusTokenizerConfig;
    readonly tokenizer: TartarusTokenizer;
    readonly weights: ModelWeightsF32;
}
export interface SamplingOptions {
    /** 0 = greedy (domyślnie); > 0 = próbkowanie stochastyczne */
    temperature?: number;
    /** Ograniczenie do top-K najwyższych logitów (0 / undefined = wyłączone) */
    topK?: number;
    /** Próg min-P względny względem max(p) (0 / undefined = wyłączone) */
    minP?: number;
    /** Kara za powtórzenia dla już wygenerowanych tokenów (1.0 = brak kary) */
    repetitionPenalty?: number;
}
export interface GenerateOptions extends SamplingOptions {
    maxNewTokens?: number;
    stopOnEos?: boolean;
    /** Opcjonalny PRNG [0, 1) dla deterministycznego samplingu w testach */
    rng?: () => number;
}
export interface StreamEvent {
    readonly step: number;
    readonly tokenId: number;
    readonly textDelta: string;
    readonly pos: number;
    readonly isEos: boolean;
    readonly layerRms: Float32Array;
}
export interface GenerateResult {
    readonly promptTokens: number[];
    readonly generatedTokens: number[];
    readonly text: string;
    readonly stoppedBy: "eos" | "ctx" | "max_tokens";
    readonly prefillMs: number;
    readonly decodeMs: number;
    readonly tokensPerSec: number;
    readonly totalPos: number;
}
export declare function computeCrc32(bytes: Uint8Array): number;
export declare function parseTNY3(input: ArrayBuffer | ArrayBufferView): ParsedTNY3;
export declare class TartarusEngine {
    readonly header: TNY3Header;
    readonly tokenizer: TartarusTokenizer;
    readonly weights: ModelWeightsF32;
    readonly padId: number;
    readonly eosId: number;
    readonly stateId: number;
    readonly userId: number;
    readonly npcId: number;
    readonly ropeCos: Float32Array;
    readonly ropeSin: Float32Array;
    readonly kCache: Float32Array;
    readonly vCache: Float32Array;
    kvPos: number;
    readonly layerRms: Float32Array;
    private readonly x;
    private readonly hNorm;
    private readonly q;
    private readonly k;
    private readonly v;
    private readonly attnOut;
    private readonly woOut;
    private readonly ffGate;
    private readonly ffUp;
    private readonly ffOut;
    private readonly attnScores;
    private readonly logits;
    private readonly workLogits;
    private readonly workProbs;
    private readonly topKIndices;
    private readonly topKValues;
    private readonly repSeen;
    constructor(parsed: ParsedTNY3);
    static fromArrayBuffer(input: ArrayBuffer | ArrayBufferView): TartarusEngine;
    encode(text: string): number[];
    decode(ids: Iterable<number>, options?: DecodeOptions): string;
    resetCache(): void;
    private runTransformerTokenAtPos;
    /**
     * Przetwarza cały prompt od pozycji 0, wypełniając KV-Cache.
     * Zwraca logity dla ostatniego tokenu promptu (ten sam bufor wewnętrzny Float32Array).
     */
    prefill(tokenIds: ArrayLike<number>): Float32Array;
    /**
     * Wykonuje pojedynczy krok autoregresji na pozycji `this.kvPos` z użyciem KV-Cache.
     */
    step(tokenId: number): Float32Array;
    /**
     * Pełny przebieg bez użycia trwałego KV-Cache (do testów referencyjnych zgodności z PyTorch).
     * Nie modyfikuje `this.kCache`, `this.vCache` ani `this.kvPos`.
     */
    forwardNoCache(tokenIds: ArrayLike<number>): Float32Array;
    sampleNextToken(logits: Float32Array, options?: SamplingOptions, seenTokens?: readonly number[], rng?: () => number): number;
    generateStream(prompt: string | readonly number[], options?: GenerateOptions): Generator<StreamEvent, GenerateResult, void>;
    generate(prompt: string | readonly number[], options?: GenerateOptions): GenerateResult;
}
