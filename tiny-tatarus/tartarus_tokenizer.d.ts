/**
 * tartarus_tokenizer.ts — Minimalny, deterministyczny tokenizer Byte-Level BPE (0 zewnętrznych zależności)
 * do projektu „Tartarus: The Fall of the Ancients”.
 *
 * Przeznaczony do bezpośredniego użycia w Node.js oraz przeglądarce (Etap 5 / TNY3):
 * konstruktor przyjmuje wyłącznie sparsowany obiekt JSON z `tokenizer.json`.
 */
export type MergeEntry = readonly [number, number] | readonly [number, number, number];
export interface TartarusTokenizerConfig {
    format?: string;
    vocab_size?: number;
    special_tokens: Record<string, number>;
    byte_offset: number;
    pretokenize_pattern: string;
    pretokenize_flags?: string;
    merges: readonly MergeEntry[];
}
export interface DecodeOptions {
    /** Pomija token <|pad|> (domyślnie true). */
    skipPad?: boolean;
    /** Pomija wszystkie tokeny specjalne (domyślnie false). */
    skipSpecial?: boolean;
}
export declare class TartarusTokenizer {
    readonly vocabSize: number;
    readonly byteOffset: number;
    readonly padId: number;
    private readonly specialToId;
    private readonly idToSpecial;
    private readonly specialRe;
    private readonly pretokenizeRe;
    private readonly idToBytes;
    private readonly mergesRank;
    private readonly chunkCache;
    private readonly textEncoder;
    private readonly textDecoder;
    constructor(config: TartarusTokenizerConfig);
    getSpecialId(token: string): number | undefined;
    getSpecialToken(id: number): string | undefined;
    getTokenBytes(id: number): Uint8Array;
    createStreamDecoder(options?: DecodeOptions): {
        push: (tid: number) => string;
        flush: () => string;
    };
    private encodeChunk;
    encode(text: string): number[];
    decode(ids: Iterable<number>, options?: DecodeOptions): string;
}
