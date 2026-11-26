# Silnik inferencji, KV Cache, Sampling, UI Przeglądarkowe i Wydajność

> Część dokumentacji systemu Tiny: [← Powrót do głównego przewodnika](./TINY_SYSTEM_DOCUMENTATION.md)  
> Powiązane dokumenty: [Architektura i Tokenizer](./TINY_ARCHITECTURE.md) · [Matematyka operacji](./TINY_MATHEMATICS.md) · [Format `.tny3`](./TINY_FORMAT_TNY3.md) · [Testy i weryfikacja](./TINY_TESTS.md)

---

## ROZDZIAŁ V – Silnik inferencji TypeScript / JavaScript (`tartarus-engine.ts`)

Plik `tartarus-engine.ts` (`1033` linie) stanowi czyste jądro obliczeniowe runtime’u. Kompiluje się w ścisłym trybie TypeScript (`tsconfig.json`: `target: ES2022`, `module: ES2022`, `lib: ["ES2022"]`, `types: []`, `strict: true`) bez żadnych zależności od DOM (`window`, `document`, HTML) i bez pakietów zewnętrznych (`node_modules`), importując jedynie `./tartarus_tokenizer.js`.

### 1. Architektura klasy `TartarusEngine` i inicjalizacja pamięci

**POTWIERDZONE** (`tartarus-engine.ts:457–556`):
Po sparsowaniu pliku `.tny3` przez `parseTNY3()`, konstruktor `new TartarusEngine(parsed)`:
1. Zapamiętuje referencje do `header`, `tokenizer`, `weights` oraz odczytuje ID tokenów specjalnych (`padId=0`, `eosId=1`, `stateId=2`, `userId=3`, `npcId=4`).
2. **Prekomputuje tablice RoPE** `ropeCos` i `ropeSin` (`ctxLen × (headDim / 2)` wartości `Float32Array`, czyli `256 × 16 = 4096` liczb na każdą tabelę) na podstawie `header.ropeTheta`. Dzięki temu w trakcie generacji nie są wywoływane kosztowne funkcje trygonometryczne `Math.cos` / `Math.sin`.
3. **Pre-alokuje pamięć KV Cache** `kCache` i `vCache` jako jednowymiarowe tablice `Float32Array(nLayers * ctxLen * dModel)` oraz bufor telemetrii `layerRms = new Float32Array(nLayers)`.
4. **Pre-alokuje jednorazowo wszystkie bufory robocze pojedynczego kroku** (`x`, `hNorm`, `q`, `k`, `v`, `attnOut`, `woOut`, `ffGate`, `ffUp`, `ffOut`, `attnScores`, `logits`) oraz **bufory samplingu** (`workLogits`, `workProbs`, `topKIndices`, `topKValues`, `repSeen`).

---

### 2. Pełna specyfikacja funkcji i metod (`tartarus-engine.ts`)

| Funkcja / Metoda | Lokalizacja | Argumenty i typy | Co wykonuje | Zwracany rezultat | Zależności | Możliwe błędy |
| --- | --- | --- | --- | --- | --- | --- |
| `computeCrc32` | `L144–150` | `bytes: Uint8Array` | Liczy sumę kontrolną IEEE 802.3 z tablicą `CRC32_TABLE` (`L132–142`) | `number` (`uint32` bez znaku) | Brak | Brak (zwraca `0` dla pustego bufora) |
| `readVectorF32` | `L152–165` | `dv: DataView`, `offset: number`, `length: number` | Sprawdza `offset % 4 === 0` i granice, czyta `length` liczb `float32` LE | `{ vec: Float32Array, nextOffset: number }` | `DataView` | `Error` przy złym wyrównaniu lub wyjściu poza bufor |
| `readMatrixInt8Dequant` | `L167–199` | `buffer: ArrayBufferLike`, `baseByteOffset`, `dv`, `offset`, `rows`, `cols` | Czyta po wierszach `scale (f32)` + `cols × int8` i dekwantyzuje do `Float32Array` | `{ mat: Float32Array, nextOffset: number }` | `DataView`, `Int8Array` | `Error` przy `cols % 4 !== 0`, złym offsecie lub wyjściu poza bufor |
| `parseTNY3` | `L201–381` | `input: ArrayBuffer \| ArrayBufferView` | Weryfikuje 64 B nagłówek, flagi, alignment, padding, CRC-32, parsuje JSON tokenizera i dekwantyzuje wagi w kolejności kanonicznej | `ParsedTNY3` | `computeCrc32`, `readVectorF32`, `readMatrixInt8Dequant`, `TartarusTokenizer`, `TextDecoder` | `Error` dla uszkodzonego pliku, złego magic, CRC-32, flag, wersji lub `vocabSize` |
| `rmsNormInto` | `L383–400` | `out`, `x`, `xOffset`, `g: Float32Array`, `dim`, `eps` | Liczy `invRms` i wpisuje `x * invRms * g` do `out` | `void` (zapis w miejscu do `out`) | `Math.sqrt` | Brak jawnych (zakłada poprawne długości tablic) |
| `matVecInto` | `L402–423` | `out`, `outOffset`, `w`, `x`, `xOffset`, `rows`, `cols` | Mnożenie macierz × wektor z odwijaniem pętli po `4` (`c += 4`) | `void` (zapis do `out`) | Brak | Wymaga `cols % 4 === 0` (zapewnione przy ładowaniu) |
| `applyRopeHalfSplitInPlace` | `L425–447` | `vec`, `vecOffset`, `pos`, `nHeads`, `headDim`, `cosTable`, `sinTable` | Obraca w miejscu pary `(i, i + half)` dla wszystkich głowic | `void` | Prekomputowane tabele RoPE | Zakłada `0 <= pos < ctxLen` |
| `silu` | `L449–455` | `z: number` | Stabilne numerycznie $z \cdot \sigma(z)$ | `number` | `Math.exp` | Brak |
| `TartarusEngine.fromArrayBuffer` | `L554–556` | `input: ArrayBuffer \| ArrayBufferView` | Skrót: `new TartarusEngine(parseTNY3(input))` | `TartarusEngine` | `parseTNY3` | Błędy z `parseTNY3` |
| `encode` / `decode` | `L558–564` | `text: string` / `ids: Iterable<number>, options?` | Deleguje do osadzonego `this.tokenizer` | `number[]` / `string` | `TartarusTokenizer` | `RangeError` przy złym ID w `decode` |
| `resetCache` | `L566–568` | Brak | Zeruje licznik `this.kvPos = 0` (bez potrzeby zerowania całych tablic `kCache`/`vCache`) | `void` | Brak | Brak |
| `runTransformerTokenAtPos` | `L570–685` | `tokenId: number`, `pos: number`, `computeLogits: boolean` | Pełny forward pojedynczego tokenu na pozycji `pos` przez `L` warstw z zapisem/odczytem KV Cache; opcjonalnie wylicza `this.logits` | `void` (aktualizuje `this.x`, `kCache`, `vCache`, `layerRms`, opcjonalnie `logits`) | `rmsNormInto`, `matVecInto`, `applyRopeHalfSplitInPlace`, `silu` | `RangeError` gdy `tokenId ∉ [0, V)` lub `pos ∉ [0, CTX)` |
| `prefill` | `L687–698` | `tokenIds: ArrayLike<number>` | Resetuje `kvPos=0`, przetwarza tokeny `0 .. n-1` (licząc logity dopiero dla `i === n-1`), ustawia `kvPos = n` | `Float32Array` (`this.logits`) | `resetCache`, `runTransformerTokenAtPos` | `RangeError` gdy `n <= 0` lub `n > ctxLen` |
| `step` | `L703–708` | `tokenId: number` | Przetwarza pojedynczy token na pozycji `this.kvPos` (zawsze z `computeLogits=true`) i inkrementuje `this.kvPos` | `Float32Array` (`this.logits`) | `runTransformerTokenAtPos` | `RangeError` z `runTransformerTokenAtPos` gdy `kvPos >= ctxLen` |
| `forwardNoCache` | `L714–821` | `tokenIds: ArrayLike<number>` | Referencyjny pełny przebieg całej sekwencji `[0..T-1]` w nowo alokowanych tablicach bez modyfikacji `kCache`/`vCache`/`kvPos` | Nowy `Float32Array(vocabSize)` | `rmsNormInto`, `matVecInto`, `applyRopeHalfSplitInPlace`, `silu` | `RangeError` gdy `T <= 0` lub `T > ctxLen` lub zły `tid` |
| `sampleNextToken` | `L823–941` | `logits: Float32Array`, `options?: SamplingOptions`, `seenTokens?`, `rng?` | Kopiuje `logits` do `workLogits`, aplikuje unikalną karę `repetitionPenalty`, greedy lub `temperature + topK + minP` | `number` (`nextId`) | `Math.exp`, `rng` | Brak (zakłada poprawny rozmiar `logits`) |
| `generateStream` | `L943–1022` | `prompt: string \| readonly number[]`, `options?: GenerateOptions` | Generator: wykonuje `prefill`, a w pętli `sampleNextToken` + `step` z przyrostowym dekodowaniem UTF-8 i zwrotem statystyk | `Generator<StreamEvent, GenerateResult, void>` | `encode`, `prefill`, `step`, `sampleNextToken`, `createStreamDecoder` | `Error` dla pustego promptu; `RangeError` gdy `promptIds.length >= ctxLen` |
| `generate` | `L1024–1031` | `prompt: string \| readonly number[]`, `options?: GenerateOptions` | Konsumuje cały generator `generateStream` synchronicznie do końca | `GenerateResult` | `generateStream` | Jak w `generateStream` |

---

### 3. Porównanie `prefill(tokenIds)` vs `step(tokenId)` vs `forwardNoCache(tokenIds)`

| Właściwość | `prefill(tokenIds)` | `step(tokenId)` | `forwardNoCache(tokenIds)` |
| --- | --- | --- | --- |
| **Przeznaczenie** | Wczytanie całego promptu wejściowego na starcie generacji | Wygenerowanie kolejnego kroku na bazie istniejącego kontekstu | Wyłącznie **testy referencyjne** (`verify_ts.mjs`) |
| **Wejście** | Ciąg `N` tokenów (`1 <= N <= CTX`) | Pojedynczy token (`1` liczba) | Cała sekwencja `T` tokenów |
| **Czy czyści KV Cache?** | Tak (`resetCache()` → `kvPos = 0`) | Nie (dopisuje na `kvPos` i zwiększa `kvPos += 1`) | Nie używa i nie narusza `kCache`/`vCache`/`kvPos` |
| **Kiedy liczy `lmHead` (`logits`)?** | Wyłącznie dla **ostatniego** tokenu promptu (`i === n - 1`) — oszczędność $O((N-1) \cdot V \cdot D)$ mnożeń | Zawsze (`computeLogits = true`) | Wyłącznie dla ostatniego tokenu `(T - 1)` |
| **Alokacje w pamięci podczas wywołania** | **Zero** (zwraca wprost `this.logits`) | **Zero** (zwraca wprost `this.logits`) | Alokuje `12` tablic tymczasowych (`xSeq`, `qSeq`, `kSeq`, ...) |

**Ważna optymalizacja w `prefill()` (`tartarus-engine.ts:694`):**  
Dla promptu o długości `78` tokenów (jak `golden_pl_state_query`), dla pierwszych `77` tokenów interesują nas wyłącznie ich wektory **Kluczy (`K`) i Wartości (`V`)** zapisywane do KV Cache. Dopiero na 78. tokenie (`i === n - 1`, czyli na `<|npc|>`) potrzebujemy wektora `2048` logitów, by wybrać pierwsze słowo odpowiedzi NPC. Pominięcie `rms_final` i `lmHead` na `77` pierwszych pozycjach oszczędza $77 \times 2048 \times 192 \approx 30{,}3\text{ miliona}$ operacji mnożenia-dodawania przy każdym prompcie!

---

### 4. Przepływ pojedynczego promptu przez silnik (przykład z `golden_pl_state_query`)

1. Tekst `"<|state|> lokacja=Gaj Świętogora; ... <|user|> Ile punktów życia i złota ma nasza armia? <|npc|>"` trafia do `encode()`, zamieniając się w tablicę `78` identyfikatorów: `[2, 300, 66, ..., 4]`.
2. `generateStream()` wywołuje `prefill(promptIds)`:
   - Dla `i = 0 .. 76`: `runTransformerTokenAtPos(promptIds[i], i, false)` przechodzi `4` warstwy (RMSNorm → Q, K, V → RoPE → zapis `k, v` do `kCache`/`vCache` pod pozycją `i` → Attention po `0..i` → `wo` + residual → RMSNorm → SwiGLU + residual).
   - Dla `i = 77` (token `4` = `<|npc|>`): `runTransformerTokenAtPos(4, 77, true)` dodatkowo normuje wynik przez `rmsFinal` i mnoży przez `lmHead`, wypełniając `this.logits` (`2048` liczb). `kvPos` staje się równe `78`.
3. Pętla generacji (przy `temperature = 0`):
   - **Krok 0:** `sampleNextToken(this.logits)` wybiera najwyższy logit: indeks **`1296`** (`6.9162`), czyli `" Armia"`. Dekoder strumieniowy oddaje `" Armia"`. Ponieważ `1296 !== eosId`, wywołuje `step(1296)`.
   - `step(1296)` liczy tylko 1 token na pozycji `pos = 78` (korzystając z gotowych `K, V` w pozycjach `0..77` oraz dopisując pozycję `78`), zwiększa `kvPos` do `79` i zwraca nowe logity.
   - **Kroki 1..6:** Silnik dopisuje kolejne tokeny składające się na zdanie `" Armia przebywa w stolicy. "`, aż w kroku `6` wybiera token **`1`** (`<|eos|>`), zamyka dekoder strumieniowy i kończy z `stoppedBy = "eos"`.

---

## ROZDZIAŁ VI – KV Cache

**KV Cache** (*Key-Value Cache*) to pamięć podręczna przechowująca wektory Kluczy ($K$) oraz Wartości ($V$) już przetworzonych tokenów we wszystkich warstwach modelu.

### 1. Dlaczego przeliczanie całego kontekstu bez KV Cache jest kosztowne?

- **Poziom A:** Wyobraź sobie, że piszesz list słowo po słowie, ale przed dopisaniem każdego nowego słowa musisz przeczytać od nowa cały list i przepisać wszystkie wcześniejsze notatki na brudno. Jeśli prompt ma `100` słów, a odpowiedź `50` słów, to bez pamięci podręcznej 100 początkowych słów musiałoby przejść przez wszystkie macierze `wq, wk, wv, wo, w_gate, w_up, w_down` aż **50 razy** — mimo że wcześniejsze słowa w ogóle się nie zmieniły i przez maskę przyczynową nigdy nie patrzą w przyszłość!
- **Poziom B/C:** **POTWIERDZONE** — Bez KV Cache koszt wygenerowania $G$ tokenów po prompcie długości $P$ rośnie kwadratowo/sześciennie względem samych projekcji: wszystkie macierze liniowe musiałyby zostać pomnożone $\sum_{t=P}^{P+G-1} t$ razy. Z KV Cache każdy token (zarówno z promptu, jak i nowo wygenerowany) przechodzi przez macierze projekcji **dokładnie jeden raz**. Dla pozycji `pos` w kroku `step()` wykonujemy rzutowania tylko dla jednego wektora długości `D`, a w pętli Attention odczytujemy gotowe klucze i wartości z pamięci podręcznej dla `t = 0 .. pos`.

---

### 2. Prealokacja pamięci i układ `L × CTX × D`

**POTWIERDZONE** (`tartarus-engine.ts:472–474`, `529–530`):
```ts
this.kCache = new Float32Array(nLayers * ctxLen * dModel);
this.vCache = new Float32Array(nLayers * ctxLen * dModel);
```

- **Dlaczego `Float32Array` zamiast zwykłej tablicy `number[]`?**
  1. Zwykła tablica JS (`Array`) przechowuje referencje lub dynamicznie pakowane wartości i zmienia rozmiar przy `push()`, a wielowymiarowe tablice zagnieżdżone (`float[][][]`) tworzą setki osobnych obiektów w stercie JS, co powoduje skoki opóźnień (GC) i gorszą lokalność pamięci procesora (cache CPU).
  2. `Float32Array` to pojedynczy, ciągły blok surowych bajtów w pamięci (4 bajty na liczbę), bezpośrednio odpowiadający tablicom w C/Rust, w którym silnik V8 w przeglądarce i Node.js generuje szybki kod maszynowy bez sprawdzania typów w locie.
- **Rozmiar `L × CTX × D`:**
  - `L` (`nLayers`): każda warstwa ma własny niezależny zestaw projekcji `K` i `V`.
  - `CTX` (`ctxLen`): maksymalna liczba pozycji w kontekście (`256`).
  - `D` (`dModel`): ponieważ $D = H \times d_h$ (np. $192 = 6 \times 32$ w Profilu `S`), wszystkie `H` głowic z danej pozycji mieszczą się spójnie w jednym odcinku długości `D`. (Dla porównania w Pythonie `model_tartarus.py:194–197` używa kształtu 5D `[L, B, H, CTX, head_dim]`, co matematycznie zajmuje dokładnie tyle samo elementów przy `B=1`).

**Dokładny wzór indeksu w `kCache` i `vCache` (`tartarus-engine.ts:590, 599, 608, 616, 639`):**

$$\text{index}(l, \text{pos}, h, d) = l \cdot (\text{CTX} \cdot D) + \text{pos} \cdot D + h \cdot d_h + d$$

Gdzie:
- `layerStride = ctxLen * dModel` (`256 * 192 = 49 152` elementów dla jednej warstwy w Profilu `S`),
- `cacheLayerBase = l * layerStride`,
- `cachePosBase = cacheLayerBase + pos * dModel`,
- głowica `h` zaczyna się pod przesunięciem `hOff = h * headDim` (`h * 32`),
- `kCache.set(this.k, cachePosBase)` i `vCache.set(this.v, cachePosBase)` kopiują cały wektor `K`/`V` bieżącego tokenu (wszystkie głowice naraz) jednym wywołaniem natywnym bez alokacji.

---

### 3. Schemat działania KV Cache w pętli generacji

```mermaid
sequenceDiagram
    participant P as Prompt (N tokenow)
    participant E as TartarusEngine
    participant KV as kCache / vCache [L x CTX x D]

    Note over E,KV: Faza 1: prefill(promptIds)
    E->>E: resetCache() ustawia kvPos = 0
    loop i = 0 .. N-1
        E->>E: RMSNorm -> Q, K, V -> RoPE(pos=i)
        E->>KV: Zapis K[i], V[i] dla warstw 0..L-1
        E->>KV: Attention czyta K[0..i], V[0..i]
    end
    E->>E: Tylko dla i = N-1: rmsFinal + lmHead -> logits
    E->>E: kvPos = N

    Note over E,KV: Faza 2: Petla generacji (step)
    loop dopoki nie EOS i kvPos < CTX
        E->>E: sampleNextToken(logits) -> nextId
        E->>E: step(nextId) na pozycji pos = kvPos
        E->>KV: Zapis K[pos], V[pos] dla warstw 0..L-1
        E->>KV: Attention czyta K[0..pos], V[0..pos]
        E->>E: rmsFinal + lmHead -> nowe logits; kvPos += 1
    end
```

---

### 4. Pamięć robocza vs KV Cache oraz dlaczego brak alokacji chroni przed Garbage Collectorem

- **Pamięć robocza (`x, hNorm, q, k, v, attnOut, woOut, ffGate, ffUp, ffOut, attnScores, logits`):** Jest nadpisywana w miejscu przy **każdym pojedynczym tokenie** i trzyma wyłącznie stan bieżącego kroku.
- **KV Cache (`kCache`, `vCache`):** Jest pamięcią **stanową między tokenami** — wartości zapisane pod pozycją `0 .. kvPos - 1` nie mogą zostać nadpisane w trakcie jednej odpowiedzi, bo każdy kolejny krok korzysta z nich w Attention.
- **Znaczenie dla JavaScriptu i Garbage Collectora (GC):**
  JavaScript posiada automatyczne odśmiecanie pamięci (Garbage Collector). Jeśli w pętli wykonywanej kilkadziesiąt razy na sekundę tworzysz nowe obiekty typu `new Float32Array(...)` albo `[]`, zapełniasz tzw. *nursery/young generation* w silniku V8/SpiderMonkey. Gdy pamięć tymczasowa się zapełni, przeglądarka na chwilę zamraża wątek główny, by posprzątać stare tablice (*GC pause*), co w grze lub streamingu powoduje mikroprzycięcia klatek i wyższe zużycie baterii. Dzięki prealokacji i pętli `for (let i = 0; i < dModel; i++) this.x[i] = tokEmb[embBase + i]` (zamiast `.subarray()`, które tworzyłoby drobny obiekt widoku) metoda `step()` wykonuje **zero alokacji tablic** na token.

---

### 5. Obsługa `kvPos`, limitu `CTX` i ograniczenia tej implementacji

- **Jak działa `kvPos`:** Przy `resetCache()` ustawiane jest tylko `this.kvPos = 0` (`tartarus-engine.ts:567`) — nie trzeba nawet zerować megabajtów tablic `kCache` i `vCache`, ponieważ pętla Attention zawsze czyta wyłącznie indeksy `t = 0 .. pos`, czyli nigdy nie sięga poza aktualny `kvPos`.
- **Limit `CTX`:**
  - `prefill()` rzuca `RangeError`, jeśli `promptIds.length > ctxLen`.
  - `generateStream()` wymaga `promptIds.length < ctxLen` (gdyż musi zostać miejsce na minimum 1 nowy token), oblicza `ctxRemaining = ctxLen - promptIds.length`, przycina `maxNew = Math.min(requestedMaxNew, ctxRemaining)` i po wyczerpaniu okna kończy z `stoppedBy = "ctx"` (`tartarus-engine.ts:948–1000`).
- **OGRANICZENIA obecnego KV Cache:**
  1. **Jedna konwersacja naraz (`resetCache()` przy każdym `prefill()`):** Nie ma metody dopisującej tylko nowy fragment dialogu bez przeliczania wcześniejszych tur od `pos = 0`.
  2. **Brak przesuwnego okna (*Sliding Window / Rolling Cache*):** Po dojściu do `256` tokenów generacja bezwzględnie się zatrzymuje.
  3. **Sztywny `batch_size = 1`:** Cache nie ma wymiaru batcha (odpowiada pojedynczemu strumieniowi rozmowy).

---

## ROZDZIAŁ VII – Sampling i generowanie tekstu

**POTWIERDZONE** (`tartarus-engine.ts:823–1031`):

1. **Greedy decoding (`temperature <= 0`):** Wybiera deterministycznie token o najwyższym logicie (`argmax`). W `verify_ts.mjs` na `golden_pl_state_query` generuje zawsze `7` tokenów: `" Armia przebywa w stolicy. "` + `<|eos|>`.
2. **Temperature (`temperature > 0`):** Dzieli logity przez `temperature` przed softmaxem (szczegóły i przykład w [Matematyce Tiny](./TINY_MATHEMATICS.md#82-greedy-decoding-texttemperature-le-0)).
3. **Top-K (`topK > 0 && topK < vocabSize`):** W `tartarus-engine.ts:857–889` działa **sortowanie przez wstawianie dla stałego K w prealokowanych tablicach** `topKValues` i `topKIndices` (zero tworzenia par `[idx, val]` i zero `Array.prototype.sort`).
4. **Min-P (`0 < minP < 1`):** Odfiltrowuje ogon rozkładu poniżej `minP * maxProb` *po* policzeniu `Math.exp`, zachowując co najmniej najlepszego kandydata (`maxProb` ma próg `1.0 >= minP`).
5. **Repetition Penalty (`repetitionPenalty > 1.0`):** Przegląda `seenTokens` (w `generateStream()` są to wyłącznie `generatedTokens`, **nie** tokeny promptu) i za pomocą tablicy flag `this.repSeen: Uint8Array(vocabSize)` nakłada karę **ściśle jeden raz na każdy unikalny widziany token**, po czym zeruje tylko dotknięte indeksy w `repSeen` (`O(seenTokens.length)` zamiast czyszczenia całych `2048` bajtów).
6. **Zatrzymywanie generacji:**
   - Gdy `stopOnEos !== false` oraz `nextId === this.eosId` → `stoppedBy = "eos"`.
   - Gdy wyczerpie się `ctxLen - promptIds.length` przed osiągnięciem `requestedMaxNew` → `stoppedBy = "ctx"`.
   - W przeciwnym razie po `requestedMaxNew` krokach → `stoppedBy = "max_tokens"`.
7. **Jak działa `generateStream()` (`tartarus-engine.ts:943–1022`):**
   Jest to funkcja generatorowa JS (`function*`). Wykonuje najpierw `prefill(promptIds)`, mierzy czas `prefillMs`, a następnie w każdym kroku `stepIdx`:
   - losuje `nextId = sampleNextToken(...)`,
   - dekoduje przyrost tekstu `textDelta` przez `streamDecoder` (`flush()` na `<|eos|>` lub ostatnim kroku, inaczej `push(nextId)`),
   - oddaje zdarzenie `yield { step, tokenId, textDelta, pos, isEos, layerRms }` (dzięki czemu UI może rysować tekst i słupki warstw na bieżąco po każdym tokenie albo przerwać pętlę przez `abortFlag`),
   - jeśli nie jest to ostatni krok i nie zatrzymano pętli, wywołuje `currentLogits = this.step(nextId)` pod kolejny krok,
   - na koniec zwraca obiekt podsumowania `GenerateResult` (`promptTokens`, `generatedTokens`, pełny `text`, `stoppedBy`, `prefillMs`, `decodeMs`, `tokensPerSec`, `totalPos`).

---

## ROZDZIAŁ IX – Interfejs przeglądarkowy (`tartarus-browser.html` i `build_browser.mjs`)

Plik `/home/user/tartarus-browser.html` (`2415` linii, `88 592 B`) jest ewolucją koncepcji nieobecnego w repozytorium pliku `tiny-2.1-browser-v2.html`. Jest składany deterministycznie przez skrypt `/home/user/build_browser.mjs` (`1418` linii), który bierze skompilowane pliki `tartarus_tokenizer.js` oraz `tartarus-engine.js`, usuwa nagłówki modułowe `import`/`export` i osadza oba moduły bezpośrednio wewnątrz jednego tagu `<script>` (IIFE). Dzięki temu strona działa bez serwera (`file://`), w piaskownicy Arena (`sandbox="allow-scripts"` bez dostępu do sieci) i przez zwykły serwer HTTP.

### 1. Podział odpowiedzialności: Silnik vs Interfejs demonstracyjny

| Element / Funkcja | Gdzie należy? | Lokalizacja | Opis działania |
| --- | --- | --- | --- |
| Parsowanie `.tny3`, CRC-32, dekwantyzacja, `TartarusTokenizer`, `prefill`, `step`, `generateStream`, sampling, `layerRms` | **Silnik** (`tartarus-engine.ts` + `tartarus_tokenizer.ts`) | `tartarus-browser.html:833–1860` | Czysta logika matematyczna i tokenizacyjna; nic nie wie o HTML |
| Wczytywanie pliku z dysku (`<input type="file">`) i **Drag & Drop** na całe okno | **Interfejs UI** | `L2157–2166`, `L2318–2341` | Czyta plik przez `file.arrayBuffer()` i przekazuje surowy bufor do `loadModelBuffer()` |
| **Auto-fetch** `./model.tny3` | **Interfejs UI** | `L2169–2184`, `L2399–2407` | Wykonywany cicho przy starcie wyłącznie gdy `location.protocol.startsWith("http")`, lub na kliknięcie przycisku „Auto-fetch ./model.tny3” |
| **Symulator `<|state|>`**, `<|user|>`, 5 presetów RPG, **tryb surowego promptu** | **Interfejs UI** | `L1865–1889`, `L1975–1995`, `L2343–2365` | Skleja tekst `<|state|> ... <|user|> ... <|npc|>` (lub bierze surowy wpis) i podświetla tagi specjalne w podglądzie |
| **Zabezpieczenie limitu `CTX`** przed uruchomieniem | **Interfejs UI** (+ walidacja w silniku) | `L2058–2103` | Na każdą zmianę tekstu liczy `engine.encode(promptStr).length`, rysuje pasek `X / CTX` (żółty `>= 80%`, czerwony `>= 100%`) i wyłącza przycisk `btnGenerate` przy braku miejsca w `CTX` |
| **Sterowanie samplingiem** (suwaki `Temperature`, `Top-K`, `Min-P`, `Rep. Penalty`, `Max New Tokens`, `Stop na <|eos|>`, przycisk `Greedy 0.0`) | **Interfejs UI** → przekazywane jako `opts` do `engine.generateStream()` | `L2214–2221`, `L2376–2389` | Tłumaczy pozycje suwaków na obiekt `GenerateOptions` |
| **Monitor warstw (RMS)** o dynamicznej liczbie wierszy | **Interfejs UI** zasilany przez `ev.layerRms` | `buildDynamicLayersUI` (`L1999–2031`), `updateLayerTelemetry` (`L2033–2046`) | Tworzy dokładnie `h.nLayers` pasków (`L=4` dla Profilu `S`, `L=6` dla `M`) po wczytaniu `.tny3`. Wewnątrz pętli silnika `layerRms[l] = Math.sqrt(sumSq / dModel)` mierzy pierwiastek ze średniego kwadratu (RMS) strumienia rezydualnego `this.x` po zakończeniu warstwy `l` (`tartarus-engine.ts:669–675`). W UI pasek skaluje się względem `Math.max(maxRms * 1.15, 2.0)`. |
| **Streaming odpowiedzi**, `tok/s`, **monitor KV Cache**, **inspektor tokenów**, **historia testów** | **Interfejs UI** | `runStreamingGeneration` (`L2190–2311`), `updateKvMeter` (`L2048–2056`) | Woła `gen.next()`, wypisuje przyrosty `ev.textDelta`, dopisuje kolorowe żetony `#id` + znak (lub bajty `[0xc5]` przy połówce znaku UTF-8), liczy `tok/s` z `performance.now()`, oddaje kontrolę przeglądarce przez `await nextFrameYield()` (`setTimeout(resolve, 0)`), aby odmalować ekran i umożliwić kliknięcie „Przerwij” (`abortFlag`) |

---

## ROZDZIAŁ XI – Wydajność i pamięć

Wszystkie wzory i liczby zostały wyliczone wprost z kodu `tartarus-engine.ts`, `model_tartarus.py` oraz zweryfikowane skryptem pomiarowym.

### 1. Wzory na zapotrzebowanie na pamięć RAM w silniku `TartarusEngine`

Niech $V = \text{vocabSize}$, $\text{CTX} = \text{ctxLen}$, $D = \text{dModel}$, $L = \text{nLayers}$, $H = \text{nHeads}$, $d_h = D / H$, $D_{\text{ff}} = \text{dFf}$.

1. **Liczba parametrów (przy `tiedEmbeddings = true`):**
   $$N_{\text{param}} = V \cdot D + L \cdot \big(2D + 4D^2 + 3 D \cdot D_{\text{ff}}\big) + D$$
2. **Pamięć wag po dekwantyzacji W8A32 (bajtów):**
   $$M_{\text{weights}} = 4 \cdot N_{\text{param}} \quad \text{(gdy } \text{tied}=\text{false: } + 4 \cdot V \cdot D\text{)}$$
3. **Pamięć KV Cache (`kCache` + `vCache`, bajtów):**
   $$M_{\text{KV}} = 2 \cdot L \cdot \text{CTX} \cdot D \cdot 4$$
   Zauważ liniową zależność od **wszystkich trzech** wymiarów: podwojenie `L`, `CTX` albo `D` podwaja pamięć KV Cache.
4. **Tablice RoPE (`ropeCos` + `ropeSin`, bajtów):**
   $$M_{\text{RoPE}} = 2 \cdot \text{CTX} \cdot \frac{d_h}{2} \cdot 4 = 4 \cdot \text{CTX} \cdot d_h$$
5. **Bufory robocze kroku, samplingu i telemetrii (`tartarus-engine.ts:531–548`, bajtów):**
   $$M_{\text{scratch}} = 4 \cdot (7D + 3D_{\text{ff}} + \text{CTX} + 4V + L) + 4V + V = 4(7D + 3D_{\text{ff}} + \text{CTX} + L) + 21V$$
   *(Bufory `Float32Array`: `x, hNorm, q, k, v, attnOut, woOut` = $7D$; `ffGate, ffUp, ffOut` = $3D_{\text{ff}}$; `attnScores` = $\text{CTX}$; `logits, workLogits, workProbs, topKValues` = $4V$; `layerRms` = $L$; do tego `topKIndices: Int32Array(V)` = $4V$ B oraz `repSeen: Uint8Array(V)` = $1V$ B).*

Sprawdźmy dokładną sumę z uwzględnieniem wszystkich 4 tablic `Float32Array(V)`:
- Dla Profilu `S` (`V=2048`): `4*(7*192 + 3*512 + 256 + 4) + 21*2048 = 4*(1344 + 1536 + 260) + 43008 = 12560 + 43008 = 55 568 B` (`54.27 KiB`).
- Dla Profilu `M` (`V=2048`): `4*(7*256 + 3*640 + 256 + 6) + 21*2048 = 4*(1792 + 1920 + 262) + 43008 = 15896 + 43008 = 58 904 B` (`57.52 KiB`).

### 2. Zestawienie zużycia pamięci: Profil `S` vs Profil `M`

**POTWIERDZONE** (obliczone z rzeczywistych struktur w `tartarus-engine.ts` i zweryfikowane rozmiarami plików `.tny3`):

| Składnik | Profil `S` (`model.tny3`) | Profil `M` (`/tmp/model_m.tny3`) |
| --- | --- | --- |
| Parametry ($N_{\text{param}}$) | `2 164 416` (`2.16M`) | `5 049 600` (`5.05M`) |
| Rozmiar pliku `.tny3` na dysku | `2 390 228 B` (`2.28 MiB`) | `5 309 908 B` (`5.06 MiB`) |
| W tym sekcja wag INT8 (`weights_len`) | `2 209 536 B` (`2.11 MiB`) | `5 129 216 B` (`4.89 MiB`) |
| RAM wag po dekwantyzacji W8A32 | `8 657 664 B` (`8.26 MiB`) | `20 198 400 B` (`19.26 MiB`) |
| RAM KV Cache (`kCache + vCache`) | `1 572 864 B` (`1536.0 KiB` / `1.50 MiB`) | `3 145 728 B` (`3072.0 KiB` / `3.00 MiB`) |
| RAM tablic RoPE (`cos + sin`) | `32 768 B` (`32.0 KiB`) | `32 768 B` (`32.0 KiB`) |
| RAM buforów roboczych + samplingu | `55 568 B` (`54.3 KiB`) | `58 904 B` (`57.5 KiB`) |
| **Łącznie tablice silnika w RAM (bez obiektów JS tokenizera i tymczasowego `ArrayBuffer` z odczytu pliku)** | **`10 318 864 B` (`9.84 MiB`)** | **`23 435 800 B` (`22.35 MiB`)** |

*(Uwaga: w trakcie wywołania `parseTNY3()`, jeśli caller trzyma nadal źródłowy `ArrayBuffer` w pamięci — np. plik `.tny3` z dysku — chwilowy szczyt pamięci obejmuje także ten bufor `2.28 MiB` lub `5.06 MiB`, dopóki po zakończeniu `parseTNY3()` referencja do `ArrayBuffer` nie zostanie zwolniona).*

### 3. Zmierzona wydajność (Node.js vs środowisko przeglądarkowe)

| Środowisko pomiarowe | Profil | Ładowanie + Dekwantyzacja | `prefill` (prompt `77` tok) | Prędkość `decode` (`tok/s`) | Czas na 1 token | Status i źródło |
| --- | --- | --- | --- | --- | --- | --- |
| **Node.js v20.20.2** (CPU sandbox, 2 vCPU, brak GPU) | **S** (`L=4, D=192`) | `29.0 ms` | `228.9 ms` (`~2.97 ms/tok`) | **`242.8 tok/s`** | `4.12 ms` | **POTWIERDZONE** (`verify_ts.mjs`, 10×64 tok po rozgrzaniu JIT; wcześniej `236.5–268.0 tok/s`) |
| **Node.js v20.20.2** (CPU sandbox, 2 vCPU) | **M** (`L=6, D=256`) | — (niezmierzone osobno) | — | **`118.8 tok/s`** | `~8.4 ms` | **POTWIERDZONE** (`verify_ts.mjs`, `16` tok po EOS; wcześniej `97.7–122.5 tok/s`) |
| **Symulowany DOM w `node:vm`** (`tartarus-browser.html`, z `setTimeout(0)` co token) | **S** (`L=4, D=192`) | — | — | **`154.3 tok/s`** | `~6.5 ms` (wliczając `setTimeout(0)` + budowę DOM) | **POTWIERDZONE** wyłącznie dla symulatora `node:vm` |
| **Prawdziwa przeglądarka GUI** (Chrome / Firefox / Safari / urządzenia mobilne) | **S / M** | — | — | — | — | **NIEZWERYFIKOWANE** (w środowisku brak binarnej przeglądarki; w HTML5 zagnieżdżony `setTimeout(0)` może dodatkowo ograniczyć streaming od ~5. tokenu do ok. 4 ms/krok) |

**Wpływ parametrów architektury na koszt obliczeń pojedynczego kroku `step()`:**
- **Liczba warstw $L$ i wymiar $D$:** Koszt mnożeń macierz × wektor w jednej warstwie wynosi $O(4D^2 + 3D \cdot D_{\text{ff}})$. Przejście z Profilu `S` ($L=4, D=192, D_{\text{ff}}=512 \implies 1{,}769{,}472$ mnożeń/warstwa) do Profilu `M` ($L=6, D=256, D_{\text{ff}}=640 \implies 3{,}014{,}656$ mnożeń/warstwa) zwiększa koszt warstw o ok. **$2{,}56\times$** — co niemal dokładnie tłumaczy spadek z `242.8 tok/s` do `118.8 tok/s` (`~2.04×` wolniej, bo stały narzut `lmHead` $V \cdot D$ rośnie tylko $1{,}33\times$).
- **Wpływ `CTX`:** Dopóki nie zmienia się faktyczna długość sekwencji `pos`, sama większa pojemność `CTX` zmienia głównie rozmiar `kCache`/`vCache` w RAM, a w czasie obliczeń koszt Attention rośnie dopiero liniowo z bieżącą pozycją `pos` ($O(L \cdot \text{pos} \cdot D)$), co przy `pos <= 256` jest ułamkiem kosztu mnożeń macierzy liniowych.
- **Rola Garbage Collectora w benchmarku Node.js:** W teście `10 × 64 = 640` wygenerowanych tokenów przyrost `process.memoryUsage().heapUsed` wyniósł `404.5 KiB` (ok. `647 B` na token *włącznie* z logowaniem i przed uruchomieniem pełnego cyklu GC dla obiektów zwracanych przez generator `StreamEvent` i stringów wynikowych). Wewnątrz samej ścieżki obliczeń tensorowych (`step()`) nie powstaje żadna nowa tablica numeryczna.
