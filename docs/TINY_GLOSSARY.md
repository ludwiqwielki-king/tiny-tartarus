# Słownik pojęć projektu Tiny (Tartarus)

> Część dokumentacji systemu Tiny: [← Powrót do głównego przewodnika](./TINY_SYSTEM_DOCUMENTATION.md)  
> Powiązane dokumenty: [Architektura i Tokenizer](./TINY_ARCHITECTURE.md) · [Matematyka](./TINY_MATHEMATICS.md) · [Format `.tny3`](./TINY_FORMAT_TNY3.md) · [Silnik i KV Cache](./TINY_RUNTIME.md) · [Ścieżka nauki](./TINY_LEARNING_PATH.md)

Każdy termin wyjaśniono w trzech punktach: **prostym językiem**, **technicznie** oraz **na przykładzie w Tiny** (z dokładnym odwołaniem do kodu i danych projektu).

---

## Indeks alfabetyczny / tematyczny (46 wymaganych haseł + pojęcia pomocnicze)

1. [Attention](#1-attention) · [Autoregresja](#2-autoregresja) · [BPE](#3-bpe) · [Buffer](#4-buffer) · [Byte-level](#5-byte-level)
2. [Context window](#6-context-window) · [CRC-32](#7-crc-32) · [CSR (pomocnicze)](#47-csr-compressed-sparse-row--format-treningowy) · [CTX](#8-ctx) · [Decode](#9-decode)
3. [Dekwantyzacja](#10-dekwantyzacja) · [DOM](#11-dom) · [Embedding](#12-embedding) · [ESM](#13-esm) · [Float32](#14-float32)
4. [Garbage Collector](#15-garbage-collector) · [Golden test](#16-golden-test) · [Greedy decoding](#17-greedy-decoding) · [Inferencja](#18-inferencja) · [INT8](#19-int8)
5. [Key](#20-key) · [KV Cache](#21-kv-cache) · [Kwantyzacja](#22-kwantyzacja) · [Little-Endian (pomocnicze)](#48-little-endian) · [Logit parity](#23-logit-parity)
6. [Logits](#24-logits) · [Macierz](#25-macierz) · [Min-P](#26-min-p) · [Numerical tolerance](#27-numerical-tolerance) · [Prealokacja](#28-prealokacja)
7. [Prefill](#29-prefill) · [Query](#30-query) · [Repetition penalty](#31-repetition-penalty) · [RMSNorm](#32-rmsnorm) · [RoPE](#33-rope)
8. [Runtime](#34-runtime) · [Sampling](#35-sampling) · [Softmax](#36-softmax) · [SwiGLU](#37-swiglu) · [Temperature](#38-temperature)
9. [Tensor](#39-tensor) · [Tied embeddings (pomocnicze)](#49-tied-embeddings) · [Token](#40-token) · [Tokenizer](#41-tokenizer) · [Top-K](#42-top-k)
10. [Transformer](#43-transformer) · [Value](#44-value) · [W8A32](#45-w8a32) · [Wektor](#46-wektor)

---

### 1. Attention
- **Prostym językiem:** Mechanizm „rozglądania się wstecz”, dzięki któremu bieżące słowo potrafi odnaleźć w wcześniejszej części zdania potrzebne informacje (np. na końcu pytania o złoto potrafi przypomnieć sobie liczbę sprzed kilku linijek).
- **Technicznie:** Mechanizm *Multi-Head Causal Self-Attention*, w którym dla każdej głowicy liczy się znormalizowane przez Softmax iloczyny skalarne $Q \cdot K^T / \sqrt{d_h}$ wyłącznie dla pozycji $t \le \text{pos}$, a wynik jest ważoną sumą wektorów $V$ rzutowaną przez $W_o$.
- **Przykład w Tiny:** `TartarusBlock._attention()` w `model_tartarus.py:237–276` oraz pętla głowic w `tartarus-engine.ts:601–654` (`6` głowic w Profilu `S`, `8` w Profilu `M`).

### 2. Autoregresja
- **Prostym językiem:** Sposób pisania tekstu po jednym kawałku naraz, gdzie każde świeżo dopisane słowo staje się od razu częścią kontekstu przy wyborze kolejnego słowa.
- **Technicznie:** Generacja iteracyjna $P(x_t \mid x_0, x_1, \dots, x_{t-1})$, w której wybrany w kroku $t$ token $x_t$ jest natychmiast podawany jako wejście w kroku $t+1$.
- **Przykład w Tiny:** Pętla w `TartarusEngine.generateStream()` (`tartarus-engine.ts:963–1003`), w której wybrany `nextId` wędruje bezpośrednio do `this.step(nextId)`.

### 3. BPE
- **Prostym językiem:** *Byte Pair Encoding* — metoda budowania słownika polegająca na wielokrotnym sklejaniu dwóch najczęściej sąsiadujących ze sobą klocków w jeden większy klocek.
- **Technicznie:** Algorytm kompresji i podziału podwyrazowego (*subword*), który podczas treningu zapisuje uporządkowaną listę scaleń (`merges`), a podczas kodowania stosuje je według priorytetu rangi (`rank`: najwcześniej nauczone scalenie wygrywa).
- **Przykład w Tiny:** `1787` par w `tokenizer.json` (np. `[202, 135] -> 262`, czyli bajty `0xC5 0x82` sklejone w polską literę `'ł'`), stosowanych przez `encodeChunk()` (`tartarus_tokenizer.ts:204–257`).

### 4. Buffer
- **Prostym językiem:** Wydzielony w pamięci RAM ciągły „pojemnik” bajtów lub liczb, po którym program porusza się bezpośrednio po adresach bez ciągłego tworzenia nowych małych obiektów.
- **Technicznie:** Niskopoziomowy obszar pamięci liniowej reprezentowany w JS przez `ArrayBuffer` (surowe bajty) oraz widoki typowane (`Uint8Array`, `Int8Array`, `Float32Array`, `DataView`).
- **Przykład w Tiny:** Plik `model.tny3` czytany jest jako jeden `ArrayBuffer` w `parseTNY3()` (`tartarus-engine.ts:201`), a bufor `byteBuf` w `decode()` zbiera bajty tokenów przed konwersją UTF-8 (`tartarus_tokenizer.ts:301`).

### 5. Byte-level
- **Prostym językiem:** Podejście, w którym podstawowymi cegiełkami słownika nie są litery alfabetu ani słowa, lecz pojedyncze bajty `0..255` zapisu komputerowego UTF-8, więc żaden znak świata nigdy nie okaże się „nieznany”.
- **Technicznie:** Kodowanie mapujące każdy bajt wejściowego ciągu UTF-8 na bazowy identyfikator `byteOffset + byte` (w Tiny `5..260`) przed wykonaniem scaleń BPE.
- **Przykład w Tiny:** `tartarus_tokenizer.ts:116–120` oraz `210–214` koduje polskie znaki 2-bajtowe (np. `'ż'` → bajty `0xC5 0xBC` → `[202, 193]` → scalone do ID `309`).

### 6. Context window
- **Prostym językiem:** Maksymalna pojemność „krótkotrwałej pamięci” modelu — łączna liczba tokenów (stan gry + pytanie + odpowiedź), jaką model potrafi objąć uwagą w jednym momencie.
- **Technicznie:** Maksymalny rozmiar osi czasu w tablicach RoPE i KV Cache, poza którym nie istnieją pozycje pozycyjne i generacja musi się zatrzymać.
- **Przykład w Tiny:** Okno długości `256` tokenów (`ctx_len = 256` w `model_tartarus.py:35` i `header.ctxLen` w `model.tny3`).

### 7. CRC-32
- **Prostym językiem:** 32-bitowa „cyfra kontrolna” całego bloku danych pliku — wystarczy zmiana nawet jednego bitu w wagach lub tokenizerze, aby kontrolna suma się nie zgodziła i program odrzucił zepsuty plik.
- **Technicznie:** Cykliczny kod nadmiarowy IEEE 802.3 z wielomianem `0xEDB88320`, wyliczany w bajtach przez `zlib.crc32()` przy eksporcie oraz `computeCrc32()` w `tartarus-engine.ts:144–150` nad `raw[64 .. EOF)`.
- **Przykład w Tiny:** Dla `model.tny3` wynosi `0x938fb800`; w `verify_ts.mjs:92–100` odwrócenie 1 bitu payloadu natychmiast powoduje błąd CRC-32 w `parseTNY3()`.

### 8. CTX
- **Prostym językiem:** Krótki zapis rozmiaru okna kontekstowego (*Context window*).
- **Technicznie:** Stała konfiguracyjna `ctx_len` (`ctxLen` w `TNY3Header`), wyznaczająca długość alokowanego `kCache`/`vCache` (`L × CTX × D`) i warunek stopu `"ctx"`.
- **Przykład w Tiny:** W `tartarus-browser.html:2073–2095` pasek „Budżet kontekstu (Prompt / CTX)” pilnuje limitu `256` tokenów i blokuje przycisk generacji przy `remaining <= 0`.

### 9. Decode
- **Prostym językiem:** Ma dwa powiązane znaczenia: (1) w tokenizerze – zamiana numerów tokenów z powrotem na czytelny polski tekst; (2) w silniku inferencji – faza dopisywania kolejnych tokenów odpowiedzi po wczytaniu promptu.
- **Technicznie:** (1) `TartarusTokenizer.decode(ids, options)` składa bajty tokenów i dekoduje UTF-8 (`tartarus_tokenizer.ts:296`); (2) faza autoregresji wykonywana przez `step(tokenId)` po `prefill()`, mierzona w `decodeMs` i `tokensPerSec` (`tartarus-engine.ts:963–1019`).
- **Przykład w Tiny:** W benchmarku `verify_ts.mjs:289` faza decode Profilu `S` osiąga w Node.js `242.8 tok/s` (`4.12 ms/tok`), a `decode([1296])` daje `" Armia"`.

### 10. Dekwantyzacja
- **Prostym językiem:** Odtworzenie przybliżonych liczb ułamkowych (`Float32`) z liczb całkowitych (`INT8`) zapisanych w małym pliku na dysku przez pomnożenie ich przez zapisany mnożnik skali wiersza.
- **Technicznie:** Operacja $\widehat{W}_{r, c} = q_{r, c} \cdot \text{scale}_r$ wykonywana w locie podczas parsowania sekcji wag w `readMatrixInt8Dequant()`.
- **Przykład w Tiny:** `tartarus-engine.ts:189–196` odtwarza przy wczytaniu `model.tny3` (`29 ms` w Node.js) wagi macierzy z `int8` do `Float32Array`.

### 11. DOM
- **Prostym językiem:** *Document Object Model* — struktura elementów strony internetowej w przeglądarce (przyciski, pola tekstowe, paski postępu, paragrafy), którą JavaScript modyfikuje, żeby coś pokazać na ekranie.
- **Technicznie:** Przeglądarkowe API (`window`, `document`, `HTMLElement`) oddzielone celowo od silnika obliczeniowego.
- **Przykład w Tiny:** Pliki `tartarus-engine.ts` i `tartarus_tokenizer.ts` mają **zero** odwołań do DOM (kompilują się bez `lib.dom.d.ts`), natomiast `tartarus-browser.html:1890–2410` używa DOM wyłącznie w warstwie interfejsu użytkownika.

### 12. Embedding
- **Prostym językiem:** Tabela przypisująca każdemu numerowi tokenu jego „wizytówkę matematyczną” — listę liczb opisującą znaczenie i kontekst użycia danego klocka tekstu.
- **Technicznie:** Macierz słownika $W^{\text{emb}} \in \mathbb{R}^{V \times D}$, gdzie odczyt wiersza o indeksie `tokenId` zamienia dyskretny symbol na ciągły wektor w przestrzeni ukrytej $\mathbb{R}^{D}$.
- **Przykład w Tiny:** `tok_emb.weight` o wymiarze `[2048, 192]` w Profilu `S` (`model_tartarus.py:323`, `tartarus-engine.ts:579–583`).

### 13. ESM
- **Prostym językiem:** *ECMAScript Modules* — nowoczesny, oficjalny standard dzielenia kodu JavaScript na osobne pliki za pomocą słów `import` i `export`, działający zarówno w przeglądarkach, jak i w Node.js.
- **Technicznie:** Statyczny system modułów JS skonfigurowany w `tsconfig.json` (`"module": "ES2022"`), emitujący pliki `.js` / `.mjs` obsługiwane przez `import ... from "./tartarus-engine.mjs"`.
- **Przykład w Tiny:** `verify_ts.mjs:19` ładuje `tartarus-engine.mjs` przez dynamiczny `import(ENGINE_URL)`, a `tartarus-engine.mjs:12` importuje `TartarusTokenizer` z `./tartarus_tokenizer.mjs`.

### 14. Float32
- **Prostym językiem:** Standardowy format przechowywania liczb ułamkowych w komputerach na **32 bitach (4 bajtach)**, dający około 7 cyfr dokładności po przecinku.
- **Technicznie:** Liczba zmiennoprzecinkowa pojedynczej precyzji IEEE-754 (*binary32*), reprezentowana w JS przez bufor `Float32Array` i czytana metodą `DataView.getFloat32(offset, true)`.
- **Przykład w Tiny:** Wektory wag norm (`rms_att.g`, `rms_ffn.g`, `rms_final.g`), wartości `scale` macierzy INT8, tablice `ropeCos`/`ropeSin`, `kCache`/`vCache` oraz zdekwantyzowane wagi w pamięci RAM są typu `Float32`.

### 15. Garbage Collector
- **Prostym językiem:** „Sprzątacz pamięci” wbudowany w przeglądarkę i Node.js, który co jakiś czas zatrzymuje na ułamek sekundy program i wyrzuca z pamięci RAM niepotrzebne już, stare tablice i obiekty.
- **Technicznie:** Automatyczny mechanizm zarządzania stertą (np. V8 Scavenger + Mark-Sweep-Compact); częste alokacje krótkotrwałych obiektów w gorących pętlach powodują presję na GC i skoki latencji.
- **Przykład w Tiny:** `TartarusEngine` prealokuje wszystkie `Float32Array` z góry w konstruktorze (`L529–551`), dzięki czemu metoda `step()` nie tworzy żadnych nowych tablic i nie obciąża GC.

### 16. Golden test
- **Prostym językiem:** „Złoty wzorzec testowy” — zamrożone dane wejściowe wraz z dokładnymi wynikami liczbowymi z oryginalnego programu w Pythonie, służące do sprawdzenia, czy wersja w TypeScript liczy tak samo.
- **Technicznie:** Plik referencyjny `golden_test.json` (`182 843 B`) wygenerowany w Etapie 4 przez `export_tny3.py:440–521`, zawierający 3 przypadki z `token_ids` i pełnymi `last_step_logits` (`2048` floatów każdy).
- **Przykład w Tiny:** `verify_ts.mjs:102–169` przepuszcza `golden_test.json` przez `forwardNoCache`, `prefill` i `prefill + 5×step`, osiągając `maxAbs <= 2.146e-6`.

### 17. Greedy decoding
- **Prostym językiem:** Najprostszy sposób wyboru kolejnego tokenu: zawsze bierzemy ten dokładnie jeden kandydat, który dostał najwyższą ocenę od modelu (zero losowości).
- **Technicznie:** Deterministyczna reguła wyboru $\arg\max_v z_v$, uruchamiana w `sampleNextToken()` gdy `temperature <= 0.0` (`tartarus-engine.ts:843–852`).
- **Przykład w Tiny:** Przycisk `Ustaw Greedy (T=0.0)` w `tartarus-browser.html:700, 2383–2386` ustawia temperaturę `0.00`, dając zawsze powtarzalną odpowiedź `" Armia przebywa w stolicy. "` dla pierwszego presetu.

### 18. Inferencja
- **Prostym językiem:** Używanie gotowego, już nauczonego modelu do udzielania odpowiedzi — w przeciwieństwie do treningu, model niczego już się nie uczy i nie zmienia swoich wag.
- **Technicznie:** Przebieg obliczeń wyłącznie w przód (*forward pass*) na zamrożonych wagach w trybie ewaluacyjnym (`model.eval()` / `TartarusEngine`).
- **Przykład w Tiny:** Cały plik `tartarus-engine.ts` wykonuje wyłącznie inferencję (brak gradientów, brak funkcji straty i brak wstecznej propagacji).

### 19. INT8
- **Prostym językiem:** Mała liczba całkowita mieszcząca się w **1 bajcie (8 bitach)**, obejmująca ze znakiem wartości od `-128` do `+127` (w Tiny wykorzystuje się symetryczny przedział `-127..+127`).
- **Technicznie:** 8-bitowa liczba całkowita ze znakiem (`int8_t` / `Int8Array`), zajmująca 4× mniej miejsca na dysku niż `Float32`.
- **Przykład w Tiny:** Wszystkie wagi macierzy 2D w `model.tny3` pakuje `pack_matrix_int8()` (`export_tny3.py:116–128`) i odczytuje `new Int8Array(...)` (`tartarus-engine.ts:191`).

### 20. Key
- **Prostym językiem:** „Etykieta” każdego wcześniejszego tokenu w pamięci podręcznej Attention — mówi przyszłym tokenom, jakiego rodzaju informację ten token przechowuje.
- **Technicznie:** Wektor $K = W_k \, h_{\text{norm}}$ po obrocie RoPE, porównywany przez iloczyn skalarny z zapytaniem $Q$ i zapisywany w `kCache`.
- **Przykład w Tiny:** `this.k` (`Float32Array(dModel)`) rzutowany przez `lw.wk` i kopiowany do `this.kCache` w `tartarus-engine.ts:592–600`.

### 21. KV Cache
- **Prostym językiem:** Brudnopis z gotowymi etykietami (`Key`) i zawartościami (`Value`) dla wszystkich słów, które model już przeczytał — żeby nie przeliczać ich od zera przy każdym kolejnym wygenerowanym słowie.
- **Technicznie:** Prealokowane bufory `kCache` i `vCache` o rozmiarze `nLayers × ctxLen × dModel`, do których `prefill()` i `step()` zapisują nowe wektory pod `kvPos` i z których `Attention` czyta zakres `0 .. pos`.
- **Przykład w Tiny:** `768.0 KiB + 768.0 KiB = 1.50 MiB` dla Profilu `S` (`tartarus-engine.ts:529–530`). Szczegóły w [Rozdziale VI](./TINY_RUNTIME.md#rozdział-vi--kv-cache).

### 22. Kwantyzacja
- **Prostym językiem:** Zmniejszenie wagi pliku modelu przez „zaokrąglenie” precyzyjnych liczb ułamkowych z `Float32` do siatki 255 małych kroczków `INT8` (jak zamiana dokładnego wzrostu `178.342 cm` na skalę od `-127` do `+127`).
- **Technicznie:** Symetryczne skalowanie wierszowe `scale = max(|W[r]|)/127` i zapis `round(W/scale)` w typie `int8` przy zachowaniu bardzo małego błędu uogólnienia.
- **Przykład w Tiny:** `quantize_per_row_int8()` w `export_tny3.py:98–113` oraz `quantized_copy()` w `train_tartarus.py:212–230` (spadek `val_loss` po kwantyzacji w ewaluacji Etapu 3 wyniósł zaledwie `+0.0009`).

### 23. Logit parity
- **Prostym językiem:** Zgodność surowych ocen punktowych (`logitów`) pomiędzy dwiema różnymi wersjami silnika (Python PyTorch vs TypeScript/JavaScript) na tym samym tekście wejściowym.
- **Technicznie:** Weryfikacja, że $\max_v |\text{logits}_{\text{TS}}[v] - \text{logits}_{\text{PyTorch}}[v]| < \varepsilon_{\text{tol}}$ dla każdego wektora w `golden_test.json`.
- **Przykład w Tiny:** `verify_ts.mjs` potwierdza `logit parity` na poziomie `2.146e-6` (`< 1e-4`) dla `forwardNoCache`, `prefill` i `prefill + 5×step`.

### 24. Logits
- **Prostym językiem:** Surowe „punkty sympatii” przyznane przez model każdemu z `2048` tokenów w słowniku na końcu obliczeń — zanim jeszcze zamienimy je na procenty (prawdopodobieństwa). Mogą być dodatnie lub ujemne.
- **Technicznie:** Wektor wyjściowy $z \in \mathbb{R}^{V}$ z warstwy projekcji słownikowej `lmHead(rmsFinal(x))`.
- **Przykład w Tiny:** W pierwszym przypadku `golden_test.json` token `1296` (`" Armia"`) dostał logit `6.9162`, wyprzedzając drugi najwyższy logit `4.3545` (`" Pod"`).

### 25. Macierz
- **Prostym językiem:** Prostokątna tabela liczb o określonej liczbie wierszy i kolumn (np. `512` wierszy po `192` liczby). Większość wiedzy modelu siedzi właśnie w takich tabelach wag.
- **Technicznie:** Dwuwymiarowa tablica liczb (`[rows, cols]`), spłaszczona w `tartarus-engine.ts` do jednowymiarowego `Float32Array(rows * cols)` w układzie wierszowym (*row-major*: wiersz `r` zaczyna się pod `r * cols`).
- **Przykład w Tiny:** Macierz `w_gate` ma rozmiar `[512, 192]` w Profilu `S` i `[640, 256]` w Profilu `M`, mnożona przez wektor funkcją `matVecInto()` (`tartarus-engine.ts:402–423`).

### 26. Min-P
- **Prostym językiem:** Sprytny filtr odrzucający absurdalnie mało prawdopodobne słowa: jeśli najlepszy kandydat ma np. `80%` szans, a `minP = 0.06`, ignorujemy wszystko słabsze niż `6%` z tych `80%` (czyli poniżej `4.8%`).
- **Technicznie:** Adaptacyjny próg odcięcia prawdopodobieństw $p_i < \text{minP} \cdot \max_j p_j$ przed losowaniem z dystrybuanty (`tartarus-engine.ts:914–926`).
- **Przykład w Tiny:** Domyślna wartość suwaka `Min-P` w `tartarus-browser.html:683` to `0.06`.

### 27. Numerical tolerance
- **Prostym językiem:** Dopuszczalny mikroskopijny margines różnicy między wynikami na różnych komputerach lub w różnych językach programowania, wynikający z zaokrągleń ułamków dziesiętnych.
- **Technicznie:** Górny próg błędu $\max |A - B| < 10^{-4}$ (vs PyTorch) oraz `< 10^{-5}` (KV Cache vs przebieg pełny w TS), zdefiniowany w `verify_ts.mjs:119, 149`.
- **Przykład w Tiny:** Zmierzona różnica `2.146e-6` jest blisko 50× mniejsza niż tolerancja `1e-4`.

### 28. Prealokacja
- **Prostym językiem:** Zarezerwowanie z góry, jeszcze przed rozpoczęciem rozmowy, stałych kawałków pamięci RAM na wszystkie potrzeby modelu — żeby w trakcie pisania odpowiedzi nie szukać już nowego miejsca w pamięci.
- **Technicznie:** Jednorazowe utworzenie stałych instancji `Float32Array`, `Int32Array` i `Uint8Array` w konstruktorze `TartarusEngine` (`tartarus-engine.ts:529–551`).
- **Przykład w Tiny:** `kCache`, `vCache`, `x`, `q`, `k`, `v`, `workLogits` itd. powstają wyłącznie raz przy `new TartarusEngine(parsed)`.

### 29. Prefill
- **Prostym językiem:** Pierwsza faza odpowiedzi modelu: szybkie „przeczytanie i zanotowanie w brudnopisie” całego wprowadzonego promptu (stanu gry i pytania gracza) aż do tokenu `<|npc|>`.
- **Technicznie:** Metoda `prefill(tokenIds)` (`tartarus-engine.ts:687–698`), która zeruje `kvPos`, wypełnia `kCache` i `vCache` dla pozycji `0..N-1` i wylicza wektor `logits` wyłącznie dla ostatniego tokenu promptu.
- **Przykład w Tiny:** Dla `77`-tokenowego promptu `prefill` trwa w Node.js średnio `228.9 ms` (`verify_ts.mjs:289`).

### 30. Query
- **Prostym językiem:** „Pytanie wyszukiwania”, z jakim bieżący token zwraca się do wcześniejszych tokenów w mechanizmie Attention (np. „gdzie wcześniej było napisane o złocie?”).
- **Technicznie:** Wektor $Q = W_q \, h_{\text{norm}}$ po obrocie pozycyjnym RoPE (`tartarus-engine.ts:592, 596`).
- **Przykład w Tiny:** Bufor `this.q` (`Float32Array(dModel)`) dzielony na `6` lub `8` głowic przy liczeniu iloczynów z `this.kCache`.

### 31. Repetition penalty
- **Prostym językiem:** „Kara za zacinanie się” — obniża szanse tokenów, które model już przed chwilą wypowiedział, aby uniknąć zapętlenia typu „Armia, armia, armia...”.
- **Technicznie:** Współczynnik $\ge 1.0$, przez który dzielone są dodatnie logity (lub mnożone ujemne) tokenów obecnych wcześniej w wygenerowanej sekwencji.
- **Przykład w Tiny:** Domyślnie `1.15` w UI (`tartarus-browser.html:687`); w `tartarus-engine.ts:832–841` tablica `repSeen` gwarantuje, że kara nakładana jest tylko raz na każde unikalne ID tokenu.

### 32. RMSNorm
- **Prostym językiem:** Prosty „wyrównywacz skali” liczb w wektorze — dzieli liczby przez ich średnią energię (pierwiastek ze średniej kwadratów) i przemnaża przez wyuczone wagi `g`, żeby sygnał nie eksplodował ani nie wygasł w głębokich warstwach.
- **Technicznie:** *Root Mean Square Layer Normalization* bez odejmowania średniej i bez biasu, liczona ze stałą $\varepsilon = 10^{-5}$ w `float32`.
- **Przykład w Tiny:** Funkcja `rmsNormInto()` (`tartarus-engine.ts:383–400`) i klasa `RMSNorm` (`model_tartarus.py:127–139`), występująca 2× na warstwę (`rms_att`, `rms_ffn`) plus `rms_final`.

### 33. RoPE
- **Prostym językiem:** *Rotary Position Embedding* — matematyczny „obrót wskazówki” nadawany wektorom `Q` i `K`, dzięki któremu model odróżnia, które słowo było wcześniej, a które później i jak daleko od siebie stoją.
- **Technicznie:** Obroty 2D na parach współrzędnych w konwencji **half-split** `(i, i + headDim/2)` o kąt $p \cdot \Theta^{-2i/d_h}$ ($\Theta = 10000.0$).
- **Przykład w Tiny:** Tablice `ropeCos`/`ropeSin` prekomputowane w konstruktorze `TartarusEngine` (`L515–527`) i funkcja `applyRopeHalfSplitInPlace()` (`L425–447`).

### 34. Runtime
- **Prostym językiem:** Środowisko uruchomieniowe oraz sam „silnik wykonawczy”, który potrafi uruchomić gotowy model poza programem do treningu (np. w przeglądarce użytkownika albo na serwerze Node.js).
- **Technicznie:** Połączenie warstwy wykonawczej JS (`V8` w Chrome/Node) oraz kodu `TartarusEngine` + `TartarusTokenizer` realizującego inferencję bez PyTorcha.
- **Przykład w Tiny:** Samodzielne pliki `tartarus-engine.ts/.mjs` i `tartarus_tokenizer.ts/.mjs` działają w Node.js v20.20.2 oraz w izolowanym `<script>` w `tartarus-browser.html`.

### 35. Sampling
- **Prostym językiem:** Metoda wybierania następnego tokenu z ocen przyznanych przez model — od wyboru w 100% pewnego (greedy) po kontrolowane losowanie różnorodnych odpowiedzi.
- **Technicznie:** Funkcja przekształcająca wektor `logits` w konkretne ID tokenu za pomocą parametrów `temperature`, `topK`, `minP` i `repetitionPenalty`.
- **Przykład w Tiny:** Metoda `TartarusEngine.sampleNextToken()` w `tartarus-engine.ts:823–941`.

### 36. Softmax
- **Prostym językiem:** Wzór zamieniający dowolne surowe punkty (dodatnie, ujemne, wielkie, małe) na ładne prawdopodobieństwa z przedziału `0%..100%`, które zawsze sumują się dokładnie do `100%` (`1.0`).
- **Technicznie:** Funkcja wykładnicza unormowana z odejmowaniem maksimum dla stabilności numerycznej: $\text{softmax}(z)_i = \exp(z_i - m) / \sum_j \exp(z_j - m)$.
- **Przykład w Tiny:** Występuje w dwóch miejscach: w wagach uwagi `Attention` (`tartarus-engine.ts:622–630`) oraz w losowaniu kolejnego tokenu w `sampleNextToken()` (`L897–911`).

### 37. SwiGLU
- **Prostym językiem:** Dwutorowa część myślowa każdej warstwy, w której jedna ścieżka działa jak inteligentny zawór (`SiLU`), decydując, ile sygnału z drugiej ścieżki przepuścić dalej.
- **Technicznie:** Wariant sieci FFN bez biasu o równaniu $W_{\text{down}}(\text{SiLU}(W_{\text{gate}} h) \odot (W_{\text{up}} h))$, zwiększający wymiar przejściowy z $D$ do $D_{\text{ff}}$.
- **Przykład w Tiny:** Wymuszane przez flagę bitową `FLAG_SWIGLU` (`1 << 2`), z wymiarem `d_ff = 512` (Profil `S`) i `640` (Profil `M`), `tartarus-engine.ts:660–667`.

### 38. Temperature
- **Prostym językiem:** Suwak „kreatywności / ostrożności”: przy `0.0` model wybiera wyłącznie najpewniejsze słowo, przy małej wartości (`0.3`) trzyma się bezpiecznych odpowiedzi, a powyżej `1.0` chętniej ryzykuje i losuje rzadsze słowa.
- **Technicznie:** Dzielnik logitów $z / T$ przed `Softmax`; wartość `<= 0` omija `Softmax` i włącza `greedy decoding`.
- **Przykład w Tiny:** Suwak `Temperature` (`0.00` – `1.50`) w `tartarus-browser.html:675`.

### 39. Tensor
- **Prostym językiem:** Ogólna nazwa na tablicę liczb o dowolnej liczbie wymiarów — 0 wymiarów to pojedyncza liczba (skalar), 1 wymiar to wektor, 2 wymiary to macierz, a 3 lub więcej wymiarów to kostka liczb.
- **Technicznie:** Wielowymiarowa struktura danych w PyTorch (`torch.Tensor`); w silniku JS wszystkie tensory 2D i 3D spłaszczane są do jednowymiarowych `Float32Array` z jawnym obliczaniem przesunięć.
- **Przykład w Tiny:** W `model_tartarus.py:194` `KVCache.k` jest 5-wymiarowym tensorem `[n_layers, batch, n_heads, ctx_len, head_dim]`, któremu w TS odpowiada spłaszczony `Float32Array(nLayers * ctxLen * dModel)`.

### 40. Token
- **Prostym językiem:** Podstawowy „klocek” tekstu w modelu — może to być całe krótkie słowo ze spacją (np. `" złoto"`), kawałek dłuższego słowa (`"togora"`), znak przystankowy (`"="`), pojedynczy bajt (`0xC5`) albo znacznik sterujący (`<|npc|>`).
- **Technicznie:** Dyskretna jednostka słownika identyfikowana liczbą całkowitą `0 <= id < vocab_size`.
- **Przykład w Tiny:** Słownik ma `2048` tokenów: `0..4` specjalne, `5..260` surowe bajty, `261..2047` scalenia BPE; ID `1296` oznacza `" Armia"`.

### 41. Tokenizer
- **Prostym językiem:** Dwukierunkowy tłumacz stojący na wejściu i wyjściu modelu: zamienia ludzki tekst na listę numerów tokenów (`encode`) oraz z powrotem numery na tekst (`decode`).
- **Technicznie:** Moduł deterministycznego BPE realizowany przez identyczne algorytmicznie klasy `TartarusTokenizer` w `tokenizer_tartarus.py` i `tartarus_tokenizer.ts`.
- **Przykład w Tiny:** Tokenizuje `"<|state|> hp=42/75; złoto=390 <|user|> Ile złota? <|npc|>"` w `17` liczb `[2, 303, ..., 4]`.

### 42. Top-K
- **Prostym językiem:** Ograniczenie wyboru kolejnego słowa wyłącznie do `K` najlepszych propozycji modelu — np. przy `topK = 32` z `2048` możliwych tokenów w grze zostają tylko `32` najwyżej ocenione.
- **Technicznie:** Wybór `K` największych przeskalowanych logitów przed policzeniem `Softmax` w `sampleNextToken()` (`tartarus-engine.ts:857–889`). Wartość `0` wyłącza filtr.
- **Przykład w Tiny:** Domyślne ustawienie suwaka `Top-K` w `tartarus-browser.html:679` wynosi `32`.

### 43. Transformer
- **Prostym językiem:** Współczesna architektura sieci neuronowych do języka, zbudowana z powtarzanych pięter (warstw), w których każde słowo najpierw zbiera kontekst od innych słów (*Attention*), a potem go przetwarza (*MLP*).
- **Technicznie:** W Tiny: wariant dekoderowy (*causal decoder-only*) w stylu LLaMA (`RMSNorm` pre-norm, `RoPE`, bez biasu, `SwiGLU`, `tied embeddings`).
- **Przykład w Tiny:** Klasa `TartarusModel` w `model_tartarus.py:307` (`4` warstwy w Profilu `S`, `6` w Profilu `M`).

### 44. Value
- **Prostym językiem:** „Treść notatki” przypisana każdemu przetworzonemu tokenowi — kiedy zapytanie `Query` dopasuje się do klucza `Key` wcześniejszego słowa, to właśnie ważona suma jego wektorów `Value` trafia dalej do sieci.
- **Technicznie:** Wektor $V = W_v \, h_{\text{norm}}$ zapisywany w `vCache` bez obrotu RoPE (RoPE stosuje się wyłącznie do `Q` i `K`).
- **Przykład w Tiny:** `this.v` rzutowany przez `lw.wv` i kopiowany do `this.vCache` w `tartarus-engine.ts:593, 600`.

### 45. W8A32
- **Prostym językiem:** Zapis oznaczający mieszankę dwóch precyzji: **Wagi 8-bitowe** (mały plik na dysku w `INT8`), ale **Aktywacje i obliczenia 32-bitowe** (`Float32` po załadowaniu do RAM).
- **Technicznie:** Strategia przechowywania i inferencji, w której `parseTNY3()` dekwantyzuje wagi przy starcie do `Float32Array`, a cała arytmetyka forward pass przebiega w `Float32`.
- **Przykład w Tiny:** Zmniejsza rozmiar `model.tny3` do `2.28 MiB` przy zachowaniu bardzo małej różnicy `val_loss` (`+0.0009` w Etapie 3) i prostoty silnika JS.

### 46. Wektor
- **Prostym językiem:** Jednowymiarowa lista liczb o stałej długości (np. `192` liczb opisujących jeden token lub `2048` liczb z ocenami kandydatów na końcu).
- **Technicznie:** Jednowymiarowa tablica `Float32Array(dim)`.
- **Przykład w Tiny:** Wektor ukryty `this.x = new Float32Array(dModel)` oraz parametry skalujące `rmsAtt`, `rmsFfn`, `rmsFinal` o długości `dModel`.

---

### Pojęcia pomocnicze używane w dokumentacji Tiny

#### 47. CSR (*Compressed Sparse Row* – format treningowy)
- **Prostym językiem:** Sposób spakowania `20 000` zdań różnej długości do jednego ciągłego pliku bez marnowania miejsca na puste wypełniacze (`<|pad|>`), z osobnym spisem, gdzie zaczyna się każde zdanie.
- **Technicznie:** Trzy surowe pliki binarne bez nagłówka (`{split}_tokens.bin` `<u2`, `{split}_mask.bin` `u1`, `{split}_offsets.bin` `<u8` o długości $N+1$) używane przez `CSRSplit` w `train_tartarus.py:45–93` przez `np.memmap`.
- **Przykład w Tiny:** `train_tokens.bin` (`3 027 680 B`), `train_mask.bin` (`1 513 840 B`), `train_offsets.bin` (`160 008 B`).

#### 48. Little-Endian
- **Prostym językiem:** Kolejność zapisu wielobajtowych liczb na dysku, w której **najmłodszy (najmniej znaczący) bajt** idzie jako pierwszy — standard we współczesnych procesorach x86/ARM.
- **Technicznie:** Wyrażana w Pythonie prefiksem `<` (`<4s2H6I2f4I2I`) i drugim argumentem `true` w `DataView.getUint32(off, true)` / `getFloat32(off, true)`.
- **Przykład w Tiny:** Liczba `vocab_size = 2048` (`0x00000800`) w pliku `model.tny3` pod offsetem `0x08` zapisana jest jako bajty `00 08 00 00`.

#### 49. Tied embeddings
- **Prostym językiem:** Trik oszczędzający pamięć: ta sama tabela, która zamienia numer tokenu na wektor na samym wejściu modelu, służy na samym wyjściu jako tabela porównawcza do obliczenia logitów.
- **Technicznie:** Współdzielenie parametrów `tok_emb.weight` oraz projekcji wyjściowej `lmHead` (flaga `FLAG_TIED_EMBEDDINGS = 1`).
- **Przykład w Tiny:** W Profilu `S` oszczędza to `393 216` parametrów (~`15%` całego modelu).
