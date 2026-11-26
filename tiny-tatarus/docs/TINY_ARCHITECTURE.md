# Architektura modelu i Tokenizacja w projekcie Tiny (Tartarus)

> Część dokumentacji systemu Tiny: [← Powrót do głównego przewodnika](./TINY_SYSTEM_DOCUMENTATION.md)  
> Powiązane dokumenty: [Matematyka operacji](./TINY_MATHEMATICS.md) · [Format `.tny3`](./TINY_FORMAT_TNY3.md) · [Silnik i KV Cache](./TINY_RUNTIME.md) · [Słownik pojęć](./TINY_GLOSSARY.md)

Wszystkie twierdzenia oznaczono etykietami statusu zdefiniowanymi w [głównym przewodniku](./TINY_SYSTEM_DOCUMENTATION.md#legenda-statusów-stosowanych-w-całej-dokumentacji): **POTWIERDZONE**, **ZAIMPLEMENTOWANE**, **ZAŁOŻENIE PROJEKTOWE**, **NIEZWERYFIKOWANE**, **OGRANICZENIE**, **DO WYJAŚNIENIA**.

---

## ROZDZIAŁ II – Podstawy działania modelu (w zestawieniu 1:1 z Tiny)

Ten rozdział tłumaczy od zera, jak działa model językowy używany w projekcie Tiny (z nazwą domenową *Tartarus: The Fall of the Ancients*) i przyporządkowuje każdy krok do konkretnego miejsca w kodzie Pythona (`model_tartarus.py`) oraz TypeScriptu (`tartarus-engine.ts`). Szczegółowe wyprowadzenia matematyczne i kody niskopoziomowe znajdują się w [Matematyce Tiny](./TINY_MATHEMATICS.md).

### 1. Czym jest model językowy w projekcie Tiny?

- **Poziom A (intuicyjny):** Model językowy to program-zgadywacz następnego fragmentu tekstu. Dostaje dotychczasową rozmowę oraz opis sytuacji w grze (np. ilość złota i punkty życia armii) i przewiduje, jaki mały kawałek tekstu (token) powinien pojawić się jako kolejny. Powtarzając to zgadywanie token po tokenie, układa pełną odpowiedź postaci NPC (*Non-Player Character* – postaci niezależnej w grze).
- **Poziom B (techniczny):** **POTWIERDZONE** — Tiny jest **dekoderowym Transformerem autoregresyjnym** (*decoder-only causal Transformer*), zaimplementowanym w klasie `TartarusModel` (`model_tartarus.py:307–470`) oraz odtworzonym 1:1 w klasie `TartarusEngine` (`tartarus-engine.ts:457–1032`). Przyjmuje ciąg liczb całkowitych (identyfikatorów tokenów) o długości maksymalnie `ctx_len = 256` i wylicza wektor `vocab_size = 2048` ocen (logitów) dla kolejnego tokenu.
- **Poziom C (implementacyjny):** Architektura nie posiada enkodera (jak w modelach tłumaczących typu T5/BERT). Cały kontekst — stan gry, pytanie gracza i początek odpowiedzi NPC — płynie w jednym ciągu tokenów przez `n_layers` identycznych bloków (`TartarusBlock`, `4` w Profilu `S`, `6` w Profilu `M`) z maską przyczynową (token na pozycji $t$ może patrzeć wyłącznie na pozycje $0 \dots t$).

---

### 2. Czym jest token i jak tekst zamienia się na liczby?

- **Poziom A:** Komputer nie liczy na literach, tylko na liczbach. Zanim tekst trafi do sieci neuronowej, dzielony jest na krótkie, często powtarzające się klocki zwane **tokenami** (np. słowo `" złoto"`, znak `"="`, cyfry `"390"`, lub znacznik `"<|user|>"`), a każdy klocek ma swój stały numer porządkowy z tabeli o rozmiarze `2048`.
- **Poziom B:** **POTWIERDZONE** — W Tiny słownik ma rozmiar `vocab_size = 2048` (`tokenizer.json`). Numery `0..4` to tokeny sterujące (specjalne), numery `5..260` to wszystkie `256` możliwych bajtów, a numery `261..2047` to `1787` najczęstszych połączeń bajtów wyuczonych na polskim korpusie treningowym. Szczegóły w [Rozdziale III](#rozdział-iii--tokenizer-tartarus_tokenizerts-i-zgodność-z-pythonem).
- **Poziom C:** Zamianę wykonuje `TartarusTokenizer.encode(text)` (`tartarus_tokenizer.ts:259–294`), zwracając tablicę `number[]` (w JS/TS) lub `List[int]` (w Pythonie, `tokenizer_tartarus.py:239–250`), mieszczącą się w typie `uint16` (`<u2`) w plikach treningowych CSR (`dataset_meta.json`).

---

### 3. Wektory, macierze i embeddingi w kontekście Tiny

- **Poziom A:**
  - **Wektor** to po prostu lista liczb ułożona w jednym rzędzie – jak jeden wiersz w arkuszu kalkulacyjnym. W Profilu `S` każdy pojedynczy token opisywany jest przez listę `192` liczb (`d_model = 192`).
  - **Macierz** to prostokątna tabela liczb (wiele wierszy i wiele kolumn). Wagi warstw modelu są takimi właśnie tabelami.
  - **Embedding** to „słownik tłumaczący numer tokenu na wektor liczb”: jeśli w tabeli `tok_emb` znajduje się `2048` wierszy po `192` liczby, to token nr `354` oznacza pobranie wiersza `354` z tej tabeli.
- **Poziom B:** **POTWIERDZONE** — Zarówno `model_tartarus.py:323` (`self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)`), jak i `tartarus-engine.ts:579–583` odczytują dla tokenu o identyfikatorze `tid` wiersz o długości `dModel` ze spłaszczonej tablicy `Float32Array` o długości `vocabSize * dModel`.
- **Poziom C:** W `tartarus-engine.ts:579–583` w pamięci leży płaska tablica jednowymiarowa `tokEmb` o długości `2048 × 192 = 393 216` liczb zmiennoprzecinkowych (`Float32Array`). Wiersz dla tokenu `tid` zaczyna się pod indeksem `embBase = tid * dModel`, więc wektor wejściowy `x` otrzymuje wartości `tokEmb[embBase + 0 .. embBase + dModel - 1]`.

---

### 4. Czym jest warstwa sieci neuronowej i jak działa Transformer w Tiny?

- **Poziom A:** Jedna warstwa to pojedynczy etap przetwarzania informacji. Wyobraź sobie biurko redakcyjne z `4` (Profil `S`) lub `6` (Profil `M`) stanowiskami ustawionymi jedno za drugim. Na każdym stanowisku wektor tokenu najpierw sprawdza, co powiedziały wcześniejsze tokeny (**Attention**), a potem samodzielnie przetwarza zebraną wiedzę (**MLP / SwiGLU**). Wynik nie zastępuje poprzedniego stanu, lecz **dodaje się** do niego (połączenie rezydualne: *residual connection*), dzięki czemu podstawowa informacja nie ginie po drodze.
- **Poziom B:** **POTWIERDZONE** — Pojedyncza warstwa to klasa `TartarusBlock` (`model_tartarus.py:211–299`), a w TypeScript pętla `for (let l = 0; l < nLayers; l++)` w `runTransformerTokenAtPos` (`tartarus-engine.ts:588–677`) oraz w `forwardNoCache` (`tartarus-engine.ts:744–814`). Tiny używa wariantu **Pre-Norm** (normalizacja `RMSNorm` jest wykonywana *przed* blokiem Attention oraz *przed* blokiem FFN, a nie po nich).
- **Poziom C:** Przepływ jednego bloku w Tiny dla stanu `x` (wektor długości `D`):

```
x ──► RMSNorm(rms_att) ──► Attention(wq, wk, wv, RoPE, causal mask, wo) ──(+)──► x'
│                                                                           ▲
└───────────────────────────────────────────────────────────────────────────┘

x' ──► RMSNorm(rms_ffn) ──► SwiGLU FFN(w_gate, w_up, SiLU, w_down) ────────(+)──► x_out
│                                                                           ▲
└───────────────────────────────────────────────────────────────────────────┘
```

---

### 5. Mechanizm Attention oraz rola Q, K i V

- **Poziom A:** Gdy model przetwarza słowo `"złoto"` albo znak `":"` na końcu pytania, musi przypomnieć sobie, jaka liczba była napisana wcześniej przy `złoto=390`. Mechanizm **Attention (Uwaga)** działa jak wyszukiwarka w notatkach:
  - **Q (Query – Zapytanie):** „Czego w tym momencie szukam?” (np. aktualny token szuka liczby powiązanej ze złotem).
  - **K (Key – Klucz):** „Etykieta na grzbiecie teczki” każdego wcześniejszego tokenu (np. token `złoto` wystawia etykietę oznaczającą kategorię zasobu, a `390` etykietę wartości liczbowej).
  - **V (Value – Wartość):** „Faktyczna zawartość teczki”, która zostanie przekazana dalej, jeśli zapytanie Q dopasuje się do klucza K.
- **Poziom B:** **POTWIERDZONE** — Tiny stosuje **Multi-Head Causal Self-Attention** bez wyrazów wolnych (`bias=False`). Wektor ukryty o długości `d_model` rzutowany jest trzema macierzami (`wq`, `wk`, `wv`) na wektory `Q`, `K`, `V`, a następnie dzielony na `n_heads` niezależnych „głowic” (`6` głowic po `32` liczby w Profilu `S`; `8` głowic po `32` liczby w Profilu `M`). Dzięki temu jedna głowica może śledzić relacje gramatyczne, a inna powiązania między liczbami ze stanu gry.
- **Poziom C:** W `model_tartarus.py:246–273` w treningu i referencji PyTorch używa `F.scaled_dot_product_attention(..., is_causal=True)`. W `tartarus-engine.ts:601–654` silnik robi dokładnie to samo jawnie: oblicza iloczyn skalarny $Q_h \cdot K_{t,h} / \sqrt{\text{headDim}}$ dla $t \le \text{pos}$, stabilizuje numerycznie przez odjęcie maksimum (`maxScore`), liczy `Math.exp` i normuje przez `sumExp` (softmax), sumuje wektory `V` z ważeniem prawdopodobieństwami i na końcu scala głowice macierzą wyjściową `wo`. Szczegóły w [Matematyce Tiny](./TINY_MATHEMATICS.md#5-multi-head-causal-self-attention-q-k-v).

---

### 6. Normalizacja (`RMSNorm`), `RoPE` i `MLP` (`SwiGLU`)

- **Normalizacja (`RMSNorm`):**
  - **Poziom A:** W trakcie wielokrotnego dodawania liczb przez kolejne warstwy wartości w wektorze mogą urosnąć lub zmaleć zbyt mocno, co psuje obliczenia. `RMSNorm` przed każdym ważnym krokiem przeskalowuje wektor tak, aby jego „energia” (średnia kwadratowa wartości) wróciła do kontrolowanej skali, mnożąc potem wynik przez wyuczone współczynniki `g`.
  - **Poziom B/C:** **POTWIERDZONE** — Zamiast starszego `LayerNorm` (który odejmuje średnią i używa biasu), Tiny stosuje `RMSNorm` (`model_tartarus.py:127–139`, `tartarus-engine.ts:383–400`) ze stałą `eps = 1e-5` (zapisywaną i odczytywaną jako `float32`).
- **Pozycyjne osadzenia obrotowe (`RoPE` – *Rotary Position Embedding*):**
  - **Poziom A:** Sam mechanizm Attention patrzy na zestaw słów jak na worek – nie wiedziałby, czy `"Wojewoda pokonał Smoka"`, czy `"Smok pokonał Wojewodę"`. `RoPE` nadaje każdemu wektorowi `Q` i `K` drobny „obrót” zależny od numeru pozycji tokenu w zdaniu (`pos = 0, 1, 2, ...`), dzięki czemu odległość między tokenami przekłada się bezpośrednio na kąt między ich wektorami.
  - **Poziom B/C:** **POTWIERDZONE** — Tiny używa konwencji **half-split** (`rope_theta = 10000.0`, flaga bit `1 == 0`), gdzie pierwsza połowa współrzędnych każdej głowicy (`0 .. headDim/2 - 1`) tworzy pary z drugą połową (`headDim/2 .. headDim - 1`). Kod: `apply_rope_half_split` (`model_tartarus.py:162–177`) oraz `applyRopeHalfSplitInPlace` (`tartarus-engine.ts:425–447`).
- **Blok `MLP` z bramką `SwiGLU`:**
  - **Poziom A:** Jeśli Attention to narada między tokenami, to MLP (*Multi-Layer Perceptron*) jest chwilą cichego przemyślenia dla każdego tokenu z osobna. Wariant **SwiGLU** ma dwie równoległe gałęzie: jedna oblicza treść (`w_up`), a druga pełni rolę zaworu/bramki (`w_gate` z nieliniowością `SiLU`), która decyduje, co przepuścić do wyjścia (`w_down`).
  - **Poziom B/C:** **POTWIERDZONE** — `model_tartarus.py:278–279` i `tartarus-engine.ts:660–675` rzutują wektor z wymiaru `d_model` (`192` lub `256`) do szerszego wymiaru ukrytego `d_ff` (`512` w Profilu `S`, `640` w Profilu `M`), liczą $\text{SiLU}(W_{\text{gate}} h) \odot (W_{\text{up}} h)$ i wracają przez $W_{\text{down}}$ do `d_model`.

---

### 7. Powstawanie logitów, wybór tokenu oraz różnica między treningiem a inferencją

- **Logity i Tied Embeddings:**
  - **Poziom A:** Po przejściu przez wszystkie warstwy końcowy wektor (po finalnym `RMSNorm`) porównywany jest ze wszystkimi `2048` tokenami ze słownika. Im wyższy wynik podobieństwa dla danego słowa ze słownika, tym większa szansa, że to właśnie ono powinno pojawić się dalej. Te surowe wyniki podobieństwa nazywamy **logitami**.
  - **Poziom B/C:** **POTWIERDZONE** — Tiny stosuje **tied embeddings** (współdzielone wagi: macierz końcowej projekcji `lmHead` to dokładnie ta sama macierz `tok_emb.weight`, co oszczędza parametry). Zapisuje to bit `0` flagi w nagłówku `.tny3`, wymusza to `TartarusConfig.validate()` (`model_tartarus.py:72`) i wykorzystuje `tartarus-engine.ts:342–344` (`lmHead = tokEmb`).
- **Wybór tokenu (Sampling):** Z wektora `2048` logitów wybiera się jeden identyfikator `nextId` za pomocą `sampleNextToken()` (`tartarus-engine.ts:823–941`) — albo biorąc wprost maksimum (`temperature = 0`, greedy), albo przekształcając logity na rozkład prawdopodobieństwa (softmax) i losując po odfiltrowaniu (`topK`, `minP`, `repetitionPenalty`). Szczegóły w [Rozdziale VII](./TINY_RUNTIME.md#rozdział-vii--sampling-i-generowanie-tekstu).
- **Trening vs Inferencja:**

| Cecha | Trening (`train_tartarus.py`) | Inferencja (`tartarus-engine.ts` / `tartarus-browser.html`) |
| --- | --- | --- |
| **Cel** | Zmiana wag modelu, aby przewidywał właściwe odpowiedzi z przykładów `train.jsonl` | Użycie zamrożonych wag do wygenerowania nowej odpowiedzi |
| **Kierunek obliczeń** | W przód (forward) + wstecz (backpropagation / gradienty) + aktualizacja AdamW | Wyłącznie w przód (`prefill` i `step`) |
| **Przetwarzanie tokenów** | Cała znana sekwencja z pliku CSR naraz (`inp = x[:, :-1]`), równolegle w batchach (np. 32 lub 128 próbek) | Sekwencyjnie token po tokenie po fazie `prefill` (`batch_size = 1`) |
| **Maska straty (`loss_mask`)** | Wykorzystuje `mask=0` na prompt i `mask=1` wyłącznie na odpowiedź NPC do liczenia `cross_entropy` | Nie liczy błędu (brak `loss_mask`); przetwarza prompt, a następnie generuje odpowiedź aż do `<|eos|>` lub `CTX` |
| **Precyzja wag** | `float32` na CPU lub `bfloat16`/`float16` AMP na GPU | Wagi zapisane jako `INT8` per-row w `.tny3`, dekwantyzowane przy ładowaniu do `Float32Array` (**W8A32**) |
| **KV Cache** | Nie używa podczas treningu (`kv_cache=None`, SDPA `is_causal=True`) | Kluczowy bufor (`L × CTX × D`), eliminujący ponowne przeliczanie kontekstu |

---

## ROZDZIAŁ III – Tokenizer (`tartarus_tokenizer.ts` i zgodność z Pythonem)

Źródła analizowane w tym rozdziale: `tartarus_tokenizer.ts` (`339` linii), `tokenizer_tartarus.py` (`1248` linii), `tokenizer.json` (`381 048 B`), `tokenizer_parity.json` (`88 365 B`).

### 1. Byte-level BPE i dlaczego tokenizer operuje na bajtach?

- **Poziom A:** Zwykły słownik oparty na literach lub całych wyrazach ma dwa problemy: po pierwsze nie zmieściłby wszystkich odmian polskich słów (np. *złoto, złota, złotem, złocie*), a po drugie gdyby gracz wpisał nietypowy symbol albo literówkę (np. `Jötunn`, emotikon czy obcą nazwę), tokenizer nie wiedziałby co z tym zrobić i wypisałby błąd „nieznane słowo” (`<unk>`). Tokenizer **Byte-level BPE** traktuje tekst jako ciąg surowych bajtów UTF-8 (liczby od `0` do `255`), więc potrafi zakodować **każdy możliwy napis w Unicode** bez wyjątku, a często powtarzające się fragmenty skleja w większe cegiełki.
- **Poziom B:** **POTWIERDZONE** — Tokenizer w Tiny jest w pełni deterministycznym kodowaniem **Byte-Level Byte Pair Encoding (BPE)** w formacie `"tartarus-bpe-v1"`. Podstawą słownika po `5` tokenach specjalnych (`byte_offset = 5`) jest `256` atomowych bajtów (`ID = 5 + bajt`). Następnie `1787` par sąsiednich tokenów sklejono w nowe tokeny o numerach `261..2047` (łącznie `vocab_size = 2048`).
- **Poziom C:** Konstruktor `TartarusTokenizer` (`tartarus_tokenizer.ts:69–163`) odtwarza wyłącznie z konfiguracji JSON (`merges` bez potrzeby użycia tablicy `vocab`):
  1. Wypełnia `idToBytes[0..4]` pustymi buforami `Uint8Array(0)`.
  2. Wypełnia `idToBytes[5..260]` jednoelementowymi tablicami `new Uint8Array([b])`.
  3. Dla każdego wpisu `rank = 0 .. 1786` z `config.merges` (trójka `[left, right, newId]`, gdzie `newId = 261 + rank`): skleja bajty `idToBytes[left]` i `idToBytes[right]` w `idToBytes[newId]` i zapisuje w mapie `mergesRank` klucz `left * 1048576 + right` → `{ rank, newId }`.
  - **Wyjaśnienie `PAIR_STRIDE = 1048576` ($2^{20}$):** Ponieważ maksymalne ID to `2047 < 2^{20}`, wyrażenie `left * 1048576 + right` pakuje dwie liczby całkowite w jedną unikalną liczbę w bezpiecznym zakresie JavaScript (`Number.MAX_SAFE_INTEGER = 2^{53}-1`), eliminując konieczność tworzenia par napisów typu `"123,456"` i alokacji obiektów podczas wyszukiwania par w `encodeChunk()`. **OGRANICZENIE:** Gdyby w przyszłości słownik przekroczył `1 048 576` tokenów, stała `PAIR_STRIDE` musiałaby zostać zwiększona.

---

### 2. Podział tekstu, tokeny specjalne i problem `\w` w Python vs JS

- **Poziom A:** Zanim zacznie się sklejanie bajtów, tekst jest cięty na logiczne kawałki: oddziela się tagi sterujące gry (`<|state|>`, `<|user|>`, `<|npc|>`, `<|eos|>`, `<|pad|>`), a resztę dzieli się na słowa, krótkie grupy cyfr (po maksymalnie `3` cyfry) i pojedyncze znaki interpunkcyjne (np. `;`, `=`, `?`). Dzięki temu spacja przed słowem skleja się z tym słowem (`" złoto"`), a nie robi osobnych dziwnych połączeń przez znaki przestankowe.
- **Poziom B:** **POTWIERDZONE** — Kolejność jest ściśle dwuetapowa:
  1. **Wydzielenie tokenów specjalnych przed BPE:** `specialRe` (`tartarus_tokenizer.ts:100–106`, `tokenizer_tartarus.py:85–90`) sortuje nazwy tokenów od najdłuższej do najkrótszej i dzieli tekst za pomocą `text.split(this.specialRe)` (z grupą przechwytującą `(...)`, dzięki czemu dopasowane tagi zostają w wynikowej tablicy segmentów). Token specjalny nigdy nie jest dzielony ani scalany przez BPE.
  2. **Pre-tokenizacja jawnie zdefiniowanym wyrażeniem regularnym:** W Pythonie skrót `\w` w `re` domyślnie obejmuje polskie litery (`ą, ć, ę, ł, ń, ó, ś, ź, ż`), natomiast w JavaScript `/\w/` obejmuje wyłącznie ASCII `[A-Za-z0-9_]`. Z tego powodu użycie `\w` zostało w kodzie zakazane (`tokenizer_tartarus.py:16–19`). Zamiast niego oba języki używają identycznego wzorca z jawną klasą liter (`LETTERS_CLASS`):

```
 ?[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźżÀ-ÖØ-öø-ÿ]+| ?[0-9]{1,3}| ?[^A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźżÀ-ÖØ-öø-ÿ0-9 \t\r\n]|[ \t\r\n]+
```

Zwróć uwagę na cztery gałęzie tego wzorca:
- ` ?[LETTERS_CLASS]+` — opcjonalna pojedyncza spacja + ciąg liter (w tym polskie znaki diakrytyczne oraz np. `ö`/`Ö` dla nazw z korpusu takich jak *Jötunn* czy *Völva*).
- ` ?[0-9]{1,3}` — opcjonalna spacja + maksymalnie `3` cyfry (większe liczby są rozbijane w grupach po 3 cyfry, dzięki czemu model nie potrzebuje osobnego tokenu dla każdej wielkiej liczby).
- ` ?[^LETTERS_CLASS 0-9 \t\r\n]` — opcjonalna spacja + pojedynczy znak interpunkcyjny / symbol (`=`, `;`, `/`, `?`, `„`, itp.).
- `[ \t\r\n]+` — ciągi białych znaków (jeśli nie zostały pochłonięte jako pojedyncza spacja wiodąca).

---

### 3. Tokeny specjalne i sposób ich przechowywania

**POTWIERDZONE** (`tokenizer_tartarus.py:48–62`, `tartarus_tokenizer.ts:81–120`, `dataset_meta.json:7–13`):

| ID | Token specjalny | Rola w korpusie, treningu i silniku |
| --- | --- | --- |
| `0` | `<|pad|>` | Token dopełnienia (padding) w batchach podczas treningu (`train_tartarus.py:181`). **Nigdy nie występuje w plikach CSR** (`tokenizer_tartarus.py:761`) ani nie jest traktowany jako odpowiedź przy dekodowaniu (`skipPad=true` domyślnie). |
| `1` | `<|eos|>` | *End of Sequence* — sygnał zakończenia wypowiedzi NPC. Wykrycie `nextId === eosId` zatrzymuje pętlę `generateStream()` (gdy `stopOnEos=true`). |
| `2` | `<|state|>` | Otwiera sekcję stanu gry (`lokacja=...; hp=...; złoto=...`). W treningu CSR maska straty wynosi `0` od tego tokenu aż do `<|npc|>` włącznie. |
| `3` | `<|user|>` | Otwiera sekcję wypowiedzi/pytania gracza. |
| `4` | `<|npc|>` | Sygnalizuje początek odpowiedzi postaci niezależnej; jest ostatnim tokenem promptu wysyłanego do `prefill()`. |

Identyfikatory tokenów przechowywane są w:
- dwukierunkowych mapach `specialToId: Map<string, number>` i `idToSpecial: Map<number, string>` w `TartarusTokenizer` (`tartarus_tokenizer.ts:62–63`),
- polach skrótowych `padId`, `eosId`, `stateId`, `userId`, `npcId` bezpośrednio na instancji `TartarusEngine` (`tartarus-engine.ts:462–466`, `506–510`).

---

### 4. Analiza metod klasy `TartarusTokenizer` (`tartarus_tokenizer.ts`)

1. **`encode(text: string): number[]` (`tartarus_tokenizer.ts:259–294`):**
   - Dla pustego ciągu zwraca `[]`.
   - Dzieli `text` przez `specialRe`. Jeśli segment znajduje się w `specialToId`, od razu dopisuje jego ID (`0..4`).
   - W przeciwnym razie resetuje `this.pretokenizeRe.lastIndex = 0` i w pętli `exec()` wyodrębnia kolejne kawałki (`chunk`), przekazując każdy do `encodeChunk(chunk)`.
   - `encodeChunk(chunk)` (`tartarus_tokenizer.ts:204–257`): sprawdza pamięć podręczną `chunkCache` (limit `50 000` wpisów), zamienia napis na bajty UTF-8 (`TextEncoder.encode`), zamienia bajty na początkowe ID (`byteOffset + b`), a następnie w pętli `while (ids.length >= 2)` znajduje w `mergesRank` parę o **najniższym `rank`** (czyli najwcześniej nauczonym scaleniu w BPE). Wszystkie nienachodzące na siebie wystąpienia tej najlepszej pary w kawałku skleja od lewej do prawej w jeden token `bestNewId` i powtarza proces, aż żadna sąsiadująca para w `ids` nie istnieje w `mergesRank`.
2. **`decode(ids: Iterable<number>, options?: DecodeOptions): string` (`tartarus_tokenizer.ts:296–338`):**
   - Sprawdza zakres każdego tokenu `[0, vocabSize)`.
   - Dla tokenów nie-specjalnych (`tid >= byteOffset`) **nie dekoduje od razu pojedynczego tokenu do tekstu**, lecz dopisuje surowe bajty z `idToBytes[tid]` do tymczasowego bufora `byteBuf: number[]`.
   - Dopiero przed napotkaniem tokenu specjalnego (`tid < byteOffset`) lub na końcu pętli wywołuje `flushBytes()`, dekodując cały nagromadzony blok bajtów naraz przez `TextDecoder("utf-8", { fatal: false })`.
   - Opcje domyślne (`DecodeOptions`): `skipPad = true` (pomija `<|pad|>`), `skipSpecial = false` (wypisuje `<|state|>`, `<|eos|>` itd. jako tekst; w `generateStream()` silnik celowo podaje `skipSpecial: true`, aby użytkownik widział czystą kwestię NPC).
3. **`getSpecialToken(id: number): string | undefined` (`tartarus_tokenizer.ts:169–171`):**
   - Zwraca nazwę tokenu specjalnego dla podanego `id` z mapy `idToSpecial` (np. `1 -> "<|eos|>"`) lub `undefined`, jeśli to zwykły token BPE. Wykorzystywane m.in. przez inspektora tokenów w `tartarus-browser.html:2275`.
4. **`getTokenBytes(id: number): Uint8Array` (`tartarus_tokenizer.ts:173–178`):**
   - Waliduje `Number.isInteger(id)` i zakres `[0, vocabSize)`, rzucając `RangeError` w razie błędu, i zwraca surowy widok bajtów `this.idToBytes[id]`. Dla tokenów specjalnych zwraca pusty `Uint8Array(0)`. Używane w `tartarus-browser.html:2278` do pokazywania bajtów szesnastkowych `[0xc5]` dla tokenów będących połówką znaku UTF-8.
5. **`createStreamDecoder(options?: DecodeOptions)` (`tartarus_tokenizer.ts:180–202`):**
   - Tworzy obiekt `{ push(tid): string, flush(): string }` z własną instancją `TextDecoder("utf-8", { fatal: false })`.
   - Kiedy `push(tid)` dostaje zwykły token BPE (`tid >= byteOffset`), wywołuje `dec.decode(this.idToBytes[tid], { stream: true })`.
   - Kiedy `push(tid)` napotyka token specjalny (`tid < byteOffset`), najpierw zamyka bufor bajtów wywołaniem `dec.decode(new Uint8Array(0), { stream: false })`, a następnie — jeśli nie pomijamy — dokleja nazwę tokenu specjalnego.

---

### 5. Dlaczego zwykłe dekodowanie fragment po fragmencie psuje polskie znaki i jak działa streaming UTF-8?

- **Poziom A:** W zapisie komputerowym UTF-8 zwykłe litery angielskie (`a`, `b`, `c`) zajmują **1 bajt**, ale polskie litery (`ą, ć, ę, ł, ń, ó, ś, ź, ż`) zajmują **2 bajty** (np. `Ś` = `C5 9A`, `ż` = `C5 BC`). Ponieważ BPE skleja najczęściej występujące bajty, czasem rozcina polską literę na dwa różne tokeny: pierwszy token ma tylko bajt `C5`, a następny ma bajt `9A`. Gdybyś próbował natychmiast zamienić sam bajt `C5` na literę, przeglądarka nie wiedziałaby, jaka to litera i wstawiłaby znak zastępczy błędu UTF-8 (`U+FFFD`).
- **Poziom B/C (rzeczywisty przykład z Tiny!):** **POTWIERDZONE** — Podczas wykonania diagnostyki bezpośrednio na rzeczywistym `tokenizer.json` słowo `"Świętogora"` (12 bajtów UTF-8: `c5 9a 77 69 c4 99 74 6f 67 6f 72 61`) rozbija się dokładnie na 4 tokeny:
  - Token `202` → 1 bajt `0xC5` (niepełny początek litery `Ś`!)
  - Token `159` → 1 bajt `0x9A` (druga połowa litery `Ś`!)
  - Token `809` → 4 bajty `77 69 c4 99` (`"wię"`)
  - Token `819` → 6 bajtów `74 6f 67 6f 72 61` (`"togora"`)

Gdyby aplikacja wywoływała `decode([tid])` osobno dla każdego tokenu w pętli:
- `decode([202])` zwróciłoby `""` (`\uFFFD`),
- `decode([159])` zwróciłoby `""` (`\uFFFD`),
- wynik dla początku słowa byłby uszkodzony: `"więtogora"`.

Natomiast `createStreamDecoder()` przekazuje do `TextDecoder.decode(bytes, { stream: true })`:
1. `push(202)` → przekazuje `[0xC5]` ze stanem `stream: true`. `TextDecoder` widzi niekompletną sekwencję dwubajtową UTF-8, **zapamiętuje `0xC5` w swoim wewnętrznym buforze** i zwraca `""` (pusty napis).
2. `push(159)` → przekazuje `[0x9A]` ze stanem `stream: true`. `TextDecoder` łączy zapamiętane `0xC5` z `0x9A`, rozpoznaje poprawny znak `U+015A` i zwraca czystą literę `"Ś"`.
3. `push(809)` → zwraca `"wię"`.
4. `push(819)` → zwraca `"togora"`.
5. `flush()` (lub token `<|eos|>`) → wywołuje `dec.decode(new Uint8Array(0), { stream: false })`, domykając ewentualne urwane bajty na końcu tekstu.

---

### 6. Konkretny przykład krok po kroku (zweryfikowany na rzeczywistym `tokenizer.json`)

**Tekst wejściowy:**
`"<|state|> hp=42/75; złoto=390 <|user|> Ile złota? <|npc|>"`

1. **Etap 1 (`split_special`):**
   `['<|state|>', ' hp=42/75; złoto=390 ', '<|user|>', ' Ile złota? ', '<|npc|>']`
2. **Etap 2 (`pretokenize` + kodowanie na bajty `UTF-8` i przesunięcie o `byte_offset = 5`):**
   - `'<|state|>'` → Token specjalny → `ID = 2`
   - `' hp'` → bajty `20 68 70` → ID bazowe `[37, 109, 117]` → po scaleniach BPE → `[303]`
   - `'='` → bajt `3d` → `[66]` → brak dalszych scaleń → `[66]`
   - `'42'` → bajty `34 32` → `[57, 55]` → BPE → `[1780]`
   - `'/'` → bajt `2f` → `[52]`
   - `'75'` → bajty `37 35` → `[60, 58]` → BPE → `[356]`
   - `';'` → bajt `3b` → `[64]`
   - `' złoto'` → bajty `20 7a c5 82 6f 74 6f` (`ł` = `c5 82`) → `[37, 127, 202, 135, 116, 121, 116]` → BPE scala m.in. parę `[202, 135]` (`merge #1`, ID `262`, czyli samo `ł`!) i dalej aż do jednego tokenu → `[354]`
   - `'='` → `[66]`
   - `'390'` → bajty `33 39 30` → `[56, 62, 53]` → BPE → `[1147]`
   - `' '` → bajt `20` → `[37]`
   - `'<|user|>'` → Token specjalny → `ID = 3`
   - `' Ile'` → bajty `20 49 6c 65` → `[37, 78, 113, 106]` → BPE → `[1569]`
   - `' złota'` → bajty `20 7a c5 82 6f 74 61` → `[37, 127, 202, 135, 116, 121, 102]` → BPE → `[490]`
   - `'?'` → bajt `3f` → `[68]`
   - `' '` → `[37]`
   - `'<|npc|>'` → Token specjalny → `ID = 4`
3. **Wynikowy ciąg identyfikatorów (`17` tokenów):**
   `[2, 303, 66, 1780, 52, 356, 64, 354, 66, 1147, 37, 3, 1569, 490, 68, 37, 4]`
4. **Dekodowanie (`decode(ids, { skipPad: true, skipSpecial: false })`):**
   - `2` → zapisuje tekst `"<|state|>"`
   - `303..37` → dopisuje do `byteBuf` bajty wszystkich 10 tokenów (`20 68 70 3d ... 20`), po czym przed tokenem specjalnym `3` dekoduje cały bufor naraz jako `" hp=42/75; złoto=390 "`
   - `3` → `"<|user|>"`
   - `1569..37` → dekoduje bufor jako `" Ile złota? "`
   - `4` → `"<|npc|>"`
   - **Tekst wynikowy:** dokładnie `"<|state|> hp=42/75; złoto=390 <|user|> Ile złota? <|npc|>"` (bezzwrotny round-trip 1:1).

---

### 7. Co oznacza „100% zgodności tokenizera” i czego taki wynik NIE dowodzi?

**POTWIERDZONE:** W testach `verify_tokenizer.mjs` (`119/119` przypadków z `tokenizer_parity.json` + `1000/1000` rekordów z `val_tokens.bin` + `1000/1000` z `test_tokens.bin`) oraz `verify_ts.mjs` (`3/3` przypadki z `golden_test.json`) implementacja `tartarus_tokenizer.ts` uzyskała 100% identyczności tablic tokenów oraz zdekodowanych napisów z `tokenizer_tartarus.py`.

**Czego to dowodzi:**
- Że w tych `2122` zróżnicowanych napisach (w tym zdaniach z polskimi znakami, wielokrotnym `ł/ś/ż`, liczbami, interpunkcją i tagami specjalnymi) podział `specialRe`, wyrażenie `pretokenizeRe`, ranking par BPE i rekonstrukcja UTF-8 dają w JS/TS dokładnie te same liczby i napisy co w Pythonie.

**Czego to NIE dowodzi (granice dowodu):**
- **NIE dowodzi** identyczności dla ekstremalnych lub nietestowanych sekwencji Unicode spoza testowanego zbioru — np. znaków spoza BMP (emoji, surogaty UTF-16 w JS vs punkty kodowe Unicode w Pythonie przy dopasowaniu negacji klasy `[^...]`), łączonych znaków diakrytycznych (*combining marks* NFD vs NFC) ani błędnych sekwencji wejściowych zawierających niesparowane surogaty UTF-16 w JavaScript.
- **NIE dowodzi** jakości językowej modelu ani tego, że wygenerowany przez model tekst będzie mądry lub prawdziwy — sprawdza wyłącznie wierność translatora `tekst ↔ liczby`.
