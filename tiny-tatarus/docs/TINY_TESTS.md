# Weryfikacja matematyczna i testy projektu Tiny (Tartarus)

> Część dokumentacji systemu Tiny: [← Powrót do głównego przewodnika](./TINY_SYSTEM_DOCUMENTATION.md)  
> Powiązane dokumenty: [Architektura i Tokenizer](./TINY_ARCHITECTURE.md) · [Matematyka operacji](./TINY_MATHEMATICS.md) · [Silnik i KV Cache](./TINY_RUNTIME.md) · [Ograniczenia i plan testów](./TINY_SYSTEM_DOCUMENTATION.md#rozdział-xii--ograniczenia-ryzyka-i-dalsza-walidacja)

---

## ROZDZIAŁ VIII – Weryfikacja matematyczna

Źródła analizowane bezpośrednio w tym rozdziale: `verify_ts.mjs` (`318` linii), `verify_tokenizer.mjs` (`158` linii), `golden_test.json` (`182 843 B`), `export_tny3.py:440–521`, `model_tartarus.py:492–550`, `checkpoints/train_history.json` oraz raporty uruchomień w środowisku uruchomieniowym.

### 1. Poziom A/B/C: czym jest test referencyjny („Złoty test”) i dlaczego PyTorch jest punktem odniesienia?

- **Poziom A (intuicyjny):** Kiedy przepisujesz skomplikowane równania z jednego języka (Python/PyTorch) na drugi (TypeScript/JavaScript), łatwo o drobną pomyłkę: inne zaokrąglenie, zamienione dwie pętle, brak jednego minusa przy obrocie RoPE albo przesunięty o 1 indeks w KV Cache. Taki błąd nie zawsze daje oczywisty krach — częściej powoduje, że model „prawie” działa, ale zaczyna pisać od rzeczy. Dlatego bierze się zamrożony model z PyTorcha, przepuszcza przez niego wybrane przykłady i zapisuje dokładne surowe liczby wynikowe (**logity**) do pliku `golden_test.json`, a potem wymaga, by silnik TypeScript wyliczył **praktycznie te same liczby** na tych samych danych wejściowych.
- **Poziom B (techniczny):** **POTWIERDZONE** — Punktem odniesienia (ground truth) jest `TartarusModel` w PyTorchu, ponieważ:
  1. Na nim był uczony model w `train_tartarus.py`.
  2. Eksporter `export_tny3.py` najpierw porównuje zrekonstruowany z pliku `.tny3` model PyTorcha (`load_tny3()`) z modelem referencyjnym po kwantyzacji (`quantized_copy`), sprawdzając `max_weight_diff == 0.0` oraz `max_logit_diff == 0.0` (`golden_test.json:33–36`).
  3. Następnie `export_tny3.py:462–516` zapisuje do `golden_test.json` zarówno wejściowe ciągi tokenów `token_ids`, jak i cały `2048`-elementowy wektor `last_step_logits` dla każdego z 3 przypadków testowych.

---

### 2. Logity, arytmetyka zmiennoprzecinkowa, `maxAbs` i tolerancja `1e-4`

- **Dlaczego liczby nie muszą zgadzać się co do ostatniego bitu (`0.0` vs `2.146e-6`)?**
  W komputerze liczby ułamkowe w `float32` (pojedyncza precyzja) mają ok. 7 cyfr znaczących dokładności. Dodawanie $a + b + c$ w innej kolejności (np. zrównoleglone operacje wektorowe w C++ w PyTorchu vs rozwinięta po `4` elementy pętla `c += 4` w JS oraz użycie wewnętrznej precyzji `float64` dla zmiennych JS `number` przed zapisem do `Float32Array`) wprowadza mikroskopijne różnice na poziomie ~7. cyfry po przecinku.
- **Czym jest `maxAbs` (`verify_ts.mjs:29–48`):**
  To maksymalna wartość bezwzględna różnicy pomiędzy odpowiadającymi sobie elementami dwóch wektorów logitów o długości `V = 2048`:
  $$\text{maxAbs}(A, B) = \max_{0 \le i < 2048} |A_i - B_i|$$
- **Dlaczego ustalono próg akceptacji `1e-4` (`0.0001`):**
  Typowe logity w `golden_test.json` mieszczą się w przedziale od ok. `-3` do `+7` (np. top-1 w pierwszym przypadku wynosi `6.9162`, a top-2 `4.3545`). Różnica poniżej `0.0001` jest o 4 rzędy wielkości mniejsza od dystansu między kolejnymi tokenami i nie ma praktycznie wpływu na kolejność ani prawdopodobieństwa, natomiast niemal każdy prawdziwy błąd implementacji (np. zły znak w RoPE, brak `rms_final`, pomylony wymiar w `w_down` lub brak skali INT8) powoduje błędy rzędu `0.1` do `10.0`.
- **Co oznacza zmierzony wynik `2.146e-6` (`0.000002146`):**
  Maksymalny błąd bezwzględny na wszystkich `3 × 2048 = 6144` porównywanych logitach jest niemal **50-krotnie mniejszy** od przyjętego limitu `1e-4` i leży na granicy szumu zaokrągleń arytmetyki 32-bitowej IEEE-754.
- **Co oznacza wynik `0.000e+0` dla KV Cache (`cacheVsFullΔ` oraz 16-krokowa autoregresja):**
  Wewnątrz samego TypeScriptu ścieżka `prefill + step` (używająca `kCache` i `vCache`) oraz ścieżka bez pamięci podręcznej `forwardNoCache` wykonują identyczne operacje w identycznej kolejności na `Float32Array`. Wynik `0.000e+0` oznacza, że nie tylko wynik mieści się w tolerancji `< 1e-5`, lecz że **każdy z 2048 logitów w obu ścieżkach był identyczny co do pojedynczego bitu** w każdym z testowanych kroków.

---

### 3. Metody weryfikacji w `verify_ts.mjs`

1. **Zgodność `forwardNoCache` (`verify_ts.mjs:115–125`):** Przepuszcza całą sekwencję promptu naraz przez bezstanowy wariant bez KV Cache i porównuje wynik z `c.last_step_logits` z `golden_test.json`.
2. **Zgodność `prefill` (`verify_ts.mjs:128–137`):** Wywołuje `engine.prefill(tsIds)`, sprawdzając czy zbudowanie całego KV Cache od zera od pozycji `0` daje te same logity dla ostatniego tokenu co Python.
3. **Zgodność `prefill(prefix) + step(ogon)` (`verify_ts.mjs:140–151`):** Rozcina każdy prompt z `golden_test.json` przed ostatnimi 5 tokenami (`splitAt = tsIds.length - 5`): najpierw robi `prefill` na prefiksie, a potem 5 razy wywołuje `engine.step()` dla ostatnich 5 tokenów. To krytyczny test sprawdzający, czy `step()` poprawnie odczytuje historyczne `K, V` utworzone przez `prefill()`, poprawnie dopisuje własne `K, V` i przesuwa `kvPos` w RoPE.
4. **Test 16-krokowej autoregresji (`verify_ts.mjs:171–189`):** Startując od `golden_pl_state_query` (`78` tokenów), wykonuje `16` kolejnych kroków generacji (aż do długości `94` tokenów). W **każdym z 16 kroków** niezależnie uruchamia `forwardNoCache(seq)` od początku dla aktualnej sekwencji i porównuje wszystkie `2048` logity z wynikiem z inkrementalnego `step()`. Wykazuje to brak akumulujących się błędów indeksowania cache podczas faktycznej generacji tekstu.

---

### 4. Zestawienie rzeczywistych wyników testów

Poniższe tabele rozdzielają wyniki po kategoriach wraz z dokładnym wskazaniem źródła i daty ostatniego zweryfikowanego uruchomienia w bieżącej sesji (po ostatnich edycjach kodu z Etapu 5).

#### Tabela A – Tokenizacja i dane CSR

| Test | Zakres | Wynik | Źródło / Skrypt | Status |
| --- | --- | --- | --- | --- |
| Parzystość Python ↔ TS (`tokenizer_parity.json`) | `119` przypadków (diakrytyki, liczby, znaki specjalne, tagi, dekodowanie z i bez `<|pad|>`) | **`119/119` (100%)** | `node verify_tokenizer.mjs` (`L60–113`) | **POTWIERDZONE** |
| Minimalna konfiguracja bez tablicy `vocab` (symulacja TNY3) | Przypadek końcowy `tokenizer_parity.json` | **PASS (100%)** | `node verify_tokenizer.mjs` (`L97–111`) | **POTWIERDZONE** |
| Zgodność JS/TS przeciwko binarnym paczkom CSR (`val` i `test`) | `1000/1000` rekordów `val_tokens.bin` + `1000/1000` rekordów `test_tokens.bin` + round-trip UTF-8 | **`2000/2000` (100%)** | `node verify_tokenizer.mjs` (`L116–151`) | **POTWIERDZONE** |
| Zgodność pełnego podziału `train + val + test` (CSR w Pythonie) | `22 000` rekordów (łącznie `1 748 814` tokenów) | **`22 000/22 000`** | `prepare_tartarus_dataset.py` / `verify_csr_dataset` | **POTWIERDZONE** (zapisane w `dataset_meta.json`, z Etapu 2) |
| Zgodność tokenizacji dla `golden_test.json` | `3` prompty (`78`, `77`, `54` tok) + round-trip `decode(encode(text))` | **`3/3` (100%)** | `node verify_ts.mjs` (`L105–113`) | **POTWIERDZONE** |

#### Tabela B – Zgodność numeryczna wag, logitów i KV Cache (`model.tny3`, Profil `S`)

| Przypadek testowy (`golden_test.json`) | Długość (`tok`) | `forwardNoCache` vs PyTorch (`maxAbs`) | `prefill` vs PyTorch (`maxAbs`) | `prefill+5×step` vs PyTorch (`maxAbs`) | `KV Cache` vs `NoCache` | `argmax` (TS == PY) | Źródło |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `golden_pl_state_query` | `78` | `1.91e-6` | `1.91e-6` | `1.91e-6` | `0.00e+0` | `1296` (`" Armia"` @ `6.9162`) | `node verify_ts.mjs` |
| `golden_pl_diacritics_lore` | `77` | `2.15e-6` | `2.15e-6` | `2.15e-6` | `0.00e+0` | `931` (`" Brak"` @ `5.6749`) | `node verify_ts.mjs` |
| `golden_pl_refusal_short` | `54` | `1.91e-6` | `1.91e-6` | `1.91e-6` | `0.00e+0` | `552` (`" Nie"` @ `4.8441`) | `node verify_ts.mjs` |
| **Ekwiwalent 16-krokowy autoregresji** | `78 -> 94` | — | — | — | **`0.00e+0`** na każdym z 16 kroków | 100% zgodne | `node verify_ts.mjs` (`L171–189`) |
| **Round-trip eksportu Python `load_tny3` == `quantized_copy`** | wagi + 3 przypadki | `max_weight_diff = 0.00e+0` | `max_logit_diff = 0.00e+0` | — | `max_diff = 9.54e-7` (`model_tartarus.py` self_test) | 100% zgodne | `export_tny3.py`, `golden_test.json`, `model_tartarus.py` |

#### Tabela C – Sampling, CRC-32, CTX, UTF-8, kompilacja oraz rozdzielenie Node.js vs Przeglądarka

| Obszar | Szczegóły testu | Wynik | Środowisko i źródło | Status |
| --- | --- | --- | --- | --- |
| **CRC-32 (odporność na uszkodzenia)** | Odwrócenie 1 bitu w payloadzie (`tampered[len-17] ^= 1`) | Odrzucony przez `parseTNY3()` komunikatem o błędzie CRC-32 (w Pythonie też `ValueError`) | Node.js `verify_ts.mjs:92–100` (oraz Python Etap 4) | **POTWIERDZONE** |
| **Prealokacja buforów KV i brak re-alokacji logitów** | Sprawdzenie rozmiaru `L*CTX*D = 196 608 f32` (`768 KiB` K + `768 KiB` V) oraz `logitsPrefill === logitsSplit` | Identyczna referencja `Float32Array`, zero alokacji w `step()` | Node.js `verify_ts.mjs:81–89, 154–157` | **POTWIERDZONE** |
| **Sampling (`greedy`, `topK+minP`, `repPenalty`)** | Greedy (`" Armia przebywa w stolicy. "`, `7` tok, `stoppedBy=eos`); `T=0.75, topK=32, minP=0.06, rep=1.15`; duplikaty `[100,100,100]` przy `rep=1.2` karane 1× | Zaliczone | Node.js `verify_ts.mjs:192–259` | **POTWIERDZONE** |
| **Granica `CTX`** | Prompt długości `252` tok (`ctxLen - 4`), `maxNewTokens=64`, `stopOnEos=false` | Zatrzymany dokładnie po `4` tokenach z `stoppedBy="ctx"` | Node.js `verify_ts.mjs:262–272` | **POTWIERDZONE** |
| **Streaming UTF-8** | Porównanie sklejonych `ev.textDelta` z końcowym `decode()`, sprawdzanie braku `\ufffd` | `streamedText === greedyRes.text`, brak `\ufffd` | Node.js `verify_ts.mjs:199–217` | **POTWIERDZONE** |
| **Kompilacja TypeScript (DOM-free)** | `npx tsc --ignoreConfig --noEmit --strict --lib ES2022` bez `@types/node` i bez `dom` na `tartarus_tokenizer.ts` + `tartarus-engine.ts` | `exit_code = 0`, 0 błędów | CLI `tsc 7.0.2` | **POTWIERDZONE** |
| **Składanie i weryfikacja `tartarus-browser.html`** | `node build_browser.mjs` + kompilacja `vm.Script` + test E2E w mockowanym DOM (`node:vm`) | Profil `S` (`L=4`, `78/256`, `" Armia przebywa w stolicy. "`, `154.3 tok/s`), Profil `M` (`L=6`, 6 pasków), guard `LIMIT CTX!` (`750/256`) | **Symulator `node:vm` w Node.js** (nie prawdziwa przeglądarka!) | **POTWIERDZONE wyłącznie dla `node:vm`** |
| **Działanie w prawdziwej przeglądarce GUI (Chrome / Firefox / Safari)** | Rzeczywisty Drag & Drop pliku, renderowanie CSS/Canvas/DOM, realny `tok/s` z throttlingiem `setTimeout(0)` | Nie uruchamiano (brak binariów przeglądarki w sandboxie) | — | **NIEZWERYFIKOWANE** |
| **Trening CPU checkpointu `model.tny3`** | `160` kroków (`bs=32, grad_accum=2`), ewaluacja `FLOAT` vs `INT8`, `16` próbek jakościowych | `val_loss`: `5.7904` (FLOAT) / `5.7913` (INT8, `Δ=+0.0009`); `EOS=16/16`, `refusal=4/4`, `state_query=0/4` | `train_tartarus.py`, `checkpoints/train_history.json` (Etap 3) | **POTWIERDZONE** (**OGRANICZENIE**: krótki trening smoke CPU) |

---

### 5. Co potwierdza każdy test, a czego NIE potwierdza?

To kluczowe rozróżnienie chroniące przed fałszywym poczuciem bezpieczeństwa:

1. **Testy `golden_test.json` (`3` przypadki) + `16`-krokowa autoregresja:**
   - **Potwierdzają:** identyczną implementację matematyki (`RMSNorm`, `RoPE half-split`, `QKV`, `Causal Attention`, `SwiGLU`, `Tied Embeddings`, dekwantyzację `INT8`) dla konkretnych wymiarów modelu Profilu `S` (`L=4, D=192, H=6, d_ff=512, V=2048, CTX=256`), w zakresie sekwencji długości `54..94` tokenów, przy `seed`/wagach z `model.tny3`.
   - **NIE potwierdzają:** poprawności numerycznej względem PyTorcha dla **Profilu `M`** na poziomie logitów (`golden_test.json` zawiera wyłącznie Profil `S`; dla `/tmp/model_m.tny3` sprawdzono w `verify_ts.mjs` jedynie parsowanie `L=6` i uruchomienie `generate()`, ale nie porównywano logitów TS vs PyTorch — **NIEZWERYFIKOWANE** numerycznie w TS dla `M`), ani dla modelu z `tiedEmbeddings = false` (gałąź `lmHead` osobno w `tartarus-engine.ts:345–348` jest **ZAIMPLEMENTOWANA**, ale nieprzetestowana na żywym pliku `.tny3`, bo `TartarusConfig.validate()` wymaga `tied_embeddings=True`).
   - **NIE potwierdzają:** zachowania przy ekstremalnych sekwencjach o pełnej długości `255 -> 256` tokenów przeciwko PyTorchowi (w `verify_ts.mjs` test `CTX` sprawdził wyłącznie licznik zatrzymania `252 + 4 = 256` na powtórzonym ID `10`, bez porównania logitów z PyTorchem).
2. **Benchmark `heapΔ = 404.5 KiB` w `verify_ts.mjs`:**
   - **Potwierdza:** brak dużych wycieków buforów wielomegabajtowych przy `10` krótkich wywołaniach.
   - **NIE potwierdza:** całkowitego braku długofalowego dryftu pamięci przy tysiącach zapytań z bardzo różnorodnym słownictwem (pamięć podręczna `chunkCache` rośnie do limitu `50 000` wpisów), ani zachowania pamięci w przeglądarce po wielokrotnym ładowaniu kolejnych plików `.tny3`.
3. **Symulacja DOM w `node:vm`:**
   - **Potwierdza:** brak błędów składniowych w `tartarus-browser.html`, brak kolizji nazw po połączeniu modułów w IIFE, poprawne podpięcie zdarzeń i aktualizację liczników/pasków w obiekcie DOM.
   - **NIE potwierdza:** wyglądu wizualnego CSS, ergonomii drag-and-drop w systemach Windows/macOS/Linux, kompatybilności starszych przeglądarek ani rzeczywistej płynności interfejsu w prawdziwym Chrome/Firefox/Safari.
4. **Jakość odpowiedzi w `model.tny3` (`" Armia przebywa w stolicy. "`):**
   - **Potwierdza:** działanie całego łańcucha end-to-end (tekst → BPE → W8A32 → KV Cache → polskie UTF-8 + `<|eos|>`).
   - **NIE potwierdza:** merytorycznej inteligencji ani wierności odczytu liczb ze `<|state|>` (obecne wagi to wynik zaledwie 160 kroków treningowych na CPU; w ewaluacji Etapu 3 wynik `state_query` wyniósł `0/4`). Do wysokiej jakości merytorycznej potrzebny jest pełny trening docelowy na GPU.
