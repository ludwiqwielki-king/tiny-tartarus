Utworzono kompletną dokumentację techniczną, edukacyjną i referencyjną projektu **Tiny (Tartarus)** w strukturze modułowej (przewodnik główny + 7 powiązanych dokumentów w katalogu `docs/`). Nie zmodyfikowano żadnych plików kodu, wag, konfiguracji ani testów poza `docs/`.

---

### 1. Lista utworzonych plików dokumentacji

1. **`/home/user/docs/TINY_SYSTEM_DOCUMENTATION.md`** — główny punkt wejścia otwarty w podglądzie:
   - Legenda statusów (`POTWIERDZONE`, `ZAIMPLEMENTOWANE`, `ZAŁOŻENIE PROJEKTOWE`, `NIEZWERYFIKOWANE`, `OGRANICZENIE`, `DO WYJAŚNIENIA`),
   - Kompletny spis treści z linkami względnymi do wszystkich rozdziałów i dodatków,
   - **Rozdział I** (historia Etapów 1–5 i diagram Mermaid 7 warstw przepływu danych),
   - **Rozdział X** (przenośność czystego ES2022 ESM, separacja gry vs silnika, przykładowy projekt API integracji oznaczony jako `[ZAŁOŻENIE PROJEKTOWE]`),
   - **Rozdział XII** (krytyczna analiza statusów, ograniczenia i 5-obszarowy plan dalszej walidacji),
   - **Mapa Kodu** (kompletny indeks plików z dokładnymi rozmiarami/liniami i tabela `Funkcja → Plik → Zakres linii → Wejścia → Wyjścia → Zależności`),
   - **Lista kontrolna do weryfikacji pracy agentów (10 pytań)**.
2. **`/home/user/docs/TINY_ARCHITECTURE.md`** — **Rozdziały II i III** (podstawy LLM i Transformera odwzorowane 1:1 w Tiny oraz tokenizer Byte-Level BPE `tartarus_tokenizer.ts`, obejście braku `\w` Unicode dla Pythona vs JS, `createStreamDecoder()` dla 2-bajtowych polskich liter na przykładzie `"Świętogora"` oraz granice testów tokenizera).
3. **`/home/user/docs/TINY_MATHEMATICS.md`** — wyjaśnienie operacji na **Poziomie A / B / C** wraz z ręcznymi przykładami liczbowymi z Tiny: kwantyzacja/dekwantyzacja `INT8 per-row (W8A32)`, `RMSNorm` (`eps=1e-5`), `RoPE half-split` ($\Theta = 10000$), `Multi-Head Causal Self-Attention`, `SwiGLU`, `tied embeddings` i matematyka samplingu.
4. **`/home/user/docs/TINY_FORMAT_TNY3.md`** — **Rozdział IV** (diagram układu pliku `.tny3`, kompletna tabela wszystkich pól 64-bajtowego nagłówka Little-Endian, flagi bitowe, 4-bajtowy alignment i padding `0x00`, `CRC-32`, dokładna kolejność tensorów i bilans formatu W8A32).
5. **`/home/user/docs/TINY_RUNTIME.md`** — **Rozdziały V, VI, VII, IX i XI** (specyfikacja metod `tartarus-engine.ts`, porównanie `prefill` vs `step` vs `forwardNoCache`, KV Cache `L × CTX × D` i brak alokacji dla GC z diagramem sekwencji, sampling `greedy`/`temperature`/`topK`/`minP`/`repetitionPenalty`, podział UI vs silnik w `tartarus-browser.html`, wzory na zużycie RAM oraz rozdzielone wyniki dla Node.js i symulacji HTML dla Profili `S` i `M`).
6. **`/home/user/docs/TINY_TESTS.md`** — **Rozdział VIII** (weryfikacja matematyczna `verify_ts.mjs` przeciwko `golden_test.json`, pełne tabele wyników `maxAbs = 2.146e-6`, `KV Cache vs NoCache = 0.000e+0`, 16 kroków `maxΔ = 0.00e+0` oraz precyzyjne wskazanie czego te testy **nie** dowodzą).
7. **`/home/user/docs/TINY_GLOSSARY.md`** — słownik wszystkich 46 wymaganych haseł (plus 3 pojęcia pomocnicze: `CSR`, `Little-Endian`, `Tied embeddings`), z których każde opisano prostym językiem, technicznie oraz konkretnym przykładem z Tiny.
8. **`/home/user/docs/TINY_LEARNING_PATH.md`** — 12-etapowa praktyczna ścieżka nauki dla właściciela projektu (cel, wyjaśnienie z analogiami DOS/plików binarnych, dokładne miejsce w kodzie, gotowy eksperyment w terminalu, oczekiwany rezultat i pytania kontrolne).

*(Wszystkie 137 linków wewnętrznych, 3 diagramy Mermaid, 46 haseł słownika, 12 etapów × 6 pól oraz brak jakichkolwiek zmian poza `docs/` sprawdzono automatycznym skryptem po ostatniej edycji.)*

---

### 2. Rzeczy niejednoznaczne / niezweryfikowane / otwarte ryzyka

- **`[OGRANICZENIE]` / `[NIEZWERYFIKOWANE]` Stan wytrenowania wag w `model.tny3`:** Obecny model Profilu `S` pochodzi wyłącznie z próbnego treningu CPU na `160` kroków (`checkpoints/train_history.json`: `val_loss_float = 5.7904`, `val_loss_int8 = 5.7913`), a nie z pełnego 20-epokowego treningu GPU. Silnik TypeScript jest matematycznie zgodny z PyTorchem (`maxAbs = 2.146e-6`), lecz jakość merytoryczna generowanych odpowiedzi nie nadaje się jeszcze do gry.
- **`[NIEZWERYFIKOWANE]` Prawdziwa przeglądarka WWW:** W środowisku brak przeglądarki Chromium/Firefox oraz GPU. Działanie `tartarus-browser.html` (profile `S`/`M`, streaming, strażnik `LIMIT CTX!`) zweryfikowano w Node.js za pomocą `vm.Script` z mockowanym DOM, natomiast rzeczywisty rendering CSS, obsługa zdarzeń Drag & Drop w przeglądarce oraz rzeczywista wydajność na sprzęcie użytkownika pozostają nieprzetestowane.
- **`[NIEZWERYFIKOWANE]` Weryfikacja `golden_test` dla Profilu `M` i trwałość `/tmp/model_m.tny3`:** Dla Profilu `M` wykonano test ładowania `TNY3`, `prefill` i benchmark (`118.8 tok/s` w Node.js) na tymczasowym pliku `/tmp/model_m.tny3`, lecz nie wygenerowano osobnego `golden_test` przeciwko PyTorch i plik ten znajduje się w `/tmp` (poza trwałym `/home/user`).
- **`[DO WYJAŚNIENIA]` Nazwa pliku UI:** W repozytorium nie istnieje plik `tiny-2.1-browser-v2.html`; faktycznym jednoplikowym interfejsem przeglądarkowym wygenerowanym przez `build_browser.mjs` jest `/home/user/tartarus-browser.html` (`88 592 B`).

---

### 3. Krótkie podsumowanie architektury Tiny

Tiny rozdziela **fabrykę treningową w Pythonie/PyTorch** (generator korpusu RPG `uploads/tartarus_data.py` → tokenizer BPE `V=2048` i pakiety treningowe `tartarus-csr-v1` → dekoderowy Transformer `RoPE half-split + RMSNorm + SwiGLU + tied embeddings` → kwantyzacja `INT8 per-row` do samoopisowego kontenera `.tny3` chronionego przez `CRC-32`) od **niezależnego runtime’u `ES2022 ESM` bez żadnych zależności i bez DOM** (`tartarus_tokenizer.ts` + `tartarus-engine.ts`), który ładuje `.tny3` w trybie **W8A32**, utrzymuje prealokowany bufor `KV Cache` (`L × CTX × D`) bez alokacji w pętli `step()` i pozwala osadzić model zarówno w jednoplikowym interfejsie offline `tartarus-browser.html`, jak i bezpośrednio w Node.js lub przyszłym adapterze gry.