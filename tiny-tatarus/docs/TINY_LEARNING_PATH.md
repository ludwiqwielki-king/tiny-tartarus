# Jak zrozumieć Tiny od podstaw – ścieżka nauki właściciela projektu

> Część dokumentacji systemu Tiny: [← Powrót do głównego przewodnika](./TINY_SYSTEM_DOCUMENTATION.md)  
> Powiązane dokumenty: [Słownik pojęć](./TINY_GLOSSARY.md) · [Architektura i Tokenizer](./TINY_ARCHITECTURE.md) · [Format `.tny3`](./TINY_FORMAT_TNY3.md) · [Silnik i KV Cache](./TINY_RUNTIME.md) · [Testy](./TINY_TESTS.md)

Ten przewodnik nie zakłada znajomości TypeScriptu, Pythona ani uczenia maszynowego. Zamiast uczyć programowania „od zmiennych i klas”, prowadzi Cię przez **12 krótkich, praktycznych kroków** opartych wyłącznie na prawdziwych plikach projektu Tiny. Analogią do znanego Ci świata DOS będą tu katalogi, pliki wykonywalne, rekordy binarne o stałym nagłówku (jak `.EXE`, `.WAV` czy `.BMP`) oraz rejestry buforów w pamięci.

---

## Etap 1: Jak czytać strukturę projektu (katalog i przepływ danych)

- **Cel:** Zrozumieć, które pliki w katalogu są „fabryką treningową”, które są „gotowym produktem”, a które „odtwarzaczem”.
- **Wyjaśnienie:** Wyobraź sobie drukarnię i czytnik książek. Większość dużych plików (`train.jsonl`, `train_tokens.bin`, `checkpoints/best.pt`, `train_tartarus.py`) to **maszyny i surowce fabryki** (Python + PyTorch) potrzebne tylko wtedy, gdy uczysz nowy model. Natomiast do **samego uruchomienia** gotowego modelu potrzebujesz zaledwie dwóch rzeczy: pliku z danymi modelu (`model.tny3`) oraz odtwarzacza (`tartarus-browser.html` albo `tartarus-engine.mjs` + `tartarus_tokenizer.mjs`).
- **Konkretne miejsce w kodzie:** Spis wszystkich plików znajduje się w [Mapie Kodu](./TINY_SYSTEM_DOCUMENTATION.md#mapa-kodu), a diagram przepływu w [Rozdziale I](./TINY_SYSTEM_DOCUMENTATION.md#rozdział-i--historia-i-ogólna-architektura-tiny).
- **Mały eksperyment praktyczny (w terminalu w `/home/user`):**
  ```bash
  ls -lh model.tny3 tartarus-browser.html tartarus-engine.ts tartarus_tokenizer.ts golden_test.json
  ```
- **Oczekiwany rezultat:** Zobaczysz, że cały gotowy model to zaledwie `~2.3 MB` (`model.tny3`), a cały działający interfejs z wbudowanym silnikiem to jeden plik `~87 KB` (`tartarus-browser.html`). Nie potrzeba ani PyTorcha, ani 26-megabajtowego pliku `checkpoints/best.pt` do działania u gracza.
- **Pytania kontrolne:**
  1. Które dwa pliki wystarczą, aby uruchomić gotowy model na innym komputerze w przeglądarce?
  2. Do czego służy `checkpoints/best.pt` i dlaczego nie wrzuca się go do przeglądarki?

---

## Etap 2: Jak czytać funkcję TypeScript (składnia bez strachu)

- **Cel:** Nauczyć się czytać sygnaturę i ciało funkcji w TypeScript, ignorując wizualny „szum” adnotacji typów.
- **Wyjaśnienie:** W językach takich jak C albo Pascal przed lub po nazwie zmiennej pisze się jej rodzaj (np. `int`, `string`). TypeScript działa tak samo, tyle że pisze typ **po dwukropku** (`nazwa: typ`), a wynik funkcji po dwukropku na końcu nawiasu. Słowo `export` oznacza „tę funkcję można wywołać z innego pliku” (jak `PUBLIC` w assemblerze/Pascalu), a `const` oznacza nazwę wartości, której nie podmienimy na inny obiekt.
- **Konkretne miejsce w kodzie:** Funkcja `silu` w `tartarus-engine.ts:449–455` oraz `rmsNormInto` w `tartarus-engine.ts:383–400`.
  ```ts
  function silu(z: number): number {
    if (z >= 0) {
      return z / (1.0 + Math.exp(-z));
    }
    const ez = Math.exp(z);
    return (z * ez) / (1.0 + ez);
  }
  ```
  Czytasz to po polsku: *„Funkcja `silu` bierze liczbę `z`, a oddaje liczbę. Jeśli `z >= 0`, oddaje `z / (1 + e^-z)`. W przeciwnym razie liczy `e^z` i oddaje `(z * e^z) / (1 + e^z)`”*.
- **Mały eksperyment praktyczny:** Otwórz `tartarus-engine.ts` na liniach `449–455` i porównaj z definicją klasy `RMSNorm` w Pythonie (`model_tartarus.py:135–139`) — zobaczysz dokładnie tę samą matematykę, zapisaną tylko innymi nawiasami (`{ ... }` zamiast wcięć).
- **Oczekiwany rezultat:** Przekonasz się, że zapis `x: Float32Array` to nie „czarna magia”, tylko wyraźna etykieta mówiąca „tu musi wejść tablica liczb ułamkowych Float32”.
- **Pytania kontrolne:**
  1. W nagłówku `function computeCrc32(bytes: Uint8Array): number` — co wchodzi do funkcji, a co z niej wychodzi?
  2. Co w TypeScript oznacza znak `?` przy argumencie `options?: DecodeOptions`? *(Podpowiedź: opcjonalny argument, można go nie podawać).*

---

## Etap 3: Jak rozumieć typy danych i tablice (`ArrayBuffer`, `DataView`, `Float32Array`)

- **Cel:** Zrozumieć, jak JavaScript czyta surowe bajty z pliku binarnego i operuje na ciągłych blokach pamięci bez obciążania systemu.
- **Wyjaśnienie:** W świecie DOS pamiętasz zapewne segment pamięci albo bufor odczytany z pliku przez `fread`. W JS odpowiednikiem surowego bloku bajtów jest **`ArrayBuffer`**. Sam w sobie nie wie, czy bajty to tekst, czy liczby — to po prostu ciąg zer i jedynek. Dopiero nakładamy na niego „okulary” (widok):
  - `Uint8Array` patrzy na każdy bajt z osobna jako liczbę `0..255`.
  - `Int8Array` patrzy na każdy bajt jako liczbę ze znakiem `-128..127`.
  - `Float32Array` patrzy na każde 4 bajty jako liczbę ułamkową.
  - `DataView` pozwala wyciągnąć konkretną liczbę dowolnego typu spod wybranego adresu (offsetu), np. `dv.getUint32(8, true)` (gdzie `true` oznacza porządek bajtów Little-Endian znany z procesorów x86).
- **Konkretne miejsce w kodzie:** `tartarus-engine.ts:210–248` (`dv.getUint16`, `dv.getUint32`, `dv.getFloat32`) oraz `tartarus-engine.ts:529–551` (tworzenie buforów `Float32Array`).
- **Mały eksperyment praktyczny:** Podejrzyj pierwsze 64 bajty pliku `model.tny3` z perspektywy nagłówka TNY3:
  ```bash
  node -e '
    const fs = require("node:fs");
    const b = fs.readFileSync("model.tny3");
    const dv = new DataView(b.buffer, b.byteOffset, 64);
    console.log("Magic:", b.subarray(0,4).toString("ascii"));
    console.log("Wersja:", dv.getUint16(4, true), "Flagi:", dv.getUint16(6, true));
    console.log("Vocab:", dv.getUint32(8, true), "CTX:", dv.getUint32(12, true), "D:", dv.getUint32(16, true), "L:", dv.getUint32(20, true));
  '
  ```
- **Oczekiwany rezultat:** Zobaczysz na ekranie: `Magic: TNY3`, `Wersja: 1 Flagi: 5`, `Vocab: 2048 CTX: 256 D: 192 L: 4` — dokładnie to samo, co robi funkcja `parseTNY3()` przy otwarciu pliku!
- **Pytania kontrolne:**
  1. Dlaczego w `dv.getUint32(8, true)` drugi argument to `true`?
  2. Ile bajtów w pamięci zajmuje `new Float32Array(100)`?

---

## Etap 4: Jak model przechowuje liczby i wagi (`.tny3` i dekwantyzacja)

- **Cel:** Zobaczyć na własne oczy, jak z jednego bajtu `INT8` i skali `Float32` odtwarza się wagę modelu w pamięci RAM.
- **Wyjaśnienie:** W nagłówkach plików graficznych czy dźwiękowych w DOS najpierw był stały nagłówek (np. 54 bajty w `.BMP`), a potem surowe piksele. W `model.tny3` po 64-bajtowym nagłówku leży kawałek tekstu JSON (słownik), potem `0..3` bajty zerowego dopełnienia, a potem rząd za rzędem macierze wag. Każdy wiersz macierzy zaczyna się od 4-bajtowego mnożnika `scale`, po którym idzie `cols` jednobajtowych liczb całkowitych. Gdy model wczytuje plik, mnoży każdy bajt przez `scale` swojej linii i wpisuje wynik do `Float32Array`.
- **Konkretne miejsce w kodzie:** `readMatrixInt8Dequant` w `tartarus-engine.ts:167–199` i `quantize_per_row_int8` w `export_tny3.py:98–113`.
- **Mały eksperyment praktyczny:** Odczytaj pierwszy wiersz tabeli embeddingów prosto z `model.tny3` (spod `weights_offset = 180692`) i policz pierwsze 3 odtworzone wagi:
  ```bash
  node -e '
    const fs = require("node:fs");
    const b = fs.readFileSync("model.tny3");
    const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    const wOff = dv.getUint32(48, true);
    const scale = dv.getFloat32(wOff, true);
    const q = new Int8Array(b.buffer, b.byteOffset + wOff + 4, 3);
    console.log("weights_offset=", wOff, "scale=", scale, "q[0..2]=", [...q], "F32=", [...q].map(x => x * scale));
  '
  ```
- **Oczekiwany rezultat:** Zobaczysz `weights_offset= 180692`, małą wartość `scale`, 3 liczby całkowite z przedziału `-127..127` oraz odtworzone małe ułamki `Float32`.
- **Pytania kontrolne:**
  1. Dlaczego po sekcji tokenizera między bajtem `180691` a `180692` dopisano jeden zerowy bajt `0x00`?
  2. Co się stanie, jeśli w pliku `model.tny3` zmienisz choćby 1 bajt wag? Sprawdź rozdział o `CRC-32`.

---

## Etap 5: Jak działa tokenizer (tekst ↔ bajty ↔ ID ↔ UTF-8)

- **Cel:** Zrozumieć podział na tokeny specjalne, słowa, polskie litery dwubajtowe i odtwarzanie tekstu bez „krzaczków”.
- **Wyjaśnienie:** Tokenizer najpierw wycina tagi gry (`<|state|>`, `<|user|>`, `<|npc|>`), potem dzieli resztę na słowa i znaki. Każde słowo rozbija na pojedyncze bajty UTF-8 (`+ 5` za 5 tagów specjalnych), a następnie skleja pary bajtów według listy scaleń zapisanej w `tokenizer.json` (i wbudowanej w `model.tny3`). Przy zamianie liczb z powrotem na tekst nie wolno zamieniać każdego tokenu osobno na literę, bo polskie znaki (`ą, ć, ę, ł, ń, ó, ś, ź, ż`) składają się z 2 bajtów, które BPE potrafi rozdzielić między dwa sąsiednie tokeny.
- **Konkretne miejsce w kodzie:** `tartarus_tokenizer.ts:180–202` (`createStreamDecoder`), `204–257` (`encodeChunk`), `259–338` (`encode` / `decode`).
- **Mały eksperyment praktyczny:** Zobacz, jak rozbijane jest słowo `"Świętogora"` (z rozciętą polską literą `Ś`) i jak `createStreamDecoder` składa je bezbłędnie:
  ```bash
  node -e '
    import("./tartarus_tokenizer.mjs").then(({ TartarusTokenizer }) => {
      const fs = require("node:fs");
      const tok = new TartarusTokenizer(JSON.parse(fs.readFileSync("tokenizer.json", "utf8")));
      const ids = tok.encode("Świętogora");
      console.log("IDs:", ids);
      console.log("Błędne dekodowanie kawałek po kawałku:", ids.map(id => new TextDecoder().decode(tok.getTokenBytes(id))));
      const sd = tok.createStreamDecoder();
      console.log("Poprawny streamDecoder krok po kroku:", ids.map(id => sd.push(id)));
    });
  '
  ```
- **Oczekiwany rezultat:** Zobaczysz `IDs: [202, 159, 809, 819]`. Przy błędnym dekodowaniu po jednym tokenie dla `202` i `159` pojawią się znaki zastępcze błędu UTF-8 (`U+FFFD`), natomiast `streamDecoder.push(202)` zwróci pusty napis `""`, a `push(159)` od razu kompletną literę `"Ś"`.
- **Pytania kontrolne:**
  1. Jakie stałe numery mają `<|pad|>`, `<|eos|>`, `<|state|>`, `<|user|>` i `<|npc|>`?
  2. Co się stanie, jeśli w pliku `tokenizer.json` zmienimy kolejność dwóch wpisów w tablicy `merges`?

---

## Etap 6: Jak działa pojedyncza warstwa Transformer

- **Cel:** Zobaczyć dwie połówki każdej warstwy (`Attention` + `SwiGLU`) i zasadę połączenia rezydualnego (`x += ...`).
- **Wyjaśnienie:** Wektor tokenu `x` to notatnik z `192` liczbami. Warstwa działa jak dwustopniowy filtr:
  1. Wyrównaj skalę notatnika (`RMSNorm`), spytaj poprzednie tokeny o kontekst (`Attention`) i **dodaj** zebrane informacje do `x` (`this.x[i] += this.woOut[i]`).
  2. Znowu wyrównaj skalę (`RMSNorm`), przefiltruj myśli przez dwutorowy blok `SwiGLU` i **dodaj** wynik do `x` (`this.x[i] += this.ffOut[i]`).
  W `tartarus-browser.html` słupki „Warstwa 0 .. Warstwa L-1” pokazują na żywo średnią wielkość liczb w notatniku `x` (tzw. RMS) po wyjściu z każdej warstwy.
- **Konkretne miejsce w kodzie:** Pętla warstw `for (let l = 0; l < nLayers; l++)` w `tartarus-engine.ts:588–677` oraz odpowiadająca jej klasa `TartarusBlock` w `model_tartarus.py:211–299`.
- **Mały eksperyment praktyczny:** Podejrzyj jak zmienia się wartość RMS w notatniku `x` po przejściu warstw 0, 1, 2 i 3 dla pierwszego przypadku testowego:
  ```bash
  node -e '
    import("./tartarus-engine.mjs").then(({ TartarusEngine }) => {
      const fs = require("node:fs");
      const eng = TartarusEngine.fromArrayBuffer(fs.readFileSync("model.tny3"));
      const gen = eng.generateStream("<|state|> hp=42/75; złoto=390 <|user|> Ile złota? <|npc|>", { temperature: 0, maxNewTokens: 3 });
      for (const ev of gen) {
        console.log("Krok", ev.step, "token", ev.tokenId, JSON.stringify(ev.textDelta), "layerRMS=", Array.from(ev.layerRms).map(v => v.toFixed(3)));
      }
    });
  '
  ```
- **Oczekiwany rezultat:** Zobaczysz na żywo `4` liczby `layerRMS` dla każdego tokenu (np. rosnące wartości między warstwą 0 a 3). Gdybyś podał `/tmp/model_m.tny3`, zobaczysz automatycznie `6` liczb!
- **Pytania kontrolne:**
  1. Dlaczego warstwa nie nadpisuje `this.x`, tylko dodaje `this.x[i] += ...`?
  2. Co robi `applyRopeHalfSplitInPlace` przed zapisaniem `K` do cache i dlaczego nie stosuje się go do `V`?

---

## Etap 7: Jak model przetwarza cały prompt (`prefill`)

- **Cel:** Zrozumieć, co dzieje się od kliknięcia „Generuj” do pojawienia się pierwszego słowa odpowiedzi.
- **Wyjaśnienie:** Kiedy wpisujesz prompt o długości `78` tokenów, model musi „przeczytać” je po kolei od pozycji `0` do `77`. W tej fazie (**`prefill`**) dla pierwszych `77` tokenów interesuje nas wyłącznie zapamiętanie ich `K` i `V` w buforze KV Cache — dopiero na ostatnim tokenie (`<|npc|>`, indeks `77`) silnik przelicza wektor wyników `logits` dla wszystkich `2048` słów ze słownika.
- **Konkretne miejsce w kodzie:** Metoda `prefill(tokenIds)` w `tartarus-engine.ts:687–698` (zwróć uwagę na warunek `i === n - 1` przekazywany jako `computeLogits`).
- **Mały eksperyment praktyczny:** Zmierz osobno czas `prefill` i zobacz `top-3` najlepiej ocenione pierwsze tokeny po `<|npc|>`:
  ```bash
  node -e '
    import("./tartarus-engine.mjs").then(({ TartarusEngine }) => {
      const fs = require("node:fs");
      const eng = TartarusEngine.fromArrayBuffer(fs.readFileSync("model.tny3"));
      const ids = eng.encode("<|state|> lokacja=Gaj Świętogora; typ=zamek; teren=las; tura=14; frakcja=Plemiona Słowiańskie; dowódca=Wojewoda Siemomysł; hp=42/75; mana=9/16; złoto=390; ruch=5; jednostki=Leszyj Puszczyx2,Wołkodłakx3; wrogowie=Królestwo Egiptu(4p) <|user|> Ile punktów życia i złota ma nasza armia? <|npc|>");
      const t0 = performance.now();
      const logits = eng.prefill(ids);
      const dt = performance.now() - t0;
      const top = Array.from(logits).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
      console.log("Prompt tok:", ids.length, "kvPos:", eng.kvPos, "prefill ms:", dt.toFixed(1));
      console.log("Top-3:", top.map(([v, i]) => ({ id: i, piece: eng.decode([i], {skipSpecial:false}), logit: Number(v.toFixed(3)) })));
    });
  '
  ```
- **Oczekiwany rezultat:** `Prompt tok: 78`, `kvPos: 78`, a w `Top-3` na 1. miejscu token `1296` (`" Armia"`, logit `6.916`).
- **Pytania kontrolne:**
  1. Co dzieje się z wartością `eng.kvPos` po wywołaniu `prefill(ids)` dla `78` tokenów?
  2. Dlaczego `prefill` przekazuje `false` jako `computeLogits` dla `i = 0 .. 76`?

---

## Etap 8: Jak działa KV Cache (`step` i `kvPos`)

- **Cel:** Zrozumieć, dlaczego dopisywanie kolejnych słów odpowiedzi jest wielokrotnie szybsze niż ponowne czytanie promptu i dlaczego w pętli nie wolno tworzyć nowych tablic.
- **Wyjaśnienie:** Po fazie `prefill` w buforach `kCache` i `vCache` leży gotowa historia pozycji `0..77`. Kiedy model wylosuje token `1296` (`" Armia"`), funkcja `step(1296)` liczy macierze tylko dla tego **jednego** słowa na pozycji `78`, dokleja jego `K` i `V` pod pozycją `78`, robi szybki przegląd Attention po wszystkich pozycjach `0..78` i przesuwa `kvPos` na `79`. Wszystkie tablice buforów były przygotowane przy starcie silnika (`prealokacja`), więc ani razu nie budujemy nowego obiektu tablicy.
- **Konkretne miejsce w kodzie:** `kCache`, `vCache`, `kvPos` i `step()` w `tartarus-engine.ts:472–474`, `588–647`, `703–708`.
- **Mały eksperyment praktyczny:** Uruchom `node verify_ts.mjs` i przyjrzyj się linijce z wynikiem testu 16-krokowego.
- **Oczekiwany rezultat:** Zobaczysz wpis `PASS [16-krokowa zgodność autoregresyjna KV Cache == forwardNoCache]: maxΔ=0.00e+0`. Oznacza to, że `step()` z cache daje co do bitu identyczny wynik jak pełne przeliczenie całej sekwencji od zera w `forwardNoCache()`.
- **Pytania kontrolne:**
  1. Dlaczego `resetCache()` nie musi zerować całego półtoramegabajtowego bufora `kCache`/`vCache`, a wystarczy ustawienie `this.kvPos = 0`?
  2. Co by się stało z pamięcią i płynnością przeglądarki, gdyby `step()` przy każdym słowie robił `new Float32Array(...)`?

---

## Etap 9: Jak powstaje kolejny token (Sampling i zatrzymywanie)

- **Cel:** Zrozumieć różnicę między trybem deterministycznym (`temperature = 0`) a losowaniem kontrolowanym (`topK`, `minP`, `repetitionPenalty`).
- **Wyjaśnienie:** Silnik po każdym kroku ma listę `2048` ocen (`logits`). W trybie `temperature = 0` po prostu bierze najwyższą ocenę (zawsze to samo zdanie). Jeśli chcesz urozmaicenia, zwiększasz `temperature` (wyrównujesz szanse słabszych słów), przycinasz listę do `topK` faworytów, odcinasz śladowe resztki przez `minP` i nakładasz karę za powtórzenia (`repetitionPenalty`). Generacja kończy się w momencie pojawienia się tokenu `1` (`<|eos|>`) albo po zapełnieniu okna `CTX` (`256`).
- **Konkretne miejsce w kodzie:** `sampleNextToken()` w `tartarus-engine.ts:823–941` oraz kontrolki suwaków w `tartarus-browser.html:672–701`.
- **Mały eksperyment praktyczny:** Otwórz `tartarus-browser.html` (albo uruchom `node verify_ts.mjs`) i porównaj wynik z `temperature = 0` vs `temperature = 1.1, topK = 40, minP = 0.04` (w przeglądarce zobaczysz też w inspektorze pod spodem numery ID każdego wylosowanego tokenu i czerwony lub zielony żeton `<|eos|>` na końcu).
- **Oczekiwany rezultat:** Przy `temperature = 0` odpowiedź jest zawsze identyczna co do spacji; przy wyższej temperaturze zmieniają się wybory tokenów, a suwaki w UI nie zmieniają wag modelu — zmieniają tylko sposób losowania z tych samych logitów.
- **Pytania kontrolne:**
  1. Co w tym modelu sygnalizuje token `1` (`<|eos|>`)?
  2. Dlaczego w `sampleNextToken` kara `repetitionPenalty` dla `seenTokens = [100, 100, 100]` powinna zadziałać tylko raz dla ID `100`?

---

## Etap 10: Jak działa cały runtime (od otwarcia pliku do ekranu)

- **Cel:** Połączyć w głowie cały łańcuch zdarzeń od opuszczenia pliku `.tny3` na okno przeglądarki do zaktualizowania interfejsu.
- **Wyjaśnienie:** W `tartarus-browser.html` żyją dwa światy obok siebie w jednym pliku:
  1. **Świat silnika** (`TartarusTokenizer` + `TartarusEngine`, linie `836–1860`) — zero wiedzy o przyciskach i kolorach.
  2. **Świat interfejsu** (`loadModelBuffer`, `updatePromptAndCtx`, `runStreamingGeneration`, linie `1865–2408`) — słucha zdarzeń myszy/klawiatury, liczy budżet tokenów po każdym naciśnięciu klawisza, wywołuje w pętli generator `engine.generateStream()` i po każdym jednym tokenie oddaje sterowanie przeglądarce (`setTimeout(0)`), żebyś widział pojawiające się na bieżąco litery i mógł nacisnąć „Przerwij”.
- **Konkretne miejsce w kodzie:** `tartarus-browser.html:2105–2311` (ładowanie pliku + pętla streamingu) oraz `build_browser.mjs` (skrypt budujący ten plik z `.js`).
- **Mały eksperyment praktyczny:** Uruchom w `/home/user` jednorazową komendę symulującą przebieg w przeglądarce albo otwórz `tartarus-browser.html` w podglądzie pliku / przez `python3 -m http.server 8000` (jeśli lokalnie na własnym komputerze) i przeciągnij plik `model.tny3`.
- **Oczekiwany rezultat:** W tabeli nagłówka od razu pojawi się `TNY3 v1`, `4 × 192`, `CRC 0x938fb800`, w monitorze telemetrii pojawią się `4` paski warstw, a po kliknięciu „Generuj odpowiedź” tekst wypisze się płynnie token po tokenie.
- **Pytania kontrolne:**
  1. Czy silnik `tartarus-engine.ts` potrafi działać w Node.js bez `tartarus-browser.html`? Dlaczego?
  2. Co kontroluje liczbę wierszy w panelu „Aktywność strumienia rezydualnego warstw (RMS)”?

---

## Etap 11: Jak samodzielnie przeprowadzać testy i weryfikować pracę agentów

- **Cel:** Umieć samodzielnie uruchomić kompletny zestaw testów automatycznych w terminalu i odróżnić prawdziwe dowody działania od deklaracji.
- **Wyjaśnienie:** Nigdy nie musisz wierzyć agentowi „na słowo”, że kod działa. W repozytorium masz trzy niezależne „bramki prawdy”:
  1. Kompilator TypeScript (`tsc`) sprawdza, czy kod ma spójne typy i nie dotyka DOM ani nieistniejących bibliotek.
  2. `verify_tokenizer.mjs` sprawdza, czy JS/TS dzieli tekst identycznie jak Python na `119` przypadkach testowych oraz na całych `2000` rekordach `val` i `test`.
  3. `verify_ts.mjs` sprawdza plik `model.tny3`, sumę CRC-32, wszystkie `2048` logity przeciwko Pythonowi (`golden_test.json`), `16` kroków KV Cache przeciwko pełnemu przebiegowi, ochronę limitu `CTX`, streaming UTF-8 i prędkość `tok/s`.
- **Konkretne miejsce w kodzie:** `verify_tokenizer.mjs` oraz `verify_ts.mjs`.
- **Mały eksperyment praktyczny:** Uruchom cały pakiet testów w `/home/user`:
  ```bash
  npx -y -p typescript tsc --ignoreConfig --noEmit --strict --target ES2022 --module ES2022 --lib ES2022 tartarus_tokenizer.ts tartarus-engine.ts
  node verify_tokenizer.mjs
  node verify_ts.mjs
  ```
- **Oczekiwany rezultat:** Wszystkie trzy polecenia muszą zakończyć się statusem sukcesu (`exit code 0`), `verify_tokenizer.mjs` ma wypisać `119/119` i `1000/1000` (`val` i `test`), a `verify_ts.mjs` końcowe podsumowanie `=== WSZYSTKIE BRAMKI ETAPU 5 ZALICZONE ===` z `maxAbs < 1e-4` i `KV Cache vs NoCache = 0.000e+0`.
- **Pytania kontrolne:**
  1. Jeżeli zobaczysz w raporcie `maxAbs = 2.146e-6`, czy to jest błąd, czy norma? A co jeśli zobaczysz `0.35`?
  2. Dlaczego `verify_ts.mjs` testuje oddzielnie `prefill + step` oraz `forwardNoCache`?

---

## Etap 12: Jak diagnozować najczęstsze błędy

- **Cel:** Rozpoznawać typowe komunikaty błędów i wiedzieć, w którym miejscu łańcucha szukać przyczyny.
- **Wyjaśnienie:** Każdy etap walidacji w Tiny rzuca precyzyjny komunikat błędu po polsku. Poniżej masz mapę diagnostyczną „objaw → przyczyna → gdzie sprawdzić”:
  - **„Błąd sumy kontrolnej CRC-32 payloadu TNY3”:** Plik `.tny3` został ucięty przy kopiowaniu albo uszkodzony. Wygeneruj ponownie z checkpointu przez `export_tny3.py` lub pobierz ponownie.
  - **„Przekroczono CTX=256” / `RangeError: Prompt ... przekracza limit`:** Za długi tekst w `<|state|>` + `<|user|>`. Skróć stan gry; model ma sztywny limit okna `256` tokenów.
  - **„Rozbieżność vocabSize” albo `tokenizer_sha256` mismatch:** Zmieszano plik `model.tny3` / checkpoint z innego treningu ze złym `tokenizer.json`. W `.tny3` tokenizer jest wbudowany w plik, więc w runtime nigdy nie trzeba dobierać go osobno.
  - **Znak błędu UTF-8 (`U+FFFD`, tzw. *replacement character*) w odpowiedzi:** Ktoś wywołał `decode([id])` pojedynczo zamiast `createStreamDecoder()` lub całego `decode(wszystkie_id)`.
  - **`ReferenceError: exports is not defined in ES module scope` w Node.js:** Ktoś skompilował `.ts` do CommonJS zamiast ESM (`"module": "ES2022"` w `tsconfig.json`).
  - **Odpowiedzi brzmią po polsku, ale zmyślają liczby ze `<|state|>`:** To **nie jest błąd silnika JS** — `verify_ts.mjs` pokazuje zgodność `2.146e-6` z PyTorchem. Przyczyną są aktualne wagi w `model.tny3`, które trenowano zaledwie `160` kroków na CPU do celów testu całego rurociągu (patrz [Rozdział XII](./TINY_SYSTEM_DOCUMENTATION.md#rozdział-xii--ograniczenia-ryzyka-i-dalsza-walidacja)).
- **Konkretne miejsce w kodzie:** Wszystkie kontrole `throw new Error(...)` / `throw new RangeError(...)` w `parseTNY3()` (`tartarus-engine.ts:206–353`), `TartarusTokenizer` (`tartarus_tokenizer.ts:70–162`) oraz `loadModelBuffer()` w `tartarus-browser.html:2105–2165`.
- **Mały eksperyment praktyczny:** Zasymuluj uszkodzenie pliku `.tny3` w pamięci (bez psucia `model.tny3` na dysku) i zobacz komunikat parsera:
  ```bash
  node -e '
    import("./tartarus-engine.mjs").then(({ parseTNY3 }) => {
      const fs = require("node:fs");
      const buf = new Uint8Array(fs.readFileSync("model.tny3"));
      buf[buf.length - 1] ^= 0xff;
      try { parseTNY3(buf); } catch (e) { console.log("Złapano oczekiwany błąd:", e.message); }
    });
  '
  ```
- **Oczekiwany rezultat:** Program natychmiast odmówi wczytania zepsutych wag, wypisując `Złapano oczekiwany błąd: Błąd sumy kontrolnej CRC-32 payloadu TNY3: ...`.
- **Pytania kontrolne:**
  1. Jeśli po zmianie wag model wypluwa zmyślone liczby, jak w 10 sekund sprawdzisz, czy winny jest silnik JavaScript, czy wytrenowanie modelu w Pythonie?
  2. Jeśli zobaczysz błąd kompilacji `Cannot find name 'document'` podczas kompilacji `tartarus-engine.ts`, co to oznacza i dlaczego to dobrze, że kompilator go zatrzymał?
