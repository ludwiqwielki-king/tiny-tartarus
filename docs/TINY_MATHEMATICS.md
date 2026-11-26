# Matematyka operacji w projekcie Tiny (Tartarus)

> Część dokumentacji systemu Tiny: [← Powrót do głównego przewodnika](./TINY_SYSTEM_DOCUMENTATION.md)  
> Powiązane dokumenty: [Architektura i Tokenizer](./TINY_ARCHITECTURE.md) · [Format `.tny3`](./TINY_FORMAT_TNY3.md) · [Silnik i KV Cache](./TINY_RUNTIME.md) · [Testy](./TINY_TESTS.md)

Każdy mechanizm opisany jest w układzie trzypoziomowym: **Poziom A (intuicyjny)**, **Poziom B (techniczny)** oraz **Poziom C (matematyczny i implementacyjny w kodzie Tiny)**.

---

## 1. Reprezentacja danych: wektory, macierze i precyzja liczb

### Poziom A
Cała wiedza sieci neuronowej mieści się w tablicach liczb rzeczywistych z ułamkami (np. `-0.0412`, `1.285`). Aby plik modelu był lekki na dysku (`2.28 MiB` w Profilu `S`), główne tabele wag zapamiętano w postaci zgrubnych liczb całkowitych od `-127` do `+127` (**`INT8`**) wraz z jednym mnożnikiem skali dla każdego wiersza, a podczas ładowania do pamięci RAM odtwarza się je z powrotem do liczb ułamkowych **`Float32`** (tryb **W8A32**: *8-bit weights on disk, 32-bit float activations & math in RAM*).

### Poziom B
**POTWIERDZONE** — Zarówno w `tartarus-engine.ts`, jak i w trybie ewaluacyjnym `export_tny3.py` wszystkie obliczenia forward pass podczas inferencji odbywają się w 32-bitowej arytmetyce zmiennoprzecinkowej IEEE-754 (`Float32Array` w JS/TS, przy czym akumulatory pętli `number` w JS pracują wewnętrznie w IEEE-754 `float64`, a wyniki zapisywane do `Float32Array` zaokrąglają się do `float32`). Wymiary dla obu zdefiniowanych profili (`model_tartarus.py:85–119`):

| Parametr | Symbol | Profil `S` (`model.tny3`) | Profil `M` (`/tmp/model_m.tny3`) | Rola matematyczna |
| --- | --- | --- | --- | --- |
| Rozmiar słownika | $V$ (`vocab_size`) | `2048` | `2048` | Liczba wierszy tabeli embeddingów i długość wektora logitów |
| Okno kontekstu | $\text{CTX}$ (`ctx_len`) | `256` | `256` | Maksymalna długość sekwencji (pozycje $0 \dots 255$) |
| Wymiar ukryty | $D$ (`d_model`) | `192` | `256` | Długość wektora reprezentującego pojedynczy token |
| Liczba warstw | $L$ (`n_layers`) | `4` | `6` | Liczba bloków `TartarusBlock` |
| Liczba głowic | $H$ (`n_heads`) | `6` | `8` | Liczba równoległych podprzestrzeni w Attention |
| Wymiar głowicy | $d_h$ (`head_dim`) | `32` (`192 / 6`) | `32` (`256 / 8`) | Rozmiar wektora w pojedynczej głowicy ($D / H$) |
| Wymiar FFN | $D_{\text{ff}}$ (`d_ff`) | `512` | `640` | Rozmiar warstwy ukrytej w SwiGLU |
| Stała normalizacji | $\varepsilon$ (`rms_eps`) | `1e-5` | `1e-5` | Zabezpieczenie przed dzieleniem przez zero w `RMSNorm` |
| Baza częstotliwości RoPE | $\Theta$ (`rope_theta`) | `10000.0` | `10000.0` | Skala kątów obrotu pozycyjnego |

---

## 2. Kwantyzacja i dekwantyzacja INT8 per-row (W8A32)

### Poziom A
Zwykła liczba ułamkowa `Float32` zajmuje **4 bajty**. Jeśli zmieścimy ją w `INT8` (od `-127` do `127`), zajmie tylko **1 bajt** — niemal 4 razy mniej miejsca. Ponieważ wartości w jednym wierszu tabeli mogą mieć różną wielkość (np. raz od `-0.1` do `+0.1`, a w innym wierszu od `-3.0` do `+3.0`), każdy wiersz dostaje własny „współczynnik powiększenia” (`scale`).

### Poziom B
**POTWIERDZONE** — Kwantyzacja symetryczna `INT8 per-row` działa na wszystkich macierzach dwuwymiarowych (`tok_emb`, `wq`, `wk`, `wv`, `wo`, `w_gate`, `w_up`, `w_down`). Jednowymiarowe wektory skalujące norm (`rms_att.g`, `rms_ffn.g`, `rms_final.g`) nie są kwantyzowane — pozostają w pełnym `float32` (`export_tny3.py:167–189`, `train_tartarus.py:212–230`).

### Poziom C (wzory i przykład liczbowy)

Niech $W[r, :] \in \mathbb{R}^{C}$ będzie $r$-tym wierszem macierzy o $C$ kolumnach.

**Kwantyzacja podczas eksportu (`export_tny3.py:98–113`):**
$$\text{scale}_r = \max\!\left(\frac{\max_{c \in [0, C)} |W_{r, c}|}{127.0},\; 10^{-8}\right) \quad (\text{zapisane jako float32})$$
$$q_{r, c} = \text{clamp}\!\left(\text{round}\!\left(\frac{W_{r, c}}{\text{scale}_r}\right),\; -127,\; 127\right) \quad (\text{zapisane jako int8})$$

**Dekwantyzacja podczas ładowania `.tny3` (`tartarus-engine.ts:187–197`):**
$$\widehat{W}_{r, c} = q_{r, c} \cdot \text{scale}_r$$

- **Znaczenie zmiennych:** $W_{r, c}$ — oryginalna waga `float32`; $\text{scale}_r$ — maksymalny moduł w wierszu podzielony przez `127` (ograniczony od dołu przez $10^{-8}$, aby wiersz złożony z samych zer nie powodował dzielenia $0/0$); $q_{r, c}$ — liczba całkowita ze znakiem ze zbioru $\{-127, \dots, 127\}$ (wartość `-128` nie jest używana, aby przedział był symetryczny); $\widehat{W}_{r, c}$ — odtworzona waga w `Float32Array`.
- **Przykład liczbowy:** Jeśli wiersz zawiera 4 wagi $[0.508,\; -0.254,\; 0.000,\; 0.127]$:
  1. $\max|W| = 0.508 \implies \text{scale} = 0.508 / 127 = 0.004$.
  2. $q = [\text{round}(127.0), \text{round}(-63.5), 0, \text{round}(31.75)] = [127, -64, 0, 32]$ (w `torch.round` stosowane jest zaokrąglanie do parzystej *round-to-even*, stąd `-63.5 -> -64`).
  3. Po odtworzeniu w silniku JS: $\widehat{W} = [0.508,\; -0.256,\; 0.000,\; 0.128]$. Błąd kwantyzacji wynosi co najwyżej $\pm 0.5 \cdot \text{scale} = \pm 0.002$.

**Implementacja w `tartarus-engine.ts:189–196`:**
```ts
const scale = dv.getFloat32(cursor, true);
const qRow = new Int8Array(buffer, baseByteOffset + cursor + 4, cols);
const dstBase = r * cols;
for (let c = 0; c < cols; c++) {
  mat[dstBase + c] = qRow[c] * scale;
}
```

---

## 3. Normalizacja RMSNorm (`eps = 1e-5`)

### Poziom A
Przed każdą ważną operacją w warstwie wektor liczb `x` przechodzi przez „stabilizator głośności”. Oblicza się typową wielkość liczb w wektorze (pierwiastek ze średniego kwadratu – RMS), dzieli się przez nią cały wektor (tak żeby jego średnia energia wynosiła około `1`), a na końcu przemnaża przez wyuczone współczynniki wagi `g`.

### Poziom B
**POTWIERDZONE** — W odróżnieniu od `LayerNorm`, `RMSNorm` **nie odejmuje średniej** i **nie ma wyrazu wolnego (bias)**. Zarówno w PyTorch (`model_tartarus.py:135–139`), jak i w TypeScript (`tartarus-engine.ts:383–400`) wariancja obliczana jest w precyzji `float32`.

### Poziom C

Dla wektora $x \in \mathbb{R}^{D}$, wektora skalującego $g \in \mathbb{R}^{D}$ oraz $\varepsilon = 10^{-5}$:

$$\text{inv\_rms}(x) = \frac{1}{\sqrt{\frac{1}{D}\sum_{i=0}^{D-1} x_i^2 + \varepsilon}}$$
$$\text{RMSNorm}(x)_i = x_i \cdot \text{inv\_rms}(x) \cdot g_i$$

- **Implementacja (`tartarus-engine.ts:391–399`):**
```ts
let sumSq = 0.0;
for (let i = 0; i < dim; i++) {
  const v = x[xOffset + i];
  sumSq += v * v;
}
const invRms = 1.0 / Math.sqrt(sumSq / dim + eps);
for (let i = 0; i < dim; i++) {
  out[i] = x[xOffset + i] * invRms * g[i];
}
```
- **Przykład liczbowy ($D=4$, $x=[2, -2, 2, -2]$, $g=[1, 1, 1, 1]$, $\varepsilon=10^{-5}$):**
  Średnia kwadratów = $(4+4+4+4)/4 = 4.0$. $\text{inv\_rms} = 1/\sqrt{4.00001} \approx 0.499999$. Wyjście: $[1.0, -1.0, 1.0, -1.0]$.

---

## 4. Pozycyjne osadzenia obrotowe RoPE (konwencja half-split)

### Poziom A
Każdy token w głowicy ma wektor długości `32` liczb (`headDim = 32`). Traktujemy te `32` liczby jako **16 dwuwymiarowych punktów na płaszczyźnie** (parując liczbę `0` z `16`, `1` z `17`, ..., `15` z `31`). W zależności od numeru pozycji tokenu w tekście (`pos`), każdy z tych 16 punktów obracamy o określony kąt: pierwsze pary kręcą się bardzo szybko (śledząc sąsiednie słowa), a ostatnie bardzo wolno (śledząc daleki kontekst zdania).

### Poziom B
**POTWIERDZONE** — Istnieją dwie popularne konwencje parowania współrzędnych w RoPE: *interleaved* (pary sąsiednie `(0,1), (2,3)`) oraz **half-split** (pierwsza połowa z drugą połową `(i, i + headDim/2)`). Tiny używa wyłącznie konwencji **half-split** (bit `1` flagi TNY3 musi wynosić `0`; gdyby był `1`, `parseTNY3()` w `tartarus-engine.ts:227–229` natychmiast przerywa ładowanie błędem). Tablice $\cos$ i $\sin$ są obliczane z góry w konstruktorze `TartarusEngine` (`tartarus-engine.ts:515–527`) oraz w `precompute_rope_tables()` (`model_tartarus.py:142–159`).

### Poziom C

Dla wymiaru głowicy $d_h$ (`32`), połowy wymiaru $M = d_h / 2$ (`16`), indeksu pary $i \in \{0, \dots, M-1\}$, pozycji $p \in \{0, \dots, \text{CTX}-1\}$ oraz $\Theta = 10000.0$:

$$\text{inv\_freq}_i = \frac{1}{\Theta^{\,2i / d_h}} = 10000^{-2i / d_h}$$
$$\theta_{p, i} = p \cdot \text{inv\_freq}_i$$

Dla wektora głowicy $u \in \mathbb{R}^{d_h}$, gdzie $x_1 = u_i$ oraz $x_2 = u_{M + i}$:

$$\begin{bmatrix} u'_i \\ u'_{M + i} \end{bmatrix} = \begin{bmatrix} \cos(\theta_{p, i}) & -\sin(\theta_{p, i}) \\ \sin(\theta_{p, i}) & \cos(\theta_{p, i}) \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} x_1 \cos(\theta_{p, i}) - x_2 \sin(\theta_{p, i}) \\ x_2 \cos(\theta_{p, i}) + x_1 \sin(\theta_{p, i}) \end{bmatrix}$$

Zwróć uwagę na identyczność w kodzie:
- Python (`model_tartarus.py:175–176`): `out1 = x1 * c - x2 * s`, `out2 = x2 * c + x1 * s`.
- TypeScript (`tartarus-engine.ts:442–443`): `vec[headBase + i] = x1 * c - x2 * s`, `vec[headBase + half + i] = x2 * c + x1 * s`.

**Dlaczego RoPE działa przy porównywaniu $Q$ i $K$?** Z własności trygonometrycznych iloczyn skalarny obróconego wektora $Q$ na pozycji $p$ oraz obróconego wektora $K$ na pozycji $t$ zależy wyłącznie od **względnej odległości $(p - t)$**, a nie od bezwzględnych miejsc w tekście.

---

## 5. Multi-Head Causal Self-Attention (Q, K, V)

### Poziom A
Dla bieżącego tokenu na pozycji `pos` każda z `H` głowic zadaje pytanie (`Q`) i porównuje je z kluczami (`K`) wszystkich tokenów od początku zdania aż do siebie samego (`t = 0 .. pos`). Tokeny w przyszłości (`t > pos`) są całkowicie ukryte (stąd nazwa *causal* – przyczynowa). Im bardziej klucz `K` wcześniejszego słowa pasuje do zapytania `Q`, tym większą wagę dostaje jego wartość `V` przy budowaniu podsumowania.

### Poziom B
**POTWIERDZONE** — Najpierw obliczane są projekcje liniowe bez biasu (`matVecInto`, `tartarus-engine.ts:402–423`):
- $q = W_q \, h_{\text{norm}}$, $k = W_k \, h_{\text{norm}}$, $v = W_v \, h_{\text{norm}}$ (każdy o wymiarze $D$).
Po zastosowaniu RoPE na $q$ i $k$ wartości $k$ oraz $v$ są dopisywane do `kCache` i `vCache` pod pozycją `pos`, a następnie wykonywana jest pętla po głowicach $h = 0 \dots H - 1$ (`tartarus-engine.ts:601–654`).

### Poziom C

Dla głowicy $h$, pozycji zapytania $p = \text{pos}$ i kolejnych pozycji kluczy $t \in \{0, \dots, p\}$:

1. **Skalowany iloczyn skalarny (Raw Attention Score):**
$$s_t = \frac{1}{\sqrt{d_h}} \sum_{d=0}^{d_h - 1} q_{h, d} \cdot K^{(l)}_{t, h, d}$$
gdzie $\frac{1}{\sqrt{d_h}} = \frac{1}{\sqrt{32}} \approx 0.1767767$ zapobiega przepełnieniu eksponenty przy dużych wymiarach głowicy.

2. **Stabilizowany numerycznie Softmax:**
$$m = \max_{0 \le t \le p} s_t, \qquad \alpha_t = \frac{\exp(s_t - m)}{\sum_{j=0}^{p} \exp(s_j - m)}$$
Odjęcie maksimum $m$ matematycznie nie zmienia wyniku ułamka, ale gwarantuje, że argument funkcji $\exp(\cdot)$ wynosi co najwyżej $0$, więc nigdy nie wystąpi przepełnienie `Infinity` w `Float32`.

3. **Ważona suma wartości i projekcja wyjściowa:**
$$o_{h, d} = \sum_{t=0}^{p} \alpha_t \cdot V^{(l)}_{t, h, d}, \qquad \text{attn\_delta} = W_o \, [o_0 \,\|\, o_1 \,\|\, \dots \,\|\, o_{H-1}]$$
4. **Połączenie rezydualne:** $x \leftarrow x + \text{attn\_delta}$.

> **Szczegół implementacyjny (`matVecInto`):** W `tartarus-engine.ts:414–420` iloczyn skalarny w mnożeniu macierz × wektor jest ręcznie rozwinięty po `4` elementy (`c += 4`), co jest legalne wyłącznie dzięki temu, że zarówno `dModel % 4 === 0`, jak i `dFf % 4 === 0` są wymuszane w konstruktorze modelu i w `parseTNY3()`.

---

## 6. Warstwa MLP z bramką SwiGLU

### Poziom A
Po zebraniu kontekstu przez Attention wektor tokenu (po drugim `RMSNorm`) trafia do bloku MLP. W starszych sieciach była to jedna zwykła warstwa rozszerzająca i druga ściskająca. **SwiGLU** rozdziela rozszerzenie na dwie ścieżki: jedna (`w_gate`) działa jak płynny zawór otwierający się mocniej dla dużych dodatnich sygnałów (funkcja `SiLU`), a druga (`w_up`) niesie treść; mnożąc je przez siebie element po elemencie, sieć wybiera, które cechy przekazać do wyjścia (`w_down`).

### Poziom B/C
**POTWIERDZONE** — Implementacja w `model_tartarus.py:278–279` (`self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))`) oraz `tartarus-engine.ts:660–675`:

$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$
$$\text{SwiGLU}(h) = W_{\text{down}} \Big( \text{SiLU}(W_{\text{gate}} \, h) \odot (W_{\text{up}} \, h) \Big)$$
$$x_{\text{out}} = x + \text{SwiGLU}(\text{RMSNorm}_{\text{ffn}}(x))$$

**Zabezpieczenie numeryczne funkcji `silu(z)` (`tartarus-engine.ts:449–455`):**
- Gdy $z \ge 0$: liczy `z / (1.0 + Math.exp(-z))` (ponieważ $-z \le 0$, więc $e^{-z} \in (0, 1]$ — brak ryzyka `Infinity`).
- Gdy $z < 0$: liczy `(z * Math.exp(z)) / (1.0 + Math.exp(z))` (ponieważ $z < 0$, więc $e^{z} \in (0, 1)$ — znowu brak ryzyka przepełnienia przy bardzo dużych ujemnych wartościach).

Symbol $\odot$ oznacza mnożenie po współrzędnych (*element-wise*): `this.ffGate[i] = silu(this.ffGate[i]) * this.ffUp[i]` dla $i = 0 \dots D_{\text{ff}} - 1$.

---

## 7. Końcowe logity i Tied Embeddings

### Poziom A/B/C
**POTWIERDZONE** — Po przejściu wszystkich $L$ warstw wektor $x$ przechodzi ostatnią normalizację `rms_final`, a następnie mnoży się przez macierz słownika `lmHead` (`model_tartarus.py:373–375`, `tartarus-engine.ts:679–682`):

$$h_{\text{final}} = \text{RMSNorm}_{\text{final}}(x), \qquad \text{logits}_v = \sum_{d=0}^{D-1} W^{\text{emb}}_{v, d} \cdot h_{\text{final}, d} \quad \text{dla } v = 0 \dots V-1$$

Ponieważ w obu profilach `tiedEmbeddings = true`, macierz $W^{\text{emb}}$ ma dokładnie wymiar `[2048, 192]` (Profil `S`) lub `[2048, 256]` (Profil `M`) i nie jest powielana w pliku `.tny3` ani w pamięci RAM.

---

## 8. Matematyka samplingu (wybór kolejnego tokenu)

Wszystkie operacje wykonuje `TartarusEngine.sampleNextToken()` (`tartarus-engine.ts:823–941`).

### 8.1. Repetition Penalty (kara jednokrotna)
Jeśli $\text{repPenalty} > 1.0$, dla każdego **unikalnego** identyfikatora tokenu $t$, który wystąpił wcześniej w `seenTokens` (z pomocą bufora `repSeen: Uint8Array` zapobiegającego wielokrotnemu karaniu duplikatów):
$$z'_t = \begin{cases} z_t / \text{repPenalty}, & \text{gdy } z_t > 0 \\ z_t \cdot \text{repPenalty}, & \text{gdy } z_t \le 0 \end{cases}$$

**Przykład liczbowy (`verify_ts.mjs:238–244`):** Niech token `100` ma logit $+5.0$, token `101` ma logit $+4.6$, a `repPenalty = 1.2`. Po jednokrotnym ukaraniu tokenu `100`: $z'_{100} = 5.0 / 1.2 = 4.1667 < 4.6$, więc wygrywa token `101`. Gdyby powtórzono karę 3 razy dla `seenTokens = [100, 100, 100]`, logit spadłby aż do $5.0 / 1.2^3 = 2.89$, co zniekształciłoby rozkład — dlatego używana jest tablica obecności `repSeen`.

### 8.2. Greedy decoding ($\text{temperature} \le 0$)
Silnik omija softmax i zwraca indeks o największym (po ewentualnej karze za powtórzenia) logicie:
$$\text{nextId} = \arg\max_{v \in [0, V)} z'_v$$

### 8.3. Temperatura, Top-K, Min-P oraz losowanie z dystrybuanty
Gdy $T = \text{temperature} > 0$:
1. **Skalowanie temperaturą:** $u_v = z'_v / T$.
   - $T < 1$ (np. $0.5$) wyostrza różnice między logitami (większa pewność modelu, mniejsza losowość).
   - $T > 1$ (np. $1.4$) spłaszcza różnice (zwiększa różnorodność i ryzyko błędów).
   - **Przykład:** dla dwóch logitów $[6.0,\; 4.0]$ różnica przed softmaxem wynosi $2.0$. Przy $T = 0.5$ wzrasta do $4.0$ (prawdopodobieństwa `98.2%` vs `1.8%`). Przy $T = 2.0$ spada do $1.0$ (`73.1%` vs `26.9%`).
2. **Top-K (`topK > 0`):** Utrzymuje bez alokacji tablicę `topK` największych wartości $u_v$, odrzucając pozostałe $V - K$ tokenów (np. przy `topK = 32` z `2048` tokenów zostaje tylko `32` najlepszych kandydatów).
3. **Softmax po wybranych kandydatach:**
   $$e_i = \exp\!\left(u_i - \max_j u_j\right), \qquad p_{\max} = \max_i e_i$$
4. **Min-P (`0 < minP < 1`):** Wyzerowuje każdego kandydata, którego nieunormowana waga $e_i$ jest mniejsza niż ułamek `minP` najlepszego kandydata:
   $$\text{jeśli } e_i < \text{minP} \cdot p_{\max} \implies e_i \leftarrow 0$$
   **Przykład:** Jeśli najlepszy token ma `prob = 0.60`, a `minP = 0.06`, próg odcięcia wynosi `0.06 × 0.60 = 0.036`. Wszystkie słowa z prawdopodobieństwem poniżej `3.6%` odpadają, nawet jeśli zmieściły się w `topK`. Gdy dominuje jeden pewny token (`0.95`), `minP = 0.06` odcina prawie całą resztę poniżej `5.7%`, a gdy model waha się między wieloma równoważnymi słowami (`max = 0.15`), próg obniża się dynamicznie do `0.9%`.
5. **Losowanie z dystrybuanty (CDF):** Generuje liczbę losową $r \in [0, \sum e_i)$ i zwraca pierwszy token, dla którego skumulowana suma przekroczy $r$.
