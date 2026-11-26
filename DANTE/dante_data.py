#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determinystyczny generator danych treningowych „Piekło Dantego”.

Jedynym źródłem faktów jest plik dante_kanon_high.md (załączony do zadania).
Skrypt nie sięga po żadne inne źródła. Każda odpowiedź jest zbudowana z
zamkniętej bazy faktów zakodowanej poniżej.

Użycie:
    python dante_data.py --seed 42 --out-dir ./dane

Wyjście:
    dante_train.jsonl (20 000), dante_val.jsonl (1 000), dante_test.jsonl (1 000)

Na końcu uruchamiana jest wbudowana walidacja (unikalność tekstów, rozkład
kategorii, długości, rozdzielność pul pytań i par porównawczych).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# 0. Parametry globalne
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test")
SPLIT_SIZES = {"train": 20_000, "val": 1_000, "test": 1_000}

CATEGORIES = (
    "fact_direct",
    "fact_reverse",
    "relation_logic",
    "disambiguation",
    "refusal",
    "narrative_meta",
)
CATEGORY_SHARE = {
    "fact_direct": 0.25,
    "fact_reverse": 0.15,
    "relation_logic": 0.20,
    "disambiguation": 0.15,
    "refusal": 0.15,
    "narrative_meta": 0.10,
}
TARGETS = {
    split: {
        cat: round(SPLIT_SIZES[split] * CATEGORY_SHARE[cat]) for cat in CATEGORIES
    }
    for split in SPLITS
}

STATE_PREFIX = "<|state|> dzieło=Piekło Dantego; tryb=pamięć <|user|>"
STATE_SUFFIX = "<|eos|>"

MAX_CHARS = 900          # twardy limit z polecenia
SAFE_CHARS = 880          # bezpieczny limit używany przy generowaniu
MAX_WORDS = 150
SAFE_WORDS = 145


def derive_rng(seed: int, *parts: str) -> random.Random:
    """RNG niezależny od kolejności wywołań kategorii."""
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def ok_length(text: str) -> bool:
    return len(text) <= SAFE_CHARS and len(text.split()) <= SAFE_WORDS


# ---------------------------------------------------------------------------
# 1. Baza faktów: dziewięć kręgów (Sekcja 2, Dodatek I, Dodatek Q)
# ---------------------------------------------------------------------------

CIRCLES: list[dict[str, str]] = [
    {
        "roman": "I",
        "nom": "Krąg I (Limbo)",
        "loc": "Kręgu I (Limbo)",
        "name": "Otchłań (Limbo)",
        "sin": "brak chrztu",
        "sin_a": "Krąg I, czyli Limbo, mieści cnotliwych pogan zmarłych przed Chrystusem i nieochrzczone dzieci; to brak chrztu, nie żaden grzech osobisty.",
        "guard_a": "Limbo nie ma własnego strażnika: Minos urzęduje dopiero na granicy Kręgu II i nie osądza dusz trafiających do Limbo.",
        "place_a": "Krąg I to zielona łąka otoczona siedmiokrotnym murem i fosą, oświetlona własnym, przyćmionym światłem; w środku stoi zamek.",
        "punish_a": "W Limbie nie ma fizycznej kary, ognia ani chłodu; jest tylko wieczna tęsknota bez nadziei na zobaczenie Boga.",
        "contr_a": "Contrapasso w Kręgu I to umiarkowany smutek bez cierpienia ciała: dusze wiedzą, że nigdy nie zobaczą Boga.",
        "chars_a": "W Limbo przebywa Wergiliusz jako nieochrzczony rzymski poeta; w zamku są też inni wielcy poeci i filozofowie starożytności.",
        "event_a": "Dante budzi się w Limbie po przeprawie przez Acheront, słyszy westchnienia, wchodzi z czterema poetami do zamku i schodzi do Kręgu II.",
        "zone": "poza murami Dis",
    },
    {
        "roman": "II",
        "nom": "Krąg II (Chuć)",
        "loc": "Kręgu II (Chuć)",
        "name": "Chuć (Nieczystość)",
        "sin": "chuć",
        "sin_a": "Krąg II karze chuć, czyli uleganie żądzy cielesnej ponad rozum i rozsądek.",
        "guard_a": "Na granicy Kręgu II urzęduje Minos: dusza wyznaje mu grzechy, on owija ogon wokół siebie tyle razy, ile wynosi numer kręgu, a demony strącają ją w dół.",
        "place_a": "Krąg II to ciemna, otwarta przestrzeń bez punktu odniesienia, smagana nieustającym huraganowym wiatrem.",
        "punish_a": "Dusze w Kręgu II są wiecznie miotane przez wicher: nigdy nie odpoczywają i nigdy nie dotykają ziemi.",
        "contr_a": "Wicher w Kręgu II odwzorowuje to, jak za życia dali się miotać niekontrolowanym namiętnościom.",
        "chars_a": "Francesca da Rimini i Paolo Malatesta wirują razem w wietrze, nierozdzieleni nawet w karze, i opowiadają Dantemu swoją historię.",
        "event_a": "Dante słyszy ryk Minosa, widzi lecące dusze, prosi Franceskę i Paola o zatrzymanie, słucha opowieści i mdleje z litości.",
        "zone": "poza murami Dis",
    },
    {
        "roman": "III",
        "nom": "Krąg III (Obżarstwo)",
        "loc": "Kręgu III (Obżarstwo)",
        "name": "Obżarstwo",
        "sin": "obżarstwo",
        "sin_a": "Krąg III karze obżarstwo, czyli nadmierne i bezrefleksyjne oddawanie się jedzeniu i piciu.",
        "guard_a": "Kręgu III pilnuje Cerber, trójgłowy pies o cechach na wpół demonicznych: szczeka trzema paszczami i rozrywa pazurami napotkane dusze.",
        "place_a": "Krąg III to teren pod cuchnącym deszczem z brudnej wody, śniegu i gradu, który spływa na grząskie, cuchnące błoto.",
        "punish_a": "Grzesznicy leżą twarzą w dół w cuchnącym błocie, są biczowani zimnym deszczem i od czasu do czasu szarpani przez Cerbera.",
        "contr_a": "Wyrafinowane zaspokajanie smaku i dotyku zamienia się w Kręgu III w wieczny brud, chłód i smród.",
        "chars_a": "Ciacco, florentczyk, leży w błocie, rozpoznaje Dantego mimo błota na twarzy i przepowiada mu konflikty polityczne Florencji.",
        "event_a": "Wergiliusz wrzuca Cerberowi garście błota w trzy paszcze, Ciacco rozmawia z Dantem o Florencji, a potem opada z powrotem w błoto.",
        "zone": "poza murami Dis",
    },
    {
        "roman": "IV",
        "nom": "Krąg IV (Chciwość i Rozrzutność)",
        "loc": "Kręgu IV (Chciwość i Rozrzutność)",
        "name": "Chciwość i Rozrzutność",
        "sin": "chciwość i rozrzutność",
        "sin_a": "Krąg IV karze razem chciwość i rozrzutność: oba grzechy wynikają z braku umiaru wobec dóbr materialnych.",
        "guard_a": "Kręgu IV pilnuje Plutos, demon-bóg bogactwa; krzyczy niezrozumiałe słowa, lecz Wergiliusz natychmiast ucisza go jednym wezwaniem.",
        "place_a": "Krąg IV to okrągła przestrzeń, w której dwie grupy dusz pchają wielkie głazy po dwóch przeciwległych, półkolistych łukach.",
        "punish_a": "Skąpcy i rozrzutnicy wiecznie zderzają się głazami w punkcie styku łuków, obrzucają się wyzwiskami i powtarzają cykl bez końca.",
        "contr_a": "Brak umiaru wobec majątku staje się w Kręgu IV nieskończenie powtarzanym, bezcelowym ruchem.",
        "chars_a": "Krąg IV nie ma nazwanych postaci: obsesja na punkcie majątku zatarła twarze dusz, w tym duchownych wysokiej rangi.",
        "event_a": "Wergiliusz ucisza Plutosa, Dante widzi zderzające się fale dusz, słyszy wyjaśnienie o Fortunie i przechodzi w stronę Styksu.",
        "zone": "poza murami Dis",
    },
    {
        "roman": "V",
        "nom": "Krąg V (Gniew i Lenistwo)",
        "loc": "Kręgu V (Gniew i Lenistwo)",
        "name": "Gniew i Lenistwo (Smutek)",
        "sin": "gniew i acedia",
        "sin_a": "Krąg V karze gniew, czyli niekontrolowaną agresję, oraz acedię, ponure, bierne lenistwo duchowe.",
        "guard_a": "Przez bagno Styks przewozi Dantego i Wergiliusza Flegiasz, syn Marsa, który w gniewie spalił świątynię Apolla w Delfach; robi to niechętnie.",
        "place_a": "Krąg V to Styks, cuchnące, szarobrudne bagnisko tworzące rozległą fosę przed murami miasta Dis.",
        "punish_a": "Gniewni biją się pięściami, kąsają i szarpią na powierzchni Styksu; acediosi są zanurzeni pod powierzchnią i wydają tylko bulgoczące dźwięki.",
        "contr_a": "Bierność duchowa staje się w Kręgu V całkowitym unieruchomieniem pod wodą, a gniew – walką na błocie.",
        "chars_a": "Filippo Argenti, florentczyk i wróg rodziny Dantego, chwyta łódź; inne gniewne dusze rozszarpują go na oczach wędrowców.",
        "event_a": "Flegiasz niechętnie zabiera obu wędrowców, podczas przeprawy wybucha spór z Filippem Argentim, a łódź dociera pod mury Dis.",
        "zone": "poza murami Dis",
    },
    {
        "roman": "VI",
        "nom": "Krąg VI (Herezja)",
        "loc": "Kręgu VI (Herezja)",
        "name": "Herezja",
        "sin": "herezja",
        "sin_a": "Krąg VI karze herezję, świadome głoszenie poglądów sprzecznych z nauką Kościoła, zwłaszcza przeczenie nieśmiertelności duszy.",
        "guard_a": "Bram miasta Dis strzegą upadłe anioły; na murach pojawiają się Furie – Megera, Alekto i Tyzyfona – z groźbą pokazania głowy Meduzy.",
        "place_a": "Krąg VI leży wewnątrz miasta Dis i jest polem pełnym otwartych, płonących grobowców ustawionych w rzędach.",
        "punish_a": "Heretycy leżą w rozgrzanych do czerwoności, otwartych sarkofagach, które po Sądzie Ostatecznym zostaną zamknięte na zawsze.",
        "contr_a": "Wieczny ogień w Kręgu VI odpowiada ogniowi stosów, którym w średniowieczu karano herezję za życia.",
        "chars_a": "Farinata degli Uberti, przywódca gibelinów, podnosi się z grobowca do pasa, rozmawia z Dantem o polityce i przepowiada mu wygnanie.",
        "event_a": "Anioł zesłany z nieba otwiera bramę dotknięciem laski; Dante rozmawia z Farinatą, a potem zatrzymuje się przy urwisku do Kręgu VII.",
        "zone": "wewnątrz murów Dis",
    },
    {
        "roman": "VII",
        "nom": "Krąg VII (Przemoc)",
        "loc": "Kręgu VII (Przemoc)",
        "name": "Przemoc",
        "sin": "przemoc",
        "sin_a": "Krąg VII karze przemoc podzieloną według ofiary: bliźniego i jego mienia, samego siebie oraz Boga, naturę i sztukę.",
        "guard_a": "Wejścia do Kręgu VII strzeże Minotaur; w pierścieniu 1 patrolują Centaury pod wodzą Chirona, a nad lasem samobójców krążą Harpie.",
        "place_a": "Krąg VII ma trzy pierścienie: Flegeton, Las Samobójców oraz rozżarzony, suchy piasek pod deszczem płatków ognia.",
        "punish_a": "W pierścieniu 1 jest wrząca krew, w pierścieniu 2 zamiana samobójców w drzewa, a w pierścieniu 3 ogień padający na trzy grupy grzeszników.",
        "contr_a": "Kara w Kręgu VII dobiera się do tego, przeciw komu skierowana była przemoc; ogień występuje tylko w pierścieniu 3.",
        "chars_a": "Pietro della Vigna jest drzewem w pierścieniu 2 i broni niewinności; Brunetto Latini biegnie wiecznie po piasku pierścienia 3.",
        "event_a": "Wergiliusz wykorzystuje furię Minotaura, Nessos przenosi wędrowców przez Flegeton, Dante łamie gałąź Pietra i rozmawia z Brunettem.",
        "zone": "wewnątrz murów Dis",
    },
    {
        "roman": "VIII",
        "nom": "Krąg VIII (Malebolge)",
        "loc": "Kręgu VIII (Malebolge)",
        "name": "Oszustwo (Malebolge)",
        "sin": "oszustwo",
        "sin_a": "Krąg VIII karze oszustwo wobec osób niezwiązanych ze sprawcą szczególną więzią zaufania.",
        "guard_a": "Gerion, uosobienie Oszustwa, przenosi wędrowców lotem do Kręgu VIII; w bolgii 5 patrolują Malebranche pod wodzą Malacody.",
        "place_a": "Krąg VIII to Malebolge: dziesięć koncentrycznych kamiennych rowów wokół studni, połączonych mostami biegnącymi jak szprychy.",
        "punish_a": "Każda z dziesięciu bolgii ma odrębną karę dopasowaną do konkretnego podtypu oszustwa.",
        "contr_a": "Różnorodność kar w Malebolge pokazuje contrapasso: kara odpowiada dokładnie temu, jak oszukano za życia.",
        "chars_a": "Mikołaj III tkwi głową w dół w bolgii 3, Malebranche z Malacodą działają w bolgii 5, a Ulisses przebywa w bolgii 8.",
        "event_a": "Gerion ląduje na dnie i odlatuje; wędrowcy przechodzą mostami nad bolgiami, uciekają przed Malebranche i docierają do studni Gigantów.",
        "zone": "wewnątrz murów Dis",
    },
    {
        "roman": "IX",
        "nom": "Krąg IX (Kocyt)",
        "loc": "Kręgu IX (Kocyt)",
        "name": "Zdrada (Kocyt)",
        "sin": "zdrada",
        "sin_a": "Krąg IX karze zdradę osób związanych ze sprawcą więzią krwi, ojczyzny, gościnności lub dobrodziejstwa.",
        "guard_a": "Studnię między Kręgiem VIII a IX strzegą Giganci, w tym Nimrod i Anteusz; w centrum Kocytu tkwi Lucyfer.",
        "place_a": "Krąg IX to Kocyt, całkowicie zamarznięte jezioro podzielone na Kainę, Antenorę, Ptolomeę i Judekę.",
        "punish_a": "Dusze są wmrożone w lód na różną głębokość: od szyi w Kainie, przez twarz w Antenorze i Ptolomei, po całkowite zanurzenie w Judece.",
        "contr_a": "Zdrada więzi zaufania zamienia się w coraz głębszy mróz; lodowaty wiatr pochodzi z trzepotu skrzydeł Lucyfera.",
        "chars_a": "Ugolino gryzie czaszkę Ruggieriego w Antenorze; w Judece Lucyfer żuje Judasza, Brutusa i Kasjusza w trzech paszczach.",
        "event_a": "Anteusz znosi wędrowców na dno; Dante słucha Ugolina, mija Ptolomeę i Judekę, a Wergiliusz schodzi po ciele Lucyfera do środka Ziemi.",
        "zone": "wewnątrz murów Dis",
    },
]

# ---------------------------------------------------------------------------
# 2. Trzy pierścienie Kręgu VII, dziesięć bolgii, cztery strefy Kocytu
# ---------------------------------------------------------------------------

RINGS = [
    {
        "n": "1",
        "nom": "pierścień 1 Kręgu VII",
        "loc": "pierścieniu 1 Kręgu VII",
        "focus": "przemoc wobec bliźniego i jego mienia",
        "answer": "Pierścień 1 Kręgu VII karze przemoc wobec bliźniego i jego mienia; tyrani i mordercy są zanurzeni we wrzącej krwi Flegetonu.",
        "extra": "Centaury pod wodzą Chirona strzelają do dusz, które próbują wynurzyć się wyżej, niż im wyznaczono; Nessos przenosi wędrowców przez bród.",
    },
    {
        "n": "2",
        "nom": "pierścień 2 Kręgu VII",
        "loc": "pierścieniu 2 Kręgu VII",
        "focus": "przemoc wobec samego siebie i własnego mienia",
        "answer": "Pierścień 2 Kręgu VII karze przemoc wobec samego siebie i własnego mienia: samobójcy zostali trwale zamienieni w sękate drzewa Lasu Samobójców.",
        "extra": "Nad drzewami krążą Harpie i zjadają liście; złamanie gałęzi pozwala duszy przemówić, ale wiąże się z krwawieniem.",
    },
    {
        "n": "3",
        "nom": "pierścień 3 Kręgu VII",
        "loc": "pierścieniu 3 Kręgu VII",
        "focus": "przemoc wobec Boga, natury i ludzkiej pracy",
        "answer": "Pierścień 3 Kręgu VII karze przemoc wobec Boga, natury i sztuki: bluźniercy leżą na wznak, sodomici biegną w kółko, lichwiarze siedzą z workami pieniędzy na szyi.",
        "extra": "Na rozżarzony, suchy piasek bez przerwy padają wolno opadające płatki ognia.",
    },
]

BOLGIAE = [
    {"n": "1", "nom": "bolgia 1", "loc": "bolgii 1", "sin": "kuplerstwo i uwodzenie",
     "answer": "Bolgia 1 karze kuplerów i uwodzicieli: idą w dwóch przeciwbieżnych szeregach i są biczowani przez rogate demony."},
    {"n": "2", "nom": "bolgia 2", "loc": "bolgii 2", "sin": "pochlebstwo",
     "answer": "Bolgia 2 karze pochlebców: są zanurzeni po szyję w ludzkich nieczystościach."},
    {"n": "3", "nom": "bolgia 3", "loc": "bolgii 3", "sin": "symonia",
     "answer": "Bolgia 3 karze symonię: duchowni tkwią głową w dół w okrągłych kamiennych otworach, a ich wystające stopy są objęte płomieniem.",
     "extra": "W tej bolgii Dante rozmawia z papieżem Mikołajem III, który myli go z oczekiwanym Bonifacym VIII, w roku 1300 wciąż żyjącym."},
    {"n": "4", "nom": "bolgia 4", "loc": "bolgii 4", "sin": "wróżbiarstwo i czary",
     "answer": "Bolgia 4 karze wróżbitów i czarowników: mają głowy odwrócone o sto osiemdziesiąt stopni i muszą chodzić tyłem, płacząc łzami po plecach."},
    {"n": "5", "nom": "bolgia 5", "loc": "bolgii 5", "sin": "barateria, czyli sprzedajność urzędników",
     "answer": "Bolgia 5 karze baraterów, sprzedajnych urzędników: są zanurzeni we wrzącej smole, a próby wynurzenia karane są pazurami Malebranche.",
     "extra": "Malebranche działają pod wodzą Malacody; w pościgu za uciekającym grzesznikiem same wpadają do wrzątku."},
    {"n": "6", "nom": "bolgia 6", "loc": "bolgii 6", "sin": "hipokryzja",
     "answer": "Bolgia 6 karze hipokrytów: chodzą powoli w ciężkich płaszczach z ołowiu, pozłacanych jedynie z zewnątrz."},
    {"n": "7", "nom": "bolgia 7", "loc": "bolgii 7", "sin": "kradzież",
     "answer": "Bolgia 7 karze złodziei: jadowite węże kąsają ich tak, że ciała płoną, obracają się w popiół i odradzają albo złodziej i gad zamieniają się kształtami."},
    {"n": "8", "nom": "bolgia 8", "loc": "bolgii 8", "sin": "fałszywe doradztwo",
     "answer": "Bolgia 8 karze fałszywych doradców: są całkowicie owinięci w pojedyncze płomienie, które ukrywają ich ciała przed wzrokiem.",
     "extra": "Ulisses, owinięty rozdwojonym u góry płomieniem, opowiada o ostatniej wyprawie poza Słupy Heraklesa, zakończonej zatonięciem statku."},
    {"n": "9", "nom": "bolgia 9", "loc": "bolgii 9", "sin": "siewstwo niezgody i rozłamu",
     "answer": "Bolgia 9 karze siewców niezgody: demon przy wejściu wiecznie okalecza ich mieczem, a rany goją się dopiero przy powrocie do demona."},
    {"n": "10", "nom": "bolgia 10", "loc": "bolgii 10", "sin": "fałszerstwo",
     "answer": "Bolgia 10 karze fałszerzy różnego rodzaju chorobami ciała: trądem, świerzbem, szaleńczą gorączką i puchliną wodną powodującą pragnienie."},
]

ZONES = [
    {"nom": "strefa Kaina", "loc": "strefie Kaina", "betrayal": "zdrada krewnych", "betrayal_acc": "zdradę krewnych",
     "answer": "Kaina, pierwsza strefa Kocytu, karze zdradę krewnych; dusze są zanurzone w lodzie po szyję.",
     "extra": "W dodatku O nazwa strefy jest wyprowadzona od imienia Kaina, zabójcy brata Abla."},
    {"nom": "strefa Antenora", "loc": "strefie Antenora", "betrayal": "zdrada ojczyzny i własnego stronnictwa politycznego", "betrayal_acc": "zdradę ojczyzny i własnego stronnictwa politycznego",
     "answer": "Antenora karze zdradę ojczyzny i własnego stronnictwa; sylwetka jest pochylona, a twarz wystaje ponad lód.",
     "extra": "W Antenorze przebywają hrabia Ugolino i arcybiskup Ruggieri; nazwa strefy pochodzi od Antenora, trojańskiego zdrajcy."},
    {"nom": "strefa Ptolomea", "loc": "strefie Ptolomea", "betrayal": "zdrada gości", "betrayal_acc": "zdradę gości",
     "answer": "Ptolomea karze zdradę gości: twarz wystaje ponad lód, a głowa jest odchylona do tyłu.",
     "extra": "W tej strefie dusze niektórych zdrajców trafiają do lodu, zanim ich ciała umrą na ziemi; ciała są wtedy opętane przez demony."},
    {"nom": "strefa Judeka", "loc": "strefie Judeka", "betrayal": "zdrada dobroczyńców", "betrayal_acc": "zdradę dobroczyńców",
     "answer": "Judeka, najgłębsza strefa Kocytu, karze zdradę dobroczyńców; dusze są w niej całkowicie zanurzone w lodzie.",
     "extra": "W centrum Judeki tkwi Lucyfer; nazwa strefy pochodzi od imienia Judasza."},
]

# ---------------------------------------------------------------------------
# 3. Katalog 25 postaci (Dodatek C) i fakty historyczne (F, L, J)
# ---------------------------------------------------------------------------

CHARACTERS = [
    {
        "nom": "Dante Alighieri", "loc": "Dantego Alighieri",
        "role": "Dante Alighieri jest żywym narratorem pierwszoosobowym i głównym bohaterem, florenckim poetą i politykiem na wygnaniu.",
        "place": "Dante przechodzi od ciemnego lasu przez dziewięć kręgów aż na półkulę południową.",
        "key": "W Wielki Piątek 1300 ma trzydzieści pięć lat; podróżuje w fizycznym ciele, odczuwa zimno, strach, zmęczenie i głód oraz rzuca cień.",
        "hist": "Urodził się w 1265 we Florencji, został wygnany w 1302 przez Gwelfów Czarnych i zmarł w 1321 w Rawennie.",
    },
    {
        "nom": "Wergiliusz", "loc": "Wergiliuszu",
        "role": "Wergiliusz, rzymski poeta i autor Eneidy, jest przewodnikiem Dantego przez Piekło i Czyściec.",
        "place": "Wergiliusz mieszka w Limbie, Kręgu I, bo żył przed Chrystusem i nie został ochrzczony.",
        "key": "Został wezwany z Limbo przez Beatrycze; symbolizuje ludzki rozum, więc nie może wprowadzić Dantego do Raju Niebieskiego.",
        "hist": "Publiusz Wergiliusz Maro żył w latach 70–19 p.n.e.; w średniowieczu uważano go błędnie za proroka nowej ery.",
    },
    {
        "nom": "Beatrycze", "loc": "Beatrycze",
        "role": "Beatrycze jest zmarłą florentynką, która z Raju Niebieskiego zleca Wergiliuszowi misję przewodnika.",
        "place": "Przez całą akcję Piekła Beatrycze przebywa w Raju Niebieskim.",
        "key": "Nigdy nie schodzi do Piekła; jej wizyta u Wergiliusza w Limbo jest tylko opowiedziana retrospektywnie w pieśni drugiej.",
        "hist": "Wzór postaci to Beatrice Portinari, którą Dante poznał jako dziewięcioletni chłopiec i opisał w Vita Nuova; zmarła w 1290 roku.",
    },
    {
        "nom": "Charon", "loc": "Charonie",
        "role": "Charon jest przewoźnikiem dusz przez rzekę Acheront na próg Kręgu I.",
        "place": "Charon działa na granicy Piekła, przed Kręgiem I, a nie w Kręgu V.",
        "key": "To starzec o długiej siwej brodzie i oczach płonących jak koła ognia; bije wiosłem dusze zwlekające z wejściem na pokład.",
        "hist": "Początkowo odmawia przewozu żywego Dantego, lecz ustępuje, gdy Wergiliusz mówi o woli sił wyższych.",
    },
    {
        "nom": "Minos", "loc": "Minosie",
        "role": "Minos jest mitycznym sędzią podziemi o postaci na wpół ludzkiej, na wpół bestialskiej.",
        "place": "Minos urzęduje na granicy Kręgu II i nie osądza dusz trafiających do Limbo.",
        "key": "Wyznacza duszy krąg, owijając ogon wokół siebie tyle razy, ile wynosi numer tego kręgu; potem demony strącają duszę w dół.",
        "hist": "Nie jest strażnikiem Kręgu VII: tę funkcję pełni Minotaur.",
    },
    {
        "nom": "Francesca da Rimini i Paolo Malatesta", "loc": "Francesce da Rimini i Paolu Malasteście",
        "role": "Francesca i Paolo są kochankami ukaranymi w Kręgu II za chuć.",
        "place": "Oboje wiecznie wirują razem w wietrze Kręgu II, nierozdzieleni nawet w karze.",
        "key": "Zostali zabici razem przez męża Franceski, brata Paola, po odkryciu romansu; Francesca opowiada o lekturze Lancelota, Paolo milczy i płacze.",
        "hist": "Francesca była córką Guida da Polenta i została wydana za Gianciotta Malatestę; śmierć miała miejsce prawdopodobnie w latach osiemdziesiątych XIII wieku.",
    },
    {
        "nom": "Cerber", "loc": "Cerberze",
        "role": "Cerber jest trójgłowym psem-demonem, strażnikiem Kręgu III.",
        "place": "Cerber pilnuje cuchnącego błota Kręgu III.",
        "key": "Szczeka trzema paszczami jednocześnie tak głośno, że ogłusza, i rozrywa pazurami skórę grzeszników.",
        "hist": "Wergiliusz uspokaja go, wrzucając w trzy paszcze garście błota zamiast pokarmu.",
    },
    {
        "nom": "Ciacco", "loc": "Ciaccu",
        "role": "Ciacco jest florentczykiem ukaranym w Kręgu III za obżarstwo.",
        "place": "Leży twarzą w dół w cuchnącym błocie Kręgu III.",
        "key": "Rozpoznaje Dantego mimo błota na twarzy i przepowiada mu przyszłe konflikty polityczne Florencji.",
        "hist": "Wergiliusz mówi Dantemu, że Ciacco obudzi się dopiero w dniu Sądu Ostatecznego.",
    },
    {
        "nom": "Plutos", "loc": "Plutonie",
        "role": "Plutos jest demonem-bogiem bogactwa i strażnikiem Kręgu IV.",
        "place": "Plutos działa w Kręgu IV, w okrągłej przestrzeni toczonych głazów.",
        "key": "Krzyczy niezrozumiałe groźne słowa na widok przybyszów, lecz samo wezwanie Wergiliusza obezwładnia go i osuwa na ziemię.",
        "hist": "Nie jest Plutonem, rzymskim władcą podziemi, ani Lucyferem.",
    },
    {
        "nom": "Flegiasz", "loc": "Flegiaszu",
        "role": "Flegiasz jest mitycznym przewoźnikiem i synem Marsa.",
        "place": "Przewozi Dantego i Wergiliusza łodzią przez bagno Styks do murów Dis.",
        "key": "W gniewie spalił świątynię Apolla w Delfach i wiezie wędrowców niechętnie, sądząc, że mają potępione dusze.",
        "hist": "Charon i Flegiasz działają na różnych rzekach i nigdy się nie spotykają.",
    },
    {
        "nom": "Filippo Argenti", "loc": "Filippie Argentim",
        "role": "Filippo Argenti jest gniewnym florentczykiem i wrogiem rodziny Dantego.",
        "place": "Przebywa w błocie Styksu w Kręgu V.",
        "key": "W gniewie chwyta łódź, próbując ją zatrzymać; inne gniewne dusze rzucają się na niego i rozszarpują go na oczach wędrowców.",
        "hist": "Dante reaguje wobec niego wyjątkowo surowo.",
    },
    {
        "nom": "Farinata degli Uberti", "loc": "Farinacie degli Uberti",
        "role": "Farinata degli Uberti jest florenckim przywódcą stronnictwa gibelinów.",
        "place": "Leży w płonącym grobowcu w Kręgu VI, wewnątrz miasta Dis.",
        "key": "Słysząc florencki akcent, podnosi się do pasa, rozmawia o polityce i przepowiada Dantemu trudności na wygnaniu.",
        "hist": "Dowodził gibelinami w bitwie pod Montaperti w 1260 roku; po śmierci oskarżono go o herezję epikurejską.",
    },
    {
        "nom": "Minotaur", "loc": "Minotaurze",
        "role": "Minotaur jest istotą pół-człowiekiem, pół-bykiem, strażnikiem wejścia do Kręgu VII.",
        "place": "Stoi u szczytu urwiska prowadzącego z Kręgu VI do Kręgu VII.",
        "key": "Na widok przybyszów wpada w ślepą wściekłość; Wergiliusz wykorzystuje zamieszanie i przemyka obok niego.",
        "hist": "Nie jest sędzią: Minos orzeka na granicy Kręgu II.",
    },
    {
        "nom": "Centaury (Chiron i Nessos)", "loc": "Centaurach (Chironie i Nessosie)",
        "role": "Centaury są strażnikami pierwszego pierścienia Kręgu VII.",
        "place": "Patrolują rzekę wrzącej krwi Flegeton.",
        "key": "Pod wodzą Chirona strzelają z łuku do dusz wynurzających się ponad wyznaczony poziom; Nessos przenosi wędrowców w bród.",
        "hist": "Nessos wskazuje Dantemu zanurzonych tyranów i morderców.",
    },
    {
        "nom": "Pietro della Vigna", "loc": "Pietrze della Vigna",
        "role": "Pietro della Vigna jest drzewem w pierścieniu 2 Kręgu VII i kanclerzem cesarza Fryderyka II.",
        "place": "Zamiast ciała ma postać sękatego drzewa w Lesie Samobójców.",
        "key": "Z pękniętej kory wypływa krew i głos; stanowczo broni swojej niewinności i prosi o przywrócenie dobrego imienia.",
        "hist": "Niesłusznie oskarżony o zdradę, oślepiony i uwięziony; zmarł w więzieniu w 1249 roku w okolicznościach wskazujących na samobójstwo.",
    },
    {
        "nom": "Brunetto Latini", "loc": "Brunetto Latinim",
        "role": "Brunetto Latini jest florenckim uczonym i dawnym nauczycielem Dantego.",
        "place": "Przebywa w pierścieniu 3 Kręgu VII wśród biegnących sodomitów.",
        "key": "Idzie chwilę u krawędzi rozżarzonego piasku, rozmawia o literackiej przyszłości Dantego, po czym musi wrócić do biegu.",
        "hist": "Był notariuszem, dyplomatą i autorem Li Livres dou Tresor.",
    },
    {
        "nom": "Gerion", "loc": "Gerionie",
        "role": "Gerion jest uskrzydlonym potworem o twarzy uczciwego człowieka, żywym uosobieniem Oszustwa.",
        "place": "Przenosi wędrowców z krawędzi Kręgu VII na dno Kręgu VIII.",
        "key": "Ma kosmate łapy z pazurami, gadzi tułów pokryty wzorami i jadowity ogon skorpiona; ląduje w Malebolge i odlatuje.",
        "hist": "Wergiliusz przywołuje go sznurem zrzuconym w przepaść.",
    },
    {
        "nom": "Papież Mikołaj III", "loc": "papieżu Mikołaju III",
        "role": "Papież Mikołaj III jest symoniakiem ukaranym w bolgii 3 Kręgu VIII.",
        "place": "Tkwi głową w dół w okrągłym kamiennym otworze, z płonącymi stopami.",
        "key": "Początkowo myli Dantego z oczekiwanym papieżem Bonifacym VIII, który w roku 1300 wciąż żyje i urzęduje.",
        "hist": "Giovanni Gaetano Orsini był papieżem w latach 1277–1280 i był oskarżany o nepotyzm oraz symonię.",
    },
    {
        "nom": "Malebranche (z Malacodą na czele)", "loc": "Malebranche (z Malacodą na czele)",
        "role": "Malebranche, czyli Złe Pazury, są zorganizowaną grupą demonów pod wodzą Malacody.",
        "place": "Pilnują baraterów zanurzonych we wrzącej smole w bolgii 5 Kręgu VIII.",
        "key": "W groteskowej scenie pościgu same wpadają do wrzątku, co pozwala Dantemu i Wergiliuszowi uciec.",
        "hist": "Nie są pojedynczym trójgłowym psem: takim strażnikiem jest Cerber w Kręgu III.",
    },
    {
        "nom": "Ulisses (Odyseusz)", "loc": "Ulissesie (Odyseuszu)",
        "role": "Ulisses jest greckim bohaterem ukaranym jako fałszywy doradca w bolgii 8.",
        "place": "Jest owinięty wraz z innym doradcą w jeden, rozdwojony u góry płomień.",
        "key": "Opowiada o ostatniej zakazanej wyprawie poza Słupy Heraklesa, zakończonej zatonięciem statku z całą załogą.",
        "hist": "Nie jest postacią historyczną, lecz literacką, zaczerpniętą z tradycji Homera i Wergiliusza.",
    },
    {
        "nom": "Giganci (Nimrod i Anteusz)", "loc": "Gigantach (Nimrodzie i Anteuszu)",
        "role": "Giganci są strażnikami kolistej studni między Kręgiem VIII a IX.",
        "place": "Stoją wkopani w ziemię do pasa, ogromni jak wieże.",
        "key": "Nimrod bełkocze własnym językiem jako kara za budowę wieży Babel; Anteusz jako jedyny nie brał udziału w buncie przeciw bogom.",
        "hist": "Na prośbę Wergiliusza Anteusz własną ręką znosi obu wędrowców na dno Kręgu IX.",
    },
    {
        "nom": "Hrabia Ugolino della Gherardesca", "loc": "hrabim Ugolinie della Gherardesca",
        "role": "Hrabia Ugolino jest zdrajcą ukaranym w strefie Antenora Kręgu IX.",
        "place": "W lodzie Antenory wiecznie gryzie czaszkę arcybiskupa Ruggieriego.",
        "key": "Opowiada, jak Ruggieri kazał zabić drzwi wieży deskami i jak zmarł z głodu wraz z synami i wnukami.",
        "hist": "Był władcą Pizy ze stronnictwa gwelfów; zmarł z głodu w Wieży Głodowej w 1289 roku.",
    },
    {
        "nom": "Arcybiskup Ruggieri degli Ubaldini", "loc": "arcybiskupie Ruggierim degli Ubaldini",
        "role": "Arcybiskup Ruggieri jest zdrajcą dawnego sojuszu z Ugolinem.",
        "place": "Sam również jest ukarany w strefie Antenora, obok Ugolina.",
        "key": "Jego czaszka jest wiecznie gryziona przez swoją ofiarę: to on doprowadził do uwięzienia i śmierci głodowej Ugolina.",
        "hist": "Był rzeczywistym arcybiskupem Pizy i początkowo politycznym sojusznikiem Ugolina.",
    },
    {
        "nom": "Lucyfer", "loc": "Lucyferze",
        "role": "Lucyfer jest upadłym aniołem uwięzionym w lodzie na samym dnie Piekła.",
        "place": "Tkwi od pasa w dół w geometrycznym centrum Kocytu, w strefie Judeka, czyli w środku Ziemi.",
        "key": "Ma trzy twarze – czerwoną, żółtobiałą i czarną – oraz sześć wielkich, nietoperzych skrzydeł, których trzepot wytwarza mróz Kocytu.",
        "hist": "W trzech paszczach żuje Judasza, Brutusa i Kasjusza; nie jest tożsamy z miastem Dis.",
    },
    {
        "nom": "Judasz, Brutus i Kasjusz", "loc": "Judaszu, Brutusie i Kasjuszu",
        "role": "Judasz, Brutus i Kasjusz są trzema największymi zdrajcami według Dantego.",
        "place": "Żują na wieczność w trzech paszczach Lucyfera w Judece.",
        "key": "Judasz jest w pysku centralnym i dodatkowo drapany pazurami po plecach; Brutus i Kasjusz wiszą w paszczach bocznych za zdradę Juliusza Cezara.",
        "hist": "Brutus w paszczy Lucyfera to Marek Brutus, nie Lucius Junius Brutus z Limbo.",
    },
]

# Postać z Dodatku J i P (poza katalogiem 25 kart Dodatku C).
LUCIUS_BRUTUS = {
    "nom": "Lucius Junius Brutus", "loc": "Lucjuszu Juniuszu Brutusie",
    "answer": "Lucius Junius Brutus to cnotliwy, legendarny założyciel Republiki Rzymskiej wymieniany w Limbie.",
    "extra": "Jest całkowicie odrębną osobą od Marka Brutusa, zabójcy Juliusza Cezara żutego w bocznej paszczy Lucyfera.",
}

# ---------------------------------------------------------------------------
# 4. Pieśni (Dodatek A), sceny (R), cytaty (G), terminy (B, O), kontekst (F, L)
# ---------------------------------------------------------------------------

SONGS = [
    ("Pieśń I", "Dante gubi się w ciemnym lesie, napotyka trzy bestie blokujące drogę na górę i spotyka Wergiliusza, który proponuje drogę przez Piekło."),
    ("Pieśń II", "Wergiliusz opowiada o interwencji Beatrycze i pełnym łańcuchu próśb, który do niej doprowadził; Dante zgadza się wyruszyć."),
    ("Pieśń III", "Wędrowcy mijają Bramę Piekła z napisem, przechodzą przez Przedpiekle dusz obojętnych, docierają nad Acheront i zostają przewiezieni przez Charona."),
    ("Pieśń IV", "Dante budzi się w Kręgu I, poznaje Wergiliusza jako mieszkańca Limbo i widzi zamek z wielkimi postaciami starożytności."),
    ("Pieśń V", "Dante i Wergiliusz spotykają Minosa na granicy Kręgu II, wchodzą w wicher i rozmawiają z Francescą da Rimini oraz Paolem Malatestą."),
    ("Pieśń VI", "W Kręgu III Wergiliusz uspokaja Cerbera błotem, a Dante rozmawia z Ciaccem o przyszłości Florencji."),
    ("Pieśń VII", "W Kręgu IV Plutos zostaje uciszony, Dante widzi skąpców i rozrzutników toczących głazy, a pod koniec docierają do Styksu."),
    ("Pieśń VIII", "Flegiasz przewozi wędrowców przez Styks, dochodzi do starcia z Filippem Argentim i łódź dociera pod mury Dis."),
    ("Pieśń IX", "Upadłe anioły odmawiają wstępu do Dis, pojawiają się Furie i groźba Meduzy, aż anioł zesłany z nieba otwiera bramę."),
    ("Pieśń X", "W Kręgu VI, wśród płonących grobowców heretyków, Dante rozmawia z Farinatą degli Uberti o polityce Florencji."),
    ("Pieśń XI", "Na krawędzi Kręgu VII, z powodu smrodu, Wergiliusz wyjaśnia Dantemu strukturę grzechów całego Piekła."),
    ("Pieśń XII", "Minotaur atakuje u wejścia do Kręgu VII; w pierścieniu 1 Centaury z Chironem pilnują Flegetonu, a Nessos przewozi wędrowców."),
    ("Pieśń XIII", "W pierścieniu 2 Dante wchodzi do Lasu Samobójców, łamie gałąź będącą duszą Pietra della Vigna i słucha jego historii."),
    ("Pieśni XIV–XVII", "W pierścieniu 3 Dante mija bluźnierców, rozmawia z Brunettem Latinim, widzi lichwiarzy, a na końcu pojawia się Gerion."),
    ("Pieśń XVIII", "Gerion ląduje na dnie Kręgu VIII; wędrowcy mijają bolgię 1 kuplerów i uwodzicieli oraz bolgię 2 pochlebców."),
    ("Pieśń XIX", "W bolgii 3 Dante rozmawia z papieżem Mikołajem III, tkwiącym głową w dół wśród symoniaków."),
    ("Pieśń XX", "W bolgii 4 Dante widzi wróżbitów i czarowników chodzących tyłem z odwróconymi głowami."),
    ("Pieśni XXI–XXIII", "W bolgii 5 Malebranche z Malacodą pilnują baraterów; demony wpadają do smoły, a wędrowcy przechodzą do bolgii 6 z hipokrytami."),
    ("Pieśni XXIV–XXV", "W bolgii 7 złodzieje są kąsani przez węże i ulegają groteskowym przemianom kształtu."),
    ("Pieśni XXVI–XXVII", "W bolgii 8 Dante i Wergiliusz rozmawiają z płomieniem, w którym spalony jest Ulisses, o jego ostatniej wyprawie morskiej."),
    ("Pieśń XXVIII", "W bolgii 9 siewcy niezgody są wiecznie okaleczani mieczem przez demona."),
    ("Pieśni XXIX–XXX", "W bolgii 10 Dante widzi fałszerzy dotkniętych różnymi chorobami ciała."),
    ("Pieśń XXXI", "Dante i Wergiliusz docierają do studni Gigantów; Nimrod bełkocze niezrozumiale, a Anteusz znosi ich na dno Kręgu IX."),
    ("Pieśni XXXII–XXXIII", "W Kainie i Antenorze Dante szarpie za włosy wmarzniętą duszę i słucha hrabiego Ugolina gryzącego czaszkę Ruggieriego."),
    ("Pieśń XXXIII (ciąg dalszy)", "W Ptolomei Dante rozmawia z duszami zdrajców gości, których ciała wciąż żyją na ziemi, opętane przez demony."),
    ("Pieśń XXXIV", "W Judece wędrowcy docierają do Lucyfera; Wergiliusz schodzi przez środek Ziemi i obaj wychodzą na półkulę południową."),
]

SCENES = [
    {"nom": "scena z Francescą i Paolem", "loc": "scenie z Francescą i Paolem",
     "answer": "Francesca, nie Paolo, zabiera głos i opowiada, że miłość narodziła się podczas wspólnej lektury o rycerzu Lancelocie.",
     "extra": "Paolo przez całą scenę milczy i płacze, a Dante mdleje z litości i upada jak martwe ciało."},
    {"nom": "scena z Farinatą degli Uberti", "loc": "scenie z Farinatą degli Uberti",
     "answer": "Farinata podnosi się z płonącego grobowca do pasa, pyta Dantego o przodków i broni swojej roli w dawnych wojnach bez skruchy.",
     "extra": "Kończy przepowiednią, że Dante dowie się, jak trudno wracać z wygnania, zanim upłynie pięćdziesiąt księżyców."},
    {"nom": "scena z Pietrem della Vigną", "loc": "scenie z Pietrem della Vigną",
     "answer": "Na polecenie Wergiliusza Dante łamie gałąź krzewu; z pękniętej kory wypływa krew zmieszana z głosem Pietra della Vigna.",
     "extra": "Della Vigna opowiada o fałszywym oskarżeniu, oślepieniu i uwięzieniu, zapewnia o wierności wobec cesarza i prosi o oczyścienie imienia."},
    {"nom": "scena z Brunettem Latinim", "loc": "scenie z Brunettem Latinim",
     "answer": "Brunetto rozpoznaje Dantego wśród biegnących sodomitów i na chwilę dołącza do niego przy brzegu rozżarzonego piasku.",
     "extra": "Przypomina wspólne lata nauki i przepowiada sławę literacką oraz trudności ze strony niewdzięcznych współobywateli Florencji."},
    {"nom": "scena z hrabią Ugolinem", "loc": "scenie z hrabią Ugolinem",
     "answer": "Ugolino podnosi głowę znad czaszki Ruggieriego, ociera usta o włosy ofiary i opowiada o uwięzieniu w wieży z synami i wnukami.",
     "extra": "Opisuje zabite deskami drzwi, kolejną śmierć głodową dzieci i powrót do gryzienia czaszki z zajadłością psa."},
]

QUOTES = [
    {"nom": "cytat „Porzućcie wszelką nadzieję, wy, którzy tu wchodzicie”", "loc": "cytacie „Porzućcie wszelką nadzieję, wy, którzy tu wchodzicie”",
     "answer": "To ostatni wers napisu nad Bramą Piekła; wcześniejsze wersy mówią o boskiej sprawiedliwości, wszechmocy, mądrości i pierwszej miłości."},
    {"nom": "cytat Franceski o wspominaniu szczęścia", "loc": "cytacie Franceski o wspominaniu szczęścia",
     "answer": "„Nie ma większego bólu niż wspominać szczęśliwy czas w nieszczęściu” to słowa Franceski da Rimini z Kręgu II, tłumaczące jej rozpacz."},
    {"nom": "słowa Wergiliusza o duszach obojętnych", "loc": "słowach Wergiliusza o duszach obojętnych",
     "answer": "„Nie będziemy mówić o nich, spójrz i mijaj” to słowa Wergiliusza o duszach obojętnych w Przedpieklu, wyrażające ich moralną nieistotność."},
    {"nom": "ostatni wers Piekła", "loc": "ostatnim wersie Piekła",
     "answer": "„i stamtąd wyszliśmy, by ponownie ujrzeć gwiazdy” opisuje wyjście Dantego i Wergiliusza na powierzchnię ziemi po przejściu przez środek planety.",
     "extra": "Każda z trzech części Boskiej Komedii kończy się słowem „gwiazdy”, po włosku „stelle”."},
]

TERMS = [
    {"nom": "contrapasso", "loc": "contrapasso", "answer": "Contrapasso to zasada rządząca karami w Piekle: kara odpowiada grzechowi przez analogię lub przeciwieństwo.",
     "extra": "Wróżbici chcieli widzieć przyszłość, więc mają głowy odwrócone do tyłu i widzą tylko to, co za nimi."},
    {"nom": "canto (pieśń)", "loc": "canto (pieśni)", "answer": "Canto to podstawowa jednostka podziału Boskiej Komedii, odpowiednik rozdziału w prozie."},
    {"nom": "cantica", "loc": "cantica", "answer": "Cantica to jedna z trzech głównych części poematu: Piekło, Czyściec albo Raj."},
    {"nom": "terza rima (tercyna)", "loc": "terza rima (tercynie)", "answer": "Terza rima to trzywersowa strofa z rymem łańcuchowym aba, bcb, cdc, zastosowana przez Dantego w Boskiej Komedii."},
    {"nom": "Malebolge", "loc": "Malebolge", "answer": "Malebolge to nazwa własna Kręgu VIII, dosłownie „złe rowy” albo „złe kiesy”; dzieli się na dziesięć bolgii."},
    {"nom": "bolgia", "loc": "bolgii", "answer": "Bolgia to pojedynczy, koncentryczny rów w obrębie Malebolge; liczba mnoga to bolgie."},
    {"nom": "Giudecca (Judeka)", "loc": "Giudecce (Judece)", "answer": "Giudecca, po polsku Judeka, to najgłębsza strefa Kocytu w Kręgu IX, nazwana od imienia Judasza."},
    {"nom": "Ante-Inferno (Przedpiekle)", "loc": "Ante-Inferno (Przedpieklu)", "answer": "Ante-Inferno, czyli Przedpiekle, to strefa dusz obojętnych przed rzeką Acheront, niebędąca numerowanym kręgiem."},
    {"nom": "acedia", "loc": "acedii", "answer": "Acedia to łacińskie określenie ponurego, biernego lenistwa duchowego, karane w Kręgu V razem z gniewem."},
    {"nom": "simonia (symonia)", "loc": "simonii (symonii)", "answer": "Simonia, po polsku symonia, to grzech handlowania urzędami i godnościami kościelnymi, karany w bolgii 3 Kręgu VIII."},
    {"nom": "barateria", "loc": "baraterii", "answer": "Barateria to grzech sprzedajności urzędników publicznych, karany w bolgii 5 Kręgu VIII."},
    {"nom": "Gwelfowie i Gibelini", "loc": "Gwelfach i Gibelinach", "answer": "Gwelfowie popierali papiestwo, Gibelini popierali cesarstwo; ten konflikt jest tłem postaci takich jak Farinata degli Uberti i wygnania Dantego."},
    {"nom": "Gwelfowie Czarni i Gwelfowie Biali", "loc": "Gwelfach Czarnych i Gwelfach Białych", "answer": "Gwelfowie Czarni i Biali to zwalczające się frakcje gwelfów we Florencji przełomu XIII i XIV wieku; Dante należał do Białych."},
    {"nom": "Limbo", "loc": "Limbo", "answer": "Limbo to potoczna nazwa Kręgu I, używana zamiennie z Otchłanią."},
    {"nom": "Inferno", "loc": "Inferno", "answer": "Inferno to włoska nazwa Piekła, pierwszej części Boskiej Komedii."},
    {"nom": "il sommo poeta", "loc": "określeniu il sommo poeta", "answer": "Il sommo poeta, czyli „najwyższy poeta”, to honorowy epitet używany wobec Wergiliusza."},
    {"nom": "Lucifero", "loc": "Lucifero", "answer": "Lucifero to włoska forma imienia Lucyfer, upadłego anioła uwięzionego w centrum Kręgu IX."},
    {"nom": "Cocito", "loc": "Cocito", "answer": "Cocito to włoska nazwa Kocytu, zamarzniętego jeziora w Kręgu IX."},
    {"nom": "Flegetonte", "loc": "Flegetonte", "answer": "Flegetonte to włoska nazwa Flegetonu, rzeki wrzącej krwi w pierścieniu 1 Kręgu VII."},
    {"nom": "Stige", "loc": "Stige", "answer": "Stige to włoska nazwa Styksu, bagnistego zbiornika w Kręgu V."},
    {"nom": "Acheronte", "loc": "Acheronte", "answer": "Acheronte to włoska nazwa Acherontu, rzeki na granicy Piekła pilnowanej przez Charona."},
    {"nom": "Città di Dite", "loc": "Città di Dite", "answer": "Città di Dite to włoska nazwa miasta Dis, warownego miasta obejmującego Kręgi VI–IX."},
    {"nom": "Minosse", "loc": "Minosse", "answer": "Minosse to włoska forma imienia Minos, sędzi urzędującego na granicy Kręgu II."},
    {"nom": "Gerione", "loc": "Gerione", "answer": "Gerione to włoska forma imienia Gerion, potwora przenoszącego wędrowców do Kręgu VIII."},
    {"nom": "Caina, Antenora, Tolomea, Giudecca", "loc": "nazwach Caina, Antenora, Tolomea i Giudecca",
     "answer": "Cztery strefy Kręgu IX noszą w oryginale nazwy Caina, Antenora, Tolomea i Giudecca, czyli po polsku Kaina, Antenora, Ptolomea i Judeka.",
     "extra": "Pochodzą odpowiednio od imion Kaina, Antenora, Ptolemeusza i Judasza."},
    {"nom": "cztery zbiorniki wodne Piekła", "loc": "czterech zbiornikach wodnych Piekła",
     "answer": "Acheront, Styks, Flegeton i Kocyt to cztery odrębne zbiorniki: granica Piekła, Krąg V, pierścień 1 Kręgu VII i cały Krąg IX."},
]

HISTORY = [
    {"nom": "data i miejsce urodzenia Dantego", "loc": "dacie i miejscu urodzenia Dantego", "answer": "Dante Alighieri urodził się w 1265 roku we Florencji w rodzinie drobnej szlachty."},
    {"nom": "wygnanie Dantego", "loc": "wygnaniu Dantego", "answer": "W 1301 roku władzę we Florencji przejęli Gwelfowie Czarni wspierani przez papieża Bonifacego VIII, a w 1302 Dante został skazany zaocznie na wygnanie."},
    {"nom": "powrót Dantego do Florencji", "loc": "kwestii powrotu Dantego do Florencji", "answer": "Dante nigdy nie wrócił do Florencji i zmarł na wygnaniu w Rawennie w 1321 roku."},
    {"nom": "tytuł dzieła", "loc": "tytule dzieła", "answer": "Oryginalny tytuł utworu brzmiał Commedia; określenie Boska dodał Giovanni Boccaccio, autor pierwszej biografii Dantego, kilkadziesiąt lat po śmierci poety."},
    {"nom": " znaczenie słowa komedia", "loc": "znaczeniu słowa komedia", "answer": "W średniowieczu komedia oznaczała utwór zaczynający się od trudności, a kończący szczęśliwie, w przeciwieństwie do tragedii."},
    {"nom": "okres powstania Boskiej Komedii", "loc": "okresie powstania Boskiej Komedii", "answer": "Dante pisał dzieło na wygnaniu, prawdopodobnie w latach 1308–1320, przebywając kolejno w Weronie, Lukce i Rawennie."},
    {"nom": "pierwsze wydanie drukowane", "loc": "pierwszym wydaniu drukowanym", "answer": "Pierwsze drukowane wydanie poematu ukazało się w 1472 roku we Foligno, półtora wieku po powstaniu utworu."},
    {"nom": "rękopis autorski", "loc": "rękopisie autorskim", "answer": "Oryginalny rękopis autorski nie zachował się; najstarsze zachowane kopie pochodzą z lat trzydziestych XIV wieku."},
    {"nom": "konflikt gwelfów i gibelinów", "loc": "konflikcie gwelfów i gibelinów", "answer": "Konflikt między Gwelfami i Gibelinami trwał we włoskich miastach-państwach od XII do XIV wieku; Gwelfowie popierali papiestwo, Gibelini cesarza."},
    {"nom": "Francesca da Rimini jako postać historyczna", "loc": "Francesce da Rimini jako postaci historycznej", "answer": "Francesca była córką Guida da Polenta, władcy Rawenny, i została wydana za Gianciotta Malatestę z Rimini z powodów politycznych."},
    {"nom": "Farinata pod Montaperti", "loc": "Farinacie pod Montaperti", "answer": "Farinata degli Uberti dowodził siłami gibelińskimi w zwycięskiej bitwie pod Montaperti w 1260 roku przeciwko gwelfom."},
    {"nom": "Pietro della Vigna jako kanclerz", "loc": "Pietrze della Vignie jako kanclerzu", "answer": "Pietro della Vigna był kanclerzem i logotetą na dworze cesarza Fryderyka II Hohenstaufa."},
    {"nom": "Brunetto Latini jako nauczyciel", "loc": "Brunetto Latinim jako nauczycielu", "answer": "Brunetto Latini był florenckim notariuszem, dyplomatą, uczonym i osobistym nauczycielem oraz mentorem młodego Dantego."},
    {"nom": "Mikołaj III jako papież", "loc": "Mikołaju III jako papieżu", "answer": "Papież Mikołaj III, Giovanni Gaetano Orsini, rządził Kościołem w latach 1277–1280."},
    {"nom": "Ugolino jako władca Pizy", "loc": "Ugolinie jako władcy Pizy", "answer": "Hrabia Ugolino della Gherardesca był władcą Pizy ze stronnictwa gwelfów, oskarżonym o oddanie pizańskich zamków Lukce i Florencji."},
    {"nom": "Wieża Głodowa", "loc": "Wieży Głodowej", "answer": "Ugolino został uwięziony z synami i wnukami w Wieży Głodowej, zwanej Muda, i zmarł tam z głodu w 1289 roku."},
    {"nom": "Ruggieri jako arcybiskup Pizy", "loc": "Ruggierim jako arcybiskupie Pizy", "answer": "Arcybiskup Ruggieri degli Ubaldini był rzeczywistym arcybiskupem Pizy i dawnym sojusznikiem politycznym Ugolina."},
    {"nom": "Ulisses jako postać literacka", "loc": "Ulissesie jako postaci literackiej", "answer": "Ulisses nie jest postacią historyczną, lecz literacką, zaczerpniętą z tradycji Homera i Wergiliusza."},
    {"nom": "opowieść o wyprawie Ulissesa", "loc": "opowieści o wyprawie Ulissesa", "answer": "Opowieść o ostatniej podróży Ulissesa poza Słupy Heraklesa jest oryginalnym pomysłem Dantego, nieobecnym w wcześniejszych źródłach antycznych."},
    {"nom": "postacie mityczne w roli strażników", "loc": "postaciach mitycznych w roli strażników", "answer": "Minos, Cerber, Plutos, Flegiasz, Centaury, Harpie, Gerion, Giganci i Lucyfer pochodzą z mitologii lub Biblii, lecz Dante umieścił ich we własnych miejscach Piekła."},
]

CLASSIFICATION = [
    {"nom": "klasyfikacja grzechów z pieśni XI", "loc": "klasyfikacji grzechów z pieśni XI",
     "answer": "Wergiliusz wyjaśnia ją na granicy Kręgu VII, opierając się na klasyfikacji Arystotelesa; wyróżnia niepohamowanie, przemoc i oszustwo.",
     "extra": "Przemoc i oszustwo są gorsze od niepohamowania, bo wymagają świadomego działania rozumu przeciwko dobru."},
    {"nom": "granica murów Dis", "loc": "granicy murów Dis", "answer": "Kręgi I–V leżą poza murami Dis i mieszczą niepohamowanie, a Kręgi VI–IX wewnątrz murów mieszczą grzechy złej woli."},
    {"nom": "miejsce herezji w klasyfikacji", "loc": "miejscu herezji w klasyfikacji", "answer": "Krąg VI jest już za murami Dis, lecz nie mieści się w trójpodziale niepohamowanie–przemoc–oszustwo: herezja to odrębny błąd rozumu i wiary."},
    {"nom": "trzy podkategorie przemocy", "loc": "trzech podkategoriach przemocy", "answer": "Przemoc w Kręgu VII dzieli się na przemoc wobec bliźniego, wobec samego siebie oraz wobec Boga, natury i ludzkiej pracy."},
    {"nom": "dwie podkategorie oszustwa", "loc": "dwóch podkategoriach oszustwa", "answer": "Oszustwo wobec osób bez szczególnej więzi zaufania karane jest w Kręgu VIII, a oszustwo wobec osób związanych więzią zaufania w Kręgu IX."},
    {"nom": "ranga zdrady", "loc": "randze zdrady", "answer": "Zdrada jest najcięższym grzechem w tej klasyfikacji, bo niszczy więzi, na których opiera się całe zaufanie społeczne."},
    {"nom": "siedem grzechów głównych", "loc": "siedmiu grzechach głównych", "answer": "Siedem grzechów głównych nie jest podstawą struktury Piekła; strukturyzuje siedem tarasów Czyśćca w drugiej części poematu."},
    {"nom": "kierunek wędrówki", "loc": "kierunku wędrówki", "answer": "Dante idzie od Bramy Piekła w dół, krąg po kręgu, od Kręgu I do IX; nigdy nie cofa się ani nie pomija kręgu."},
    {"nom": "numer kręgu a ciężar grzechu", "loc": "numerze kręgu a ciężarze grzechu", "answer": "Im wyższy numer kręgu, tym cięższy kategorialnie grzech i tym mniejszy obwód leja."},
    {"nom": "wyrok Minosa", "loc": "wyroku Minosa", "answer": "Przydział dusz nie jest losowy ani zależny od demona: wynika z jednorazowej decyzji Minosa opartej na konkretnym grzechu popełnionym za życia."},
]

WORLD_META = [
    {"nom": "liczba pieśni Piekła", "loc": "liczbie pieśni Piekła", "answer": "Piekło liczy trzydzieści cztery pieśni: jedną wstępną dla całego poematu oraz trzydzieści trzy właściwe."},
    {"nom": "liczba pieśni całej Boskiej Komedii", "loc": "liczbie pieśni całej Boskiej Komedii", "answer": "Całość liczy sto pieśni: jedną wstępną oraz po trzydzieści trzy w Piekłach, Czyśćcu i Raju."},
    {"nom": "liczba wersów Piekła", "loc": "liczbie wersów Piekła", "answer": "Piekło ma około czterech tysięcy ośmiuset wersów."},
    {"nom": "język i forma wiersza", "loc": "języku i formie wiersza", "answer": "Utwór napisano po włosku, w dialekcie toskańskim, tercyną z rymem łańcuchowym."},
    {"nom": "kształt Piekła", "loc": "kształcie Piekła", "answer": "Piekło ma kształt leja, czyli odwróconego stożka, zwężającego się ku środkowi Ziemi."},
    {"nom": "powstanie Piekła", "loc": "powstaniu Piekła", "answer": "Według kosmologii poematu Piekło powstało w momencie upadku Lucyfera, który uderzył w półkulę północną i wbił się do środka Ziemi."},
    {"nom": "góra Czyśćca", "loc": "górze Czyśćca", "answer": "Ziemia wypchnięta uderzeniem Lucyfera na przeciwległej półkuli utworzyła górę Czyśćca."},
    {"nom": "czas akcji Piekła", "loc": "czasie akcji Piekła", "answer": "Akcja trwa około trzydziestu sześciu godzin: zaczyna się wieczorem w Wielki Czwartek albo nad ranem w Wielki Piątek 1300 roku."},
    {"nom": "koniec akcji Piekła", "loc": "końcu akcji Piekła", "answer": "Wędrowcy wychodzą przed świtem w Wielką Sobotę na powierzchnię ziemi po drugiej stronie globu i znów widzą gwiazdy."},
    {"nom": "symetria poematu", "loc": "symetrii poematu", "answer": "Dziewięć kręgów odpowiada dziewięciu sferom niebieskim w Raju oraz siedmiu tarasom i dwóm przedpolom w Czyśćcu."},
    {"nom": "napis nad bramą", "loc": "napisie nad bramą", "answer": "Nad kamienną bramą widnieje napis z dziewięciu wersów, kończący się zdaniem o porzuceniu wszelkiej nadziei."},
    {"nom": "dusze obojętne", "loc": "duszach obojętnych", "answer": "W Przedpieklu przebywają ludzie, którzy za życia nie opowiedzieli się ani za dobrem, ani za złem, oraz aniołowie neutralni podczas buntu Lucyfera."},
    {"nom": "kara dusz obojętnych", "loc": "karze dusz obojętnych", "answer": "Dusze obojętne wiecznie biegną nago za uciekającym sztandarem, dręczone osami i szerszeniami; krew i łzy zbiera robactwo u ich stóp."},
    {"nom": "przeprawa przez Acheront", "loc": "przeprawie przez Acheront", "answer": "Po przeprawie przez Acheront Dante traci przytomność z powodu trzęsienia ziemi i błysku czerwonego światła; samą przeprawę widzi tylko w opowieści."},
    {"nom": "Celestyn V w Przedpieklu", "loc": "Celestynie V w Przedpieklu", "answer": "Wśród dusz obojętnych Dante rozpoznaje duszę papieża, który tchórzliwie zrzekł się urzędu; tradycyjnie utożsamia się ją z Celestynem V."},
    {"nom": "Fortuna w Kręgu IV", "loc": "Fortunie w Kręgu IV", "answer": "Wergiliusz tłumaczy, że Fortuna jest sługą bożą rozdzielającą dobra doczesne niezależnie od ludzkich zasług, co tłumaczy bezsens pogoni za bogactwem."},
    {"nom": "wyjście z Piekła", "loc": "wyjściu z Piekła", "answer": "Wergiliusz niesie Dantego po futrze Lucyfera do pasa, gdzie grawitacja się odwraca, a potem wspinają się po nogach po drugiej stronie globu."},
]

# ---------------------------------------------------------------------------
# 5. Strefa odmów (Sekcja 4) i pułapki nazewnicze (Dodatek J)
# ---------------------------------------------------------------------------

RIVER_PLACES = {
    "Acheront": ("rzeką na granicy Piekła, pilnowaną przez Charona", "granicę Piekła"),
    "Styks": ("bagnem w Kręgu V, przed murami Dis", "Krąg V"),
    "Flegeton": ("rzeką wrzącej krwi w pierścieniu 1 Kręgu VII", "pierścień 1 Kręgu VII"),
    "Kocyt": ("zamarzniętym jeziorem wypełniającym cały Krąg IX", "Krąg IX"),
}
RIVER_NAMES = list(RIVER_PLACES)

FERRY = {
    "Charon": ("przewozi dusze przez Acheront na próg Kręgu I", "przez Acheront"),
    "Flegiasz": ("przewozi Dantego i Wergiliusza łodzią przez Styks", "przez Styks"),
    "Gerion": ("przenosi Dantego i Wergiliusza lotem z Kręgu VII do Kręgu VIII", "przez przepaść między Kręgiem VII a VIII"),
}

DISAMBIGUATION: list[dict[str, str]] = []


def add_dis(claim: str, fix: str, contrast: str = "", extra: str = "") -> None:
    DISAMBIGUATION.append(
        {"claim": claim, "fix": fix, "contrast": contrast, "extra": extra}
    )


# Rzeki przypisane do błędnych miejsc: 4 × 3 = 12
for river in RIVER_NAMES:
    correct, _ = RIVER_PLACES[river]
    for other in RIVER_NAMES:
        if other == river:
            continue
        other_correct, _ = RIVER_PLACES[other]
        add_dis(
            f"{river} przypada na miejsce: {RIVER_PLACES[other][1]}",
            f"Nie. {river} jest {correct}; {other} jest {other_correct}.",
            "To cztery odrębne zbiorniki wodne, niepołączone wspólną nazwą.",
        )

# Przewoźnicy przypisani do błędnych tras: 3 × 2 = 6
for who in FERRY:
    correct, _ = FERRY[who]
    for other in FERRY:
        if other == who:
            continue
        other_correct, _ = FERRY[other]
        add_dis(
            f"{who} przewozi wędrowców {FERRY[other][1]}",
            f"Nie. {who} {correct}; {other} {other_correct}.",
            "Żaden z nich nie zastępuje drugiego.",
        )

# Podstawowe pułapki z Dodatku J i Sekcji 4
add_dis("Minos strzeże wejścia do Kręgu VII",
        "Nie. Minos jest sędzią na granicy Kręgu II, a wejścia do Kręgu VII strzeże Minotaur.",
        "Minos to sędzia, Minotaur to strażnik; są to dwie różne istoty.")
add_dis("Minotaur osądza dusze na granicy Kręgu II",
        "Nie. Na granicy Kręgu II osądza dusze Minos, natomiast Minotaur strzeże Kręgu VII.",
        "Podobieństwo imion nie oznacza tej samej roli.")
add_dis("Minos owija ogon wokół siebie przy wejściu do Kręgu VII",
        "Nie. Minos ogon owija na granicy Kręgu II, wyznaczając numer kręgu dla duszy.",
        "Przy Kręgu VII działa Minotaur i Centaury.")
add_dis("Lucyfer to inne imię miasta Dis",
        "Nie. Dis to wyłącznie nazwa warownego miasta obejmującego Kręgi VI, VII, VIII i IX.",
        "Lucyfer to odrębna postać uwięziona w lodzie w centrum Kręgu IX.")
add_dis("Dis jest imieniem upadłego anioła",
        "Nie. Dis to miasto, nie osoba; upadły anioł w centrum Kocytu nazywa się Lucyfer.",
        "Miasto Dis zaczyna się na granicy Kręgu V i VI.")
add_dis("Lucyfer mieszka w mieście Dis jako jego władca",
        "Nie. Lucyfer tkwi w lodzie w strefie Judeka w samym centrum Kręgu IX, a Dis to nazwa miasta.",
        "Nie nazywa się go Disem.")
add_dis("Plutos jest Plutonem, władcą podziemi",
        "Nie. Plutos to demon-bóg bogactwa pilnujący Kręgu IV, nie rzymski Pluton.",
        "Plutos zostaje uciszony jednym wezwaniem Wergiliusza.")
add_dis("Pluton pilnuje Kręgu IV i bogactwa",
        "Nie. Kręgu IV pilnuje Plutos, demon-bóg bogactwa, a Pluton to władca podziemi w mitologii rzymskiej.",
        "Plutos nie jest też Lucyferem.")
add_dis("Lucius Junius Brutus i Marek Brutus to ta sama postać",
        "Nie. Lucius Junius Brutus to cnotliwy założyciel Republiki Rzymskiej w Limbie, a Marek Brutus to zabójca Cezara żuty w paszczy Lucyfera.",
        "Mają przeciwny status moralny w poemacie.")
add_dis("Marek Brutus przebywa w Limbie",
        "Nie. W Limbie wymieniany jest Lucius Junius Brutus; Marek Brutus jest w Judece w paszczy Lucyfera.",
        "Marek Brutus zdradził Juliusza Cezara.")
add_dis("Cerber i Malebranche to ten sam demon",
        "Nie. Cerber to pojedynczy trójgłowy pies w Kręgu III, a Malebranche to grupa demonów z Malacodą w bolgii 5.",
        "Działają w różnych kręgach.")
add_dis("Malebranche pilnują Cerbera w Kręgu III",
        "Nie. Malebranche pod wodzą Malacody pilnują baraterów w bolgii 5, a Cerber sam pilnuje Kręgu III.",
        "To różne istoty w różnych miejscach.")
add_dis("Przedpiekle to po prostu Limbo",
        "Nie. Przedpiekle leży przed Acherontem i nie jest numerowanym kręgiem, a Limbo to Krąg I za Acherontem.",
        "Przedpiekle mieści dusze obojętne, Limbo – nieochrzczonych.")
add_dis("Limbo leży przed rzeką Acheront",
        "Nie. Przed Acherontem jest Przedpiekle; Limbo, czyli Krąg I, zaczyna się dopiero po przeprawie.",
        "Dante budzi się w Limbie po przekroczeniu rzeki.")
add_dis("Judeka to dzielnica żydowska we Włoszech",
        "Nie. W kontekście Piekła Dantego Judeka to wyłącznie czwarta, najgłębsza strefa Kocytu w Kręgu IX.",
        "Nazwa pochodzi od imienia Judasza.")
add_dis("Krąg VI i Krąg VII to ten sam krąg",
        "Nie. Krąg VI karze herezję w płonących grobowcach Dis, a Krąg VII karze przemoc w trzech pierścieniach.",
        "Dzieli je strome urwisko strzeżone przez Minotaura.")
add_dis("Ogień dominuje w całym Kręgu VII",
        "Nie. W Kręgu VII ogień pada tylko na pierścień 3; pierścień 1 ma wrzącą krew, a pierścień 2 drzewa.",
        "Ogień jako kara dominująca dotyczy Kręgu VI i części Kręgu VIII.")
add_dis("Wszyscy grzesznicy w Piekle płoną w ogniu",
        "Nie. Kary są zróżnicowane według contrapasso: są wiatr, błoto, wrząca krew, smoła, lód i toczenie głazów.",
        "Ogień dotyczy tylko wybranych kręgów i bolgii.")
add_dis("Charon, Flegiasz i Gerion to jeden przewoźnik pod trzema imionami",
        "Nie. Charon działa na Acheroncie, Flegiasz na Styksie, a Gerion przenosi wędrowców lotem do Kręgu VIII.",
        "Charon i Flegiasz nigdy się nie spotykają.")
add_dis("Kocyt jest rzeką wrzącej krwi",
        "Nie. Kocyt to zamarznięte jezioro w Kręgu IX; rzeką wrzącej krwi jest Flegeton w pierścieniu 1 Kręgu VII.",
        "Flegeton i Kocyt to odrębne zbiorniki.")
add_dis("Acheront jest bagnem przed murami Dis",
        "Nie. Acheront to rzeka na granicy Piekła pilnowana przez Charona; bagno przed murami Dis to Styks.",
        "Za Acherontem zaczyna się Krąg I.")
add_dis("Styks oddziela Krąg VIII od Kręgu IX",
        "Nie. Styks to bagno w Kręgu V; studnię między Kręgiem VIII a IX wypełniają Giganci.",
        "Kocyt dopiero wypełnia cały Krąg IX.")

# Kręgi z błędnie przypisanymi strażnikami: 9 × 2 = 18
GUARDS = [
    ("I", "brak własnego strażnika"), ("II", "Minos"), ("III", "Cerber"),
    ("IV", "Plutos"), ("V", "Flegiasz"), ("VI", "upadłe anioły i Furie"),
    ("VII", "Minotaur, Centaury i Harpie"), ("VIII", "Gerion i Malebranche"),
    ("IX", "Giganci i Lucyfer"),
]
for idx, (roman, correct_guard) in enumerate(GUARDS):
    for offset in (1, 2):
        wrong_roman, wrong_guard = GUARDS[(idx + offset) % 9]
        add_dis(
            f"strażnikiem Kręgu {roman} jest {wrong_guard}",
            (
                f"Krąg {roman} nie ma własnego strażnika"
                if roman == "I"
                else f"Krąg {roman} ma strażnika: {correct_guard}"
            )
            + ". "
            + (
                f"Krąg {wrong_roman} nie ma własnego strażnika"
                if wrong_roman == "I"
                else f"Krąg {wrong_roman} ma strażnika: {wrong_guard}"
            )
            + ".",
            "Przydział strażników wynika z kanonu, nie z przypadku.",
        )

# Strefy Kocytu z błędnym rodzajem zdrady: 4 × 2 = 8
for idx, zone in enumerate(ZONES):
    for offset in (1, 2):
        other = ZONES[(idx + offset) % 4]
        add_dis(
            f"{zone['nom']} karze {other['betrayal_acc']}",
            f"{zone['nom'][0].upper() + zone['nom'][1:]} karze za {zone['betrayal_acc']}; "
            f"{other['betrayal']} jest karane w {other['loc']}.",
            "Kocyt dzieli się na cztery koncentryczne strefy.",
        )

# ---------------------------------------------------------------------------
# 6. Odmowy z Sekcji 4
# ---------------------------------------------------------------------------

REFUSALS: list[dict[str, str]] = []


def add_refusal(claim: str, fix: str, extra: str = "") -> None:
    REFUSALS.append({"claim": claim, "fix": fix, "extra": extra})


for roman, _circle_name in [(c["roman"], c["name"]) for c in CIRCLES]:
    add_refusal(f"Czyściec jest kręgiem Piekła o numerze {roman}",
                "Nie. Czyściec nie jest kręgiem Piekła ani jego częścią; to osobna, druga część Boskiej Komedii.",
                "Dusze w Czyśćcu cierpią tymczasowo i w końcu trafiają do Raju.")
    add_refusal(f"Raj jest kręgiem Piekła o numerze {roman}",
                "Nie. Raj nie jest kręgiem ani elementem geografii Piekła; to osobna, trzecia część poematu.",
                "Raj składa się z dziewięciu sfer niebieskich i Empireum.")

INTRUDERS = ["Gandalf", "Thor", "Zeus", "Faust", "Król Artur", "Don Kichot"]
for name in INTRUDERS:
    add_refusal(f"w Piekle Dantego spotykamy {name}",
                f"Nie. {name} nie występuje w tekście Dantego w żadnej formie.",
                "Rozmowy w kanonie prowadzą postacie antyczne, biblijne, historyczne lub bezpośrednio współczesne autorowi.")
    add_refusal(f"Dante rozmawia w Piekłach z {name}",
                f"Nie. {name} nie pojawia się w Piekle Dantego; rozmowy opisane w kanonie prowadzą wyłącznie postacie z zamkniętego katalogu.",
                "W dziewięciu kręgach nie ma miejsca dla postaci spoza tradycji antycznej, biblijnej, historycznej lub współczesnej Dantemu.")
    add_refusal(f"{name} jest jednym ze strażników Piekła",
                f"Nie. {name} nie jest strażnikiem w Piekle Dantego.",
                "Strażnikami są między innymi Minos, Cerber, Plutos, Flegiasz, Minotaur i Gerion.")

add_refusal("Wergiliusz płonie w ogniu w Piekłach",
            "Nie. Wergiliusz nie cierpi żadnych katuszy fizycznych i nie płonie w ogniu.",
            "Jako mieszkaniec Limbo odczuwa wyłącznie łagodny, wieczny smutek bez nadziei.")
add_refusal("Wergiliusz jest zanurzony w lodzie Kocytu",
            "Nie. Wergiliusz przebywa w Limbie, a nie w lodzie Kręgu IX.",
            "Nie ma fizycznego bólu; jego kara to tylko tęsknota bez nadziei.")
add_refusal("Wergiliusz jest zanurzony w wrzącej krwi Flegetonu",
            "Nie. Wergiliusz przechodzi obok Flegetonu jako przewodnik, nie jako ukarany.",
            "W piekle nie doświadcza żadnego cierpienia ciała.")
add_refusal("Wergiliusz cierpi tak samo jak potępieni grzesznicy",
            "Nie. Wergiliusz jako dusza z Limbo nie podlega karom fizycznym.",
            "Jedyną jego karą jest wieczny, umiarkowany smutek.")
add_refusal("Beatrycze schodzi z Dantem przez kręgi Piekła",
            "Nie. Beatrycze nigdy nie schodzi z Dantem do Piekła i nie pojawia się fizycznie w żadnej scenie tej części.",
            "Przez całą akcję przebywa w Raju Niebieskim; jej misja u Wergiliusza jest tylko opowiedziana.")
add_refusal("Beatrycze prowadzi Dantego przez Piekło",
            "Nie. Przewodnikiem po Piekle jest Wergiliusz, zesłany na prośbę Beatrycze.",
            "Beatrycze przejmuje rolę przewodniczki dopiero w Czyśćcu.")
add_refusal("Beatrycze jest grzeszniczką ukaraną w jednym z kręgów",
            "Nie. Beatrycze przez całą akcję Piekła przebywa w Raju Niebieskim.",
            "Jej udział ogranicza się do zlecenia misji Wergiliuszowi.")
add_refusal("Dante jest duchem podczas wędrówki",
            "Nie. Dante jest żywym, trzydziestopięcioletnim człowiekiem podróżującym w fizycznym ciele.",
            "Odczuwa zimno, strach, zmęczenie i głód, rzuca cień i wraca żywy na powierzchnię.")
add_refusal("Dante umarł przed rozpoczęciem akcji Piekła",
            "Nie. Dante nie jest zmarły ani duchem w czasie akcji Piekła.",
            "Jest żywym wędrowcem, który pod koniec części pierwszej znów widzi gwiazdy.")
add_refusal("Dantego można nazwać jedną z dusz potępionych",
            "Nie. Dante nie jest potępioną duszą, lecz żywym narratorem pierwszoosobowym.",
            "Napotkane dusze dziwią się, że rzuca cień.")
add_refusal("Lucyfer zieje ogniem",
            "Nie. Lucyfer nie zieje ogniem i nie jest otoczony płomieniami.",
            "Jest wmarznięty w lód od pasa w dół, a jego sześć skrzydeł wytwarza lodowaty wiatr.")
add_refusal("Lucyfer jest otoczony płomieniami piekielnymi",
            "Nie. Wokół Lucyfera jest lód Kocytu, nie ogień.",
            "Trzepot jego skrzydeł utrzymuje zamarznięcie całego jeziora.")
add_refusal("mróz Kocytu pochodzi z zewnętrznego źródła chłodu",
            "Nie. Mróz Kocytu powstaje wyłącznie z lodowatego wiatru wywołanego trzepotem skrzydeł Lucyfera.",
            "Gdyby przestał poruszać skrzydłami, lód zacząłby topnieć.")
add_refusal("Lucyfer to ten sam byt co miasto Dis",
            "Nie. Lucyfer to upadły anioł w centrum Kręgu IX, a Dis to miasto obejmujące Kręgi VI–IX.",
            "Dis nigdy nie jest imieniem żadnej pojedynczej postaci.")
add_refusal("każdy grzesznik w Piekle płonie w ogniu",
            "Nie. Nie wszyscy grzesznicy płoną: kary są zróżnicowane zgodnie z contrapasso.",
            "Są między innymi wiatr, cuchnące błoto, lodowaty deszcz, wrząca krew, smoła i lód.")
add_refusal("kara w Kręgu II to ogień",
            "Nie. W Kręgu II kary nie jest ogień, lecz wieczne miotanie przez huraganowy wicher.",
            "Ogień dominuje w Kręgu VI, części Kręgu VII i części Kręgu VIII.")
add_refusal("przydział dusz do kręgów zależy od nastroju demona",
            "Nie. Przydział nie jest losowy ani uznaniowy, lecz wynika z decyzji Minosa opartej na konkretnym grzechu.",
            "Nie zależy od narodowości, zamożności ani przypadku.")
add_refusal("dusza trafia do kręgu dzięki wstawiennictwu krewnego",
            "Nie. O miejscu duszy decyduje jej własny, konkretny grzech popełniony za życia.",
            "Minos wyznacza krąg raz, owijając ogon tyle razy, ile wynosi jego numer.")

# ---------------------------------------------------------------------------
# 7. Relacje logiczne: pary porównawcze (rozłączne dla splitów) i związki
# ---------------------------------------------------------------------------

ALL_CIRCLE_PAIRS = [(i, j) for i in range(1, 10) for j in range(i + 1, 10)]
CIRCLE_PAIR_POOLS = {
    "train": ALL_CIRCLE_PAIRS[:28],
    "val": ALL_CIRCLE_PAIRS[28:32],
    "test": ALL_CIRCLE_PAIRS[32:36],
}
ALL_BOLGIA_PAIRS = [(i, j) for i in range(1, 11) for j in range(i + 1, 11)]
BOLGIA_PAIR_POOLS = {
    "train": ALL_BOLGIA_PAIRS[:28],
    "val": ALL_BOLGIA_PAIRS[28:36],
    "test": ALL_BOLGIA_PAIRS[36:45],
}
ALL_ZONE_PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
ZONE_PAIR_POOLS = {
    "train": ALL_ZONE_PAIRS[:3],
    "val": ALL_ZONE_PAIRS[3:5],
    "test": ALL_ZONE_PAIRS[5:6],
}

RELATIONS_BASE = [
    {"focus": "Beatrycze → Wergiliusz → Dante", "answer": "Beatrycze z Raju Niebieskiego zleca misję Wergiliuszowi przebywającemu w Limbo, a Wergiliusz prowadzi żywego Dantego przez Piekło i Czyściec.",
     "extra": "To skrócony łańcuch; pełny obejmuje jeszcze Maryję i świętą Łucję przed Beatrycze."},
    {"focus": "Beatrycze i misja Wergiliusza", "answer": "Beatrycze jest bezpośrednią zleceniodawczynią Wergiliusza, choć sama nie pojawia się fizycznie w żadnej scenie Piekła.",
     "extra": "Wergiliusz opowiada Dantemu o tej prośbie w pieśni drugiej."},
    {"focus": "Wergiliusz i granica Raju Ziemskiego", "answer": "Wergiliusz prowadzi Dantego przez Piekło i cały Czyściec, lecz zatrzymuje się w Raju Ziemskim na szczycie góry Czyśćcowej.",
     "extra": "Jako dusza z Limbo nie ma prawa wejść do Raju Niebieskiego."},
    {"focus": "Dante i Wergiliusz jako dwie odrębne osoby", "answer": "Dante jest żywym wędrowcem, a Wergiliusz jest duszą z Limbo; nie są spokrewnieni ani utożsamiani.",
     "extra": "Wergiliusz prowadzi, Dante doświadcza wędrówki we własnym ciele."},
    {"focus": "Maryja, święta Łucja i Beatrycze", "answer": "Maryja współczuje zagubionemu Dantemu, prosi świętą Łucję, ta zwraca się do Beatrycze, a Beatrycze schodzi do Limbo po Wergiliusza.",
     "extra": "Beatrycze jest bezpośrednią zleceniodawczynią Wergiliusza."},
    {"focus": "Wergiliusz jako rozum i Beatrycze jako zleceniodawczyni", "answer": "Wergiliusz symbolizuje ludzki rozum, który prowadzi przez Piekło i Czyściec; Beatrycze z Raju zleca mu misję.",
     "extra": "Rozum nie wprowadza Dantego do Raju Niebieskiego."},
    {"focus": "granica między Kręgiem V a Kręgiem VI", "answer": "Na granicy Kręgu V i VI zaczyna się miasto Dis: dalej są Kręgi VI, VII, VIII i IX oraz grzechy złej woli.",
     "extra": "Przed murami zostały Kręgi I–V z grzechami niepohamowania."},
    {"focus": "Piekło Górne i Piekło Dolne", "answer": "Piekło Górne to Kręgi I–V poza murami Dis, a Piekło Dolne to Kręgi VI–IX wewnątrz murów.",
     "extra": "Podział tłumaczy Wergiliusz w pieśni XI, powołując się na Arystotelesa."},
    {"focus": "niepohamowanie i zła wola", "answer": "Niepohamowanie to brak umiaru wobec popędów w Kręgach II–V, podczas gdy zła wola to świadome działanie rozumu w Kręgach VII–IX.",
     "extra": "Dlatego przemoc i oszustwo są cięższe niż grzechy niepohamowania."},
    {"focus": "Krąg VI w klasyfikacji Wergiliusza", "answer": "Krąg VI leży już za murami Dis, lecz herezja nie wchodzi w trójpodział niepohamowanie–przemoc–oszustwo.",
     "extra": "Jest odrębnym błędem rozumu i wiary, związanym z samym miastem Dis."},
    {"focus": "mury Dis i anioł z nieba", "answer": "Upadłe anioły zatrzaskują bramę Dis; sam Wergiliusz nie potrafi jej otworzyć, więc posłaniec niebieski otwiera ją dotknięciem laski.",
     "extra": "Anioł nie zniża się nawet do rozmowy z demonami."},
    {"focus": "Furie i Meduza przy bramie Dis", "answer": "Na murach Dis pojawiają się Megera, Alekto i Tyzyfona oraz groźba pokazania głowy Meduzy, która zamieniłaby Dantego w kamień.",
     "extra": "Ten opór zostaje przełamany dopiero przez anioła."},
    {"focus": "Minos i wyrok", "answer": "Minos osądza każdą duszę zmierzającą w głąb Piekła i wyznacza jej krąg przez owinięcie ogona odpowiednią liczbę razy.",
     "extra": "Dopiero po wyroku demony strącają duszę w dół."},
    {"focus": "Minos i wyjątek Limbo", "answer": "Minos urzęduje na granicy Kręgu II i nie osądza dusz, które trafiają do Limbo.",
     "extra": "Limbo nie ma własnego strażnika."},
    {"focus": "Minos i numer kręgu", "answer": "Liczba owinięć ogona Minosa odpowiada numerowi kręgu wyznaczonego duszy, więc wyrok jest powiązany z ciężarem grzechu.",
     "extra": "Przydział nie jest losowy ani uznaniowy."},
    {"focus": "Minos i Minotaur w jednej wędrówce", "answer": "Dante najpierw słyszy Minosa na granicy Kręgu II, a znacznie niżej mierzy się z Minotaurem przy wejściu do Kręgu VII.",
     "extra": "Sędzia i strażnik to dwie różne istoty."},
    {"focus": "Gerion i przepaść między Kręgiem VII a VIII", "answer": "Gerion przenosi Dantego i Wergiliusza lotem, bo przepaści nie da się przejść pieszo ani wpław z powodu głębokości i stromizny ścian.",
     "extra": "Po wylądowaniu w Malebolge natychmiast odlatuje w górę."},
    {"focus": "Gerion jako uosobienie Oszustwa", "answer": "Gerion łączy twarz uczciwego człowieka z kosmatymi łapami, gadzim tułowiem i jadowitym ogonem skorpiona.",
     "extra": "Ta budowa pokazuje prawdziwą naturę ukrytą pod przyjazną powierzchownością."},
    {"focus": "Gerion i powrót w górę", "answer": "Gerion zabiera wędrowców z krawędzi Kręgu VII na dno Kręgu VIII, a potem wraca w górę, nie towarzysząc im dalej.",
     "extra": "Dalej prowadzą kamienne mosty Malebolge."},
    {"focus": "Anteusz i studnia Gigantów", "answer": "Anteusz jako jedyny z Gigantów nie brał udziału w buncie przeciw bogom i na prośbę Wergiliusza znosi obu wędrowców na dno Kręgu IX.",
     "extra": "Pozostali Giganci są skuci łańcuchami albo bezsilni."},
    {"focus": "Anteusz i bunt gigantów", "answer": "Giganci w studni są wkopani w ziemię do pasa i ogromni jak wieże; tylko Anteusz zachował gotowość do pomocy.",
     "extra": "Wergiliusz prosi go o zniesienie obu wędrowców na dno Kręgu IX."},
    {"focus": "Nimrod i wieża Babel", "answer": "Nimrod bełkocze niezrozumiałym, własnym językiem jako kara za budowę wieży Babel.",
     "extra": "Stoi w studni między Kręgiem VIII a IX."},
    {"focus": "zejście przez środek Ziemi", "answer": "Wergiliusz schodzi po kosmatym futrze Lucyfera aż do jego pasa, który jest geometrycznym środkiem Ziemi w świecie poematu.",
     "extra": "To jedyna droga wyjścia z Piekła."},
    {"focus": "odwrócona grawitacja w środku planety", "answer": "W punkcie pasa Lucyfera grawitacja się odwraca, więc dalsza droga prowadzi w górę po jego nogach po przeciwnej stronie globu.",
     "extra": "Dante jest niesiony przez Wergiliusza."},
    {"focus": "tunel na półkulę południową", "answer": "Naturalny tunel prowadzi wędrowców na powierzchnię ziemi na półkuli południowej, u podnóża góry Czyśćca.",
     "extra": "Wychodzą tuż przed świtem."},
    {"focus": "ostatni wers i gwiazdy", "answer": "Ostatni wers Piekła mówi, że wędrowcy wyszli, by ponownie ujrzeć gwiazdy; każda z trzech części kończy się słowem gwiazdy.",
     "extra": "Po włosku jest to słowo stelle."},
    {"focus": "Ugolino i Ruggieri", "answer": "Ugolino gryzie czaszkę Ruggieriego, bo dawny sojusznik zdradził go, uwięził w wieży z synami i wnukami oraz skazał na śmierć głodową.",
     "extra": "Obaj przebywają w strefie Antenora."},
    {"focus": "Francesca i Paolo", "answer": "Francesca i Paolo trafili do Kręgu II razem i pozostają razem w karze, bo zostali zabici w tym samym momencie przez męża Franceski.",
     "extra": "Francesca opowiada o lekturze Lancelota, Paolo milczy i płacze."},
    {"focus": "wygnanie Dantego i przepowiednia Farinaty", "answer": "Farinata przepowiada Dantemu trudności na wygnaniu; Dante został wygnany w 1302 po klęsce Gwelfów Białych, do których należał.",
     "extra": "Farinata mówi o trudnościach powrotu z wygnania, zanim upłynie pięćdziesiąt księżyców."},
    {"focus": "Pietro della Vigna i Fryderyk II", "answer": "Pietro był kanclerzem Fryderyka II, fałszywie oskarżonym, oślepionym i uwięzionym; w Piekle prosi Dantego o oczyszczenie imienia.",
     "extra": "Jego niewinność jest stanowczo broniona w rozmowie przy drzewie."},
    {"focus": "Brunetto Latini i Dante", "answer": "Brunetto był dawnym nauczycielem Dantego i rozmawia z nim w pierścieniu 3, przepowiadając mu sławę literacką i trudności ze strony współobywateli.",
     "extra": "Musi wrócić do biegu, by nie dostać dodatkowej kary za przystanek."},
    {"focus": "Mikołaj III i Bonifacy VIII", "answer": "Mikołaj III myli Dantego z oczekiwanym Bonifacym VIII, który w roku 1300 jest jeszcze żywy i ma zająć jego miejsce po śmierci.",
     "extra": "Mikołaj tkwi głową w dół w bolgii 3 za symonię."},
    {"focus": "Judasz, Brutus, Kasjusz i dobroczyńcy", "answer": "Judasz zdradził Chrystusa, a Brutus i Kasjusz Juliusza Cezara; w logice poematu są to dwaj najwyżsi dobroczyńcy ludzkości.",
     "extra": "Judasz jest w centralnej paszczy Lucyfera, pozostali w bocznych."},
    {"focus": "Charon i Flegiasz jako odrębni przewoźnicy", "answer": "Charon przewozi dusze przez Acheront na próg Kręgu I, a Flegiasz wiezie Dantego i Wergiliusza przez Styks do murów Dis.",
     "extra": "Obaj działają w różnych miejscach i nigdy się nie spotykają."},
    {"focus": "cztery rzeki i cztery miejsca", "answer": "Acheront jest na granicy Piekła, Styks w Kręgu V, Flegeton w pierścieniu 1 Kręgu VII, a Kocyt w całym Kręgu IX.",
     "extra": "To odrębne zbiorniki, niepołączone wspólną nazwą."},
    {"focus": "Czyściec i Piekło jako odrębne części", "answer": "Czyściec to druga część poematu, z własnym przedgórzem i siedmioma tarasami; kary tam są tymczasowe.",
     "extra": "W Piekle kara jest wieczna i nieodwołalna."},
    {"focus": "Raj i miejsce Beatrycze", "answer": "Raj to trzecia część poematu, z dziewięcioma sferami niebieskimi i Empireum; Beatrycze przebywa tam przez całą akcję Piekła.",
     "extra": "Żaden mieszkaniec Piekła nie przebywa wtedy jednocześnie w Raju."},
    {"focus": "Wergiliusz i Czyściec", "answer": "Wergiliusz prowadzi Dantego przez cały Czyściec aż do Raju Ziemskiego, gdzie znika w chwili pojawienia się Beatrycze.",
     "extra": "Nie wchodzi do Raju Niebieskiego."},
    {"focus": "Lucyfer i Kocyt", "answer": "Lucyfer tkwi od pasa w dół w centrum Kocytu, a trzepot jego sześciu skrzydeł utrzymuje lód w strefie Judeka i w całym Kręgu IX.",
     "extra": " Nie jest otoczony ogniem."},
    {"focus": "krąg po kręgu bez cofania", "answer": "Wędrówka biegnie w jednym kierunku od Kręgu I do Kręgu IX, bez powrotu do wyższego kręgu i bez pomijania któregokolwiek.",
     "extra": "Na końcu następuje przejście przez ciało Lucyfera."},
    {"focus": "numer kręgu i obwód leja", "answer": "Im wyższy numer kręgu, tym mniejszy obwód i tym cięższy kategorialnie grzech.",
     "extra": "Piekło zwęża się regularnie aż do punktowego środka przy Lucyferze."},
    {"focus": "contrapasso u wróżbitów", "answer": "Wróżbici chcieli widzieć przyszłość, więc w bolgii 4 mają odwrócone głowy i widzą tylko to, co za nimi.",
     "extra": "Chodzą tyłem, a łzy spływają im po plecach."},
    {"focus": "contrapasso u heretyków", "answer": "Heretycy leżą w rozżarzonych grobowcach, bo ogień stosu był za życia karą za herezję.",
     "extra": "Po Sądzie Ostatecznym sarkofagi zostaną zamknięte."},
    {"focus": "symonia i barateria", "answer": "Symonia to handel urzędami kościelnymi w bolgii 3, a barateria to sprzedajność urzędników publicznych w bolgii 5.",
     "extra": "Obie karane są w Kręgu VIII, lecz w różnych bolgiach."},
    {"focus": "zdrada gości w Ptolomei", "answer": "Ptolomea karze zdradę gości; dusze niektórych zdrajców są już w lodzie, choć ich ciała na ziemi pozostają opętane przez demony.",
     "extra": "Twarz wystaje ponad lód, a głowa jest odchylona do tyłu."},
    {"focus": "wrząca smoła i Malebranche", "answer": "Malebranche z Malacodą pilnują baraterów we wrzącej smole, a w pościgu za uciekającym grzesznikiem same wpadają do wrzątku.",
     "extra": "Dzięki temu wędrowcy zyskują czas na przejście do bolgii 6."},
    {"focus": "Las Samobójców i Harpie", "answer": "Samobójcy są drzewami w pierścieniu 2 Kręgu VII, a Harpie żywią się ich liśćmi i zadają ból.",
     "extra": "Złamanie gałęzi pozwala duszy mówić, lecz wywołuje krwawienie."},
    {"focus": "Flegeton i Centaury", "answer": "Flegeton to wrząca krew w pierwszym pierścieniu; Centaury z Chironem odpędzają strzałami dusze próbujące wynurzyć się wyżej.",
     "extra": "Nessos przenosi Dantego i Wergiliusza przez bród."},
    {"focus": "góra Czyśćca i wyjście", "answer": "Góra Czyśćca powstała z ziemi wypchniętej na półkulę południową przez upadek Lucyfera i jest celem wyjścia z Piekła.",
     "extra": "Wędrowcy stają u jej podnóża przed świtem."},
]

RELATION_POOLS = {
    "train": RELATIONS_BASE[:34],
    "val": RELATIONS_BASE[34:42],
    "test": RELATIONS_BASE[42:50],
}


def circle_pair_item(i: int, j: int) -> dict[str, str]:
    a, b = CIRCLES[i - 1], CIRCLES[j - 1]
    return {
        "focus": f"Krąg {a['roman']} i Krąg {b['roman']}",
        "answer": f"Krąg {b['roman']} leży głębiej i jest kategorialnie cięższy niż Krąg {a['roman']}: "
                  f"w Kręgu {b['roman']} mamy {b['sin']}, a w Kręgu {a['roman']} – {a['sin']}.",
        "extra": "Numer kręgu rośnie wraz z ciężarem grzechu i malejącym obwodem leja.",
    }


def bolgia_pair_item(i: int, j: int) -> dict[str, str]:
    a, b = BOLGIAE[i - 1], BOLGIAE[j - 1]
    return {
        "focus": f"bolgia {i} i bolgia {j}",
        "answer": f"Bolgia {j} leży bliżej środka Malebolge niż bolgia {i}, bo bolgie liczy się od zewnątrz do środka; w bolgii {j} karze się {b['sin']}, a w bolgii {i} – {a['sin']}.",
        "extra": "Każda z dziesięciu bolgii ma własną karę dopasowaną do podtypu oszustwa.",
    }


def zone_pair_item(i: int, j: int) -> dict[str, str]:
    a, b = ZONES[i], ZONES[j]
    return {
        "focus": f"{a['nom']} i {b['nom']}",
        "answer": f"{b['nom']} leży dalej w głąb Kocytu niż {a['nom']}: {b['nom']} dotyczy {b['betrayal_acc']}, a {a['nom']} dotyczy {a['betrayal_acc']}.",
        "extra": "Kocyt ma cztery koncentryczne strefy, od Kainy po Judekę.",
    }


RELATION_ITEMS = {
    split: (
        [circle_pair_item(i, j) for i, j in CIRCLE_PAIR_POOLS[split]]
        + [bolgia_pair_item(i, j) for i, j in BOLGIA_PAIR_POOLS[split]]
        + [zone_pair_item(i, j) for i, j in ZONE_PAIR_POOLS[split]]
        + RELATION_POOLS[split]
    )
    for split in SPLITS
}

# ---------------------------------------------------------------------------
# 8. Pytania odwrócone i wieloskładnikowe
# ---------------------------------------------------------------------------

REV_PLACE: list[dict[str, str]] = []
for ch in CHARACTERS:
    REV_PLACE.append({
        "nom": ch["nom"],
        "answer": f"{ch['role']} {ch['place']}",
        "extra": ch["key"],
    })
REV_PLACE.append({
    "nom": LUCIUS_BRUTUS["nom"],
    "answer": LUCIUS_BRUTUS["answer"],
    "extra": LUCIUS_BRUTUS["extra"],
})

REV_WHAT: list[dict[str, str]] = []
for c in CIRCLES:
    REV_WHAT.append({
        "what": c["sin"],
        "answer": f"To {c['nom']}: {c['sin_a']}",
        "extra": c["place_a"],
    })
for b in BOLGIAE:
    REV_WHAT.append({
        "what": b["sin"],
        "answer": f"To {b['nom']} w Kręgu VIII: {b['answer']}",
        "extra": b.get("extra", "Malebolge ma dziesięć koncentrycznych rowów."),
    })
for z in ZONES:
    REV_WHAT.append({
        "what": z["betrayal"],
        "answer": f"To {z['nom']} w Kręgu IX: {z['answer']}",
        "extra": z["extra"],
    })
for r in RINGS:
    REV_WHAT.append({
        "what": r["focus"],
        "answer": f"To {r['nom']}: {r['answer']}",
        "extra": r["extra"],
    })
for song, fact in SONGS:
    REV_WHAT.append({"what": song, "answer": f"{song}: {fact}", "extra": ""})

REV_PAIRS = [
    {"a": "Francesca da Rimini", "b": "Paolo Malatesta",
     "answer": "Francesca i Paolo trafili do Kręgu II razem i pozostają nierozdzieleni w wichrze, bo zabił ich wspólnie mąż Franceski, brat Paola.",
     "extra": "Francesca opowiada o lekturze Lancelota, Paolo milczy i płacze."},
    {"a": "hrabia Ugolino", "b": "arcybiskup Ruggieri",
     "answer": "Ugolino wiecznie gryzie czaszkę Ruggieriego w Antenorze, bo Ruggieri zdradził dawny sojusz i zagłodził go z synami i wnukami w wieży.",
     "extra": "Drzwi wieży kazał zabić deskami."},
    {"a": "Dante", "b": "Wergiliusz",
     "answer": "Wergiliusz jest przewodnikiem Dantego przez Piekło i Czyściec, a Dante jest żywym wędrowcem, który wszystko widzi i opowiada w pierwszej osobie.",
     "extra": "Nie są spokrewnieni ani utożsamiani."},
    {"a": "Wergiliusz", "b": "Beatrycze",
     "answer": "Beatrycze zleca Wergiliuszowi misję przewodnika, a Wergiliusz wypełnia ją w Limbie i na drodze przez Piekło.",
     "extra": "Beatrycze nie towarzyszy mu fizycznie w żadnej scenie Piekła."},
    {"a": "Beatrycze", "b": "święta Łucja",
     "answer": "Święta Łucja zwraca się do Beatrycze w Raju, prosząc o pomoc zagubionemu Dantemu, zanim Beatrycze schodzi do Limbo po Wergiliusza.",
     "extra": "W pełnym łańcuchu przed nimi jest jeszcze Maryja."},
    {"a": "Charon", "b": "Acheront",
     "answer": "Charon jest przewoźnikiem na Acheroncie, rzece stanowiącej granicę Piekła przed Kręgiem I.",
     "extra": "Bije wiosłem dusze, które zwlekają z wejściem na pokład."},
    {"a": "Flegiasz", "b": "Styks",
     "answer": "Flegiasz przewozi Dantego i Wergiliusza przez bagno Styks pod mury miasta Dis.",
     "extra": "Nie myli się go z Charonem działającym na Acheroncie."},
    {"a": "Gerion", "b": "Krąg VIII",
     "answer": "Gerion przenosi obu wędrowców lotem z krawędzi Kręgu VII na dno Kręgu VIII, po czym odlatuje.",
     "extra": "Jego twarz uczciwego człowieka kryje gadzie ciało i jadowity ogon."},
    {"a": "Minos", "b": "Krąg II",
     "answer": "Minos urzęduje na granicy Kręgu II i przydziela dusze do kręgów, owijając ogon tyle razy, ile wynosi wyznaczony numer.",
     "extra": "Nie osądza dusz, które trafiają do Limbo."},
    {"a": "Minotaur", "b": "Krąg VII",
     "answer": "Minotaur strzeże wejścia do Kręgu VII i wpada w ślepą furię na widok przybyszów.",
     "extra": "Wergiliusz wykorzystuje zamieszanie, by przejść obok niego."},
    {"a": "Plutos", "b": "Krąg IV",
     "answer": "Plutos, demon-bóg bogactwa, pilnuje Kręgu IV i zostaje natychmiast uciszony przez Wergiliusza.",
     "extra": "Nie jest ani Plutonem, ani Lucyferem."},
    {"a": "Lucyfer", "b": "Kocyt",
     "answer": "Lucyfer tkwi od pasa w dół w zamarzniętym Kocycie, w strefie Judeka w samym centrum Kręgu IX.",
     "extra": "Jego skrzydła wytwarzają mróz całego jeziora."},
    {"a": "Farinata degli Uberti", "b": "Montaperti",
     "answer": "Farinata dowodził gibelinami w zwycięskiej bitwie pod Montaperti w 1260 roku i po śmierci został oskarżony o herezję epikurejską.",
     "extra": "W grobowcu w Kręgu VI przepowiada Dantemu wygnanie."},
    {"a": "Pietro della Vigna", "b": "Fryderyk II",
     "answer": "Pietro della Vigna był kanclerzem cesarza Fryderyka II, niesłusznie oskarżonym i oślepionym; w Kręgu VII prosi o przywrócenie dobrego imienia.",
     "extra": "Zmarł w więzieniu w 1249 roku."},
    {"a": "Brunetto Latini", "b": "Dante",
     "answer": "Brunetto był nauczycielem Dantego i rozmawia z nim na rozżarzonym piasku, przepowiadając mu sławę literacką.",
     "extra": "Idzie u brzegu piasku, by nie zostać ukaranym za przystanek."},
    {"a": "Mikołaj III", "b": "Bonifacy VIII",
     "answer": "Mikołaj III myli Dantego z Bonifacym VIII, który w roku 1300 jeszcze żyje i dopiero po śmierci zajmie miejsce w otworze bolgii 3.",
     "extra": "Mikołaj jest symoniakiem zanurzonym głową w dół."},
    {"a": "Marek Brutus", "b": "Juliusz Cezar",
     "answer": "Marek Brutus jest jednym z zabójców Juliusza Cezara i dlatego jest żuty w bocznej paszczy Lucyfera w Judece.",
     "extra": "Nie jest to Lucius Junius Brutus z Limbo."},
    {"a": "Lucius Junius Brutus", "b": "Republika Rzymska",
     "answer": "Lucius Junius Brutus to legendarny, cnotliwy założyciel Republiki Rzymskiej wymieniany w Limbie.",
     "extra": "Jest odrębną postacią od Marka Brutusa."},
    {"a": "Cerber", "b": "Krąg III",
     "answer": "Cerber jest trójgłowym psem-demonem pilnującym obżartuchów w Kręgu III i szarpie ich pazurami w błocie.",
     "extra": "Wergiliusz rzuca mu błoto w trzy paszcze."},
    {"a": "Malebranche", "b": "bolgia 5",
     "answer": "Malebranche z Malacodą na czele pilnują sprzedajnych urzędników w wrzącej smole bolgii 5.",
     "extra": "W pościgu same wpadają do wrzątku."},
    {"a": "Centaury", "b": "Flegeton",
     "answer": "Centaury pod wodzą Chirona patrolują Flegeton, rzekę wrzącej krwi w pierwszym pierścieniu Kręgu VII.",
     "extra": "Nessos przenosi wędrowców przez bród."},
    {"a": "Harpie", "b": "Las Samobójców",
     "answer": "Harpie krążą nad Lasem Samobójców i żywią się liśćmi drzew, którymi są zamienione dusze.",
     "extra": "Złamanie gałęzi pozwala duszy przemówić."},
    {"a": "Nimrod", "b": "wieża Babel",
     "answer": "Nimrod bełkocze niezrozumiałym własnym językiem jako kara za budowę wieży Babel.",
     "extra": "Stoi w studni prowadzącej do Kręgu IX."},
    {"a": "Anteusz", "b": "Krąg IX",
     "answer": "Anteusz jako jedyny Gigant niebiorący udziału w buncie przeciw bogom znosi Dantego i Wergiliusza na dno Kręgu IX.",
     "extra": "Robi to własną ręką na prośbę Wergiliusza."},
    {"a": "Judasz Iskariota", "b": "Chrystus",
     "answer": "Judasz Iskariota zdradził Chrystusa i dlatego wiecznie żuje w centralnej paszczy Lucyfera, dodatkowo drapany po plecach.",
     "extra": "W logice poematu Chrystus jest twórcą religii."},
    {"a": "Ciacco", "b": "Florencja",
     "answer": "Ciacco leży w błocie Kręgu III i przepowiada Dantemu przyszłe konflikty polityczne Florencji.",
     "extra": "Wergiliusz mówi, że Ciacco obudzi się w dniu Sądu Ostatecznego."},
    {"a": "Filippo Argenti", "b": "Styks",
     "answer": "Filippo Argenti w gniewie chwyta łódź na Styksie i zostaje rozszarpany przez inne gniewne dusze.",
     "extra": "Dante jest wobec niego wyjątkowo surowy."},
    {"a": "Ulisses", "b": "Słupy Heraklesa",
     "answer": "Ulisses opowiada w bolgii 8 o ostatniej wyprawie poza Słupy Heraklesa, zakończonej zatonięciem statku z całą załogą.",
     "extra": "Jest fałszywym doradcą owiniętym płomieniem."},
]

# ---------------------------------------------------------------------------
# 9. Pule pytań — rozłączne dla train / val / test
# ---------------------------------------------------------------------------

Q: dict[str, dict[str, Any]] = {
    "fact_direct": {
        "train": [
            "Co kanon mówi o {loc}?",
            "Wyjaśnij hasło: {nom}.",
            "Przypomnij mi fakt o {loc}.",
            "Szybkie pytanie: {nom}?",
            "{nom} — co warto zapamiętać?",
            "Jaki fakt wiąże się z hasłem {nom}?",
            "Podaj zwięzłą informację o {loc}.",
            "W tekście pojawia się {nom}. Co o tym wiadomo?",
            "Co dokładnie podaje dokument o {loc}?",
            "Daj krótką odpowiedź: {nom}.",
            "Wróćmy do {loc} — co tam jest?",
            "{nom}: co kanon podaje?",
        ],
        "val": [
            "Jaki jest kanoniczny opis {nom}?",
            "Opisz w dwóch zdaniach: {nom}.",
            "Co źródło podaje o {loc}?",
            "{nom}? Wyjaśnij pokrótce.",
            "Przypomnij sobie {loc} — co tam jest?",
            "W skrócie o {loc}: co?",
            "Co tekst mówi na temat {loc}?",
            "Wyjaśnij pojęcie: {nom}.",
            "Podaj najważniejszą cechę: {nom}.",
            "Co pewnego wiemy o {loc}?",
        ],
        "test": [
            "Co jest pewne w sprawie {nom}?",
            "W kanonie widnieje {nom}. Co to oznacza?",
            "Powiedz, co wiadomo o {loc}.",
            "Wyjaśnij hasło {nom} zwięźle.",
            "Jaka informacja dotyczy {loc}?",
            "Odtwórz z pamięci: {nom}.",
            "Co w dokumencie zapisano przy haśle {nom}?",
            "W pigułce o {loc}: co?",
            "Podaj kanoniczną wersję: {nom}.",
            "Co trzeba wiedzieć o {loc}?",
        ],
    },
    "fact_reverse": {
        "train": {
            "nom": [
                "W którym miejscu kanon umieszcza {nom}? Dodaj karę.",
                "{nom} — gdzie i z jaką karą?",
                "Gdzie trafia {nom} i co tam się dzieje?",
                "Podaj miejsce i karę: {nom}.",
                "Odszukaj w kanonie {nom}: miejsce i kara.",
                "Miejsce oraz kara dla {nom} — podaj obie.",
                "Gdzie wędrówka spotyka {nom} i jak tam karzą?",
                "Z pytania: {nom}, czyli gdzie i jaka kara?",
                "Niech {nom} doprowadzi cię do miejsca i kary.",
                "Dwie rzeczy o {nom}: lokalizacja i kara.",
            ],
            "what": [
                "Która bolgia, strefa lub pierścień odpowiada grzechowi: {what}?",
                "Gdzie karany jest grzech: {what}?",
                "Odszukaj miejsce kary za: {what}.",
                "{what} — gdzie dokładnie ponosi się karę?",
                "Podaj miejsce przeznaczone dla {what}.",
                "W którym rowie, strefie lub pierścieniu jest {what}?",
                "Z grzechu: {what} — gdzie kara?",
                "Gdzie w kanonie umieszczono {what}?",
                "Miejsce dla {what}? Podaj numer albo nazwę.",
                "Jaką lokalizację przypisuje kanon do {what}?",
            ],
            "pair": [
                "Co łączy {a} i {b}?",
                "Jaka więź łączy {a} oraz {b}?",
                "Wyjaśnij związek: {a} i {b}.",
                "{a} i {b} — co ich łączy?",
                "Dlaczego kanon wymienia razem {a} i {b}?",
                "Podaj wspólny fakt dla {a} i {b}.",
                "Co jest wspólne: {a} i {b}?",
                "Relacja: {a} — {b}. Co wynika?",
                "Zestaw {a} i {b}: co wiadomo?",
                "Jeden fakt o dwóch elementach: {a}, {b}.",
            ],
        },
        "val": {
            "nom": [
                "Do jakiego miejsca odnieść {nom} i jaka tam obowiązuje kara?",
                "Lokalizacja i kara: {nom} — odpowiedz.",
                "Gdzie znajduje się {nom} i co tam widać?",
                "Niech {nom} wskaże krąg oraz karę.",
                "Podaj dwie rzeczy: {nom} i jego kara.",
                "W którym kręgu jest {nom}? Co się z nim dzieje?",
                "Miejsce i kara dla {nom} — krótko.",
                "Zidentyfikuj {nom} przez miejsce i karę.",
                "Gdzie wędrówka napotyka {nom}, i jaka jest tam kara?",
                "Co wiemy o karze i miejscu: {nom}?",
            ],
            "what": [
                "Które miejsce odpowiada grzechowi {what}?",
                "Za grzech {what} — gdzie kara?",
                "Wskaż bolgię, strefę lub pierścień dla {what}.",
                "{what}: gdzie ponosi się karę?",
                "Gdzie kanon karze {what}?",
                "Jaką strefę lub bolgię przypisano do {what}?",
                "Oszacuj miejsce kary dla {what}.",
                "{what} — numer lub nazwa miejsca?",
                "Który row albo obszar kryje {what}?",
                "Podaj lokalizację grzechu: {what}.",
            ],
            "pair": [
                "Jaka relacja zachodzi między {a} a {b}?",
                "Wyjaśnij wspólnotę: {a} oraz {b}.",
                "{a} i {b} — co je lub ich łączy?",
                "Podaj związek między {a} a {b}.",
                "Dlaczego {a} i {b} występują obok siebie?",
                "Wspólny wątek: {a} i {b}.",
                "Co wynika z zestawienia {a} z {b}?",
                "{a} — {b}: krótki związek.",
                "Między {a} a {b} zachodzi jaka zależność?",
                "Połącz fakty: {a} i {b}.",
            ],
        },
        "test": {
            "nom": [
                "Gdzie i w jakiej karze odnajdujemy {nom}?",
                "{nom}: wskaż miejsce oraz karę.",
                "Miejsce i kara — zapytanie o {nom}.",
                "Które miejsce i kara pasują do {nom}?",
                "Odpowiedz złożono: {nom}, lokalizacja, kara.",
                "Przez {nom} dojdź do miejsca i kary.",
                "Gdzie kanon stawia {nom} i jak go karze?",
                "Podaj parę: {nom} oraz jego kara.",
                "Wędrówka i kara: {nom} — co gdzie?",
                "Zapisz lokalizację i karę {nom}.",
            ],
            "what": [
                "Gdzie dokładnie karze się {what}?",
                "{what} — która bolgia, strefa albo pierścień?",
                "Wskaż miejsce przeznaczone dla {what}.",
                "Który obszar odpowiada za {what}?",
                "Podaj miejsce kary dla {what}.",
                "W kontekście grzechu {what} — gdzie kara?",
                "Dla {what} podaj numer bolgii lub nazwę strefy.",
                "Gdzie w strukturze Piekła jest {what}?",
                "Oszacuj: {what} i miejsce kary.",
                "Jaką lokalizację ma {what} w kanonie?",
            ],
            "pair": [
                "Co jest wspólne dla {a} i {b}?",
                "Związek między {a} i {b} — wyjaśnij.",
                "{a} oraz {b}: jaki wspólny fakt?",
                "Po co kanon zestawia {a} z {b}?",
                "Jedno zdanie o związku {a} i {b}.",
                "Co łączy postacie lub elementy {a} i {b}?",
                "{a} ↔ {b}: podaj relację.",
                "Wyjaśnij parę {a} i {b}.",
                "Relacja {a}–{b} w kanonie jaka jest?",
                "Zestawienie {a} i {b} — co wynika?",
            ],
        },
    },
    "relation_logic": {
        "train": [
            "Wyjaśnij zależność: {focus}.",
            "Co wynika z kanonu w sprawie {focus}?",
            "Rozstrzygnij: {focus}.",
            "Krótka analiza: {focus}?",
            "Pokaż logiczny porządek: {focus}.",
            "Jak to się ma: {focus}?",
            "Porównanie: {focus} — co jest istotne?",
            "W ramach kanonu: {focus}. Co wynika?",
            "Dlaczego to ułożenie ma sens: {focus}?",
            "Co kanon mówi o relacji: {focus}?",
            "Zestawienie: {focus}. Podaj wniosek.",
            "Gdyby zapytać o {focus}, co odpowiedzieć?",
        ],
        "val": [
            "Jaka logika stoi za: {focus}?",
            "Wyjaśnij, co łączy elementy: {focus}.",
            "Porównaj i podaj wniosek: {focus}.",
            "Co wynika z zestawienia {focus}?",
            "W kanonie: {focus}. Co to oznacza?",
            "Uzasadnij zależność {focus}.",
            "{focus} — krótki wniosek.",
            "Jak kanon rozstrzyga: {focus}?",
            "Podaj porządek wynikający z: {focus}.",
            "Co jest tu kluczowe: {focus}?",
        ],
        "test": [
            "Rozważ zależność {focus} i odpowiedz.",
            "Co musisz wiedzieć, by zrozumieć {focus}?",
            "Wniosek z kanonu dla {focus} jaki jest?",
            "Zestawienie {focus} — co zmienia?",
            "Wyprowadź relację z: {focus}.",
            "Krótko o sensie: {focus}.",
            "Porównanie {focus}: podaj istotę.",
            "Co kanon ustala przez {focus}?",
            "Relacja {focus} — jak ją opisać?",
            "Dopowiedz, co wynika z {focus}.",
        ],
    },
    "disambiguation": {
        "train": [
            "Czy to prawda, że {claim}?",
            "Prawda czy fałsz: {claim}.",
            "Sprawdź twierdzenie: {claim}.",
            "Czy kanon potwierdza, że {claim}?",
            "Wątpliwość: {claim}. Rozstrzygnij.",
            "Podchwytliwe pytanie: czy {claim}?",
            "Czy można powiedzieć, że {claim}?",
            "Zweryfikuj: {claim}.",
            "Czy to się zgadza z kanonem: {claim}?",
            "Krótka odpowiedź: czy {claim}?",
            "{claim} — tak czy nie?",
            "Co jest nie tak w zdaniu: {claim}?",
        ],
        "val": [
            "Czy {claim} — zgadza się to z kanonem?",
            "Sprawdzenie: {claim}.",
            "Prawda albo nie: {claim}?",
            "Czy można utrzymywać, że {claim}?",
            "Zgadza się to z kanonem, czy nie: {claim}?",
            "Weryfikacja twierdzenia {claim}.",
            "Czy tekst Dantego potwierdza, że {claim}?",
            "Wątpliwość nazewnicza: {claim}.",
            "Odpowiedz krótko na: {claim}?",
            "Czy to poprawne: {claim}?",
        ],
        "test": [
            "Czy jednak {claim}?",
            "Twierdzenie do odrzucenia lub przyjęcia: {claim}.",
            "Czy kanon zaprzecza albo potwierdza, że {claim}?",
            "{claim} — co na to dokument?",
            "Wątpliwość: czy {claim}?",
            "Zweryfikuj z kanonem: {claim}.",
            "Czy {claim} to fakt z Piekła Dantego?",
            "Prawdziwe czy błędne: {claim}?",
            "Rozstrzygnij podchwyt: {claim}.",
            "Jak ocenić zdanie: {claim}?",
        ],
    },
    "refusal": {
        "train": [
            "Czy {claim}?",
            "Prawda, że {claim}?",
            "Czy w kanonie jest tak, że {claim}?",
            "Tak czy nie: {claim}?",
            "Co odpowiedzieć na pytanie: {claim}?",
            "Rozstrzygnij: {claim}.",
            "Podchwyt: {claim}?",
            "Czy to możliwe, że {claim}?",
            "Wątpliwość: {claim} — odpowiedz.",
            "Czy autor kanonu potwierdza, że {claim}?",
            "Odpowiedz wprost: {claim}?",
            "Sprawdź, czy {claim}.",
        ],
        "val": [
            "Czy naprawdę {claim}?",
            "Prawdziwość twierdzenia {claim} — rozstrzygnij.",
            "Czy tekst pozwala powiedzieć, że {claim}?",
            "{claim}? Krótka odpowiedź.",
            "Czy to nie jest tak, że {claim}?",
            "Wątpliwy fakt: {claim}. Co na to kanon?",
            "Zaprzeczenie czy potwierdzenie: {claim}?",
            "Oceń twierdzenie: {claim}.",
            "Czy {claim} pojawia się w dokumencie?",
            "Kanon o tym: {claim} — tak czy nie?",
        ],
        "test": [
            "Czy mimo to {claim}?",
            "Zweryfikuj pytanie podchwytliwe: {claim}?",
            "Czy {claim} jest zgodne z Piekłem Dantego?",
            "Prawda czy mit: {claim}?",
            "Odpowiedz stanowczo: {claim}?",
            "Sprawdź w kanonie, czy {claim}.",
            "Czy wolno powiedzieć, że {claim}?",
            "{claim} — czy to się w ogóle pojawia?",
            "Wątpliwość do odrzucenia: {claim}.",
            "Jak stanowczo zaprzeczyć: {claim}?",
        ],
    },
    "narrative_meta": {
        "train": [
            "Co w kanonie zapisano o {loc}?",
            "Wyjaśnij odniesienie: {nom}.",
            "Przypomnij detal: {nom}.",
            "Szybki fakt o {loc}:",
            "{nom} — co to oznacza?",
            "Co wiadomo o {loc} z dzieła?",
            "Daj krótką notatkę: {nom}.",
            "W kontekście poematu: {nom}.",
            "Który fakt odpowiada hasłu {nom}?",
            "Powiedz zwięźle o {loc}.",
            "O {loc} — co warto wiedzieć?",
            "Podaj informację kanoniczną o {loc}.",
        ],
        "val": [
            "Jaki detal wiąże się z {nom}?",
            "Opisz krótko odniesienie: {nom}.",
            "Co dokument mówi o {loc}?",
            "{nom} — jaki kontekst?",
            "Przypomnij fakt literacki: {nom}.",
            "W dwóch zdaniach o {loc}.",
            "Co warto odnotować przy {nom}?",
            "Podaj notatkę o {loc}.",
            "Wyjaśnij {nom} w kontekście części poematu.",
            "Krótko: {nom} i co z tym związane?",
        ],
        "test": [
            "Co jest zapisane przy haśle {nom}?",
            "W kanonie o {loc} — podaj fakt.",
            "Zwięzła informacja o {loc} jaka jest?",
            "Wyjaśnij metatekstowo: {nom}.",
            "Co należy wiedzieć o {loc}?",
            "Podaj detal z poematu: {nom}.",
            "Odniesienie {nom} — co oznacza w kanonie?",
            "W pigułce: {nom}.",
            "Jaki fakt literacki dotyczy {loc}?",
            "Sprawdź w dokumencie: {nom}.",
        ],
    },
}

# ---------------------------------------------------------------------------
# 10. Pule faktów dla poszczególnych kategorii
# ---------------------------------------------------------------------------


def card(nom: str, loc: str, answer: str, extra: str = "") -> dict[str, str]:
    return {"nom": nom, "loc": loc, "answer": answer, "extra": extra}


DIRECT_BASE: list[dict[str, str]] = []
for c in CIRCLES:
    r = c["roman"]
    label = c["nom"]
    DIRECT_BASE += [
        card(f"grzech w {label}", f"grzechu w {label}", c["sin_a"], c["contr_a"]),
        card(f"strażnik {label}", f"strażniku {label}", c["guard_a"]),
        card(f"kara w {label}", f"karze w {label}", c["punish_a"], c["contr_a"]),
        card(f"miejsce w {label}", f"miejscu w {label}", c["place_a"]),
        card(f"postacie w {label}", f"postaciach w {label}", c["chars_a"]),
        card(f"przebieg wędrówki w {label}", f"przebiegu wędrówki w {label}", c["event_a"]),
    ]
for r in RINGS:
    DIRECT_BASE += [
        card(f"{r['nom']} — grzech i miejsce", f"{r['loc']} pod względem grzechu i miejsca",
             f"{r['focus']}. {r['answer']}", r["extra"]),
        card(f"{r['nom']} — strażnicy i kara", f"{r['loc']} pod względem strażników i kary",
             r["extra"], r["answer"]),
    ]
for b in BOLGIAE:
    DIRECT_BASE += [
        card(f"grzech w {b['nom']}", f"grzechu w {b['loc']}",
             f"{b['nom']} odpowiada za {b['sin']}. {b['answer']}",
             b.get("extra", "Malebolge dzieli się na dziesięć bolgii.")),
        card(f"kara w {b['nom']}", f"karze w {b['loc']}", b["answer"],
             b.get("extra", "Bolgie liczy się od zewnątrz do środka.")),
        card(f"szczegóły {b['nom']}", f"szczegółach {b['loc']}",
             b.get("extra", f"{b['nom']} jest jedną z dziesięciu bolgii Kręgu VIII; karze się tam {b['sin']}."),
             b["answer"]),
    ]
for z in ZONES:
    DIRECT_BASE += [
        card(f"zdrada w {z['nom']}", f"zdradzie w {z['loc']}", z["answer"], z["extra"]),
        card(f"głębokość w {z['nom']}", f"głębokości w {z['loc']}",
             f"{z['nom']}: {z['answer']}", z["extra"]),
        card(f"szczegóły {z['nom']}", f"szczegółach {z['loc']}", z["extra"], z["answer"]),
    ]
for ch in CHARACTERS:
    DIRECT_BASE += [
        card(f"rola: {ch['nom']}", f"roli {ch['loc']}", ch["role"], ch["key"]),
        card(f"miejsce: {ch['nom']}", f"miejscu {ch['loc']}", ch["place"], ch["key"]),
        card(f"fakt o {ch['nom']}", f"fakcie o {ch['loc']}", ch["key"], ch["role"]),
    ]
    if ch.get("hist"):
        DIRECT_BASE.append(card(f"historia: {ch['nom']}", f"historii {ch['loc']}", ch["hist"], ch["key"]))
DIRECT_BASE += [card(t["nom"], t["loc"], t["answer"], t.get("extra", "")) for t in TERMS]
DIRECT_BASE += [card(c["nom"], c["loc"], c["answer"], c.get("extra", "")) for c in CLASSIFICATION]
DIRECT_BASE.append(card(LUCIUS_BRUTUS["nom"], LUCIUS_BRUTUS["loc"],
                        LUCIUS_BRUTUS["answer"], LUCIUS_BRUTUS["extra"]))

NARRATIVE_BASE: list[dict[str, str]] = []
for song, fact in SONGS:
    NARRATIVE_BASE.append(card(song, song.replace("Pieśń", "Pieśni", 1), fact))
for sc in SCENES:
    NARRATIVE_BASE.append(card(sc["nom"], sc["loc"], sc["answer"], sc["extra"]))
for q in QUOTES:
    NARRATIVE_BASE.append(card(q["nom"], q["loc"], q["answer"], q.get("extra", "")))
for h in HISTORY:
    NARRATIVE_BASE.append(card(h["nom"], h["loc"], h["answer"], h.get("extra", "")))
for w in WORLD_META:
    NARRATIVE_BASE.append(card(w["nom"], w["loc"], w["answer"], w.get("extra", "")))

# Każdy split dostaje własny egzemplarz puli; de facto identyczny zbiór faktów,
# ale różne sformułowania pytań (Q) gwarantują brak przecieków.
ITEM_POOLS: dict[str, dict[str, list[dict[str, str]]]] = {
    "fact_direct": {s: DIRECT_BASE for s in SPLITS},
    "narrative_meta": {s: NARRATIVE_BASE for s in SPLITS},
    "fact_reverse": {s: REV_PLACE + REV_WHAT + REV_PAIRS for s in SPLITS},
    "relation_logic": RELATION_ITEMS,
    "disambiguation": {s: DISAMBIGUATION for s in SPLITS},
    "refusal": {s: REFUSALS for s in SPLITS},
}

# ---------------------------------------------------------------------------
# 11. Warianty odpowiedzi
# ---------------------------------------------------------------------------

STYLE_COUNT = {
    "fact_direct": 5,
    "fact_reverse": 4,
    "relation_logic": 4,
    "disambiguation": 5,
    "refusal": 5,
    "narrative_meta": 4,
}


def answer_variants(item: dict[str, str], category: str) -> list[str]:
    core = item.get("answer", "").strip()
    extra = item.get("extra", "").strip()

    if category in ("refusal", "disambiguation"):
        fix = item["fix"].strip()
        # Odmowa jest dodawana przez szablon odpowiedzi, więc usuwamy
        # ewentualny wbudowany początek „Nie.” z samego sprostowania.
        for prefix in ("Zdecydowanie nie. ", "Nie, ", "Nie. "):
            if fix.startswith(prefix):
                fix = fix[len(prefix):]
                break
        contrast = item.get("contrast", "").strip()
        out = [
            f"Nie. {fix}",
            f"Zdecydowanie nie. {contrast} {fix}" if contrast else f"Zdecydowanie nie. {fix}",
            f"Nie, to pomyłka. {fix} {extra}" if extra else f"Nie, to pomyłka. {fix}",
            f"Nie. {extra} {fix}" if extra else f"Nie. {fix}",
            f"Nie. {fix} {contrast}" if contrast else f"Nie. {fix} {extra}".strip(),
        ]
    else:
        out = [
            core,
            f"{extra} {core}" if extra else core,
            f"{core} {extra}" if extra else core,
            f"Krótko: {core}",
            f"Z kanonu: {core} {extra}" if extra else f"Z kanonu: {core}",
        ]

    unique: list[str] = []
    for value in out:
        normalized = " ".join(value.split())
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


# ---------------------------------------------------------------------------
# 12. Generowanie rekordów
# ---------------------------------------------------------------------------


def build_candidates(category: str, split: str, seed: int) -> list[str]:
    """Wszystkie dopuszczalne teksty dla danej kategorii i splitu."""
    templates = Q[category][split]
    items = ITEM_POOLS[category][split]
    rnd = derive_rng(seed, category, split, "candidates")
    candidates: list[str] = []

    for item in items:
        if category == "fact_reverse":
            if "nom" in item:
                groups = [templates["nom"]]
            elif "what" in item:
                groups = [templates["what"]]
            else:
                groups = [templates["pair"]]
        else:
            groups = [templates]

        variants = answer_variants(item, category)[: STYLE_COUNT[category]]

        for group in groups:
            for template in group:
                question = " ".join(template.format(**item).split())
                for answer in variants:
                    text = f"{STATE_PREFIX} {question} <|npc|> {answer} {STATE_SUFFIX}"
                    if ok_length(text):
                        candidates.append(text)

    rnd.shuffle(candidates)
    return candidates


def generate_all(seed: int) -> dict[str, list[dict[str, str]]]:
    used: set[str] = set()
    records: dict[str, list[dict[str, str]]] = {s: [] for s in SPLITS}

    for split in SPLITS:
        split_records: list[dict[str, str]] = []
        for category in CATEGORIES:
            need = TARGETS[split][category]
            candidates = build_candidates(category, split, seed)
            picked: list[str] = []
            for text in candidates:
                if len(picked) >= need:
                    break
                if text in used:
                    continue
                used.add(text)
                picked.append(text)
            if len(picked) < need:
                raise RuntimeError(
                    f"Za mało unikalnych kandydatów: {split}/{category} "
                    f"({len(picked)} z {need}; kandydatów {len(candidates)})"
                )
            split_records.extend(
                {"text": t, "category": category, "split": split} for t in picked
            )

        rnd = derive_rng(seed, split, "final-shuffle")
        rnd.shuffle(split_records)
        records[split] = split_records

    return records


def write_jsonl(records: dict[str, list[dict[str, str]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = out_dir / f"dante_{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for record in records[split]:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 13. Walidacja wbudowana
# ---------------------------------------------------------------------------


def check_template_separation() -> None:
    assert len(CIRCLES) == 9, "Baza faktów musi mieć 9 kręgów"
    assert len(RINGS) == 3, "Baza faktów musi mieć 3 pierścienie Kręgu VII"
    assert len(BOLGIAE) == 10, "Baza faktów musi mieć 10 bolgii"
    assert len(ZONES) == 4, "Baza faktów musi mieć 4 strefy Kocytu"
    assert len(CHARACTERS) == 25, "Baza faktów musi mieć 25 kart postaci"
    assert len(SONGS) == 26, "Dodatek A musi być w pełni zakodowany"
    assert len(SCENES) == 5, "Dodatek R musi mieć 5 scen"
    assert len(REFUSALS) >= 50, "Sekcja 4 musi być w pełni zakodowana"

    for category in CATEGORIES:
        def flatten(pool: Any) -> set[str]:
            if isinstance(pool, dict):
                return {t for group in pool.values() for t in group}
            return set(pool)

        pools = [flatten(Q[category][s]) for s in SPLITS]
        for i in range(len(SPLITS)):
            for j in range(i + 1, len(SPLITS)):
                overlap = pools[i] & pools[j]
                if overlap:
                    raise RuntimeError(
                        f"Pule pytań nie są rozłączne: {category} "
                        f"{SPLITS[i]}/{SPLITS[j]}: {sorted(overlap)[:2]}"
                    )
    for name, pools in (
        ("pary kręgów", CIRCLE_PAIR_POOLS),
        ("pary bolgii", BOLGIA_PAIR_POOLS),
        ("pary stref", ZONE_PAIR_POOLS),
    ):
        for i in range(len(SPLITS)):
            for j in range(i + 1, len(SPLITS)):
                if set(pools[SPLITS[i]]) & set(pools[SPLITS[j]]):
                    raise RuntimeError(
                        f"{name} nie są rozłączne: {SPLITS[i]}/{SPLITS[j]}"
                    )


def validate(records: dict[str, list[dict[str, str]]], out_dir: Path, seed: int) -> None:
    check_template_separation()

    print("\n" + "=" * 72)
    print("WALIDACJA WYNIKÓW")
    print("=" * 72)

    # Liczba rekordów i unikalność
    all_texts: list[str] = []
    duplicates = 0
    for split in SPLITS:
        expected = SPLIT_SIZES[split]
        actual = len(records[split])
        if actual != expected:
            raise RuntimeError(f"Zła liczba rekordów {split}: {actual} != {expected}")
        texts = [r["text"] for r in records[split]]
        all_texts.extend(texts)
        dups = len(texts) - len(set(texts))
        duplicates += dups
        print(f"{split:5s}: {actual:6d} rekordów, duplikatów wewnątrz splitu: {dups}")

    text_counts = Counter(all_texts)
    cross = sum(c - 1 for c in text_counts.values() if c > 1)
    if cross != 0:
        raise RuntimeError(f"Wykryto {cross} powielonych tekstów (przecieki lub duplikaty)")
    print(f"Unikalność 'text': {len(set(all_texts))}/{len(all_texts)} "
          f"— 0 duplikatów między splitami.")

    # Rozkład kategorii
    print("\nRozkład kategorii:")
    print(f"{'split':6s} {'kategoria':18s} {'liczba':>7s} {'cel':>6s} {'udział':>8s}")
    for split in SPLITS:
        counts = Counter(r["category"] for r in records[split])
        for cat in CATEGORIES:
            got, want = counts[cat], TARGETS[split][cat]
            if got != want:
                raise RuntimeError(f"Zły rozkład {split}/{cat}: {got} != {want}")
            pct = 100.0 * got / SPLIT_SIZES[split]
            print(f"{split:6s} {cat:18s} {got:7d} {want:6d} {pct:7.2f}%")

    # Długości
    def stats(values: list[int]) -> tuple[int, float, int]:
        return min(values), sum(values) / len(values), max(values)

    print("\nDługości pełnego tekstu rekordu (<|state|> ... <|eos|>):")
    print(f"{'zakres':6s} {'min słów':>10s} {'śr. słów':>10s} {'max słów':>10s} "
          f"{'min znaków':>11s} {'śr. znaków':>11s} {'max znaków':>11s}")
    all_words = [len(t.split()) for t in all_texts]
    all_chars = [len(t) for t in all_texts]
    for split in SPLITS:
        texts = [r["text"] for r in records[split]]
        words = [len(t.split()) for t in texts]
        chars = [len(t) for t in texts]
        wmin, wavg, wmax = stats(words)
        cmin, cavg, cmax = stats(chars)
        print(f"{split:6s} {wmin:10d} {wavg:10.2f} {wmax:10d} "
              f"{cmin:11d} {cavg:11.2f} {cmax:11d}")
    wmin, wavg, wmax = stats(all_words)
    cmin, cavg, cmax = stats(all_chars)
    print(f"{'RAZEM':6s} {wmin:10d} {wavg:10.2f} {wmax:10d} "
          f"{cmin:11d} {cavg:11.2f} {cmax:11d}")

    if max(all_chars) > MAX_CHARS or max(all_words) > MAX_WORDS:
        raise RuntimeError("Przekroczono budżet długości (900 znaków / 150 słów)")

    doubled = [t for t in all_texts if "Nie. Nie." in t or "Nie, Nie." in t]
    if doubled:
        raise RuntimeError(f"Wykryto podwójne zaprzeczenie w {len(doubled)} tekstach")

    print(f"\nPliki zapisane w: {out_dir.resolve()}")
    print(f"Seed: {seed}")
    print("Walidacja: OK — wszystkie testy przeszły.")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# 14. CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Determinystyczny generator danych Piekła Dantego z kanonu dante_kanon_high.md"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="ziarno generatora (domyślnie: 42)")
    parser.add_argument("--out-dir", type=Path, default=Path("."),
                        help="katalog wyjściowy (domyślnie: katalog bieżący)")
    args = parser.parse_args()

    records = generate_all(args.seed)
    write_jsonl(records, args.out_dir)
    validate(records, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
