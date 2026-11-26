#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tartarus_data.py — Deterministyczny generator korpusu dialogowo-stanowego dla świata
„Tartarus: The Fall of the Ancients” (język polski, pełne UTF-8).

Format rekordu tekstowego:
    <|state|> klucz=wartość; ... <|user|> pytanie lub akcja gracza <|npc|> odpowiedź postaci <|eos|>

Zasady higieny Train / Val / Test (anty-przeuczenie):
    1. Podział na zbiory train, val i test następuje PRZED generacją próbek.
    2. Osobne pule szablonów pytań (<|user|>), odpowiedzi (<|npc|>) i prefiksów dla każdego splitu.
    3. Odłożone encje (holdout) w val i test — w train występują wyłącznie encje train.
    4. Rozłączne kombinacje stanów gry pomiędzy train, val i test.
    5. Spójność logiczna stan ↔ odpowiedź (martwy dowódca, wrogowie=brak, brak wycieków encji val/test).
    6. Pełna zgodność polskiej gramatyki (rodzaj, liczba mnoga w nazwie, fleksja).
    7. Wewnątrz train dopuszcza się powtórzenie pary (user, npc) przy innym <|state|>, przy
       jednoczesnym bezwzględnym zakazie duplikacji całych rekordów oraz przecieków do val/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENY SPECJALNE I STAŁE FORMATU
# ═══════════════════════════════════════════════════════════════════════════════

STATE_TOKEN = "<|state|>"
USER_TOKEN = "<|user|>"
NPC_TOKEN = "<|npc|>"
EOS_TOKEN = "<|eos|>"

CATEGORIES = ("dialog", "state_query", "lore_entity", "refusal")
POLISH_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

SAFE_LOWERCASE_STARTS = {
    "ile", "ilu", "czy", "jaki", "jaka", "jakie", "jakim", "jakiego", "jaką",
    "kto", "kogo", "komu", "gdzie", "co", "czego", "czym", "dlaczego",
    "podaj", "sprawdź", "opisz", "opowiedz", "streść", "wyjaśnij", "porównaj",
    "zestaw", "scharakteryzuj", "odczytaj", "wymień", "raportuj", "oceń",
    "na", "w", "we", "z", "za", "do", "od", "o", "przed", "przy",
    "kup", "rekrutuj", "wyślij", "uderz", "rzuć", "użyj", "maszeruj",
    "złącz", "powiększ", "dołącz", "rozkaż", "aktywuj", "wyszkol", "zaciągnij",
    "wypłać", "opłać", "przejdź", "rozpocznij", "splądruj", "złóż", "przeszukaj",
    "wejdź", "zbadaj", "zaatakuj", "skieruj", "prowadź", "niech",
    "jak", "który", "która", "które", "wskaż", "określ", "zdaj", "przelicz",
    "okolica", "dodaj", "meldunek", "słowo",
}


def plural_pole(n: int) -> str:
    if n == 1:
        return "pole"
    if n % 10 in (2, 3, 4) and not (12 <= (n % 100) <= 14):
        return "pola"
    return "pól"


def plural_pole_gen(n: int) -> str:
    return "pola" if n == 1 else "pól"


# ═══════════════════════════════════════════════════════════════════════════════
# MODELE DANYCH Z POLSKĄ FLEKSJĄ I ZGODNOŚCIĄ GRAMATYCZNĄ
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class UnitEntry:
    key: str
    faction: str
    role: str
    split: str  # "train" | "val" | "test"
    nom: str
    gen: str
    dat: str
    acc: str
    ins: str
    loc: str
    pl_nom: str
    pl_gen: str
    gender: str  # "m" | "f"
    atk: int
    defense: int
    hp: int
    ini: int
    mv: int
    cost: int
    special_name: str
    special_desc: str
    fly: bool = False
    naval: bool = False
    nom_is_plural: bool = False
    pl_human_masc: bool = False  # True -> "nasi ...", False -> "nasze ..."

    @property
    def pl_our(self) -> str:
        return "nasi" if self.pl_human_masc else "nasze"

    @property
    def v_ma(self) -> str:
        return "mają" if self.nom_is_plural else "ma"

    @property
    def v_zna(self) -> str:
        return "znają" if self.nom_is_plural else "zna"

    @property
    def v_znosi(self) -> str:
        return "znoszą" if self.nom_is_plural else "znosi"

    @property
    def v_nalezy(self) -> str:
        return "należą" if self.nom_is_plural else "należy"

    @property
    def v_dysponuje(self) -> str:
        return "dysponują" if self.nom_is_plural else "dysponuje"

    @property
    def v_kosztuje(self) -> str:
        return "kosztują" if self.nom_is_plural else "kosztuje"

    @property
    def v_potrafi(self) -> str:
        return "potrafią" if self.nom_is_plural else "potrafi"

    @property
    def v_utrzyma(self) -> str:
        return "utrzymają" if self.nom_is_plural else "utrzyma"

    @property
    def v_przynosi(self) -> str:
        return "przynoszą" if self.nom_is_plural else "przynosi"

    @property
    def v_jest(self) -> str:
        return "są" if self.nom_is_plural else "jest"

    @property
    def v_pelni(self) -> str:
        return "pełnią" if self.nom_is_plural else "pełni"

    @property
    def adj_drogi(self) -> str:
        if self.nom_is_plural:
            return "drogie"
        return "droga" if self.gender == "f" else "drogi"

    @property
    def adj_wytrzymaly(self) -> str:
        if self.nom_is_plural:
            return "wytrzymałe"
        return "wytrzymała" if self.gender == "f" else "wytrzymały"

    @property
    def adj_tanszy(self) -> str:
        if self.nom_is_plural:
            return "Tańsze"
        return "Tańsza" if self.gender == "f" else "Tańszy"

    @property
    def adj_szybszy(self) -> str:
        if self.nom_is_plural:
            return "Szybsze"
        return "Szybsza" if self.gender == "f" else "Szybszy"

    @property
    def adj_zwinniejszy(self) -> str:
        if self.nom_is_plural:
            return "Zwinniejsze"
        return "Zwinniejsza" if self.gender == "f" else "Zwinniejszy"


@dataclass(frozen=True)
class LocationEntry:
    key: str
    kind: str  # "zamek" | "świątynia" | "ruiny"
    split: str  # "train" | "val" | "test"
    nom: str
    gen: str
    acc: str
    loc: str
    prep_in: str
    prep_to: str
    summary: str
    bonus_or_guard: str
    is_plural: bool = False
    gender: str = "m"  # "m" | "f" | "n" | "pl"

    @property
    def v_slynie(self) -> str:
        return "słyną" if self.is_plural else "słynie"

    @property
    def v_pelni(self) -> str:
        return "pełnią" if self.is_plural else "pełni"

    @property
    def v_jest(self) -> str:
        return "są" if self.is_plural else "jest"

    @property
    def v_stanowi(self) -> str:
        return "stanowią" if self.is_plural else "stanowi"

    @property
    def adj_sklasyfikowane(self) -> str:
        if self.is_plural:
            return "sklasyfikowane"
        if self.gender == "f":
            return "sklasyfikowana"
        if self.gender == "n":
            return "sklasyfikowane"
        return "sklasyfikowany"


@dataclass(frozen=True)
class SpellEntry:
    key: str
    school: str
    split: str
    nom: str
    gen: str
    acc: str
    ins: str
    loc: str
    mana_cost: int
    range_tiles: int
    effect: str


@dataclass(frozen=True)
class ArtifactEntry:
    key: str
    rarity: str
    slot: str
    split: str
    nom: str
    gen: str
    acc: str
    ins: str
    loc: str
    bonus: str
    gender: str = "m"

    @property
    def adj_ten(self) -> str:
        if self.gender == "f":
            return "Ta"
        if self.gender == "pl":
            return "Te"
        return "Ten"


@dataclass(frozen=True)
class FactionEntry:
    key: str
    split: str
    nom: str
    gen: str
    dat: str
    loc: str
    leader_nom: str
    leader_gen: str
    capital: str
    fortress_aura: str
    doctrine: str


@dataclass(frozen=True)
class TerrainEntry:
    key: str
    split: str
    nom: str
    gen: str
    loc: str
    prep_in: str
    move_cost: str


# ═══════════════════════════════════════════════════════════════════════════════
# KANONICZNE DANE ŚWIATA TARTARUS (Z ODŁOŻONYMI ENCJAMI VAL / TEST)
# ═══════════════════════════════════════════════════════════════════════════════

FACTIONS: Dict[str, FactionEntry] = {
    "nordic": FactionEntry(
        key="nordic",
        split="train",
        nom="Nordyccy",
        gen="Nordyckich",
        dat="Nordyckim",
        loc="Nordyckich",
        leader_nom="Jarl Ragnar Lodbrok",
        leader_gen="Jarla Ragnara Lodbroka",
        capital="Fjordoskian",
        fortress_aura="Burza Lodowa (6–10 obrażeń w 1. rundzie oblężenia)",
        doctrine="gwałtowny szturm berserkerów, zwiad kruków i najazdy drakkarami",
    ),
    "russian": FactionEntry(
        key="russian",
        split="train",
        nom="Ruskie Kniaziostwa",
        gen="Ruskich Kniaziostw",
        dat="Ruskim Kniaziostwom",
        loc="Ruskich Kniaziostwach",
        leader_nom="Kniaź Włodzimierz",
        leader_gen="Kniazia Włodzimierza",
        capital="Gniazdo Żmija",
        fortress_aura="Zatrute Bagno (-3 Obrony wroga i kwas)",
        doctrine="twardy mur tarcz drużynników, ciężkie kusze i wsparcie Żmija Gorynycza",
    ),
    "slavic": FactionEntry(
        key="slavic",
        split="train",
        nom="Plemiona Słowiańskie",
        gen="Plemion Słowiańskich",
        dat="Plemionom Słowiańskim",
        loc="Plemionach Słowiańskich",
        leader_nom="Wojewoda Siemomysł",
        leader_gen="Wojewody Siemomysła",
        capital="Gaj Świętogora",
        fortress_aura="Korzenie Zatrzymania (-4 Inicjatywy wroga w 1. rundzie)",
        doctrine="zasadzki w gęstych borach, watahy wołkodłaków i magia ognia Swaroga",
    ),
    "egypt": FactionEntry(
        key="egypt",
        split="train",
        nom="Królestwo Egiptu",
        gen="Królestwa Egiptu",
        dat="Królestwu Egiptu",
        loc="Królestwie Egiptu",
        leader_nom="Faraon Ramzes Wieczny",
        leader_gen="Faraona Ramzesa Wiecznego",
        capital="Nekropolia Siedmiu Klątw",
        fortress_aura="Oko Ra (burza piaskowa rażąca latające oddziały)",
        doctrine="ostrzał łuczników Horusa, klątwy balsamisty i pancerne sfinksy",
    ),
    "byzantine": FactionEntry(
        key="byzantine",
        split="val",
        nom="Cesarstwo Bizantyńskie",
        gen="Cesarstwa Bizantyńskiego",
        dat="Cesarstwu Bizantyńskiemu",
        loc="Cesarstwie Bizantyńskim",
        leader_nom="Basileus Konstantyn XI",
        leader_gen="Basileusa Konstantyna XI",
        capital="Twierdza Złotego Rogu",
        fortress_aura="Święty Ogień (5 obrażeń i zwrot 25% ciosów wręcz)",
        doctrine="żelazna falanga skoutaroi, ogień grecki i szarże katafraktów",
    ),
    "greek": FactionEntry(
        key="greek",
        split="test",
        nom="Hellada",
        gen="Hellady",
        dat="Helladzie",
        loc="Helladzie",
        leader_nom="Król Leonidas Niezłomny",
        leader_gen="Króla Leonidasa Niezłomnego",
        capital="Twierdza Tartar",
        fortress_aura="Gorgoneion (petryfikuje najszybszego wroga w 1. rundzie)",
        doctrine="karna falanga hoplitów, przebijające pancerz groty toxotes i mityczne bestie",
    ),
}

TERRAINS: List[TerrainEntry] = [
    TerrainEntry("trawa", "train", "równina", "równiny", "równinie", "na równinie", "2 PM"),
    TerrainEntry("las", "train", "las", "lasu", "lesie", "w lesie", "3 PM"),
    TerrainEntry("wzgorza", "train", "wzgórza", "wzgórz", "wzgórzach", "na wzgórzach", "3 PM"),
    TerrainEntry("droga", "train", "droga", "drogi", "drodze", "na trakcie", "1 PM"),
    TerrainEntry("bagna", "val", "bagna", "bagien", "bagnach", "na bagnach", "4 PM"),
    TerrainEntry("woda", "val", "morze", "morza", "morzu", "na morzu", "2 PM okrętem"),
    TerrainEntry("gory", "test", "góry", "gór", "górach", "w górach", "4 PM (górołazi/lot)"),
]

UNITS: List[UnitEntry] = [
    # --- TRAIN ---
    UnitEntry("berserker", "nordic", "piechota", "train", "Berserker", "Berserkera", "Berserkerowi", "Berserkera", "Berserkerem", "Berserkerze", "Berserkerzy", "Berserkerów", "m", 7, 2, 30, 8, 5, 65, "Furia Bojowa", "zyskuje +1 do Ataku po każdym zabitym wrogu", pl_human_masc=True),
    UnitEntry("mysliwy_ulfhednar", "nordic", "strzelec", "train", "Myśliwy Ulfhednar", "Myśliwego Ulfhednara", "Myśliwemu Ulfhednarowi", "Myśliwego Ulfhednara", "Myśliwym Ulfhednarem", "Myśliwym Ulfhednarze", "Myśliwi Ulfhednarzy", "Myśliwych Ulfhednarów", "m", 7, 3, 25, 8, 5, 120, "Strzał Łaski", "zadaje podwójne obrażenia celom poniżej 50% HP", pl_human_masc=True),
    UnitEntry("jotunn_mrozny", "nordic", "pancerz", "train", "Jötunn Mroźny", "Jötunna Mroźnego", "Jötunnowi Mroźnemu", "Jötunna Mroźnego", "Jötunnem Mroźnym", "Jötunnie Mroźnym", "Jötunnowie Mroźni", "Jötunnów Mroźnych", "m", 10, 8, 75, 4, 4, 340, "Zimowy Chłód", "mroźna aura zadaje 3 obrażenia wrogom co rundę", pl_human_masc=True),
    UnitEntry("draugr_krwi", "nordic", "tytan", "train", "Draugr Krwi", "Draugra Krwi", "Draugrowi Krwi", "Draugra Krwi", "Draugrem Krwi", "Draugrze Krwi", "Draugry Krwi", "Draugrów Krwi", "m", 16, 11, 120, 8, 6, 920, "Wampiryzm", "leczy 50% zadanych obrażeń jako własne HP", pl_human_masc=False),
    UnitEntry("druzynnik", "russian", "piechota", "train", "Drużynnik", "Drużynnika", "Drużynnikowi", "Drużynnika", "Drużynnikiem", "Drużynniku", "Drużynnicy", "Drużynników", "m", 7, 5, 36, 6, 4, 70, "Mur Tarcz", "+2 do Obrony (+4 podczas obrony twierdzy)", pl_human_masc=True),
    UnitEntry("zmij_gorynycz", "russian", "lot", "train", "Żmij Gorynycz", "Żmija Gorynycza", "Żmijowi Gorynyczowi", "Żmija Gorynycza", "Żmijem Gorynyczem", "Żmiju Gorynyczu", "Żmije Gorynycze", "Żmijów Gorynyczów", "m", 14, 9, 100, 11, 8, 780, "Trzy Głowy", "wyprowadza 3 osobne ataki w każdej rundzie", fly=True, pl_human_masc=False),
    UnitEntry("bohatyr_mocny", "russian", "pancerz", "train", "Bohatyr Mocny", "Bohatyra Mocnego", "Bohatyrowi Mocnemu", "Bohatyra Mocnego", "Bohatyrem Mocnym", "Bohatyrze Mocnym", "Bohatyrowie Mocni", "Bohatyrów Mocnych", "m", 11, 9, 85, 5, 4, 360, "Przełamanie", "trwale obniża Obronę celu o 1 przy każdym ciosie", pl_human_masc=True),
    UnitEntry("dziwozaba", "russian", "tytan", "train", "Bagienna Dziwożaba", "Bagiennej Dziwożaby", "Bagiennej Dziwożabie", "Bagienną Dziwożabę", "Bagienną Dziwożabą", "Bagiennej Dziwożabie", "Bagienne Dziwożaby", "Bagiennych Dziwożab", "f", 13, 12, 130, 4, 4, 900, "Kwas Bagienny", "wypala 4 HP na turę bez możliwości leczenia", pl_human_masc=False),
    UnitEntry("wolkodolak", "slavic", "piechota", "train", "Wołkodłak", "Wołkodłaka", "Wołkodłakowi", "Wołkodłaka", "Wołkodłakiem", "Wołkodłaku", "Wołkodłaki", "Wołkodłaków", "m", 8, 4, 34, 9, 5, 65, "Wilcza Wataha", "rośnie w siłę za każdego kolejnego wilka w oddziale", pl_human_masc=False),
    UnitEntry("siekacka_z_gaju", "slavic", "strzelec", "train", "Siekaczka z Gaju", "Siekaczki z Gaju", "Siekaczce z Gaju", "Siekaczkę z Gaju", "Siekaczką z Gaju", "Siekaczce z Gaju", "Siekaczki z Gaju", "Siekaczek z Gaju", "f", 7, 3, 25, 8, 5, 115, "Cierniste Bełty", "zadaje +3 obrażenia od cierni ignorując pancerz", pl_human_masc=False),
    UnitEntry("leszyj_puszczy", "slavic", "pancerz", "train", "Leszyj Puszczy", "Leszyja Puszczy", "Leszyjowi Puszczy", "Leszyja Puszczy", "Leszyjem Puszczy", "Leszyju Puszczy", "Leszyje Puszczy", "Leszyjów Puszczy", "m", 10, 10, 95, 4, 4, 350, "Korzenie Puszczy", "obniża Inicjatywę celu o 3 i unieruchamia wroga", pl_human_masc=False),
    UnitEntry("kaplan_swaroga", "slavic", "wsparcie", "train", "Kapłan Swaroga", "Kapłana Swaroga", "Kapłanowi Swaroga", "Kapłana Swaroga", "Kapłanem Swaroga", "Kapłanie Swaroga", "Kapłani Swaroga", "Kapłanów Swaroga", "m", 6, 4, 35, 7, 6, 310, "Święty Ogień", "+2 Ataku dla sojuszników i leczy 10 HP walczącemu", pl_human_masc=True),
    UnitEntry("gwardzista_maskowy", "egypt", "piechota", "train", "Gwardzista Maskowy", "Gwardzisty Maskowego", "Gwardziście Maskowemu", "Gwardzistę Maskowego", "Gwardzistą Maskowym", "Gwardziście Maskowym", "Gwardziści Maskowi", "Gwardzistów Maskowych", "m", 7, 5, 36, 6, 4, 65, "Nieustraszony", "odporny na strach i +2 Ataku w zwarciu", pl_human_masc=True),
    UnitEntry("lucznik_horusa", "egypt", "strzelec", "train", "Łucznik Horusa", "Łucznika Horusa", "Łucznikowi Horusa", "Łucznika Horusa", "Łucznikiem Horusa", "Łuczniku Horusa", "Łucznicy Horusa", "Łuczników Horusa", "m", 7, 3, 24, 8, 5, 120, "Oko Sokoła", "podwójne obrażenia przeciwko jednostkom latającym", pl_human_masc=True),
    UnitEntry("sfinks_ozywiony", "egypt", "pancerz", "train", "Sfinks Ożywiony", "Sfinksa Ożywionego", "Sfinksowi Ożywionemu", "Sfinksa Ożywionego", "Sfinksem Ożywionym", "Sfinksie Ożywionym", "Sfinksy Ożywione", "Sfinksów Ożywionych", "m", 11, 9, 85, 5, 4, 380, "Pancerz Pustyni", "odbija 20% obrażeń fizycznych w atakującego", pl_human_masc=False),
    UnitEntry("skoutaros", "byzantine", "piechota", "train", "Skoutaros", "Skoutarosa", "Skoutarosowi", "Skoutarosa", "Skoutarosem", "Skoutarosie", "Skoutaroi", "Skoutarosów", "m", 7, 6, 40, 6, 4, 70, "Żółw (Testudo)", "+50% do Obrony przeciwko atakom dystansowym", pl_human_masc=True),
    UnitEntry("miotacz_ognia_greckiego", "byzantine", "strzelec", "train", "Miotacz Ognia Greckiego", "Miotacza Ognia Greckiego", "Miotaczowi Ognia Greckiego", "Miotacza Ognia Greckiego", "Miotaczem Ognia Greckiego", "Miotaczu Ognia Greckiego", "Miotacze Ognia Greckiego", "Miotaczy Ognia Greckiego", "m", 9, 3, 24, 6, 4, 140, "Ogień Grecki", "podpala cel zadając +4 obrażenia płomieni", pl_human_masc=True),
    UnitEntry("katafrakt_varangianin", "byzantine", "pancerz", "train", "Katafrakt Varangianin", "Katafrakta Varangianina", "Katafraktowi Varangianinowi", "Katafrakta Varangianina", "Katafraktem Varangianinem", "Katafrakcie Varangianinie", "Katafrakci Varangianie", "Katafraktów Varangianów", "m", 12, 11, 85, 6, 5, 390, "Żelazna Kontra", "zawsze kontratakuje z premią +2 do obrażeń", pl_human_masc=True),
    UnitEntry("hoplita_nekrofagos", "greek", "piechota", "train", "Hoplita Nekrofagos", "Hoplity Nekrofagosa", "Hoplicie Nekrofagosowi", "Hoplitę Nekrofagosa", "Hoplitą Nekrofagosem", "Hoplicie Nekrofagosie", "Hoplici Nekrofagosi", "Hoplitów Nekrofagosów", "m", 7, 4, 32, 7, 4, 65, "Krwawa Szarża", "+3 Ataku w otwierającej rundzie bitwy", pl_human_masc=True),
    UnitEntry("toxotes_piekla", "greek", "strzelec", "train", "Toxotes Piekła", "Toxotesa Piekła", "Toxotesowi Piekła", "Toxotesa Piekła", "Toxotesem Piekła", "Toxotesie Piekła", "Toxotes Piekła", "Toxotesów Piekła", "m", 7, 2, 22, 8, 5, 125, "Kościane Groty", "ignoruje 75% pancerza celu", pl_human_masc=True),
    UnitEntry("hydra_lernejska", "greek", "tytan", "train", "Hydra Lernejska", "Hydry Lernejskiej", "Hydrze Lernejskiej", "Hydrę Lernejską", "Hydrą Lernejską", "Hydrze Lernejskiej", "Hydry Lernejskie", "Hydr Lernejskich", "f", 15, 8, 160, 7, 6, 990, "Regeneracja", "regeneruje 15 HP co rundę i atakuje dwukrotnie", pl_human_masc=False),
    UnitEntry("drakkar", "nordic", "okręt", "train", "Drakkar Nordycki", "Drakkaru Nordyckiego", "Drakkarowi Nordyckiemu", "Drakkar Nordycki", "Drakkarem Nordyckim", "Drakkarze Nordyckim", "Drakkary Nordyckie", "Drakkarów Nordyckich", "m", 0, 2, 30, 2, 20, 100, "Transport Morski", "przewozi do 7 oddziałów przez morza", naval=True, pl_human_masc=False),

    # --- VAL (odłożone do walidacji) ---
    UnitEntry("wrony_odyna", "nordic", "lot", "val", "Wrony Odyna", "Wron Odyna", "Wronom Odyna", "Wrony Odyna", "Wronami Odyna", "Wronach Odyna", "Wrony Odyna", "Wron Odyna", "f", 5, 4, 28, 12, 10, 220, "Zwiastun Śmierci", "obniża Atak wroga o 2 i Inicjatywę o 1", fly=True, nom_is_plural=True, pl_human_masc=False),
    UnitEntry("strzelec_samosieczki", "russian", "strzelec", "val", "Strzelec Samosieczki", "Strzelca Samosieczki", "Strzelcowi Samosieczce", "Strzelca Samosieczki", "Strzelcem Samosieczką", "Strzelcu Samosieczce", "Strzelcy Samosieczki", "Strzelców Samosieczki", "m", 8, 3, 26, 7, 5, 130, "Ciężki Bełt", "ignoruje połowę pancerza celu", pl_human_masc=True),
    UnitEntry("zywiolik_kamienny", "slavic", "tytan", "val", "Żywiołak Kamienny", "Żywiołaka Kamiennego", "Żywiołakowi Kamiennemu", "Żywiołaka Kamiennego", "Żywiołakiem Kamiennym", "Żywiołaku Kamiennym", "Żywiołaki Kamienne", "Żywiołaków Kamiennych", "m", 15, 14, 140, 3, 4, 950, "Kamienna Ziemia", "zyskuje +1 Obrony za każde utracone 20 HP", pl_human_masc=False),
    UnitEntry("balsamista_anubisa", "egypt", "wsparcie", "val", "Balsamista Anubisa", "Balsamisty Anubisa", "Balsamiście Anubisa", "Balsamistę Anubisa", "Balsamistą Anubisa", "Balsamiście Anubisie", "Balsamiści Anubisa", "Balsamistów Anubisa", "m", 6, 3, 28, 6, 6, 290, "Mumifikacja", "zadaje 10 obrażeń i zamienia zabitego w Mumię", pl_human_masc=True),
    UnitEntry("czarny_mnich", "byzantine", "wsparcie", "val", "Czarny Mnich", "Czarnego Mnicha", "Czarnemu Mnichowi", "Czarnego Mnicha", "Czarnym Mnichem", "Czarnym Mnichu", "Czarni Mnisi", "Czarnych Mnichów", "m", 5, 4, 32, 7, 6, 310, "Antymagia", "blokuje zdolności specjalne przeciwnika", pl_human_masc=True),
    UnitEntry("cyklop_brazowy", "greek", "pancerz", "val", "Cyklop Brązowy", "Cyklopa Brązowego", "Cyklopowi Brązowemu", "Cyklopa Brązowego", "Cyklopem Brązowym", "Cyklopie Brązowym", "Cyklopy Brązowe", "Cyklopów Brązowych", "m", 11, 8, 85, 4, 4, 380, "Brązowy Pancerz", "redukuje każde otrzymane obrażenia o 2", pl_human_masc=False),
    UnitEntry("dromona", "byzantine", "okręt", "val", "Dromona Cesarska", "Dromony Cesarskiej", "Dromonie Cesarskiej", "Dromonę Cesarską", "Dromoną Cesarską", "Dromonie Cesarskiej", "Dromony Cesarskie", "Dromon Cesarskich", "f", 0, 2, 30, 2, 20, 100, "Transport Morski", "ciężki okręt wojenny przewożący armię przez oceany", naval=True, pl_human_masc=False),

    # --- TEST (odłożone do testu generalizacji) ---
    UnitEntry("wolva_wieszczka", "nordic", "wsparcie", "test", "Völva Wieszczka", "Völvy Wieszczki", "Völvie Wieszczce", "Völvę Wieszczkę", "Völvą Wieszczką", "Völvie Wieszczce", "Völvy Wieszczki", "Völv Wieszczek", "f", 6, 4, 35, 7, 6, 300, "Krwawe Runy", "leczy sojusznika o 15 HP lub razi wroga magią", pl_human_masc=False),
    UnitEntry("koldun_czarnoksieznik", "russian", "wsparcie", "test", "Kołdun Czarnoksiężnik", "Kołduna Czarnoksiężnika", "Kołdunowi Czarnoksiężnikowi", "Kołduna Czarnoksiężnika", "Kołdunem Czarnoksiężnikiem", "Kołdunie Czarnoksiężniku", "Kołduni Czarnoksiężnicy", "Kołdunów Czarnoksiężników", "m", 7, 3, 30, 6, 5, 290, "Nekromancja", "zadaje 8 obrażeń magii i leczy sojusznika o 8 HP", pl_human_masc=True),
    UnitEntry("strzyga_zelazna", "slavic", "lot", "test", "Strzyga Żelazna", "Strzygi Żelaznej", "Strzydze Żelaznej", "Strzygę Żelazną", "Strzygą Żelazną", "Strzydze Żelaznej", "Strzygi Żelazne", "Strzyg Żelaznych", "f", 8, 4, 30, 10, 9, 230, "Nocna Zasadzka", "zyskuje +4 Ataku w 1. rundzie bitwy", fly=True, pl_human_masc=False),
    UnitEntry("awiz_krwawy", "egypt", "tytan", "test", "Awiz Krwawy", "Awiza Krwawego", "Awizowi Krwawemu", "Awiza Krwawego", "Awizem Krwawym", "Awizie Krwawym", "Awizy Krwawe", "Awizów Krwawych", "m", 17, 10, 110, 10, 8, 940, "Oko Ra", "zadaje +5 czystych obrażeń ignorujących pancerz", fly=True, pl_human_masc=False),
    UnitEntry("gargulec_hipodromu", "byzantine", "lot", "test", "Gargulec Hipodromu", "Gargulca Hipodromu", "Gargulcowi Hipodromu", "Gargulca Hipodromu", "Gargulcem Hipodromu", "Gargulcu Hipodromu", "Gargulce Hipodromu", "Gargulców Hipodromu", "m", 7, 5, 32, 10, 9, 240, "Miedziany Wybuch", "eksploduje po śmierci za 15 obrażeń", fly=True, pl_human_masc=False),
    UnitEntry("swiety_likwidator", "byzantine", "tytan", "test", "Święty Likwidator", "Świętego Likwidatora", "Świętemu Likwidatorowi", "Świętego Likwidatora", "Świętym Likwidatorem", "Świętym Likwidatorze", "Święci Likwidatorzy", "Świętych Likwidatorów", "m", 16, 12, 125, 7, 6, 960, "Głos Boga", "razi wszystkich wrogów na 10 obrażeń na starcie bitwy", pl_human_masc=True),
    UnitEntry("harpii_krwawa", "greek", "lot", "test", "Krwawa Harpia", "Krwawej Harpii", "Krwawej Harpii", "Krwawą Harpię", "Krwawą Harpią", "Krwawej Harpii", "Krwawe Harpie", "Krwawych Harpii", "f", 7, 3, 26, 11, 10, 220, "Wyssanie Życia", "leczy 30% zadanych obrażeń", fly=True, pl_human_masc=False),
    UnitEntry("mantyka_wyrocznia", "greek", "wsparcie", "test", "Mantyka Wyrocznia", "Mantyki Wyroczni", "Mantyce Wyroczni", "Mantykę Wyrocznię", "Mantyką Wyrocznią", "Mantyce Wyroczni", "Mantyki Wyrocznie", "Mantyk Wyroczni", "f", 7, 4, 32, 8, 6, 320, "Piorun Zeusa", "ciska gromem za 20% maks. HP celu", pl_human_masc=False),
    UnitEntry("trirema", "greek", "okręt", "test", "Trirema Hellady", "Triremy Hellady", "Triremie Hellady", "Triremę Hellady", "Triremą Hellady", "Triremie Hellady", "Triremy Hellady", "Trirem Hellady", "f", 0, 2, 30, 2, 20, 100, "Transport Morski", "szybka jednostka morska strzegąca archipelagów", naval=True, pl_human_masc=False),
]

LOCATIONS: List[LocationEntry] = [
    # --- TRAIN ---
    LocationEntry("wrota_tartaru", "zamek", "train", "Wrota Tartaru", "Wrót Tartaru", "Wrota Tartaru", "Wrotach Tartaru", "we Wrotach Tartaru", "do Wrót Tartaru", "strategiczna brama między krainami archipelagu", "mury graniczne i garnizon", is_plural=True, gender="pl"),
    LocationEntry("fjordoskian", "zamek", "train", "Fjordoskian", "Fjordoskianu", "Fjordoskian", "Fjordoskianie", "we Fjordoskianie", "do Fjordoskianu", "lodowa warownia jarlów wykuta w klifie nad oceanem", "Burza Lodowa w 1. rundzie oblężenia", gender="m"),
    LocationEntry("gniazdo_zmija", "zamek", "train", "Gniazdo Żmija", "Gniazda Żmija", "Gniazdo Żmija", "Gnieździe Żmija", "w Gnieździe Żmija", "do Gniazda Żmija", "kamienny detiniec Ruskich Kniaziostw otoczony bagnami", "Zatrute Bagno osłabiające pancerz najeźdźcy", gender="n"),
    LocationEntry("gaj_swietogora", "zamek", "train", "Gaj Świętogora", "Gaju Świętogora", "Gaj Świętogora", "Gaju Świętogora", "w Gaju Świętogora", "do Gaju Świętogora", "święty gród Słowian ukryty pośród prastarych dębów", "Korzenie Zatrzymania spowalniające wroga", gender="m"),
    LocationEntry("nekropolia", "zamek", "train", "Nekropolia Siedmiu Klątw", "Nekropolii Siedmiu Klątw", "Nekropolię Siedmiu Klątw", "Nekropolii Siedmiu Klątw", "w Nekropolii Siedmiu Klątw", "do Nekropolii Siedmiu Klątw", "bazaltowa stolica faraonów strzeżona przez sfinksy", "Oko Ra rażące jednostki latające", gender="f"),
    LocationEntry("zloty_rog", "zamek", "train", "Twierdza Złotego Rogu", "Twierdzy Złotego Rogu", "Twierdzę Złotego Rogu", "Twierdzy Złotego Rogu", "w Twierdzy Złotego Rogu", "do Twierdzy Złotego Rogu", "marmurowa metropolia Basileusa nad cieśniną", "Święty Ogień zwracający część obrażeń", gender="f"),
    LocationEntry("bialy_grod", "zamek", "train", "Biały Gród", "Białego Grodu", "Biały Gród", "Białym Grodzie", "w Białym Grodzie", "do Białego Grodu", "warowny gród z jasnego wapienia na równinach", "solidny mur obronny", gender="m"),
    LocationEntry("czarny_kruk", "zamek", "train", "Czarny Kruk", "Czarnego Kruka", "Czarny Kruk", "Czarnym Kruku", "w Czarnym Kruku", "do Czarnego Kruka", "ponura strażnica na skalistym grzbiecie", "wysoka wieża zwiadowcza", gender="m"),
    LocationEntry("smocza_jama", "zamek", "train", "Smocza Jama", "Smoczej Jamy", "Smoczą Jamę", "Smoczej Jamie", "w Smoczej Jamie", "do Smoczej Jamy", "górska twierdza zdolna szkolić smoki", "smocze legowisko i skarbiec", gender="f"),
    LocationEntry("sanktuarium_swiatla", "świątynia", "train", "Sanktuarium Światła i Ładu", "Sanktuarium Światła i Ładu", "Sanktuarium Światła i Ładu", "Sanktuarium Światła i Ładu", "w Sanktuarium Światła i Ładu", "do Sanktuarium Światła i Ładu", "alabastrowa rotunda Pierwszej Ery", "+1 Siły Wodza i pełna regeneracja many", gender="n"),
    LocationEntry("chram_swaroga", "świątynia", "train", "Chram Płomiennego Swaroga", "Chramu Płomiennego Swaroga", "Chram Płomiennego Swaroga", "Chramie Płomiennego Swaroga", "w Chramie Płomiennego Swaroga", "do Chramu Płomiennego Swaroga", "dębowy krąg ognia kowala niebios", "trwały bonus +1 Siły Dowodzenia i pełna mana", gender="m"),
    LocationEntry("oltarz_horusa", "świątynia", "train", "Słoneczny Ołtarz Horusa", "Słonecznego Ołtarza Horusa", "Słoneczny Ołtarz Horusa", "Słonecznym Ołtarzu Horusa", "przy Słonecznym Ołtarzu Horusa", "do Słonecznego Ołtarza Horusa", "pustynny obelisk ze złota i piaskowca", "+1 Siły Bohatera oraz odnowienie many", gender="m"),
    LocationEntry("ruiny_kataklizmu", "ruiny", "train", "Ruiny Kataklizmu", "Ruin Kataklizmu", "Ruiny Kataklizmu", "Ruinach Kataklizmu", "w Ruinach Kataklizmu", "do Ruin Kataklizmu", "strzaskana forteca z podziemiami pełnymi kości", "strzegą ich szkielety i gargulce", is_plural=True, gender="pl"),
    LocationEntry("grobowiec_tytanow", "ruiny", "train", "Grobowiec Tytanów", "Grobowca Tytanów", "Grobowiec Tytanów", "Grobowcu Tytanów", "w Grobowcu Tytanów", "do Grobowca Tytanów", "kamienny kurhan poległych gigantów", "strzegą go trolle i widmowi olbrzymi", gender="m"),

    # --- VAL ---
    LocationEntry("twierdza_tartar", "zamek", "val", "Twierdza Tartar", "Twierdzy Tartar", "Twierdzę Tartar", "Twierdzy Tartar", "w Twierdzy Tartar", "do Twierdzy Tartar", "cyklopowa stolica Hellady nad przepaścią", "Gorgoneion paraliżujący najszybszego napastnika", gender="f"),
    LocationEntry("wieza_wichrow", "zamek", "val", "Wieża Wichrów", "Wieży Wichrów", "Wieżę Wichrów", "Wieży Wichrów", "w Wieży Wichrów", "do Wieży Wichrów", "wysoka strażnica smagana górskim wiatrem", "szeroki zasięg widzenia nad przełęczą", gender="f"),
    LocationEntry("zelazny_most", "zamek", "val", "Żelazny Most", "Żelaznego Mostu", "Żelazny Most", "Żelaznym Moście", "na Żelaznym Moście", "do Żelaznego Mostu", "forteca broniąca jedynej przeprawy przez rzekę", "wąskie gardło łatwe do obrony", gender="m"),
    LocationEntry("sanktuarium_olimpu", "świątynia", "val", "Sanktuarium Bogów Olimpu", "Sanktuarium Bogów Olimpu", "Sanktuarium Bogów Olimpu", "Sanktuarium Bogów Olimpu", "w Sanktuarium Bogów Olimpu", "do Sanktuarium Bogów Olimpu", "dorycka świątynia Zeusa i Ateny", "dar +1 Siły oraz błyskawiczna odnowa many", gender="n"),
    LocationEntry("krypta_architektow", "ruiny", "val", "Krypta Pradawnych Architektów", "Krypty Pradawnych Architektów", "Kryptę Pradawnych Architektów", "Krypcie Pradawnych Architektów", "w Krypcie Pradawnych Architektów", "do Krypty Pradawnych Architektów", "granitowy labirynt budowniczych Tartaru", "bazaltowe golemy oraz zjawy mędrców", gender="f"),
    LocationEntry("kazamaty_bazaltu", "ruiny", "val", "Kazamaty Czarnego Bazaltu", "Kazamatów Czarnego Bazaltu", "Kazamaty Czarnego Bazaltu", "Kazamatach Czarnego Bazaltu", "w Kazamatach Czarnego Bazaltu", "do Kazamatów Czarnego Bazaltu", "wulkaniczne lochy dawnego imperium", "żelazne upiory i widmowi kaci", is_plural=True, gender="pl"),

    # --- TEST ---
    LocationEntry("burzowa_twierdza", "zamek", "test", "Burzowa Twierdza", "Burzowej Twierdzy", "Burzową Twierdzę", "Burzowej Twierdzy", "w Burzowej Twierdzy", "do Burzowej Twierdzy", "potężna warownia na pograniczu burzowych szczytów", "wzmocnione baszty oblężnicze", gender="f"),
    LocationEntry("srebrna_przystan", "zamek", "test", "Srebrna Przystań", "Srebrnej Przystani", "Srebrną Przystań", "Srebrnej Przystani", "w Srebrnej Przystani", "do Srebrnej Przystani", "nadmorski port handlowy i stocznia floty", "dostęp do szlaków morskich i dochód z ceł", gender="f"),
    LocationEntry("wzgorze_cieni", "zamek", "test", "Wzgórze Cieni", "Wzgórza Cieni", "Wzgórze Cieni", "Wzgórzu Cieni", "na Wzgórzu Cieni", "na Wzgórze Cieni", "mroczna cytadela spowita mgłą", "trudny podjazd pod strome zbocze", gender="n"),
    LocationEntry("kaplica_boreasza", "świątynia", "test", "Kaplica Mroźnego Boreasza", "Kaplicy Mroźnego Boreasza", "Kaplicę Mroźnego Boreasza", "Kaplicy Mroźnego Boreasza", "w Kaplicy Mroźnego Boreasza", "do Kaplicy Mroźnego Boreasza", "lodowcowe sanktuarium wichrów Północy", "trwałe +1 Siły Dowódcy i odnowa many", gender="f"),
    LocationEntry("czelusc_bostw", "ruiny", "test", "Czeluść Strąconych Bóstw", "Czeluści Strąconych Bóstw", "Czeluść Strąconych Bóstw", "Czeluści Strąconych Bóstw", "w Czeluści Strąconych Bóstw", "do Czeluści Strąconych Bóstw", "zapadlina zakazanego kultu pełna siarki", "demoniczne pomioty i czarnoksiężnicy", gender="f"),
    LocationEntry("podniebna_cytadela", "ruiny", "test", "Zgliszcza Podniebnej Cytadeli", "Zgliszcz Podniebnej Cytadeli", "Zgliszcza Podniebnej Cytadeli", "Zgliszczach Podniebnej Cytadeli", "w Zgliszczach Podniebnej Cytadeli", "do Zgliszcz Podniebnej Cytadeli", "roztrzaskana forteca z białego marmuru", "drapieżne wiwerny i spaczone gargulce", is_plural=True, gender="pl"),
]

SPELLS: List[SpellEntry] = [
    SpellEntry("oko_wizji", "Ład", "train", "Oko Jasnowidzenia", "Oka Jasnowidzenia", "Oko Jasnowidzenia", "Okiem Jasnowidzenia", "Oku Jasnowidzenia", 3, 12, "odsłania mgłę wojny w promieniu 4 pól"),
    SpellEntry("sanktuarium", "Ład", "train", "Sanktuarium Swaroga", "Sanktuarium Swaroga", "Sanktuarium Swaroga", "Sanktuarium Swaroga", "Sanktuarium Swaroga", 6, 99, "leczy garnizon o 20 HP i dodaje +2 Obrony grodu"),
    SpellEntry("wichry_boreasza", "Natura", "train", "Wichry Boreasza", "Wichrów Boreasza", "Wichry Boreasza", "Wichrami Boreasza", "Wichrach Boreasza", 4, 8, "przywraca armii +15 punktów ruchu"),
    SpellEntry("balsam_zywiolow", "Natura", "train", "Balsam Żywiołów", "Balsamu Żywiołów", "Balsam Żywiołów", "Balsamem Żywiołów", "Balsamie Żywiołów", 4, 8, "leczy wszystkich rannych wojowników o 25 HP"),
    SpellEntry("grom_peruna", "Wojna", "train", "Grom Peruna", "Gromu Peruna", "Grom Peruna", "Gromem Peruna", "Gromie Peruna", 5, 8, "zadaje 30–45 obrażeń jednostce wroga"),
    SpellEntry("szal_krwawego_lwa", "Wojna", "train", "Szał Krwawego Lwa", "Szału Krwawego Lwa", "Szał Krwawego Lwa", "Szałem Krwawego Lwa", "Szale Krwawego Lwa", 4, 8, "zwiększa atak dowódcy o +3 i dodaje +10 PM"),
    SpellEntry("zew_cieni", "Cień", "train", "Zew Cieni", "Zewu Cieni", "Zew Cieni", "Zewem Cieni", "Zewie Cieni", 6, 8, "przyzywa widmową jednostkę wsparcia do armii"),
    # VAL holdout
    SpellEntry("aura_natchnienia", "Ład", "val", "Aura Natchnienia", "Aury Natchnienia", "Aurę Natchnienia", "Aurą Natchnienia", "Aurze Natchnienia", 4, 8, "daje +1 Siły wszystkim jednostkom w armii"),
    SpellEntry("grzezawisko_mokradel", "Natura", "val", "Grzęzawisko Mokradeł", "Grzęzawiska Mokradeł", "Grzęzawisko Mokradeł", "Grzęzawiskiem Mokradeł", "Grzęzawisku Mokradeł", 4, 8, "odbiera wrogiej armii 14 punktów ruchu"),
    # TEST holdout
    SpellEntry("grad_ognisty", "Wojna", "test", "Grad Ognisty", "Gradu Ognistego", "Grad Ognisty", "Gradem Ognistym", "Gradzie Ognistym", 7, 8, "zadaje 10–15 obrażeń każdej wrogiej jednostce"),
    SpellEntry("calun_grozy", "Cień", "test", "Całun Grozy", "Całunu Grozy", "Całun Grozy", "Całunem Grozy", "Całunie Grozy", 4, 8, "obniża siłę wroga i odbiera mu 10 PM"),
]

ARTIFACTS: List[ArtifactEntry] = [
    ArtifactEntry("miecz_peruna", "legendarny", "broń", "train", "Miecz Peruna", "Miecza Peruna", "Miecz Peruna", "Mieczem Peruna", "Mieczu Peruna", "+6 Ataku, +4 Inicjatywy i +1 Siły armii", gender="m"),
    ArtifactEntry("ostrze_mrozu", "epicki", "broń", "train", "Mroźne Ostrze Niflheimu", "Mroźnego Ostrza Niflheimu", "Mroźne Ostrze Niflheimu", "Mroźnym Ostrzem Niflheimu", "Mroźnym Ostrzu Niflheimu", "+4 Ataku i +2 Inicjatywy", gender="m"),
    ArtifactEntry("egida_atenejska", "legendarny", "tarcza", "train", "Egida Ateny", "Egidy Ateny", "Egidę Ateny", "Egidą Ateny", "Egidzie Ateny", "+6 Obrony, +30 HP i +1 Siły armii", gender="f"),
    ArtifactEntry("kolczuga_swaroga", "epicki", "pancerz", "train", "Złota Kolczuga Swaroga", "Złotej Kolczugi Swaroga", "Złotą Kolczugę Swaroga", "Złotą Kolczugą Swaroga", "Złotej Kolczudze Swaroga", "+5 Obrony i +20 HP dowódcy", gender="f"),
    ArtifactEntry("buty_szybkosci", "rzadki", "buty", "train", "Buty Siedmiomilowe", "Butów Siedmiomilowych", "Buty Siedmiomilowe", "Butami Siedmiomilowymi", "Butach Siedmiomilowych", "+1 PM dla całej armii i +2 do marszu wodza", gender="pl"),
    ArtifactEntry("kielich_zycia", "legendarny", "relikwia", "train", "Święty Kielich Żywota", "Świętego Kielicha Żywota", "Święty Kielich Żywota", "Świętym Kielichem Żywota", "Świętym Kielichu Żywota", "regeneruje +15 HP wszystkim oddziałom co turę", gender="m"),
    # VAL holdout
    ArtifactEntry("topor_berserka", "rzadki", "broń", "val", "Topór Krwawego Jarla", "Topora Krwawego Jarla", "Topór Krwawego Jarla", "Toporem Krwawego Jarla", "Toporze Krwawego Jarla", "+4 Ataku i +2 Siły dowódcy", gender="m"),
    ArtifactEntry("oko_widzenia", "epicki", "relikwia", "val", "Kryształowe Oko Tartaru", "Kryształowego Oka Tartaru", "Kryształowe Oko Tartaru", "Kryształowym Okiem Tartaru", "Kryształowym Oku Tartaru", "rozszerza zwiad i daje +2 PM armii", gender="m"),
    # TEST holdout
    ArtifactEntry("wlocznia_slonca", "epicki", "broń", "test", "Włócznia Słoneczna Ra", "Włóczni Słonecznej Ra", "Włócznię Słoneczną Ra", "Włócznią Słoneczną Ra", "Włóczni Słonecznej Ra", "+5 Ataku i +3 Inicjatywy przeciwko pancerzom", gender="f"),
    ArtifactEntry("amulet_anubisa", "legendarny", "relikwia", "test", "Skarabeusz Nieśmiertelności", "Skarabeusza Nieśmiertelności", "Skarabeusza Nieśmiertelności", "Skarabeuszem Nieśmiertelności", "Skarabeuszu Nieśmiertelności", "+20 HP i chroni duszę bohatera przed śmiercią", gender="m"),
]

UNKNOWN_ENTITIES: Dict[str, List[Tuple[str, str, str]]] = {
    "train": [
        ("Smok Mechaniczny", "Smoka Mechanicznego", "W kronikach Tartaru nie istnieje taka machina — znamy jedynie Żmija Gorynycza i dzikie smoki."),
        ("Karabin Plazmowy", "Karabinu Plazmowego", "Taka broń nie istnieje w Tartarze; strzelcy używają łuków, kusz i syfonów ognia greckiego."),
        ("Cytadela Gondoru", "Cytadeli Gondoru", "Nie ma takiej twierdzy na naszych mapach archipelagu Tartaru."),
        ("Bomba Kryształowa", "Bomby Kryształowej", "Księga Zaklęć nie zawiera takiego czaru — znamy Grom Peruna i Grad Ognisty."),
        ("Rycerz Światłowodowy", "Rycerza Światłowodowego", "To nazwa spoza naszych kronik; żaden z rodów Tartaru nie wystawia takiego oddziału."),
        ("Czołg Oblężniczy", "Czołgu Oblężniczego", "W Tartarze nie znamy takich machin; oblężenia wspierają tytani i ciężka piechota."),
        ("Portal Międzygwiezdny", "Portalu Międzygwiezdnego", "Na archipelagu nie ma takiego obiektu — podróżujemy lądem, lotem lub okrętami."),
        ("Krasnolud Kosmiczny", "Krasnoluda Kosmicznego", "Znamy jedynie krasnoludów górskich, nie zaś przybyszów spoza nieboskłonu."),
        ("Katapulta Atomowa", "Katapulty Atomowej", "W zbrojowniach Tartaru nie buduje się takich machin oblężniczych."),
        ("Wieża Telepatyczna", "Wieży Telepatycznej", "Nie odnotowano takiej budowli w żadnej ze stref archipelagu."),
    ],
    "val": [
        ("Zeppelin Cesarski", "Zeppelina Cesarskiego", "Nie dysponujemy takim statkiem powietrznym; nieba strzegą gryfy, harpie oraz Wrony Odyna."),
        ("Zamek Winterfell", "Zamku Winterfell", "Takiej warowni nie odnotowano w rejestrach Tartaru."),
        ("Eliksir Niewidzialności", "Eliksiru Niewidzialności", "Nie znam takiego rytuału w czterech szkołach magii Tartaru."),
        ("Golem Laserowy", "Golema Laserowego", "Żadne ze źródeł nie wspomina o takim strażniku ruin."),
        ("Kawaleria Rakietowa", "Kawalerii Rakietowej", "Żadna frakcja w Tartarze nie posiada takiego oddziału."),
        ("Księga Cybernetyki", "Księgi Cybernetyki", "To nie jest relikwia z archipelagu Tartaru."),
        ("Działo Jonowe", "Działa Jonowego", "Nie znamy broni o tej nazwie na żadnym froncie Tartaru."),
    ],
    "test": [
        ("Paladyn Atomowy", "Paladyna Atomowego", "W szeregach żadnej z sześciu frakcji Tartaru nie służy taki wojownik."),
        ("Twierdza Mordoru", "Twierdzy Mordoru", "Nie odnajduję takiej warowni w granicach archipelagu Tartaru."),
        ("Zaklęcie Teleportacji Masowej", "Zaklęcia Teleportacji Masowej", "Księga Zaklęć nie zna tej inkantacji; marsz przyspieszają Wichry Boreasza."),
        ("Okręt Podwodny", "Okrętu Podwodnego", "Floty Tartaru pływają na drakkarach, dromonach, barkach, triremach, czołnach i ładjach."),
        ("Snajper Dalekosiężny", "Snajpera Dalekosiężnego", "Nie znam tej formacji; na dystans walczą łucznicy i kusznicy."),
        ("Kryształowy Reaktor", "Kryształowego Reaktora", "Taki obiekt nie występuje w ruinach ani świątyniach Tartaru."),
        ("Bateria Artyleryjska", "Baterii Artyleryjskiej", "Na polach bitew Tartaru nie stosuje się artylerii prochowej."),
    ],
}

# Rozłączne prefiksy per split (zapobiegają wyciekowi promptów między train / val / test)
USER_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "train": ("", "Krótko: ", "Panie, ", "Meldunek: ", "Doradź: "),
    "val": ("", "Raportuj: ", "Odpowiedz proszę: "),
    "test": ("", "Podaj dokładnie: ", "Oceń sytuację: "),
}


def apply_prefix(prefix: str, text: str) -> str:
    if not prefix:
        return text
    first_word = text.split(" ", 1)[0].rstrip(".,:;!?")
    if first_word.lower() in SAFE_LOWERCASE_STARTS:
        return prefix + text[0].lower() + text[1:]
    return prefix + text


# ═══════════════════════════════════════════════════════════════════════════════
# ZARZĄDZANIE PODZIAŁEM STANÓW (HIGIENA TRAIN / VAL / TEST)
# ═══════════════════════════════════════════════════════════════════════════════


def split_bucket_for_state(faction: str, loc_key: str, terrain: str, turn: int) -> str:
    """
    Deterministycznie przypisuje kombinację (frakcja, lokacja, teren, przedział_tury)
    do dokładnie jednego zbioru: train / val / test przed generacją próbek.
    """
    turn_bucket = turn // 2
    sig = f"{faction}|{loc_key}|{terrain}|{turn_bucket}".encode("utf-8")
    digest = int(hashlib.blake2b(sig, digest_size=4).hexdigest(), 16) % 100
    if digest < 70:
        return "train"
    if digest < 85:
        return "val"
    return "test"


def format_state_from_dict(st: dict, compact: bool, required_keys: Sequence[str] = ()) -> str:
    """Buduje zwięzły ciąg klucz=wartość; mieszczący się bezpiecznie w małym CTX."""
    all_fields = [
        ("lokacja", st["loc"].nom),
        ("typ", st["loc"].kind),
        ("teren", st["terrain"].nom),
        ("tura", str(st["turn"])),
        ("frakcja", st["faction"].nom),
        ("dowódca", st["faction"].leader_nom if st["hero_alive"] else "poległ"),
        ("hp", f"{st['hp']}/{st['max_hp']}"),
        ("mana", f"{st['mana']}/{st['max_mana']}"),
        ("złoto", str(st["gold"])),
        ("ruch", str(st["mv"])),
        ("jednostki", st["units_str"]),
        ("wrogowie", st["enemy_str"]),
    ]
    if not compact:
        return "; ".join(f"{k}={v}" for k, v in all_fields)

    core_keys = {"lokacja", "hp", "jednostki"} | set(required_keys)
    return "; ".join(f"{k}={v}" for k, v in all_fields if k in core_keys)


def build_state_pool(rng: random.Random, target_split: str, count: int) -> List[dict]:
    """
    Buduje zróżnicowane stany gry przypisane wyłącznie do target_split.
    KRYTYCZNE: dla `train` losuje wyłącznie lokacje, tereny i jednostki z `split == "train"`,
    dzięki czemu żadna encja holdout z `val`/`test` nie wycieka do `train`.
    """
    states: List[dict] = []
    seen_state_sigs: Set[str] = set()

    faction_keys = list(FACTIONS.keys())
    if target_split == "train":
        loc_pool = [l for l in LOCATIONS if l.split == "train"]
        terrain_pool = [t for t in TERRAINS if t.split == "train"]
    elif target_split == "val":
        loc_pool = [l for l in LOCATIONS if l.split == "val"]
        terrain_pool = [t for t in TERRAINS if t.split in ("val", "train")]
    else:
        loc_pool = [l for l in LOCATIONS if l.split == "test"]
        terrain_pool = [t for t in TERRAINS if t.split in ("test", "train")]

    attempts = 0
    max_attempts = max(20000, count * 25)
    while len(states) < count and attempts < max_attempts:
        attempts += 1
        f_key = rng.choice(faction_keys)
        faction = FACTIONS[f_key]
        loc = rng.choice(loc_pool)
        terrain = rng.choice(terrain_pool)
        turn = rng.randint(1, 36)

        if split_bucket_for_state(f_key, loc.key, terrain.key, turn) != target_split:
            continue

        max_hp = rng.choice([70, 75, 80])
        hp = rng.randint(8, max_hp)
        max_mana = rng.choice([12, 14, 16, 20])
        mana = rng.randint(2, max_mana)
        gold = rng.choice([45, 75, 110, 140, 185, 220, 275, 310, 390, 460, 540, 620, 750, 890])
        mv = rng.randint(1, 8)
        hero_alive = rng.random() > 0.15
        enemy_dist = rng.choice([0, 1, 2, 4, 6, 8, 99, 99])

        if target_split == "train":
            faction_units = [u for u in UNITS if u.faction == f_key and not u.naval and u.split == "train"]
        elif target_split == "val":
            faction_units = [u for u in UNITS if u.faction == f_key and not u.naval and u.split == "val"]
            if not faction_units:
                faction_units = [u for u in UNITS if not u.naval and u.split == "val"]
        else:
            faction_units = [u for u in UNITS if u.faction == f_key and not u.naval and u.split == "test"]
            if not faction_units:
                faction_units = [u for u in UNITS if not u.naval and u.split == "test"]

        u1 = rng.choice(faction_units)
        u2 = rng.choice(faction_units)
        c1 = rng.randint(1, 4)
        c2 = rng.randint(1, 3)
        if u1.key == u2.key:
            units_str = f"{u1.nom}x{c1}"
            total_units = c1 + (1 if hero_alive else 0)
        else:
            units_str = f"{u1.nom}x{c1},{u2.nom}x{c2}"
            total_units = c1 + c2 + (1 if hero_alive else 0)

        enemy_faction = FACTIONS[rng.choice([k for k in faction_keys if k != f_key])]
        if enemy_dist == 0:
            enemy_str = f"{enemy_faction.nom}(starcie)"
        elif enemy_dist <= 8:
            enemy_str = f"{enemy_faction.nom}({enemy_dist}p)"
        else:
            enemy_str = "brak"

        st = {
            "faction": faction,
            "loc": loc,
            "terrain": terrain,
            "turn": turn,
            "hp": hp,
            "max_hp": max_hp,
            "mana": mana,
            "max_mana": max_mana,
            "gold": gold,
            "mv": mv,
            "hero_alive": hero_alive,
            "u1": u1,
            "u2": u2,
            "c1": c1,
            "c2": c2,
            "total_units": total_units,
            "units_str": units_str,
            "enemy_dist": enemy_dist,
            "enemy_faction": enemy_faction,
            "enemy_str": enemy_str,
        }
        sig = format_state_from_dict(st, compact=False)
        if sig in seen_state_sigs:
            continue
        seen_state_sigs.add(sig)
        states.append(st)

    return states


# ═══════════════════════════════════════════════════════════════════════════════
# SZABLONY DIALOGOWE Z WARUNKAMI SPÓJNOŚCI (HERO / ENEMY)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DialogTemplate:
    user: str
    npc: str
    req_keys: Tuple[str, ...]
    req_hero: Optional[bool] = None   # True = tylko żywy wódz, False = tylko poległy, None = dowolnie
    req_enemy: Optional[bool] = None  # True = enemy_dist <= 8, False = enemy_dist == 99, None = dowolnie


DIALOG_TEMPLATES: Dict[str, List[DialogTemplate]] = {
    "train": [
        DialogTemplate(
            "Jaki jest nasz plan wymarszu {prep_in}?",
            "Ruszamy w szyku zwartym. {u1_pl} osłonią skrzydła, a {leader} poprowadzi atak ku następnemu grodowi.",
            ("lokacja", "dowódca", "jednostki"),
            req_hero=True,
        ),
        DialogTemplate(
            "Kto przejmie dowodzenie po stracie wodza {prep_in}?",
            "Skoro dowódca poległ, chorągiew {faction_gen} prowadzi zastępca; gdy dotrzemy do stolicy {faction_gen}, wskrzesimy wodza za 150 złota.",
            ("lokacja", "dowódca", "frakcja"),
            req_hero=False,
        ),
        DialogTemplate(
            "Czy {u1_nom} {u1_v_utrzyma} linię przeciwko oddziałom {enemy_gen}?",
            "{u1_nom} {u1_v_zna} swoją wartość ({u1_special}). Jeśli zachowamy szyk {terrain_prep}, nie damy się przełamać.",
            ("teren", "jednostki", "wrogowie"),
            req_enemy=True,
        ),
        DialogTemplate(
            "Na horyzoncie spokojnie — co robimy dalej {prep_in}?",
            "Wrogów brak w zasięgu zwiadu, więc {u1_pl} mogą ruszyć ku najbliższej warowni albo uzupełnić zapasy.",
            ("lokacja", "jednostki", "wrogowie"),
            req_enemy=False,
        ),
        DialogTemplate(
            "Wodzu, czy morale w chorągwi {faction_gen} nie słabnie?",
            "W {turn}. turze wojny serca wojowników wciąż biją równo. Złoto ({gold} zł) i chwała utrzymają dyscyplinę.",
            ("tura", "frakcja", "złoto", "dowódca"),
            req_hero=True,
        ),
        DialogTemplate(
            "Czy mamy dość sił w {turn}. turze, skoro dowódca poległ?",
            "Choć {u1_pl} odczuwają brak wodza, skarbiec ({gold} zł) pozwoli przetrwać i odbudować dowodzenie w stolicy.",
            ("tura", "dowódca", "złoto", "jednostki"),
            req_hero=False,
        ),
        DialogTemplate(
            "Czy powinniśmy teraz zrekrutować {u2_acc}?",
            "Przy skarbcu {gold} sztuk złota koszt {u2_cost} zł jest możliwy do udźwignięcia, jeśli zamek ma wolne miejsce.",
            ("złoto", "jednostki"),
        ),
        DialogTemplate(
            "Meldunek z przedpola: co robimy w tej turze?",
            "Jesteśmy {prep_in}. Rozważmy marsz na {enemy_check} albo wzmocnienie szyku przed kolejnym starciem.",
            ("lokacja", "wrogowie"),
            req_enemy=True,
        ),
        DialogTemplate(
            "Jakie rozkazy dla straży tylnej przed nocą?",
            "Rozstawić warty wokół miejsca postoju {prep_in}. {u1_pl} czuwają na pierwszej linii.",
            ("lokacja", "jednostki"),
        ),
        DialogTemplate(
            "Ile czasu zdołamy utrzymać szyk {terrain_prep}?",
            "{terrain_prep_cap} ruch kosztuje {terrain_cost}, lecz przy {mv} PM zachowujemy jeszcze swobodę manewru.",
            ("teren", "ruch"),
        ),
    ],
    "val": [
        DialogTemplate(
            "Panie, czy warto zatrzymać się dłużej {prep_in}?",
            "Zbyt długi postój {prep_in} odda inicjatywę przeciwnikowi ({enemy_str}). Gdy uzupełnimy siły, podejmijmy marsz.",
            ("lokacja", "wrogowie", "dowódca"),
            req_hero=True,
            req_enemy=True,
        ),
        DialogTemplate(
            "Czy możemy bezpiecznie odpocząć {prep_in}?",
            "Skoro wrogów brak w zasięgu, możemy uporządkować oddziały ({units_str}) i przygotować dalszy przemarsz.",
            ("lokacja", "jednostki", "wrogowie"),
            req_enemy=False,
        ),
        DialogTemplate(
            "Czy {u1_pl_our} {u1_pl} poradzą sobie {terrain_prep}?",
            "{terrain_prep_cap} każdy krok kosztuje {terrain_cost}, lecz {u1_nom} dobrze {u1_v_znosi} trudy tej kampanii.",
            ("teren", "jednostki"),
        ),
        DialogTemplate(
            "Co sądzisz o sile wojsk {enemy_gen} w pobliżu?",
            "Nie lekceważmy {enemy_gen} w zasięgu {enemy_dist} {enemy_pole_gen}. Wypatrujmy ich podjazdów i trzymajmy w odwodzie {u2_acc}.",
            ("wrogowie", "jednostki"),
            req_enemy=True,
        ),
        DialogTemplate(
            "Jak utrzymać porządek w oddziale bez żywego dowódcy?",
            "Zastępca przejmuje komendę nad oddziałem ({units_str}); wycofajmy się w stronę stolicy {faction_gen}, by wskrzesić wodza.",
            ("dowódca", "jednostki", "frakcja"),
            req_hero=False,
        ),
    ],
    "test": [
        DialogTemplate(
            "Jak oceniasz gotowość naszego hufca w {turn}. turze?",
            "Hufiec {faction_gen} stoi {prep_in} w sile {total_units} oddziałów. Zachowajmy czujność i chrońmy flanki.",
            ("tura", "frakcja", "lokacja", "jednostki"),
        ),
        DialogTemplate(
            "Czy mamy wyruszyć dalej czy bronić obecnych pozycji?",
            "Dopóki mamy zapas {mv} punktów ruchu i sprawne oddziały ({units_str}), możemy narzucić własne warunki.",
            ("ruch", "jednostki"),
        ),
        DialogTemplate(
            "Słowo otuchy dla żołnierzy przed starciem, dowódco!",
            "Za {leader_acc} i sztandar {faction_gen}! Niech {u1_pl} staną jak mur przeciw siłom {enemy_gen}.",
            ("dowódca", "frakcja", "jednostki", "wrogowie"),
            req_hero=True,
            req_enemy=True,
        ),
        DialogTemplate(
            "Czy brak wodza zagraża naszemu marszowi {prep_in}?",
            "Dopóki nie wskrzesimy dowódcy {faction_gen}, nie zbadamy ruin ani świątyń, lecz {u1_pl} nadal bronią kolumny.",
            ("dowódca", "lokacja", "frakcja", "jednostki"),
            req_hero=False,
        ),
        DialogTemplate(
            "Okolica wokół nas wydaje się pusta — jaki następny krok?",
            "Wrogów brak na radarze zwiadu. Wykorzystajmy {mv} PM i {gold} złota na umocnienie szeregów.",
            ("wrogowie", "ruch", "złoto"),
            req_enemy=False,
        ),
    ],
}


STATE_QUERY_TEMPLATES: Dict[str, Dict[str, List[Tuple[str, str, Tuple[str, ...]]]]] = {
    "hp": {
        "train": [
            ("Ile punktów życia ma nasz dowódca?", "Aktualny stan zdrowia wodza wynosi {hp} na {max_hp} HP.", ("hp",)),
            ("Podaj obecne HP w raporcie.", "W raporcie widnieje {hp}/{max_hp} HP.", ("hp",)),
            ("W jakiej kondycji jest nasz bohater?", "Stan bohatera to {hp}/{max_hp} HP — {hp_note}.", ("hp", "dowódca")),
            ("Sprawdź, ile zdrowia pozostało wodzowi.", "Zapis stanu podaje {hp} z {max_hp} HP.", ("hp",)),
        ],
        "val": [
            ("Czy nasz dowódca potrzebuje leczenia?", "Ma obecnie {hp} z {max_hp} HP, więc {hp_heal_advice}.", ("hp",)),
            ("Sprawdź poziom wytrzymałości wodza.", "Poziom zdrowia wynosi dokładnie {hp}/{max_hp} HP.", ("hp",)),
            ("Ile punktów życia figuruje teraz przy dowódcy?", "Aktualny odczyt to {hp}/{max_hp} HP.", ("hp",)),
        ],
        "test": [
            ("Odczytaj aktualną wartość HP ze stanu bitwy.", "Zapis stanu wskazuje {hp}/{max_hp} HP.", ("hp",)),
            ("Ile zdrowia zostało nam po ostatnich walkach?", "Pozostało nam {hp} punktów życia z maksymalnych {max_hp}.", ("hp",)),
            ("Wskaż bieżące HP naszego dowódcy.", "Obecne HP wynosi {hp}/{max_hp}.", ("hp",)),
        ],
    },
    "mana_gold": {
        "train": [
            ("Ile mamy złota i many w skarbcu?", "W skarbcu znajduje się {gold} złota oraz {mana}/{max_mana} punktów many.", ("złoto", "mana")),
            ("Podaj stan zasobów: złoto i mana.", "Zasoby królestwa: {gold} sztuk złota i {mana}/{max_mana} many.", ("złoto", "mana")),
            ("Na jak kosztowne zaklęcia wystarczy nam many?", "Dysponujemy {mana} z {max_mana} punktów many oraz {gold} złota.", ("mana", "złoto")),
            ("Przelicz nasze rezerwy złota i many.", "Stan wynosi {gold} sztuk złota oraz {mana}/{max_mana} many.", ("złoto", "mana")),
        ],
        "val": [
            ("Czy stać nas na wydatki i czary w tej turze?", "Mamy obecnie {gold} złota w skarbcu i {mana}/{max_mana} many.", ("złoto", "mana")),
            ("Raportuj stan skarbca i rezerw magicznych.", "Skarbiec mieści {gold} złota, a rezerwa many wynosi {mana}/{max_mana}.", ("złoto", "mana")),
            ("Ile dokładnie monet i many nam zostało?", "W skarbcu jest {gold} złota, a mana wynosi {mana}/{max_mana}.", ("złoto", "mana")),
        ],
        "test": [
            ("Jaki jest bilans złota i zasób many w {turn}. turze?", "W {turn}. turze posiadamy {gold} złota oraz {mana}/{max_mana} many.", ("tura", "złoto", "mana")),
            ("Sprawdź, ile mamy monet i many.", "Stan wynosi {gold} złota i {mana}/{max_mana} many.", ("złoto", "mana")),
            ("Określ stan skarbca oraz pulę many.", "Dysponujemy kwotą {gold} złota i {mana}/{max_mana} punktami many.", ("złoto", "mana")),
        ],
    },
    "location_terrain": {
        "train": [
            ("Gdzie obecnie stacjonuje nasza armia?", "Armia przebywa {prep_in} ({loc_kind}), a teren pod stopami to {terrain_nom}.", ("lokacja", "typ", "teren")),
            ("Jaka jest nasza lokacja i ukształtowanie terenu?", "Znajdujemy się {prep_in}; teren to {terrain_nom} (koszt ruchu {terrain_cost}).", ("lokacja", "teren")),
            ("Która to tura i gdzie jesteśmy?", "Trwa {turn}. tura. Stacjonujemy {prep_in} ({terrain_nom}).", ("tura", "lokacja", "teren")),
            ("Wskaż numer tury oraz typ obecnego miejsca postoju.", "Jest {turn}. tura; zajmujemy lokację {loc_nom} (typ: {loc_kind}).", ("tura", "lokacja", "typ")),
        ],
        "val": [
            ("Odczytaj z mapy naszą obecną pozycję.", "Pozycja oddziału: {loc_nom} ({loc_kind}), {terrain_prep}.", ("lokacja", "typ", "teren")),
            ("Na jakim podłożu stoimy i jak się nazywa ta lokacja?", "Stoimy {terrain_prep}, a lokacja to {loc_nom}.", ("lokacja", "teren")),
            ("Zdaj meldunek o terenie i bieżącej lokacji.", "Przebywamy {prep_in}, gdzie teren stanowi {terrain_nom}.", ("lokacja", "teren")),
        ],
        "test": [
            ("Podaj nazwę miejsca postoju oraz rodzaj terenu.", "Miejsce postoju to {loc_nom}, zaś teren stanowi {terrain_nom}.", ("lokacja", "teren")),
            ("W której turze i w jakim miejscu trwa kampania?", "Jest {turn}. tura, a nasze chorągwie stoją {prep_in}.", ("tura", "lokacja")),
            ("Jakiego typu jest nasza aktualna lokacja i na jakim leży terenie?", "{loc_nom} ma typ „{loc_kind}” i leży {terrain_prep}.", ("lokacja", "typ", "teren")),
        ],
    },
    "units_enemies": {
        "train": [
            ("Jakie oddziały mamy teraz pod rozkazami?", "Pod rozkazami mamy: {units_str} (łącznie {total_units}, dowódca: {hero_status}).", ("jednostki", "dowódca")),
            ("Czy w zasięgu widać wrogie siły?", "W meldunku zwiadu wrogowie: {enemy_desc}.", ("wrogowie",)),
            ("Ile punktów ruchu nam jeszcze zostało?", "Armia dysponuje jeszcze {mv} punktami ruchu w tej turze.", ("ruch",)),
            ("Podaj skład chorągwi oraz pozostały zapas ruchu.", "W szyku idą: {units_str}, a zapas marszu wynosi {mv} PM.", ("jednostki", "ruch")),
        ],
        "val": [
            ("Wymień skład naszego oddziału i sytuację z wrogiem.", "Skład hufca: {units_str}. Zasięg wroga: {enemy_desc}.", ("jednostki", "wrogowie")),
            ("Sprawdź nasz limit ruchu oraz stan armii.", "Pozostało {mv} PM, a w szyku idą: {units_str}.", ("ruch", "jednostki")),
            ("Czy dowódca żyje i ilu mamy przeciwników w pobliżu?", "Status dowódcy: {hero_status}. Wróg: {enemy_desc}.", ("dowódca", "wrogowie")),
        ],
        "test": [
            ("Ilu wojowników liczy nasza kolumna i czy ktoś nam zagraża?", "W kolumnie maszerują: {units_str} (dowódca: {hero_status}). Przeciwnik: {enemy_desc}.", ("jednostki", "dowódca", "wrogowie")),
            ("Jaki jest nasz zapas marszowy i najbliższy wróg?", "Zapas ruchu wynosi {mv} PM, a status wroga to: {enemy_desc}.", ("ruch", "wrogowie")),
            ("Wymień obecny skład armii oraz status dowódcy.", "Armia składa się z: {units_str}; dowódca: {hero_status}.", ("jednostki", "dowódca")),
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATORY POSZCZEGÓLNYCH KATEGORII
# ═══════════════════════════════════════════════════════════════════════════════


def _render_dialog_sample(rng: random.Random, st: dict, split: str) -> Tuple[str, str, str]:
    candidates = [
        tpl
        for tpl in DIALOG_TEMPLATES[split]
        if (tpl.req_hero is None or tpl.req_hero == st["hero_alive"])
        and (tpl.req_enemy is None or (tpl.req_enemy and st["enemy_dist"] <= 8) or (not tpl.req_enemy and st["enemy_dist"] == 99))
    ]
    if not candidates:
        candidates = [tpl for tpl in DIALOG_TEMPLATES[split] if tpl.req_hero is None and tpl.req_enemy is None]

    tpl = rng.choice(candidates)
    loc: LocationEntry = st["loc"]
    u1: UnitEntry = st["u1"]
    u2: UnitEntry = st["u2"]
    faction: FactionEntry = st["faction"]
    terrain: TerrainEntry = st["terrain"]
    enemy: FactionEntry = st["enemy_faction"]

    if st["enemy_dist"] == 0:
        enemy_check = f"bitwę z siłami {enemy.gen}"
    elif st["enemy_dist"] <= 8:
        enemy_check = f"oddziały {enemy.gen} oddalone o {st['enemy_dist']} {plural_pole(st['enemy_dist'])}"
    else:
        enemy_check = "najbliższą twierdzę lub ruiny"

    mapping = {
        "prep_in": loc.prep_in,
        "u1_nom": u1.nom,
        "u1_pl": u1.pl_nom,
        "u1_pl_our": u1.pl_our,
        "u1_special": u1.special_name,
        "u1_v_utrzyma": u1.v_utrzyma,
        "u1_v_zna": u1.v_zna,
        "u1_v_znosi": u1.v_znosi,
        "u2_acc": u2.acc,
        "u2_cost": u2.cost,
        "leader": faction.leader_nom,
        "leader_acc": faction.leader_gen,
        "faction_gen": faction.gen,
        "enemy_gen": enemy.gen,
        "enemy_dist": st["enemy_dist"],
        "enemy_pole_gen": plural_pole_gen(st["enemy_dist"]),
        "enemy_str": st["enemy_str"],
        "enemy_check": enemy_check,
        "terrain_prep": terrain.prep_in,
        "terrain_prep_cap": terrain.prep_in.capitalize(),
        "terrain_cost": terrain.move_cost,
        "turn": st["turn"],
        "gold": st["gold"],
        "mv": st["mv"],
        "total_units": st["total_units"],
        "units_str": st["units_str"],
    }
    prefix = rng.choice(USER_PREFIXES[split])
    state_str = format_state_from_dict(st, compact=(rng.random() < 0.55), required_keys=tpl.req_keys)
    user_txt = apply_prefix(prefix, tpl.user.format(**mapping))
    npc_txt = tpl.npc.format(**mapping)
    return state_str, user_txt, npc_txt


def _render_state_query_sample(rng: random.Random, st: dict, split: str) -> Tuple[str, str, str]:
    subkind = rng.choice(list(STATE_QUERY_TEMPLATES.keys()))
    tpl_u, tpl_n, req_keys = rng.choice(STATE_QUERY_TEMPLATES[subkind][split])
    loc: LocationEntry = st["loc"]
    terrain: TerrainEntry = st["terrain"]
    faction: FactionEntry = st["faction"]

    hp_ratio = st["hp"] / st["max_hp"]
    if hp_ratio >= 0.75:
        hp_note = "jest w pełni sił do walki"
        hp_heal_advice = "nie wymaga pilnego leczenia"
    elif hp_ratio >= 0.4:
        hp_note = "nosi ślady starć, lecz nadal stoi w szyku"
        hp_heal_advice = "warto rozważyć Balsam Żywiołów przed większą bitwą"
    else:
        hp_note = "jest ciężko ranny i wymaga ostrożności"
        hp_heal_advice = "pilnie potrzebuje leczenia lub odwrotu do twierdzy"

    if st["enemy_dist"] == 0:
        enemy_desc = f"bezpośrednie starcie z {st['enemy_faction'].nom}"
    elif st["enemy_dist"] <= 8:
        enemy_desc = f"{st['enemy_faction'].nom} w odległości {st['enemy_dist']} {plural_pole_gen(st['enemy_dist'])}"
    else:
        enemy_desc = "brak wykrytych oddziałów wroga w zasięgu"

    hero_status = faction.leader_nom if st["hero_alive"] else "poległ (do wskrzeszenia w stolicy)"

    mapping = {
        "hp": st["hp"],
        "max_hp": st["max_hp"],
        "hp_note": hp_note,
        "hp_heal_advice": hp_heal_advice,
        "mana": st["mana"],
        "max_mana": st["max_mana"],
        "gold": st["gold"],
        "turn": st["turn"],
        "mv": st["mv"],
        "loc_nom": loc.nom,
        "loc_kind": loc.kind,
        "prep_in": loc.prep_in,
        "terrain_nom": terrain.nom,
        "terrain_prep": terrain.prep_in,
        "terrain_cost": terrain.move_cost,
        "units_str": st["units_str"],
        "total_units": st["total_units"],
        "hero_status": hero_status,
        "enemy_desc": enemy_desc,
    }
    prefix = rng.choice(USER_PREFIXES[split])
    state_str = format_state_from_dict(st, compact=(rng.random() < 0.6), required_keys=req_keys)
    return state_str, apply_prefix(prefix, tpl_u.format(**mapping)), tpl_n.format(**mapping)


def _render_lore_sample(rng: random.Random, st: dict, split: str) -> Tuple[str, str, str]:
    """
    Generuje pytania i opisy encji świata Tartarus wyłącznie z puli encji przypisanej do `split`,
    z wagami proporcjonalnymi do pojemności kombinatorycznej podkategorii.
    """
    unit_pool = [u for u in UNITS if u.split == split]
    loc_pool = [l for l in LOCATIONS if l.split == split]
    spell_pool = [s for s in SPELLS if s.split == split]
    art_pool = [a for a in ARTIFACTS if a.split == split]
    fac_pool = [f for f in FACTIONS.values() if f.split == split] or list(FACTIONS.values())
    terr_pool = [t for t in TERRAINS if t.split == split]

    # Wagi proporcjonalne do bogactwa kombinacji w podkategorii (teren nie mieli 4 zdań tak często jak porównania!)
    weighted_choices = [
        ("unit", max(2, len(unit_pool) * 3)),
        ("unit_compare", max(6, len(unit_pool) * max(1, len(unit_pool) - 1))),
        ("location", max(2, len(loc_pool) * 2)),
        ("spell", max(1, len(spell_pool) * 2)),
        ("artifact", max(1, len(art_pool) * 2)),
        ("faction", max(1, len(fac_pool) * 2)),
        ("terrain", max(1, len(terr_pool))),
    ]
    labels, weights = zip(*weighted_choices)
    entity_type = rng.choices(labels, weights=weights, k=1)[0]

    state_str = format_state_from_dict(st, compact=(rng.random() < 0.65))
    prefix = rng.choice(USER_PREFIXES[split])

    if entity_type == "unit":
        u = rng.choice(unit_pool)
        f_nom = FACTIONS[u.faction].nom
        f_gen = FACTIONS[u.faction].gen
        if split == "train":
            prompts = [
                (f"Co {u.v_potrafi} {u.nom} w walce?", f"{u.nom} ({f_nom}, rola: {u.role}) {u.v_ma} {u.atk} Ataku, {u.defense} Obrony, {u.hp} HP i {u.ini} INI. Cecha {u.special_name}: {u.special_desc}."),
                (f"Podaj koszt i statystyki {u.gen}.", f"Rekrutacja {u.gen} kosztuje {u.cost} złota. Statystyki: ATK {u.atk}, DEF {u.defense}, HP {u.hp}, ruch {u.mv}."),
                (f"Dlaczego warto mieć w armii {u.acc}?", f"Z {u.ins} zyskujesz atut {u.special_name} — {u.special_desc} (koszt {u.cost} zł)."),
                (f"Ile punktów życia oraz inicjatywy {u.v_ma} {u.nom}?", f"{u.nom} {u.v_dysponuje} {u.hp} HP, {u.ini} Inicjatywy oraz {u.mv} PM."),
                (f"Jaką zdolnością specjalną wyróżnia się {u.nom}?", f"{u.nom} posiada cechę „{u.special_name}”: {u.special_desc}."),
            ]
        elif split == "val":
            prompts = [
                (f"Scharakteryzuj oddział: {u.nom}.", f"{u.nom} {u.v_nalezy} do {f_gen} ({u.role}). Parametry: {u.atk}/{u.defense} (ATK/DEF), {u.hp} HP, {u.mv} PM; {u.special_name}: {u.special_desc}."),
                (f"Jaką rolę na polu bitwy {u.v_pelni} {u.nom}?", f"{u.nom} to {u.role} frakcji {f_nom} i {u.v_ma} zdolność {u.special_name}: {u.special_desc}."),
                (f"Ile złota {u.v_kosztuje} {u.nom} i jak szybko się porusza?", f"Koszt {u.gen} to {u.cost} złota, a zasięg marszu wynosi {u.mv} PM (inicjatywa {u.ini})."),
                (f"Jaką obroną i atakiem {u.v_dysponuje} {u.nom}?", f"{u.nom} {u.v_ma} {u.atk} Ataku, {u.defense} Obrony oraz {u.hp} punktów życia."),
                (f"Z jakiego ludu wywodzi się {u.nom}?", f"{u.nom} służy w szeregach {f_gen} (rola: {u.role}, koszt: {u.cost} zł)."),
            ]
        else:
            prompts = [
                (f"Opowiedz o zdolnościach {u.gen} z szeregów {f_gen}.", f"{u.nom} {u.v_kosztuje} {u.cost} złota, {u.v_ma} {u.hp} HP, {u.atk} Ataku i zdolność {u.special_name}: {u.special_desc}."),
                (f"Czym wyróżniają się {u.pl_nom}?", f"{u.pl_nom} to {u.role} ({f_nom}), o sile {u.atk} ATK / {u.defense} DEF; {u.special_name}: {u.special_desc}."),
                (f"Czy {u.nom} {u.v_jest} {u.adj_drogi} w werbunku i {u.adj_wytrzymaly}?", f"Werbunek {u.gen} kosztuje {u.cost} zł, a jednostka ma {u.hp} HP i {u.defense} Obrony."),
                (f"Podaj profil bojowy dla jednostki {u.nom}.", f"Profil {u.gen}: {u.role}, {u.atk} ATK, {u.defense} DEF, {u.hp} HP, {u.ini} INI, {u.mv} PM."),
                (f"Jaką cechą szczególną dysponuje {u.nom}?", f"{u.nom} ({f_nom}) korzysta ze zdolności {u.special_name} — {u.special_desc}."),
            ]
        u_txt, n_txt = rng.choice(prompts)
        return state_str, apply_prefix(prefix, u_txt), n_txt

    if entity_type == "unit_compare":
        u1, u2 = rng.sample(unit_pool, 2)
        attr = rng.choice(("atk", "defense", "hp", "cost", "ini", "mv"))
        if attr == "atk":
            if u1.atk > u2.atk:
                ans = f"Wyższy Atak {u1.v_ma} {u1.nom} ({u1.atk} wobec {u2.atk} u {u2.gen})."
            elif u2.atk > u1.atk:
                ans = f"Wyższy Atak {u2.v_ma} {u2.nom} ({u2.atk} wobec {u1.atk} u {u1.gen})."
            else:
                ans = f"Obie jednostki ({u1.nom} i {u2.nom}) mają równy Atak: {u1.atk}."
            if split == "train":
                q = f"Kto zadaje silniejsze ciosy: {u1.nom} czy {u2.nom}?"
            elif split == "val":
                q = f"Porównaj wartość ataku {u1.gen} i {u2.gen}."
            else:
                q = f"Która jednostka ma większy Atak — {u1.nom} czy {u2.nom}?"
        elif attr == "defense":
            if u1.defense > u2.defense:
                ans = f"Twardszą Obronę {u1.v_ma} {u1.nom} ({u1.defense} wobec {u2.defense} u {u2.gen})."
            elif u2.defense > u1.defense:
                ans = f"Twardszą Obronę {u2.v_ma} {u2.nom} ({u2.defense} wobec {u1.defense} u {u1.gen})."
            else:
                ans = f"{u1.nom} oraz {u2.nom} mają taką samą Obronę ({u1.defense})."
            if split == "train":
                q = f"Kto ma lepszy pancerz: {u1.nom} czy {u2.nom}?"
            elif split == "val":
                q = f"Zestaw Obronę {u1.gen} z {u2.ins}."
            else:
                q = f"Kto lepiej wytrzymuje ciosy w obronie: {u1.nom} czy {u2.nom}?"
        elif attr == "hp":
            if u1.hp > u2.hp:
                ans = f"Więcej HP {u1.v_ma} {u1.nom} ({u1.hp} HP kontra {u2.hp} HP u {u2.gen})."
            elif u2.hp > u1.hp:
                ans = f"Więcej HP {u2.v_ma} {u2.nom} ({u2.hp} HP kontra {u1.hp} HP u {u1.gen})."
            else:
                ans = f"{u1.nom} i {u2.nom} mają identyczną pulę {u1.hp} HP."
            if split == "train":
                q = f"Kto ma więcej punktów życia: {u1.nom} czy {u2.nom}?"
            elif split == "val":
                q = f"Porównaj żywotność (HP) {u1.gen} oraz {u2.gen}."
            else:
                q = f"Który oddział ma większą pulę HP: {u1.nom} czy {u2.nom}?"
        elif attr == "ini":
            if u1.ini > u2.ini:
                ans = f"Wyższą Inicjatywę {u1.v_ma} {u1.nom} ({u1.ini} INI wobec {u2.ini} INI u {u2.gen})."
            elif u2.ini > u1.ini:
                ans = f"Wyższą Inicjatywę {u2.v_ma} {u2.nom} ({u2.ini} INI wobec {u1.ini} INI u {u1.gen})."
            else:
                ans = f"Zarówno {u1.nom}, jak i {u2.nom} mają tę samą Inicjatywę ({u1.ini})."
            if split == "train":
                q = f"Kto uderzy pierwszy w starciu: {u1.nom} czy {u2.nom}?"
            elif split == "val":
                q = f"Która jednostka ma wyższą Inicjatywę: {u1.nom} czy {u2.nom}?"
            else:
                q = f"Porównaj Inicjatywę {u1.gen} i {u2.gen}."
        elif attr == "mv":
            if u1.mv > u2.mv:
                ans = f"{u1.adj_szybszy} w marszu {u1.v_jest} {u1.nom} ({u1.mv} PM wobec {u2.mv} PM u {u2.gen})."
            elif u2.mv > u1.mv:
                ans = f"{u2.adj_szybszy} w marszu {u2.v_jest} {u2.nom} ({u2.mv} PM wobec {u1.mv} PM u {u1.gen})."
            else:
                ans = f"Zasięg ruchu obu oddziałów ({u1.nom} i {u2.nom}) jest identyczny i wynosi {u1.mv} PM."
            if split == "train":
                q = f"Kto szybciej przemierza mapę: {u1.nom} czy {u2.nom}?"
            elif split == "val":
                q = f"Zestaw punkty ruchu (PM) {u1.gen} oraz {u2.gen}."
            else:
                q = f"Który oddział ma większy zasięg marszu: {u1.nom} czy {u2.nom}?"
        else:
            if u1.cost < u2.cost:
                ans = f"{u1.adj_tanszy} w werbunku {u1.v_jest} {u1.nom} ({u1.cost} zł wobec {u2.cost} zł za {u2.acc})."
            elif u2.cost < u1.cost:
                ans = f"{u2.adj_tanszy} w werbunku {u2.v_jest} {u2.nom} ({u2.cost} zł wobec {u1.cost} zł za {u1.acc})."
            else:
                ans = f"Obie jednostki ({u1.nom} oraz {u2.nom}) kosztują dokładnie {u1.cost} złota."
            if split == "train":
                q = f"Kogo taniej zrekrutować: {u1.acc} czy {u2.acc}?"
            elif split == "val":
                q = f"Porównaj cenę najmu {u1.gen} i {u2.gen}."
            else:
                q = f"Co mniej obciąży skarbiec: {u1.nom} czy {u2.nom}?"
        return state_str, apply_prefix(prefix, q), ans

    if entity_type == "location":
        loc = rng.choice(loc_pool)
        if split == "train":
            prompts = [
                (f"Czym {loc.v_slynie} {loc.nom}?", f"{loc.nom} to {loc.kind} — {loc.summary}. Wyróżnik: {loc.bonus_or_guard}."),
                (f"Co nas czeka, gdy wejdziemy {loc.prep_to}?", f"Gdy dotrzesz {loc.prep_to}, zastaniesz tam {loc.kind}: {loc.summary} ({loc.bonus_or_guard})."),
                (f"Opisz zagrożenia i dary {loc.gen}.", f"W obrębie {loc.gen} znajduje się {loc.kind} — {loc.summary}; {loc.bonus_or_guard}."),
            ]
        elif split == "val":
            prompts = [
                (f"Podaj wpis z kroniki dotyczący {loc.gen}.", f"Kronika określa {loc.acc} jako {loc.kind}: {loc.summary}. Kluczowa cecha: {loc.bonus_or_guard}."),
                (f"Co warto wiedzieć przed marszem {loc.prep_to}?", f"To {loc.kind} ({loc.nom}): {loc.summary}. Pamiętaj: {loc.bonus_or_guard}."),
                (f"Jakiego rodzaju miejscem {loc.v_jest} {loc.nom}?", f"{loc.nom} {loc.v_jest} {loc.adj_sklasyfikowane} jako {loc.kind} — {loc.summary}."),
            ]
        else:
            prompts = [
                (f"Jaką funkcję w Tartarze {loc.v_pelni} {loc.nom}?", f"{loc.nom} {loc.v_stanowi} {loc.kind} — {loc.summary}. Na miejscu obowiązuje: {loc.bonus_or_guard}."),
                (f"Co kroniki mówią o {loc.loc}?", f"W księgach o {loc.loc} zapisano, że to {loc.kind}: {loc.summary} ({loc.bonus_or_guard})."),
                (f"Czego spodziewać się w pobliżu {loc.gen}?", f"{loc.nom} ({loc.kind}) to {loc.summary}; ważna uwaga: {loc.bonus_or_guard}."),
            ]
        u_txt, n_txt = rng.choice(prompts)
        return state_str, apply_prefix(prefix, u_txt), n_txt

    if entity_type == "spell":
        sp = rng.choice(spell_pool)
        if split == "train":
            prompts = [
                (f"Jak działa czar {sp.nom}?", f"{sp.nom} (szkoła: {sp.school}) kosztuje {sp.mana_cost} many, ma zasięg {sp.range_tiles} pól i {sp.effect}."),
                (f"Ile many wymaga rzucenie {sp.gen}?", f"Rzucenie {sp.gen} pochłania {sp.mana_cost} many i {sp.effect}."),
                (f"Jaki jest zasięg i koszt zaklęcia {sp.nom}?", f"Czar {sp.nom} ma zasięg {sp.range_tiles} pól i kosztuje {sp.mana_cost} MP."),
            ]
        elif split == "val":
            prompts = [
                (f"Co osiągniemy zaklęciem {sp.ins}?", f"Dzięki inkantacji {sp.gen} (koszt {sp.mana_cost} many) zaklęcie {sp.effect}."),
                (f"Do jakiej szkoły należy {sp.nom} i jaki ma zasięg?", f"{sp.nom} należy do szkoły {sp.school}, ma zasięg {sp.range_tiles} pól i kosztuje {sp.mana_cost} MP."),
            ]
        else:
            prompts = [
                (f"Wyjaśnij działanie rytuału {sp.gen} z Księgi Zaklęć.", f"Rytuał {sp.nom} ze szkoły {sp.school} zużywa {sp.mana_cost} many i {sp.effect}."),
                (f"Czy {sp.nom} wymaga dużego nakładu many?", f"Koszt {sp.gen} wynosi {sp.mana_cost} many (zasięg {sp.range_tiles} pól); efekt: {sp.effect}."),
            ]
        u_txt, n_txt = rng.choice(prompts)
        return state_str, apply_prefix(prefix, u_txt), n_txt

    if entity_type == "artifact":
        art = rng.choice(art_pool)
        if split == "train":
            prompts = [
                (f"Jakie premie daje {art.nom}?", f"{art.nom} ({art.rarity}, slot: {art.slot}) zapewnia dowódcy {art.bonus}."),
                (f"Czy warto zdobyć {art.acc}?", f"Zdecydowanie: noszenie {art.gen} gwarantuje {art.bonus}."),
                (f"Na jaki slot zakłada się {art.acc}?", f"{art.nom} trafia na slot „{art.slot}” i daje {art.bonus}."),
            ]
        elif split == "val":
            prompts = [
                (f"Co zyska bohater wyposażony w {art.acc}?", f"{art.adj_ten} {art.rarity} artefakt ({art.slot}) dodaje {art.bonus}."),
                (f"Jakiej rzadkości jest {art.nom} i do jakiego slotu trafia?", f"{art.nom} ma rzadkość „{art.rarity}”, pasuje na slot „{art.slot}” i daje {art.bonus}."),
            ]
        else:
            prompts = [
                (f"Opisz moc artefaktu znanego jako {art.nom}.", f"{art.nom} należy do kategorii „{art.slot}” ({art.rarity}) i daje {art.bonus}."),
                (f"Jaką korzyść w boju przynosi {art.nom}?", f"Po założeniu {art.gen} wódz otrzymuje {art.bonus}."),
            ]
        u_txt, n_txt = rng.choice(prompts)
        return state_str, apply_prefix(prefix, u_txt), n_txt

    if entity_type == "terrain":
        tr = rng.choice(terr_pool)
        if split == "train":
            u_txt = f"Ile punktów ruchu kosztuje marsz {tr.prep_in}?"
            n_txt = f"Przemarsz przez teren „{tr.nom}” kosztuje bazowo {tr.move_cost} (na skos +1 PM)."
        elif split == "val":
            u_txt = f"Jak ukształtowanie terenu ({tr.nom}) wpływa na ruch armii?"
            n_txt = f"Poruszanie się {tr.prep_in} pochłania {tr.move_cost}, chyba że armia leci lub ma specjalne zdolności."
        else:
            u_txt = f"Jaki jest koszt wejścia na pole typu {tr.nom}?"
            n_txt = f"Kafelek „{tr.nom}” wymaga {tr.move_cost} przy ruchu prostym."
        return state_str, apply_prefix(prefix, u_txt), n_txt

    f = rng.choice(fac_pool)
    if split == "train":
        prompts = [
            (f"Jaka jest taktyka {f.gen}?", f"{f.nom} pod wodzą {f.leader_gen} stosują {f.doctrine}. Stolica to {f.capital}."),
            (f"Czym broni się stolica {f.gen}?", f"Twierdza {f.capital} posiada aurę: {f.fortress_aura}."),
        ]
    elif split == "val":
        prompts = [
            (f"Kto przewodzi {f.dat} i jaka jest ich doktryna?", f"Wodzem w {f.loc} jest {f.leader_nom}. Ich siłą jest {f.doctrine}."),
            (f"Jaka stolica należy do {f.gen}?", f"Główną warownią {f.gen} jest {f.capital}, chroniona przez {f.fortress_aura}."),
        ]
    else:
        prompts = [
            (f"Streść charakterystykę frakcji {f.nom}.", f"{f.nom} (stolica {f.capital}, wódz {f.leader_nom}) opierają się na: {f.doctrine}. Obrona grodu: {f.fortress_aura}."),
            (f"Jaką aurą obronną dysponuje główna twierdza {f.gen}?", f"Stolica {f.gen} ({f.capital}) uruchamia efekt: {f.fortress_aura}."),
        ]
    u_txt, n_txt = rng.choice(prompts)
    return state_str, apply_prefix(prefix, u_txt), n_txt


def _render_refusal_sample(rng: random.Random, st: dict, split: str) -> Tuple[str, str, str]:
    """
    Tworzy przykłady negatywne i odmowy (wzorzec g_unknown + brak obiektu/zasobu +
    sprzeczne/absurdalne stany gry) przy użyciu rozłącznych encji i szablonów na split.
    """
    refusal_kind = rng.choice(("unknown_entity", "out_of_range", "missing_resource", "absurd_state"))
    compact = rng.random() < 0.6
    prefix = rng.choice(USER_PREFIXES[split])

    if refusal_kind == "unknown_entity":
        unk_nom, unk_gen, unk_reason = rng.choice(UNKNOWN_ENTITIES[split])
        state_str = format_state_from_dict(st, compact=compact)
        if split == "train":
            prompts = [
                (f"Rekrutuj oddział: {unk_nom}.", f"Nie mogę wykonać rozkazu. {unk_reason}"),
                (f"Co wiesz o {unk_gen}?", f"Nie wiem nic o takim mianie jak {unk_nom}. {unk_reason}"),
                (f"Wyślij zwiad w stronę {unk_gen}.", f"Brak takiego obiektu w rejestrach Tartaru. {unk_reason}"),
                (f"Dodaj do armii {unk_gen}.", f"Odrzucam żądanie — {unk_nom} nie istnieje w tym świecie. {unk_reason}"),
            ]
        elif split == "val":
            prompts = [
                (f"Czy w tym zamku wyszkolimy {unk_gen}?", f"Niestety nie znam takiej jednostki jak {unk_nom}. {unk_reason}"),
                (f"Użyj przeciw wrogowi: {unk_nom}.", f"Odmowa — w naszym świecie nie istnieje {unk_nom}. {unk_reason}"),
                (f"Gdzie na mapie leży {unk_nom}?", f"Nie odnajduję {unk_gen} w granicach Tartaru. {unk_reason}"),
            ]
        else:
            prompts = [
                (f"Sprawdź kroniki pod hasłem „{unk_nom}”.", f"Brak wpisu w Kodeksie Tartaru. {unk_reason}"),
                (f"Rozkaż bohaterowi zdobyć {unk_gen}.", f"Taki cel nie istnieje na mapie Tartaru. {unk_reason}"),
                (f"Opisz statystyki {unk_gen}.", f"Nie znam parametrów dla „{unk_nom}”. {unk_reason}"),
            ]
        u, n = rng.choice(prompts)
        return state_str, apply_prefix(prefix, u), n

    if refusal_kind == "out_of_range":
        mode = rng.choice(("no_enemy", "not_ruin", "no_hero"))
        if mode == "no_enemy":
            mod = dict(st)
            mod["enemy_str"] = "brak"
            state_str = format_state_from_dict(mod, compact=compact, required_keys=("wrogowie",))
            sp = rng.choice([s for s in SPELLS if s.split == split])
            if split == "train":
                u, n = rng.choice([
                    (f"Uderz w pobliżu {st['loc'].gen} na najbliższą wrogą armię!", f"Brak wrogich oddziałów w zasięgu wzroku wokół {st['loc'].gen}."),
                    (f"Wyślij {st['u1'].acc} do natarcia na wroga!", f"Nie widać żadnego przeciwnika w zasięgu marszu z {st['loc'].gen}."),
                    (f"Rzuć {sp.acc} w nadciągającego wroga!", f"Brak celu dla {sp.gen} — raport z {st['loc'].gen} wykazuje brak wrogów."),
                ])
            elif split == "val":
                u, n = rng.choice([
                    (f"Rzuć {sp.acc} w przeciwnika przed nami.", f"Brak wroga w zasięgu zaklęcia {sp.nom} — zwiad wykazuje brak przeciwników."),
                    (f"Zaatakuj sąsiedni oddział wroga z {st['loc'].gen}.", f"Odmowa ataku: wokół {st['loc'].gen} nie wykryto żadnych wrogich wojsk."),
                ])
            else:
                u, n = rng.choice([
                    (f"Rozpocznij natychmiastową szarżę {st['u1'].pl_gen} na nieprzyjaciela.", f"Odmowa: przed nami nie wykryto wrogiej armii w zasięgu natarcia."),
                    (f"Skieruj {sp.acc} przeciwko pobliskiemu wrogowi.", f"Brak celu w zasięgu — nie możemy użyć {sp.gen}, bo wrogowie=brak."),
                ])
            return state_str, apply_prefix(prefix, u), n

        if mode == "not_ruin":
            castle_locs = [l for l in LOCATIONS if l.kind == "zamek" and l.split == split]
            cloc = rng.choice(castle_locs)
            mod = dict(st)
            mod["loc"] = cloc
            state_str = format_state_from_dict(mod, compact=compact, required_keys=("lokacja", "typ"))
            if split == "train":
                u, n = f"Splądruj ruiny w miejscu {cloc.nom}!", f"Nie możemy plądrować ruin — {cloc.nom} to zamek, a nie opuszczona krypta."
            elif split == "val":
                u, n = f"Złóż ofiarę w świątyni {cloc.gen}.", f"Brak świątyni w tej lokacji; przebywamy {cloc.prep_in} (typ=zamek)."
            else:
                u, n = f"Przeszukaj podziemia ruin {cloc.prep_in}.", f"Nie ma tu ruin do zbadania — {cloc.prep_in} stoi ufortyfikowany zamek."
            return state_str, apply_prefix(prefix, u), n

        ruin_locs = [l for l in LOCATIONS if l.kind in ("ruiny", "świątynia") and l.split == split]
        rloc = rng.choice(ruin_locs)
        mod = dict(st)
        mod["loc"] = rloc
        mod["hero_alive"] = False
        state_str = format_state_from_dict(mod, compact=compact, required_keys=("lokacja", "typ", "dowódca"))
        if split == "train":
            u, n = f"Wejdź {rloc.prep_to} i odbierz skarb.", f"Brak żywego Bohatera w armii — bez dowódcy {st['u1'].pl_nom} nie zbadają {rloc.gen}."
        elif split == "val":
            u, n = f"Zbadaj {rloc.acc} bez udziału wodza.", f"Odmowa: do wejścia {rloc.prep_to} wymagana jest obecność żywego Bohatera."
        else:
            u, n = f"Rozkaż {st['u1'].dat} splądrować {rloc.acc}.", f"To niemożliwe — {rloc.prep_in} plądrować może wyłącznie armia z Bohaterem."
        return state_str, apply_prefix(prefix, u), n

    if refusal_kind == "missing_resource":
        res_mode = rng.choice(("no_mana", "no_gold", "no_mv"))
        if res_mode == "no_mana":
            sp = rng.choice([s for s in SPELLS if s.split == split and s.mana_cost >= 4] or [s for s in SPELLS if s.split == split])
            low_mana = rng.choice((0, 1, 2))
            mod = dict(st)
            mod["mana"] = low_mana
            state_str = format_state_from_dict(mod, compact=compact, required_keys=("mana",))
            if split == "train":
                u, n = f"Rzuć natychmiast {sp.acc}!", f"Za mało many — mamy {low_mana}/{st['max_mana']} MP, a {sp.nom} wymaga {sp.mana_cost} many."
            elif split == "val":
                u, n = f"Użyj zaklęcia {sp.nom} w tej turze.", f"Niewystarczająca rezerwa many ({low_mana} MP); rytuał {sp.gen} kosztuje {sp.mana_cost} many."
            else:
                u, n = f"Aktywuj czar {sp.nom}!", f"Odmowa: posiadamy zaledwie {low_mana} MP, podczas gdy {sp.nom} wymaga {sp.mana_cost} punktów many."
            return state_str, apply_prefix(prefix, u), n

        if res_mode == "no_gold":
            unit = rng.choice([u for u in UNITS if u.split == split and u.cost >= 115])
            low_gold = rng.choice((0, 15, 35, 50))
            mod = dict(st)
            mod["gold"] = low_gold
            state_str = format_state_from_dict(mod, compact=compact, required_keys=("złoto",))
            if split == "train":
                u, n = f"Wyszkol w grodzie {unit.acc}.", f"Skarbiec jest zbyt pusty — przy {low_gold} złota nie opłacimy {unit.gen} ({unit.cost} zł)."
            elif split == "val":
                u, n = f"Zaciągnij natychmiast oddział: {unit.nom}.", f"Brak złota w skarbcu ({low_gold} zł); werbunek {unit.gen} kosztuje {unit.cost} zł."
            else:
                u, n = f"Kup do garnizonu {unit.acc}.", f"Odrzucam rozkaz: przy {low_gold} sztukach złota nie stać nas na {unit.acc} ({unit.cost} zł)."
            return state_str, apply_prefix(prefix, u), n

        mod = dict(st)
        mod["mv"] = 0
        state_str = format_state_from_dict(mod, compact=compact, required_keys=("ruch",))
        if split == "train":
            u, n = f"Maszeruj jeszcze trzy pola {st['loc'].prep_to}!", f"Brak punktów ruchu (0 PM) — {st['u1'].pl_nom} muszą odpocząć do następnej tury."
        elif split == "val":
            u, n = f"Przejdź natychmiast przez teren „{st['terrain'].nom}” {st['loc'].prep_to}.", f"Nie ruszymy z miejsca przy 0 punktach ruchu ({st['u1'].nom}); poczekajmy na nową turę."
        else:
            u, n = f"Rozkaż {st['u1'].dat} rozpocząć forsowny marsz.", f"Odmowa ruchu: licznik marszowy wynosi 0 PM i bez Wichrów Boreasza {st['u1'].pl_nom} nie pójdą dalej."
        return state_str, apply_prefix(prefix, u), n

    absurd_mode = rng.choice(("dead_hp", "negative_gold", "overstack"))
    if absurd_mode == "dead_hp":
        mod = dict(st)
        mod["hp"] = 0
        state_str = format_state_from_dict(mod, compact=compact, required_keys=("hp",))
        if split == "train":
            u, n = f"Prowadź {st['faction'].leader_gen} na czoło natarcia!", f"Stan wskazuje 0/{st['max_hp']} HP — poległy dowódca nie może walczyć przed wskrzeszeniem w stolicy."
        elif split == "val":
            u, n = f"Niech {st['faction'].leader_nom} osobiście rzuci się w wir bitwy.", f"Niespójny rozkaz: przy 0/{st['max_hp']} HP wódz {st['faction'].gen} nie żyje i wymaga wskrzeszenia."
        else:
            u, n = f"Wyślij {st['faction'].leader_gen} na samotny pojedynek.", f"To niemożliwe — w raporcie widnieje 0/{st['max_hp']} HP, więc {st['faction'].leader_nom} poległ."
        return state_str, apply_prefix(prefix, u), n

    if absurd_mode == "negative_gold":
        neg_gold = rng.choice((-40, -120, -250, -500))
        mod = dict(st)
        mod["gold"] = neg_gold
        state_str = format_state_from_dict(mod, compact=compact, required_keys=("złoto",))
        if split == "train":
            u, n = f"Kup najemników i ulepsz mury {st['loc'].gen}.", f"Zapis skarbca jest błędny ({neg_gold} złota) — przy ujemnej wartości nie można wydać monet."
        elif split == "val":
            u, n = f"Wypłać żołd i zwerbuj {st['u1'].acc}.", f"Stan skarbca {st['faction'].gen} wynosi {neg_gold} złota; przy ujemnym stanie nie zrealizujemy zakupu."
        else:
            u, n = f"Opłać natychmiast nowy zaciąg {st['u1'].pl_gen}.", f"Odmowa: wartość złota ({neg_gold}) {st['loc'].prep_in} jest nieprawidłowa i uniemożliwia rekrutację."
        return state_str, apply_prefix(prefix, u), n

    c_over_1 = rng.randint(5, 7)
    c_over_2 = rng.randint(4, 6)
    total_over = c_over_1 + c_over_2
    mod = dict(st)
    if st["u1"].key == st["u2"].key:
        mod["units_str"] = f"{st['u1'].nom}x{total_over}"
    else:
        mod["units_str"] = f"{st['u1'].nom}x{c_over_1},{st['u2'].nom}x{c_over_2}"
    state_str = format_state_from_dict(mod, compact=compact, required_keys=("jednostki",))
    if split == "train":
        u, n = f"Dołącz jeszcze {st['u1'].acc} do tej armii.", f"Stan wykazuje {total_over} jednostek w stosie, co przekracza twardy limit 8 oddziałów na jednym polu."
    elif split == "val":
        u, n = f"Złącz tu dodatkowy oddział {st['u1'].pl_gen}.", f"Nie można dodać żołnierzy — w stosie widnieje {total_over} jednostek, ponad limit 8."
    else:
        u, n = f"Powiększ ten oddział — dodaj do szeregu {st['u1'].acc}.", f"Błąd pojemności stosu: {total_over} oddziałów przekracza kanoniczny limit 8 jednostek na kafelek."
    return state_str, apply_prefix(prefix, u), n


# ═══════════════════════════════════════════════════════════════════════════════
# ORKIESTRACJA GENERACJI, WALIDACJA I RAPORT STATYSTYCZNY
# ═══════════════════════════════════════════════════════════════════════════════


def format_record(state_str: str, user_text: str, npc_text: str) -> str:
    return f"{STATE_TOKEN} {state_str} {USER_TOKEN} {user_text} {NPC_TOKEN} {npc_text} {EOS_TOKEN}"


def generate_split(
    rng: random.Random,
    split: str,
    per_category: int,
    forbidden_texts: Set[str],
    forbidden_cross_pairs: Set[Tuple[str, str]],
    forbidden_cross_users: Set[str],
    forbidden_states: Set[str],
) -> List[dict]:
    """
    Generuje dokładnie `per_category` próbek dla każdej z 4 kategorii w zadanym splicie.
    - Wewnątrz `val` i `test` pilnujemy pełnej unikalności par (user, npc), żeby dokładnie mierzyć generalizację.
    - Wewnątrz `train` znosimy blokadę `local_pairs` (dopuszcza się różne stany dla tej samej pary),
      zachowując bezwzględną blokadę przecieków między splitami oraz duplikacji całych rekordów `full_text`.
    """
    pool_target = max(per_category * len(CATEGORIES) * 2, 1200)
    state_pool = build_state_pool(rng, split, pool_target)
    if len(state_pool) < per_category:
        raise RuntimeError(f"Za mało unikalnych stanów dla splitu '{split}': {len(state_pool)}")

    records: List[dict] = []
    local_pairs: Set[Tuple[str, str]] = set()
    local_users: Set[str] = set()
    enforce_local_unique_pairs = split in ("val", "test")
    state_idx = 0

    for cat in CATEGORIES:
        cat_count = 0
        attempts = 0
        max_attempts = max(4000, per_category * 40)
        while cat_count < per_category and attempts < max_attempts:
            attempts += 1
            st = state_pool[state_idx % len(state_pool)]
            state_idx += 1

            if cat == "dialog":
                state_str, user_text, npc_text = _render_dialog_sample(rng, st, split)
            elif cat == "state_query":
                state_str, user_text, npc_text = _render_state_query_sample(rng, st, split)
            elif cat == "lore_entity":
                state_str, user_text, npc_text = _render_lore_sample(rng, st, split)
            else:
                state_str, user_text, npc_text = _render_refusal_sample(rng, st, split)

            if state_str in forbidden_states:
                continue
            if user_text in forbidden_cross_users:
                continue
            pair = (user_text, npc_text)
            if pair in forbidden_cross_pairs:
                continue
            if enforce_local_unique_pairs and pair in local_pairs:
                continue

            full_text = format_record(state_str, user_text, npc_text)
            if full_text in forbidden_texts:
                continue

            forbidden_texts.add(full_text)
            forbidden_states.add(state_str)
            local_pairs.add(pair)
            local_users.add(user_text)

            records.append(
                {
                    "text": full_text,
                    "category": cat,
                    "split": split,
                    "state": state_str,
                    "user": user_text,
                    "npc": npc_text,
                }
            )
            cat_count += 1

        if cat_count < per_category:
            raise RuntimeError(
                f"Nie udało się wygenerować {per_category} unikalnych próbek dla ({split}, {cat}); "
                f"osiągnięto {cat_count}."
            )

    forbidden_cross_pairs.update(local_pairs)
    forbidden_cross_users.update(local_users)
    rng.shuffle(records)
    return records


def validate_corpus(splits: Dict[str, List[dict]]) -> dict:
    """Przeprowadza rygorystyczną walidację UTF-8, znaczników, balansu i braku przecieków."""
    text_sets: Dict[str, Set[str]] = {}
    state_sets: Dict[str, Set[str]] = {}
    user_sets: Dict[str, Set[str]] = {}
    diacritic_counter: Dict[str, int] = {ch: 0 for ch in sorted(POLISH_DIACRITICS)}

    for split_name, rows in splits.items():
        t_set: Set[str] = set()
        s_set: Set[str] = set()
        u_set: Set[str] = set()

        for row in rows:
            txt = row["text"]
            assert txt.startswith(f"{STATE_TOKEN} "), f"Brak {STATE_TOKEN} na początku: {txt}"
            assert txt.endswith(f" {EOS_TOKEN}"), f"Brak {EOS_TOKEN} na końcu: {txt}"
            for tok in (STATE_TOKEN, USER_TOKEN, NPC_TOKEN, EOS_TOKEN):
                assert txt.count(tok) == 1, f"Nieprawidłowa liczba tokenów {tok} w: {txt}"
            idx_s = txt.index(STATE_TOKEN)
            idx_u = txt.index(USER_TOKEN)
            idx_n = txt.index(NPC_TOKEN)
            idx_e = txt.index(EOS_TOKEN)
            assert idx_s < idx_u < idx_n < idx_e, f"Zła kolejność tokenów w: {txt}"

            assert "{" not in txt and "}" not in txt, f"Nierozwinięty placeholder w: {txt}"
            assert "\ufffd" not in txt, f"Błąd kodowania UTF-8 (U+FFFD) w: {txt}"
            txt.encode("utf-8").decode("utf-8")

            # Kontrola spójności logicznej stan <-> odpowiedź
            if "dowódca=poległ" in row["state"] and row["category"] == "dialog":
                for f in FACTIONS.values():
                    assert f.leader_nom not in row["npc"], f"Halucynacja żywego wodza przy dowódca=poległ: {txt}"
            if "wrogowie=brak" in row["state"] and row["category"] == "dialog":
                assert "w zasięgu" not in row["user"].lower() or "spokojnie" in row["user"].lower()

            for ch in txt:
                if ch in diacritic_counter:
                    diacritic_counter[ch] += 1

            t_set.add(txt)
            s_set.add(row["state"])
            u_set.add(row["user"])

        assert len(t_set) == len(rows), f"Wykryto duplikaty w zbiorze {split_name}!"
        text_sets[split_name] = t_set
        state_sets[split_name] = s_set
        user_sets[split_name] = u_set

    leak_tv = len(text_sets["train"] & text_sets["val"])
    leak_tt = len(text_sets["train"] & text_sets["test"])
    leak_vt = len(text_sets["val"] & text_sets["test"])
    assert leak_tv == 0 and leak_tt == 0 and leak_vt == 0, "Wykryto przeciek próbek między splitami!"

    state_leak_tv = len(state_sets["train"] & state_sets["val"])
    state_leak_tt = len(state_sets["train"] & state_sets["test"])
    assert state_leak_tv == 0 and state_leak_tt == 0, "Wykryto przeciek identycznych stanów między splitami!"

    user_leak_tv = len(user_sets["train"] & user_sets["val"])
    user_leak_tt = len(user_sets["train"] & user_sets["test"])
    assert user_leak_tv == 0 and user_leak_tt == 0, "Wykryto przeciek identycznych pytań między splitami!"

    missing_lower = [ch for ch in "ąćęłńóśźż" if diacritic_counter[ch] == 0]
    assert not missing_lower, f"Brak polskich znaków w korpusie: {missing_lower}"

    return {
        "leak_text_train_val": leak_tv,
        "leak_text_train_test": leak_tt,
        "leak_text_val_test": leak_vt,
        "leak_state_train_val": state_leak_tv,
        "leak_state_train_test": state_leak_tt,
        "leak_user_train_val": user_leak_tv,
        "leak_user_train_test": user_leak_tt,
        "unique_users_train": len(user_sets["train"]),
        "unique_users_val": len(user_sets["val"]),
        "unique_users_test": len(user_sets["test"]),
        "diacritics": diacritic_counter,
    }


def write_jsonl(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            out = {"text": row["text"], "category": row["category"], "split": row["split"]}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")


def print_report(splits: Dict[str, List[dict]], validation: dict, seed: int, out_dir: str) -> None:
    print("=" * 76)
    print("TARTARUS — RAPORT STATYSTYCZNY KORPUSU TRENINGOWEGO I WALIDACYJNEGO")
    print("=" * 76)
    print(f"Seed deterministyczny : {seed}")
    print(f"Katalog wyjściowy     : {os.path.abspath(out_dir)}")
    print()

    for split_name in ("train", "val", "test"):
        rows = splits[split_name]
        char_lens = [len(r["text"]) for r in rows]
        word_lens = [len(r["text"].split()) for r in rows]
        state_lens = [len(r["state"]) for r in rows]
        cat_counts = {c: sum(1 for r in rows if r["category"] == c) for c in CATEGORIES}
        p95_chars = sorted(char_lens)[int(0.95 * (len(char_lens) - 1))]
        uniq_u = validation[f"unique_users_{split_name}"]

        print(f"[{split_name.upper()}] Próbek: {len(rows)} (unikalne pytania <|user|>: {uniq_u})")
        print("  Kategorie : " + ", ".join(f"{k}={v}" for k, v in cat_counts.items()))
        print(
            f"  Długość (znaki): min={min(char_lens)}, śr={statistics.mean(char_lens):.1f}, "
            f"med={statistics.median(char_lens):.0f}, p95={p95_chars}, max={max(char_lens)}"
        )
        print(
            f"  Długość (słowa): min={min(word_lens)}, śr={statistics.mean(word_lens):.1f}, "
            f"max={max(word_lens)} | śr <|state|>={statistics.mean(state_lens):.1f} zn."
        )
        print()

    print("[HIGIENA PODZIAŁU I KONTROLA PRZECIEKÓW]")
    print(f"  Przeciek dokładny (train ∩ val)  : {validation['leak_text_train_val']}")
    print(f"  Przeciek dokładny (train ∩ test) : {validation['leak_text_train_test']}")
    print(f"  Przeciek dokładny (val ∩ test)   : {validation['leak_text_val_test']}")
    print(f"  Wspólne stany (train ∩ val)      : {validation['leak_state_train_val']}")
    print(f"  Wspólne stany (train ∩ test)     : {validation['leak_state_train_test']}")
    print(f"  Wspólne pytania (train ∩ val)    : {validation['leak_user_train_val']}")
    print(f"  Wspólne pytania (train ∩ test)   : {validation['leak_user_train_test']}")
    print()

    lower_chars = "ąćęłńóśźż"
    print("[WALIDACJA POLSKIEGO UTF-8]")
    print("  Wystąpienia znaków: " + "  ".join(f"{ch}:{validation['diacritics'][ch]}" for ch in lower_chars))
    print()

    print("[PRZYKŁADOWE REKORDY (PO 1 Z KATEGORII W TRAIN)]")
    for cat in CATEGORIES:
        sample = next(r for r in splits["train"] if r["category"] == cat)
        print(f"  ({cat}) {sample['text']}")
    print("=" * 76)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministyczny generator polskiego korpusu Tartarus (train/val/test JSONL)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Stałe ziarno generatora losowego (domyślnie: 42)")
    parser.add_argument("--out-dir", type=str, default=".", help="Katalog docelowy plików JSONL (domyślnie: .)")
    parser.add_argument("--train-per-cat", type=int, default=5000, help="Liczba próbek na kategorię w train (domyślnie: 5000 -> 20000 łącznie)")
    parser.add_argument("--val-per-cat", type=int, default=250, help="Liczba próbek na kategorię w val (domyślnie: 250 -> 1000 łącznie)")
    parser.add_argument("--test-per-cat", type=int, default=250, help="Liczba próbek na kategorię w test (domyślnie: 250 -> 1000 łącznie)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    forbidden_texts: Set[str] = set()
    forbidden_cross_pairs: Set[Tuple[str, str]] = set()
    forbidden_cross_users: Set[str] = set()
    forbidden_states: Set[str] = set()

    # Generacja val i test przed masową generacją train zgodnie z higieną podziału
    val_rows = generate_split(rng, "val", args.val_per_cat, forbidden_texts, forbidden_cross_pairs, forbidden_cross_users, forbidden_states)
    test_rows = generate_split(rng, "test", args.test_per_cat, forbidden_texts, forbidden_cross_pairs, forbidden_cross_users, forbidden_states)
    train_rows = generate_split(rng, "train", args.train_per_cat, forbidden_texts, forbidden_cross_pairs, forbidden_cross_users, forbidden_states)

    splits = {"train": train_rows, "val": val_rows, "test": test_rows}
    validation = validate_corpus(splits)

    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val_rows)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test_rows)

    print_report(splits, validation, args.seed, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
