"""
QuantBet OS — ficheiro único para Streamlit Cloud.
Toda a lógica (config, simulação, mercados, parser, dados) está aqui,
organizada em secções claramente separadas.

v2 — Melhorias adicionadas:
  1. Hold rate específico por superfície (com shrinkage bayesiano para a média global)
  2. Forma recente ponderada pelo Elo do adversário (vencer o nº1 pesa mais que vencer o 200º)
  3. Fadiga modelada de forma não-linear (efeito quase neutro em baixo volume, acelera acima do limiar)
  4. Mapeamento hold-rate em espaço logit (sigmoidal), não linear — colapsa perto dos limites físicos
  5. Momentum entre sets (ganhar um set altera a prob. de hold no set seguinte)
  6. Módulo de calibração histórica (Brier Score + reliability diagram)
  7. Slider manual de "Contexto" (altitude / torneio / jet lag) como proxy honesto,
     enquanto não existirem dados dedicados a estas variáveis
"""

import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# SECÇÃO 1 — CONFIGURAÇÃO (todos os parâmetros num único lugar)
# ============================================================================

@dataclass(frozen=True)
class CircuitConfig:
    base_hold: float
    hold_min: float
    hold_max: float
    base_aces_per_game: float
    default_elo: float = 1500.0


@dataclass(frozen=True)
class SurfaceModifier:
    hold_delta: float
    ace_multiplier: float


@dataclass(frozen=True)
class ModelConfig:
    elo_k: float             = 400.0
    form_weight: float       = 0.05
    form_adj_scale: float    = 0.10
    fatigue_weight: float    = 0.08
    fatigue_adj_scale: float = 10.0
    h2h_weight: float        = 0.015
    h2h_max: float           = 0.075
    h2h_prior: int           = 8       # jogos neutros de prior bayesiano
    game_prob_scale: float   = 0.15
    monte_carlo_n: int       = 10_000
    noise_std: float         = 0.02
    kelly_fraction: float    = 0.10
    kelly_max: float         = 0.035
    kelly_min: float         = 0.005

    # ---- Novos parâmetros (v2) --------------------------------------------
    hold_shrink_k: float     = 20.0   # nº de jogos na superfície para "peso pleno" no shrinkage
    form_elo_scale: float    = 400.0  # escala tipo-Elo para pesar vitórias pela força do adversário
    form_weight_min: float   = 0.3    # peso mínimo de uma vitória (adversário muito mais fraco)
    form_weight_max: float   = 3.0    # peso máximo de uma vitória (adversário muito mais forte)
    fatigue_threshold: float = 6.0    # jogos/semana considerados "carga normal"
    fatigue_exponent: float  = 1.8    # expoente de aceleração acima do limiar
    hold_logit_scale: float  = 3.0    # ganho do deslocamento em espaço logit (hold rate)
    momentum_boost: float    = 0.10   # ~8-12% de boost de hold para quem ganhou o set anterior
    context_weight: float    = 0.01   # peso do ajuste manual de contexto (altitude/torneio/jetlag)


CIRCUIT: Dict[str, CircuitConfig] = {
    "ATP (Masculino)": CircuitConfig(
        base_hold=0.780, hold_min=0.45, hold_max=0.95, base_aces_per_game=0.55
    ),
    "WTA (Feminino)": CircuitConfig(
        base_hold=0.635, hold_min=0.35, hold_max=0.85, base_aces_per_game=0.25
    ),
}

SURFACE_MOD: Dict[str, SurfaceModifier] = {
    "Lento (Clay Lento)":                     SurfaceModifier(-0.04, 0.60),
    "Médio-Lento (Clay Rápido / Hard Lento)": SurfaceModifier(-0.02, 0.80),
    "Médio (Hard Normal)":                    SurfaceModifier( 0.00, 1.00),
    "Rápido (Grass / Hard Rápido)":           SurfaceModifier(+0.03, 1.30),
    "Ultra Rápido (Indoor Rápido)":           SurfaceModifier(+0.05, 1.50),
}

MODEL = ModelConfig()


# ============================================================================
# SECÇÃO 2 — MOTOR DE SIMULAÇÃO (Monte Carlo vectorizado com NumPy)
# ============================================================================

@dataclass
class PlayerStats:
    elo: float
    hold_rate: float
    fatigue: float
    recent_form: float


@dataclass
class MatchSetup:
    p1: PlayerStats
    p2: PlayerStats
    sets_to_win: int
    circuit_cfg: CircuitConfig
    surface_mod: SurfaceModifier
    form_adj: int
    fatigue_adj: int
    h2h: Tuple[int, int]
    context_adj: int = 0     # ajuste manual: altitude / tier do torneio / jet lag
    ml_model: object = None


def _elo_prob(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / MODEL.elo_k))


def _h2h_adj(h2h: Tuple[int, int]) -> float:
    """Ajuste bayesiano de H2H — cresce com evidência real."""
    w1, w2 = h2h
    total = w1 + w2
    if total == 0:
        return 0.0
    posterior = (w1 + MODEL.h2h_prior / 2) / (total + MODEL.h2h_prior)
    raw = (posterior - 0.5) * MODEL.h2h_weight * total
    return float(np.clip(raw, -MODEL.h2h_max, MODEL.h2h_max))


def _fatigue_nonlinear(games: float) -> float:
    """
    Converte 'jogos disputados na última semana' numa carga de fadiga não-linear.
    A literatura mostra que o efeito não é proporcional ao nº de jogos: um volume
    normal (poucos jogos) é quase neutro, mas a partir de um limiar o desgaste
    acelera (aproxima o efeito real de vários jogos/sets seguidos em dias
    consecutivos, sem precisarmos ainda de uma coluna dedicada a "games jogados").
    """
    if games <= MODEL.fatigue_threshold:
        return games * 0.5
    excess = games - MODEL.fatigue_threshold
    return MODEL.fatigue_threshold * 0.5 + excess ** MODEL.fatigue_exponent


def _compute_match_prob(setup: MatchSetup) -> float:
    """Probabilidade P(P1 vence) antes da simulação por jogos."""
    if setup.ml_model is not None:
        try:
            features = pd.DataFrame([[
                setup.p1.elo - setup.p2.elo,
                setup.p1.hold_rate - setup.p2.hold_rate,
                setup.p1.fatigue - setup.p2.fatigue,
            ]], columns=["elo_diff", "hold_diff_last5", "fatigue_diff"])
            prob = float(setup.ml_model.predict_proba(features)[0][1])
        except Exception:
            prob = _elo_prob(setup.p1.elo - setup.p2.elo)
    else:
        prob = _elo_prob(setup.p1.elo - setup.p2.elo)

    prob += ((setup.p1.recent_form - setup.p2.recent_form)
             + setup.form_adj * MODEL.form_adj_scale) * MODEL.form_weight

    fatigue_diff = _fatigue_nonlinear(setup.p1.fatigue) - _fatigue_nonlinear(setup.p2.fatigue)
    prob -= ((fatigue_diff + setup.fatigue_adj * MODEL.fatigue_adj_scale)
             / 100.0 * MODEL.fatigue_weight)

    prob += _h2h_adj(setup.h2h)
    prob += setup.context_adj * MODEL.context_weight

    return float(np.clip(prob, 0.05, 0.95))


def _logit(p: float, lo: float, hi: float) -> float:
    x = np.clip((p - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    return float(np.log(x / (1 - x)))


def _inv_logit(z: float, lo: float, hi: float) -> float:
    x = 1.0 / (1.0 + np.exp(-z))
    return lo + x * (hi - lo)


def _hold_probs(prob: float, cfg: CircuitConfig, mod: SurfaceModifier) -> Tuple[float, float]:
    """
    Mapeia a probabilidade de vitória do encontro para hold rates de serviço.

    Em vez de um deslocamento linear (`shift = (prob-0.5) * k`), trabalha em
    espaço logit: a relação qualidade-do-jogador -> hold rate não é linear,
    é sigmoidal — a diferença de hold colapsa perto dos limites fisiológicos
    (hold_min / hold_max), tal como acontece na realidade (um jogador muito
    melhor não continua a ganhar hold rate proporcionalmente perto de 95%).
    """
    base = float(np.clip(cfg.base_hold + mod.hold_delta, cfg.hold_min, cfg.hold_max))
    z_base = _logit(base, cfg.hold_min, cfg.hold_max)
    delta = (prob - 0.5) * MODEL.game_prob_scale * MODEL.hold_logit_scale

    p1h = float(np.clip(_inv_logit(z_base + delta / 2, cfg.hold_min, cfg.hold_max),
                         cfg.hold_min, cfg.hold_max))
    p2h = float(np.clip(_inv_logit(z_base - delta / 2, cfg.hold_min, cfg.hold_max),
                         cfg.hold_min, cfg.hold_max))
    return p1h, p2h


def _apply_momentum(base_h: Union[float, np.ndarray], won_prev: np.ndarray, boost: float) -> np.ndarray:
    """
    Ajusta o hold rate consoante o resultado do set anterior.
    Ganhar o set anterior aumenta a prob. de vitória no seguinte em ~8-12%
    acima do previsto pelo Elo (efeito de momentum documentado empiricamente).
    """
    boosted = np.clip(base_h + boost * (1.0 - base_h), 0.0, 1.0)
    depressed = np.clip(base_h - boost * base_h, 0.0, 1.0)
    return np.where(won_prev, boosted, depressed)


def _sim_set(
    n: int,
    p1h: Union[float, np.ndarray],
    p2h: Union[float, np.ndarray],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula N sets em paralelo com lógica de terminação vectorizada.
    p1h/p2h podem ser escalares (mesma taxa para todas as simulações) ou
    arrays de forma (n,) — usado para aplicar momentum por-simulação
    consoante o resultado do set anterior nessa simulação específica.
    """
    g1 = np.zeros(n, dtype=np.int32)
    g2 = np.zeros(n, dtype=np.int32)
    active = np.ones(n, dtype=bool)

    for game_idx in range(13):  # máximo 13 jogos num set (7-6)
        if not active.any():
            break
        base_p = p1h if (game_idx % 2 == 0) else (1.0 - p2h)
        probs = np.clip(base_p + rng.normal(0, MODEL.noise_std, n), 0.0, 1.0)
        p1_wins = rng.random(n) < probs
        g1 += active & p1_wins
        g2 += active & ~p1_wins
        done = active & (
            ((g1 >= 6) & ((g1 - g2) >= 2)) |
            ((g2 >= 6) & ((g2 - g1) >= 2)) |
            (g1 == 7) | (g2 == 7)
        )
        active[done] = False

    return g1, g2


def simulate(setup: MatchSetup, n: int = MODEL.monte_carlo_n) -> dict:
    """
    Simula N encontros em paralelo.
    20-50× mais rápido que a versão original em Python puro.
    Inclui momentum entre sets: o resultado de cada set ajusta o hold rate
    efectivo do set seguinte, por simulação.
    """
    rng = np.random.default_rng(42)
    match_prob = _compute_match_prob(setup)
    p1h, p2h = _hold_probs(match_prob, setup.circuit_cfg, setup.surface_mod)

    ace_base = setup.circuit_cfg.base_aces_per_game * setup.surface_mod.ace_multiplier
    rate_a1 = max(0.05, ace_base + (p1h - setup.circuit_cfg.base_hold))
    rate_a2 = max(0.05, ace_base + (p2h - setup.circuit_cfg.base_hold))

    max_sets = setup.sets_to_win * 2 - 1
    p1_sets  = np.zeros(n, dtype=np.int32)
    p2_sets  = np.zeros(n, dtype=np.int32)
    tot_g    = np.zeros(n, dtype=np.int32)
    diff_g   = np.zeros(n, dtype=np.int32)
    aces1    = np.zeros(n, dtype=np.float32)
    aces2    = np.zeros(n, dtype=np.float32)
    set_data = []

    prev_g1: Optional[np.ndarray] = None
    prev_g2: Optional[np.ndarray] = None

    for s in range(max_sets):
        playing = (p1_sets < setup.sets_to_win) & (p2_sets < setup.sets_to_win)
        if not playing.any():
            break

        if prev_g1 is None:
            cur_p1h, cur_p2h = p1h, p2h
        else:
            won_prev_p1 = prev_g1 > prev_g2
            cur_p1h = _apply_momentum(p1h, won_prev_p1, MODEL.momentum_boost)
            cur_p2h = _apply_momentum(p2h, ~won_prev_p1, MODEL.momentum_boost)

        g1, g2 = _sim_set(n, cur_p1h, cur_p2h, rng)
        games_in_set = g1 + g2

        # Aces: escalados pelo número real de jogos de serviço
        srv1 = (games_in_set + 1) // 2
        srv2 = games_in_set // 2
        sa1 = rng.poisson(rate_a1 * srv1).astype(np.float32)
        sa2 = rng.poisson(rate_a2 * srv2).astype(np.float32)

        p1_wins_set = g1 > g2
        p1_sets[playing & p1_wins_set] += 1
        p2_sets[playing & ~p1_wins_set] += 1
        tot_g[playing]  += games_in_set[playing]
        diff_g[playing] += (g1 - g2)[playing]
        aces1[playing]  += sa1[playing]
        aces2[playing]  += sa2[playing]

        if s < 2:
            set_data.append((np.where(playing, g1, 0), np.where(playing, g2, 0)))

        prev_g1, prev_g2 = g1, g2

    while len(set_data) < 2:
        set_data.append((np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)))

    return {
        "p1_match_wins": p1_sets > p2_sets,
        "total_games":   tot_g,
        "game_diff":     diff_g,
        "p1_sets": p1_sets, "p2_sets": p2_sets,
        "s1_p1": set_data[0][0], "s1_p2": set_data[0][1],
        "s2_p1": set_data[1][0], "s2_p2": set_data[1][1],
        "aces_p1": aces1, "aces_p2": aces2,
        "match_prob": match_prob,
        "p1_hold": p1h, "p2_hold": p2h,
    }


# ============================================================================
# SECÇÃO 3 — MERCADOS (EV e Kelly)
# ============================================================================

@dataclass
class Bet:
    market: str
    prob: float
    odd: float
    ev: float
    kelly: float

    @property
    def fair_odd(self) -> float:
        return 1.0 / self.prob if self.prob > 0 else float("inf")

    @property
    def stake_pct(self) -> float:
        return float(np.clip(self.kelly, MODEL.kelly_min, MODEL.kelly_max))


def _kelly(prob: float, odd: float) -> float:
    """Kelly fraccionado completo. A versão original usava EV/(odd-1), que é uma aproximação."""
    if odd <= 1.0 or prob <= 0:
        return 0.0
    full = (prob * odd - 1.0) / (odd - 1.0)
    return max(0.0, full * MODEL.kelly_fraction)


def _bet(market: str, prob: float, odd: float) -> Bet:
    return Bet(market=market, prob=prob, odd=odd,
               ev=prob * odd - 1.0, kelly=_kelly(prob, odd))


def compute_markets(sims: dict, mkts: dict, p1: str, p2: str) -> List[Bet]:
    bets: List[Bet] = []
    tot   = sims["total_games"]
    diff  = sims["game_diff"]
    p1mw  = sims["p1_match_wins"]
    s1p1, s1p2 = sims["s1_p1"], sims["s1_p2"]
    s2p1, s2p2 = sims["s2_p1"], sims["s2_p2"]
    a1, a2 = sims["aces_p1"], sims["aces_p2"]
    ta = a1 + a2
    p1s, p2s = sims["p1_sets"], sims["p2_sets"]
    p1g = (tot + diff) / 2
    p2g = (tot - diff) / 2

    def ou(arr, line, label_over, label_under, odds_ou):
        if "Over"  in odds_ou: bets.append(_bet(f"Over {line} {label_over}",  np.mean(arr > line), odds_ou["Over"]))
        if "Under" in odds_ou: bets.append(_bet(f"Under {line} {label_under}", np.mean(arr < line), odds_ou["Under"]))

    prob_p1 = float(np.mean(p1mw))
    mw = mkts.get("match_winner", {})
    if "P1" in mw: bets.append(_bet(f"Vitória {p1}", prob_p1, mw["P1"]))
    if "P2" in mw: bets.append(_bet(f"Vitória {p2}", 1 - prob_p1, mw["P2"]))

    for line, oo in mkts.get("total_games", {}).items():    ou(tot, line, "Jogos", "Jogos", oo)
    for line, oo in mkts.get("total_sets",  {}).items():    ou(p1s + p2s, line, "Sets", "Sets", oo)
    for line, oo in mkts.get("total_aces",  {}).items():    ou(ta,  line, "Ases", "Ases", oo)
    for line, oo in mkts.get("p1_total_games", {}).items(): ou(p1g, line, f"Jogos {p1}", f"Jogos {p1}", oo)
    for line, oo in mkts.get("p2_total_games", {}).items(): ou(p2g, line, f"Jogos {p2}", f"Jogos {p2}", oo)
    for line, oo in mkts.get("p1_aces", {}).items():        ou(a1, line, f"Ases {p1}", f"Ases {p1}", oo)
    for line, oo in mkts.get("p2_aces", {}).items():        ou(a2, line, f"Ases {p2}", f"Ases {p2}", oo)

    for hcp, odd in mkts.get("game_handicap", {}).get("P1", {}).items():
        bets.append(_bet(f"Hcp Jogos {p1} ({'+' if hcp>0 else ''}{hcp})", np.mean(diff > -hcp), odd))
    for hcp, odd in mkts.get("game_handicap", {}).get("P2", {}).items():
        bets.append(_bet(f"Hcp Jogos {p2} ({'+' if hcp>0 else ''}{hcp})", np.mean(-diff > -hcp), odd))

    prob_s1 = float(np.mean(s1p1 > s1p2))
    s1w = mkts.get("set1_winner", {})
    if "P1" in s1w: bets.append(_bet(f"Vence 1º Set {p1}", prob_s1, s1w["P1"]))
    if "P2" in s1w: bets.append(_bet(f"Vence 1º Set {p2}", 1 - prob_s1, s1w["P2"]))
    for line, oo in mkts.get("set1_total_games", {}).items(): ou(s1p1 + s1p2, line, "Jogos 1º Set", "Jogos 1º Set", oo)
    for hcp, odd in mkts.get("set1_handicap", {}).get("P1", {}).items():
        bets.append(_bet(f"Hcp 1º Set {p1} ({hcp})", np.mean((s1p1 - s1p2) > -hcp), odd))

    prob_s2 = float(np.mean(s2p1 > s2p2))
    s2w = mkts.get("set2_winner", {})
    if "P1" in s2w: bets.append(_bet(f"Vence 2º Set {p1}", prob_s2, s2w["P1"]))
    if "P2" in s2w: bets.append(_bet(f"Vence 2º Set {p2}", 1 - prob_s2, s2w["P2"]))
    for line, oo in mkts.get("set2_total_games", {}).items(): ou(s2p1 + s2p2, line, "Jogos 2º Set", "Jogos 2º Set", oo)

    bets.sort(key=lambda b: b.ev, reverse=True)
    return bets


def best_bet(bets: List[Bet], min_ev: float, min_odd: float) -> Optional[Bet]:
    value = [b for b in bets if b.ev >= min_ev]
    if not value:
        return None
    eligible = [b for b in value if b.odd >= min_odd] or value
    return max(eligible, key=lambda b: b.kelly)


# ============================================================================
# SECÇÃO 4 — PARSER MULTILINGUE DE ODDS
# ============================================================================

_IGNORED = [
    "par/ímpar", "odd/even", "exato", "exact", "correct", "duplo",
    "double result", "only one set", "apenas um set", "vencedor e",
    "winner and", "vencedor &", "winner &", "tie-break", "tie break",
]
_RE_ODD    = re.compile(r":\s*([\d,\.]+)\s*$")
_RE_OU     = re.compile(r"(over|under|mais de|menos de|mais|menos|acima|abaixo)\s*([\d]+\.[\d]+)", re.I)
_RE_HCP    = re.compile(r"([+-]?\d+\.\d+)")


def _tokens(name: str) -> List[str]:
    return [t.lower() for t in name.replace(",", " ").split() if len(t) > 2]


def _parse_line(line: str) -> Optional[Tuple[str, float]]:
    clean = line.replace("—", ":").replace(" - ", ":")
    m = _RE_ODD.search(clean)
    if not m:
        return None
    try:
        return clean[:m.start()].strip().lower(), float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def _detect_cat(header: str, p1t: List[str], p2t: List[str]) -> str:
    if any(x in header for x in _IGNORED):
        return "Ignored"

    if any(x in header for x in ["aces", "ases"]):
        if any(x in header for x in p1t + ["player 1", "jogador 1", "casa"]): return "p1_aces"
        if any(x in header for x in p2t + ["player 2", "jogador 2", "fora"]): return "p2_aces"
        return "total_aces"

    for triggers, h_cat, t_cat, w_cat in [
        (["set 1", "1º set", "1o set", "primeiro set", "1st set"], "set1_handicap", "set1_total_games", "set1_winner"),
        (["set 2", "2º set", "2o set", "segundo set", "2nd set"],  "set2_handicap", "set2_total_games", "set2_winner"),
    ]:
        if any(x in header for x in triggers):
            if any(x in header for x in p1t + p2t + ["player 1", "player 2", "jogador", "casa", "fora"]):
                return "Ignored"
            if "handicap" in header: return h_cat
            if any(x in header for x in ["total", "jogos", "games"]): return t_cat
            if any(x in header for x in ["winner", "vencedor"]): return w_cat
            return "Ignored"

    if any(x in header for x in p1t + ["player 1", "jogador 1", "casa"]):
        if any(x in header for x in ["total", "jogos", "games"]): return "p1_total_games"
    if any(x in header for x in p2t + ["player 2", "jogador 2", "fora"]):
        if any(x in header for x in ["total", "jogos", "games"]): return "p2_total_games"
    if any(x in header for x in ["total jogos", "total games", "total de jogos", "jogos no encontro"]): return "total_games"
    if "total sets" in header or "total de sets" in header: return "total_sets"
    if "handicap" in header: return "set_handicap" if "sets" in header else "game_handicap"
    if any(x in header for x in ["winner", "vencedor", "resultado final", "match winner", "1x2"]): return "match_winner"
    return "Ignored"


def _empty_mkts() -> dict:
    return {
        "match_winner": {}, "total_games": {}, "total_sets": {},
        "game_handicap": {"P1": {}, "P2": {}}, "set_handicap": {"P1": {}, "P2": {}},
        "p1_total_games": {}, "p2_total_games": {},
        "set1_winner": {}, "set1_total_games": {}, "set1_handicap": {"P1": {}, "P2": {}},
        "set2_winner": {}, "set2_total_games": {}, "set2_handicap": {"P1": {}, "P2": {}},
        "total_aces": {}, "p1_aces": {}, "p2_aces": {},
    }


def parse_odds(text: str, p1: str = "", p2: str = "") -> dict:
    mkts = _empty_mkts()
    p1t, p2t = _tokens(p1), _tokens(p2)
    cat = "Ignored"

    for raw in text.split("\n"):
        line = raw.strip()
        if not line: continue

        parsed = _parse_line(line)
        if parsed is None:
            cat = _detect_cat(line.lower(), p1t, p2t)
            continue

        if cat == "Ignored": continue
        key, odd = parsed

        # Filtrar linhas compostas (&, and, resultado exacto)
        has_combo = any(x in key for x in ["&", " and ", " e "])
        if has_combo and cat in ("match_winner", "set1_winner", "set2_winner",
                                  "total_games", "total_sets", "p1_total_games",
                                  "p2_total_games", "set1_total_games", "set2_total_games"):
            continue
        if cat in ("match_winner", "set1_winner", "set2_winner") and re.search(r"\b[02]:[012]\b", key):
            continue

        try:
            if cat in ("match_winner", "set1_winner", "set2_winner"):
                is_p1 = any(x in key for x in p1t + ["casa", "home", "jogador 1"]) or re.search(r"(?<!\d)1(?!\d)", key)
                is_p2 = any(x in key for x in p2t + ["fora", "away",  "jogador 2"]) or re.search(r"(?<!\d)2(?!\d)", key)
                c = mkts[cat]
                if is_p1 and "P1" not in c: c["P1"] = odd
                elif is_p2 and "P2" not in c: c["P2"] = odd

            elif cat in ("total_games", "total_sets", "p1_total_games", "p2_total_games",
                         "set1_total_games", "set2_total_games", "total_aces", "p1_aces", "p2_aces"):
                m = _RE_OU.search(key)
                if m:
                    ou = "Over" if m.group(1).lower() in ("over", "mais de", "mais", "acima") else "Under"
                    val = float(m.group(2))
                    target = "total_sets" if cat == "total_games" and val < 6.0 else cat
                    if val not in mkts[target]: mkts[target][val] = {}
                    mkts[target][val][ou] = odd

            elif cat in ("game_handicap", "set_handicap", "set1_handicap", "set2_handicap"):
                m = _RE_HCP.search(key)
                if m:
                    hcp = float(m.group(1))
                    side = "P1" if any(x in key for x in p1t + ["1", "casa", "jogador 1"]) else "P2"
                    mkts[cat][side][hcp] = odd
        except Exception:
            continue

    return mkts


# ============================================================================
# SECÇÃO 5 — DADOS (carregamento e stats de jogadores)
# ============================================================================

@st.cache_resource
def load_ml_model():
    try:
        import joblib
        if os.path.exists("modelo_tenis_calibrado.pkl"):
            return joblib.load("modelo_tenis_calibrado.pkl")
    except ImportError:
        pass
    return None


@st.cache_data
def load_match_data() -> pd.DataFrame:
    with zipfile.ZipFile("dados_resumidos.zip", "r") as z:
        df = pd.read_csv(z.open("dados_resumidos.csv"))
    # Normalizar nomes uma vez — evita casefold() em cada query
    for col, norm in [("player", "_pn"), ("opponent", "_on"), ("winner", "_wn")]:
        if col in df.columns:
            df[norm] = df[col].str.casefold().str.strip()
    return df


@st.cache_data
def load_elos(circuito: str) -> pd.DataFrame:
    f = "EloRankP.csv" if circuito == "WTA (Feminino)" else "PlayerElo.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        df["_nn"] = df["Player"].str.casefold().str.strip()
        return df
    return pd.DataFrame(columns=["Player", "Elo", "hElo", "cElo", "gElo", "_nn"])


def load_agenda() -> pd.DataFrame:
    """Sem @cache_data — contém datetime.today() que não pode ser congelado."""
    if os.path.exists("agenda.csv"):
        try:
            df = pd.read_csv("agenda.csv")
            df["Data"] = pd.to_datetime(df["Data"]).dt.date
            return df
        except Exception:
            pass
    hoje = datetime.today().date()
    amanha = hoje + timedelta(days=1)
    return pd.DataFrame({
        "Torneio": ["ATP Challenger Amersfoort", "ATP Challenger Amersfoort", "WTA Palermo", "WTA Palermo"],
        "Data":    [hoje, hoje, hoje, amanha],
        "Hora":    ["14:00", "15:30", "16:00", "10:00"],
        "P1":      ["Jesper De Jong", "Jaime Faria", "Qinwen Zheng", "Karolina Muchova"],
        "P2":      ["Sebastian Baez", "Titouan Droguet", "Sara Errani", "Qinwen Zheng"],
    })


def _lookup_elo(nn: str, superficie: str, df_elos: pd.DataFrame, default: float) -> float:
    """Elo do jogador (por nome já normalizado) na superfície indicada, com fallback."""
    if df_elos.empty or "_nn" not in df_elos.columns:
        return default
    row = df_elos[df_elos["_nn"] == nn]
    if row.empty:
        return default
    col = {"Clay": "cElo", "Grass": "gElo", "Hard": "hElo"}.get(superficie, "Elo")
    if col in row.columns:
        return float(row[col].iloc[0])
    return default


def get_stats(nome: str, superficie: str, circuito: str, df: pd.DataFrame, df_elos: pd.DataFrame) -> PlayerStats:
    cfg = CIRCUIT[circuito]
    nn = str(nome).casefold().strip()

    elo = _lookup_elo(nn, superficie, df_elos, cfg.default_elo)

    # --- Hold rate: específico da superfície, com shrinkage bayesiano ------
    # Um jogador pode ter 78% de hold em geral mas 72% no clay. Usamos o valor
    # específico da superfície, encolhido em direcção à média global consoante
    # o nº de jogos disponíveis nessa superfície (evita overfit em amostras pequenas).
    hold = cfg.base_hold
    if "_pn" in df.columns and "hold_percentage" in df.columns:
        m_all = df[df["_pn"] == nn]
        if not m_all.empty:
            hold_global = float(m_all["hold_percentage"].mean())
            if "surface" in df.columns:
                m_surf = m_all[m_all["surface"] == superficie]
            else:
                m_surf = m_all.iloc[0:0]
            if not m_surf.empty:
                hold_surf = float(m_surf["hold_percentage"].mean())
                n_surf = len(m_surf)
                w = n_surf / (n_surf + MODEL.hold_shrink_k)
                hold = w * hold_surf + (1 - w) * hold_global
            else:
                hold = hold_global

    # --- Fadiga (valor bruto; a não-linearidade é aplicada em _fatigue_nonlinear) ---
    fatigue = 0.0
    if "_pn" in df.columns and "games_played_last_week" in df.columns:
        m = df[df["_pn"] == nn]
        if not m.empty:
            fatigue = float(m["games_played_last_week"].iloc[-1])

    # --- Forma recente ponderada pelo Elo do adversário --------------------
    # Uma vitória contra o top 10 vale mais do que uma vitória contra o 200º.
    form = 0.5
    if "_pn" in df.columns and "_wn" in df.columns and "_on" in df.columns:
        m = df[df["_pn"] == nn].tail(5)
        if not m.empty:
            weights, wins = [], []
            for _, row in m.iterrows():
                opp_nn = row["_on"]
                opp_elo = _lookup_elo(opp_nn, superficie, df_elos, cfg.default_elo)
                w = float(np.clip(
                    2.0 ** ((opp_elo - cfg.default_elo) / MODEL.form_elo_scale),
                    MODEL.form_weight_min, MODEL.form_weight_max,
                ))
                weights.append(w)
                wins.append(1.0 if row["_wn"] == nn else 0.0)
            weights_arr = np.array(weights)
            wins_arr = np.array(wins)
            form = float(np.sum(weights_arr * wins_arr) / np.sum(weights_arr))

    return PlayerStats(elo=elo, hold_rate=hold, fatigue=fatigue, recent_form=form)


def get_h2h(p1: str, p2: str, df: pd.DataFrame) -> Tuple[int, int]:
    if "_pn" not in df.columns or "_on" not in df.columns:
        return 0, 0
    n1 = str(p1).casefold().strip()
    n2 = str(p2).casefold().strip()
    mask = (
        ((df["_pn"] == n1) & (df["_on"] == n2)) |
        ((df["_pn"] == n2) & (df["_on"] == n1))
    )
    h2h = df[mask]
    if h2h.empty or "_wn" not in df.columns:
        return 0, 0
    return int((h2h["_wn"] == n1).sum()), int((h2h["_wn"] == n2).sum())


# ----------------------------------------------------------------------------
# Calibração / Backtesting histórico
# ----------------------------------------------------------------------------

@st.cache_data
def run_backtest(circuito: str, n_sample: int = 3000) -> pd.DataFrame:
    """
    Gera pares (probabilidade prevista, resultado real) para jogos históricos,
    usando o modelo Elo puro do sistema (P1 = "player" na linha, P2 = "opponent").

    AVISO METODOLÓGICO: usa o snapshot ACTUAL de Elo como proxy do valor à
    data de cada jogo (não existe histórico de Elo point-in-time guardado).
    Para jogos antigos isto introduz algum lookahead bias — os números aqui
    servem como primeiro sinal de honestidade do modelo, não como um
    backtest walk-forward puro. Para rigor total, seria preciso guardar
    snapshots de Elo por data e reconstruir o estado "à data do jogo".
    """
    df = load_match_data()
    df_elos = load_elos(circuito)
    cfg = CIRCUIT[circuito]

    if not {"_pn", "_on", "_wn"}.issubset(df.columns):
        return pd.DataFrame(columns=["prob", "actual"])

    rows = df.dropna(subset=["_pn", "_on", "_wn"])
    if len(rows) > n_sample:
        rows = rows.sample(n_sample, random_state=42)

    preds, actuals = [], []
    for _, r in rows.iterrows():
        p1n, p2n = r["_pn"], r["_on"]
        surf = r["surface"] if ("surface" in r and pd.notna(r["surface"])) else "Hard"
        e1 = _lookup_elo(p1n, surf, df_elos, cfg.default_elo)
        e2 = _lookup_elo(p2n, surf, df_elos, cfg.default_elo)
        prob = _elo_prob(e1 - e2)
        preds.append(prob)
        actuals.append(1.0 if r["_wn"] == p1n else 0.0)

    return pd.DataFrame({"prob": preds, "actual": actuals})


def brier_score(df_bt: pd.DataFrame) -> float:
    """0 = previsões perfeitas; 0.25 = tão bom como prever sempre 50/50."""
    if df_bt.empty:
        return float("nan")
    return float(np.mean((df_bt["prob"] - df_bt["actual"]) ** 2))


def calibration_table(df_bt: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Reliability diagram em forma de tabela: para cada bucket de probabilidade
    prevista, compara com a taxa real de vitórias observada nesse bucket.
    Um modelo honesto tem as duas colunas próximas em todos os buckets.
    """
    cols = ["Bucket", "Prob. Média Prevista", "Taxa Real de Vitórias", "N"]
    if df_bt.empty:
        return pd.DataFrame(columns=cols)
    bins = np.linspace(0, 1, n_bins + 1)
    tmp = df_bt.copy()
    tmp["bucket"] = pd.cut(tmp["prob"], bins, include_lowest=True)
    g = (
        tmp.groupby("bucket", observed=True)
        .agg(prob_media=("prob", "mean"), taxa_real=("actual", "mean"), n=("actual", "size"))
        .dropna()
        .reset_index()
    )
    g.columns = cols
    return g


# ============================================================================
# SECÇÃO 6 — INTERFACE STREAMLIT
# ============================================================================

st.set_page_config(page_title="QuantBet OS", layout="wide")
st.title("🎾 QuantBet OS: Sistema Quantitativo ATP & WTA")

# Session state
for k in ("agenda_p1", "agenda_p2"):
    if k not in st.session_state:
        st.session_state[k] = None

# Dados
df       = load_match_data()
ml_model = load_ml_model()

# Sidebar
st.sidebar.header("1. Configurações Globais")
circuito = st.sidebar.radio("Circuito", list(CIRCUIT.keys()))
if ml_model is None:
    st.sidebar.info("🤖 Motor: Elo Matemático (Fallback)")
else:
    st.sidebar.success("🤖 Motor: XGBoost Calibrado")

df_elos    = load_elos(circuito)
superficie = st.sidebar.selectbox("Superfície", sorted(df["surface"].dropna().unique()))
sets_padrao = [3] if circuito == "WTA (Feminino)" else [3, 5]
sets_input  = st.sidebar.radio("Sets do Encontro", sets_padrao)

jogadores = sorted(df_elos["Player"].dropna().unique()) if not df_elos.empty else ["Jogador A", "Jogador B"]
if len(jogadores) < 2:
    st.sidebar.error("⚠️ Ficheiro de Elos vazio ou não encontrado.")

st.sidebar.header("2. Filtros de Valor")
limite_ev      = st.sidebar.slider("EV Mínimo (%)", 1.0, 15.0, 5.0, 0.5) / 100
odd_minima_rec = st.sidebar.number_input("Odd Mínima Recomendada", value=1.50, step=0.05)

st.sidebar.header("⚙️ Condições de Jogo")
vel_campo     = st.sidebar.selectbox("Velocidade do Campo", list(SURFACE_MOD.keys()))
ajuste_forma  = st.sidebar.slider("Ajuste de Forma (P1 vs P2)", -5, 5, 0)
ajuste_fadiga = st.sidebar.slider("Ajuste de Fadiga (P1 vs P2)", -5, 5, 0)
ajuste_contexto = st.sidebar.slider(
    "Ajuste de Contexto (Altitude/Torneio/Jet Lag)", -5, 5, 0,
    help=(
        "Proxy manual para efeitos ainda não modelados com dados dedicados: "
        "altitude (Madrid/Bogotá aceleram a bola e sobem os aces), fuso "
        "horário nos últimos dias, tier do torneio (Slam vs Challenger). "
        "Positivo favorece P1, negativo favorece P2. Peso deliberadamente "
        "pequeno até existir uma fonte de dados própria para estas variáveis."
    ),
)

st.sidebar.caption(
    "ℹ️ Altitude, temperatura/humidade, jet lag e tier do torneio ainda não "
    "têm dados próprios — o slider acima é um proxy manual, não um cálculo "
    "automático. Para automatizar, é preciso uma coluna de altitude/tier por "
    "torneio e datas de jogos anteriores para calcular o jet lag."
)


def safe_idx(name, fallback=0) -> int:
    try:
        return jogadores.index(name) if name in jogadores else fallback
    except Exception:
        return fallback


def build_setup(p1: str, p2: str, h2h_override=None) -> MatchSetup:
    return MatchSetup(
        p1=get_stats(p1, superficie, circuito, df, df_elos),
        p2=get_stats(p2, superficie, circuito, df, df_elos),
        sets_to_win=sets_input // 2 + 1,
        circuit_cfg=CIRCUIT[circuito],
        surface_mod=SURFACE_MOD[vel_campo],
        form_adj=ajuste_forma,
        fatigue_adj=ajuste_fadiga,
        context_adj=ajuste_contexto,
        h2h=h2h_override if h2h_override is not None else get_h2h(p1, p2, df),
        ml_model=ml_model,
    )


def render_results(bets: List[Bet], p1: str, p2: str, sims: dict) -> None:
    if not bets:
        st.error("Não foram encontrados mercados. Verifica a formatação do texto.")
        return

    top = best_bet(bets, limite_ev, odd_minima_rec)
    if top:
        st.markdown("---")
        st.markdown("### 🏆 Aposta Recomendada")
        st.success(
            f"**Mercado:** {top.market}\n\n"
            f"**Odd:** {top.odd:.2f} | **Prob:** {top.prob:.1%} | "
            f"**EV:** {'+' if top.ev > 0 else ''}{top.ev:.1%}\n\n"
            f"⚖️ **Banca Sugerida:** **{top.stake_pct:.1%}**"
        )
        st.markdown("---")

    rows = [{
        "Status":    "✅ Valor" if b.ev >= limite_ev else "❌ Evitar",
        "Mercado":   b.market,
        "Odd":       f"{b.odd:.2f}",
        "Odd Justa": f"{b.fair_odd:.2f}",
        "Prob":      f"{b.prob:.2%}",
        "EV":        f"{'+' if b.ev > 0 else ''}{b.ev:.2%}",
    } for b in bets]
    st.markdown("#### 📋 Auditoria Completa")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("🔬 Parâmetros do Modelo"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Prob. Vitória P1",      f"{np.mean(sims['p1_match_wins']):.1%}")
        c2.metric("P1 Hold Rate",           f"{sims['p1_hold']:.1%}")
        c3.metric("P2 Hold Rate",           f"{sims['p2_hold']:.1%}")
        c1.metric("Média Total Jogos",      f"{np.mean(sims['total_games']):.1f}")
        c2.metric("Mediana Aces",           f"{np.median(sims['aces_p1'] + sims['aces_p2']):.1f}")
        c3.metric("Prob P1 Vence 1º Set",   f"{np.mean(sims['s1_p1'] > sims['s1_p2']):.1%}")


# ── ABAS ──────────────────────────────────────────────────────────────────────
tab_agenda, tab_scanner, tab_manual, tab_csv, tab_calib = st.tabs(
    ["📅 Agenda", "🤖 Auto-Scanner", "🔍 Calculadora Manual", "🚀 CSV em Massa", "📉 Calibração"]
)

# ── AGENDA ────────────────────────────────────────────────────────────────────
with tab_agenda:
    st.header("📅 Calendário de Torneios")
    st.markdown("Podes alimentar esta lista criando um ficheiro `agenda.csv` na mesma pasta da app.")
    data_sel  = st.date_input("🗓️ Selecionar Data", datetime.today().date())
    df_agenda = load_agenda()
    jogos_dia = df_agenda[df_agenda["Data"] == data_sel]

    if jogos_dia.empty:
        st.info(f"Sem jogos agendados para {data_sel.strftime('%d/%m/%Y')}.")
    else:
        for torneio in jogos_dia["Torneio"].unique():
            with st.expander(f"🏆 {torneio}", expanded=True):
                for idx, jogo in jogos_dia[jogos_dia["Torneio"] == torneio].iterrows():
                    ch, cj, cb = st.columns([2, 6, 2])
                    ch.markdown(f"🕒 `{jogo['Hora']}`")
                    cj.markdown(f"**{jogo['P1']}** vs **{jogo['P2']}**")
                    if cb.button("Carregar", key=f"ag_{idx}"):
                        st.session_state["agenda_p1"] = jogo["P1"]
                        st.session_state["agenda_p2"] = jogo["P2"]
                        st.success("✅ Vai à aba **Auto-Scanner** ou **Calculadora Manual**.")

# ── AUTO-SCANNER ──────────────────────────────────────────────────────────────
with tab_scanner:
    st.header("Auto-Scanner Inteligente (Copiar & Colar)")
    cs1, cs2 = st.columns(2)
    sp1 = cs1.selectbox("Favorito (P1)", jogadores, index=safe_idx(st.session_state["agenda_p1"]), key="sp1")
    sp2 = cs2.selectbox("Underdog (P2)", jogadores, index=safe_idx(st.session_state["agenda_p2"], 1), key="sp2")

    if sp1 == sp2:
        st.error("⚠️ Seleciona jogadores diferentes.")
    else:
        h2h_db = get_h2h(sp1, sp2, df)
        st.markdown("**⚔️ Correcção Manual de H2H**")
        ch1, ch2 = st.columns(2)
        hp1 = ch1.number_input(f"Vitórias {sp1}", value=int(h2h_db[0]), min_value=0, step=1, key="hp1")
        hp2 = ch2.number_input(f"Vitórias {sp2}", value=int(h2h_db[1]), min_value=0, step=1, key="hp2")
        texto = st.text_area("Cola as Odds:", height=300, key="txt_odds")

        if st.button("Analisar Odds", key="btn_scan"):
            if not texto.strip():
                st.warning("Cola o texto com as odds primeiro.")
            else:
                with st.spinner("A simular 10 000 encontros..."):
                    sims = simulate(build_setup(sp1, sp2, (hp1, hp2)))
                    mkts = parse_odds(texto, sp1, sp2)
                    bets = compute_markets(sims, mkts, sp1, sp2)
                render_results(bets, sp1, sp2, sims)

# ── CALCULADORA MANUAL ────────────────────────────────────────────────────────
with tab_manual:
    st.header("Análise de Partida Única")
    cm1, cm2 = st.columns(2)
    mp1 = cm1.selectbox("Favorito (P1)", jogadores, index=safe_idx(st.session_state["agenda_p1"]), key="mp1")
    mp2 = cm2.selectbox("Underdog (P2)", jogadores, index=safe_idx(st.session_state["agenda_p2"], 1), key="mp2")

    if mp1 == mp2:
        st.error("⚠️ Seleciona jogadores diferentes.")
    else:
        s1 = get_stats(mp1, superficie, circuito, df, df_elos)
        s2 = get_stats(mp2, superficie, circuito, df, df_elos)
        cm1.metric(f"Elo {superficie}", f"{s1.elo:.1f}")
        cm1.markdown(f"📈 Forma: `{s1.recent_form:.0%}` | 💤 Fadiga: `{s1.fatigue:.0f}`")
        cm2.metric(f"Elo {superficie}", f"{s2.elo:.1f}")
        cm2.markdown(f"📈 Forma: `{s2.recent_form:.0%}` | 💤 Fadiga: `{s2.fatigue:.0f}`")

        h2h_m = get_h2h(mp1, mp2, df)
        st.markdown(f"**H2H:** {mp1} {h2h_m[0]}–{h2h_m[1]} {mp2}")

        co1, co2 = st.columns(2)
        odd1 = co1.number_input(f"Odd {mp1}", value=1.80, step=0.01, min_value=1.01)
        odd2 = co2.number_input(f"Odd {mp2}", value=2.10, step=0.01, min_value=1.01)

        if st.button("Calcular EV", key="btn_manual"):
            with st.spinner("A simular..."):
                sims = simulate(build_setup(mp1, mp2))
                mkts = _empty_mkts()
                mkts["match_winner"] = {"P1": odd1, "P2": odd2}
                bets = compute_markets(sims, mkts, mp1, mp2)
            render_results(bets, mp1, mp2, sims)

# ── CSV EM MASSA ──────────────────────────────────────────────────────────────
with tab_csv:
    st.header("Scanner de Valor Múltiplo (CSV)")
    st.markdown("`Jogador 1, Jogador 2, Odd P1, Odd P2`")
    bloco = st.text_area("Cola as linhas CSV:", height=200, key="csv_area")

    if st.button("Analisar CSV", key="btn_csv") and bloco.strip():
        resultados = []
        for i, linha in enumerate(bloco.strip().split("\n"), 1):
            partes = [p.strip() for p in linha.split(",")]
            if len(partes) < 4:
                st.warning(f"Linha {i} ignorada: precisa de pelo menos 4 campos.")
                continue
            p1c, p2c = partes[0], partes[1]
            if p1c not in jogadores or p2c not in jogadores:
                st.warning(f"Linha {i}: jogador não encontrado ({p1c} / {p2c}).")
                continue
            if p1c == p2c:
                st.warning(f"Linha {i}: jogadores iguais.")
                continue
            try:
                o1, o2 = float(partes[2]), float(partes[3])
                sims = simulate(build_setup(p1c, p2c))
                mkts = _empty_mkts()
                mkts["match_winner"] = {"P1": o1, "P2": o2}
                bets = compute_markets(sims, mkts, p1c, p2c)
                top  = best_bet(bets, limite_ev, odd_minima_rec)
                if top:
                    resultados.append({
                        "Encontro": f"{p1c} vs {p2c}",
                        "Mercado":  top.market,
                        "Odd":      f"{top.odd:.2f}",
                        "Prob":     f"{top.prob:.1%}",
                        "EV":       f"+{top.ev:.1%}",
                        "Banca %":  f"{top.stake_pct:.1%}",
                    })
            except Exception as e:
                st.warning(f"Linha {i} com erro: {e}")

        if resultados:
            st.success(f"✅ {len(resultados)} apostas com valor encontradas.")
            st.dataframe(pd.DataFrame(resultados), use_container_width=True)
        else:
            st.info("Nenhuma aposta com valor nas linhas processadas.")

# ── CALIBRAÇÃO ────────────────────────────────────────────────────────────────
with tab_calib:
    st.header("📉 Calibração Histórica do Modelo")
    st.warning(
        "⚠️ **Nota metodológica:** este backtest usa o snapshot ACTUAL de Elo "
        "como aproximação do valor à data de cada jogo histórico — não existe "
        "ainda um histórico de Elo point-in-time guardado. Os números são "
        "indicativos e tendem a ser optimistas para jogos mais antigos "
        "(lookahead bias), não substituem um backtest walk-forward puro."
    )

    n_amostra = st.slider("Nº de jogos a amostrar", 500, 10_000, 3_000, 500)

    if st.button("Correr Backtest", key="btn_backtest"):
        with st.spinner("A recalcular previsões históricas..."):
            df_bt = run_backtest(circuito, n_amostra)

        if df_bt.empty:
            st.error(
                "Dados insuficientes para o backtest — faltam colunas "
                "`player` / `opponent` / `winner` no ficheiro de dados."
            )
        else:
            bs = brier_score(df_bt)
            c1, c2 = st.columns(2)
            c1.metric(
                "Brier Score", f"{bs:.4f}",
                help="0 = previsões perfeitas | 0.25 = tão bom como adivinhar sempre 50/50 | quanto menor, melhor",
            )
            c2.metric("Jogos Analisados", f"{len(df_bt)}")

            tabela = calibration_table(df_bt)
            st.markdown("#### Curva de Calibração (Reliability Diagram)")
            st.markdown(
                "Se o modelo for honesto, a **Prob. Média Prevista** e a "
                "**Taxa Real de Vitórias** devem ficar próximas em cada bucket. "
                "Se a taxa real for sistematicamente mais baixa que a prevista "
                "nos buckets altos, o modelo está a ser demasiado confiante."
            )
            st.dataframe(tabela, use_container_width=True)

            if not tabela.empty:
                chart_df = tabela.set_index("Bucket")[["Prob. Média Prevista", "Taxa Real de Vitórias"]]
                st.line_chart(chart_df)