"""
QuantBet OS — ficheiro único para Streamlit Cloud.
Toda a lógica (config, simulação, mercados, parser, dados) está aqui,
organizada em secções claramente separadas.

v2 — Melhorias adicionadas:
  1. Hold rate específico por superfície (com shrinkage bayesiano)
  2. Forma recente ponderada pelo Elo do adversário
  3. Fadiga modelada de forma não-linear
  4. Mapeamento hold-rate em espaço logit (sigmoidal)
  5. Momentum entre sets
  6. Módulo de calibração histórica
  7. Slider manual de "Contexto"

v3 — Dados reais do Match Charting Project:
  8. Hold rate e ace rate reais ponto-a-ponto

v4 — Refinamentos de Qualidade:
  9. Fuzzy Matching de nomes de jogadores (rapidfuzz)
 10. Cache dinâmica com TTL (1 hora)
 11. Suporte para mercado de Correct Score (Sets)
 12. Validação detalhada de lucro por superfície no Backtest
"""

import os
import re
import zipfile
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import io
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
import requests
import matplotlib.pyplot as plt
# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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
    time_decay_rate: float   = 0.005
    # ---- Parâmetros Avançados --------------------------------------------
    hold_shrink_k: float     = 20.0   
    form_elo_scale: float    = 400.0  
    form_weight_min: float   = 0.3    
    form_weight_max: float   = 3.0    
    fatigue_threshold: float = 6.0    
    fatigue_exponent: float  = 1.8    
    hold_logit_scale: float  = 3.0    
    momentum_boost: float    = 0.10   
    context_weight: float    = 0.01   


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

PTS_PER_SERVICE_GAME = 6.24
MCP_HOLD_MIN_GAMES = 40.0


# ============================================================================
# SECÇÃO 2 — MOTOR DE SIMULAÇÃO (Monte Carlo vectorizado com NumPy)
# ============================================================================

@dataclass
class PlayerStats:
    elo: float
    hold_rate: float
    return_rate: float       # NOVO: Capacidade de resposta (quebrar o serviço)
    fatigue: float
    recent_form: float
    ace_rate_100: Optional[float] = None  
    hold_std: float = MODEL.noise_std  


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
    context_adj: int = 0     
    ml_model: object = None


def _elo_prob(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / MODEL.elo_k))


def _h2h_adj(h2h: Tuple[int, int]) -> float:
    w1, w2 = h2h
    total = w1 + w2
    if total == 0:
        return 0.0
    posterior = (w1 + MODEL.h2h_prior / 2) / (total + MODEL.h2h_prior)
    raw = (posterior - 0.5) * MODEL.h2h_weight * total
    return float(np.clip(raw, -MODEL.h2h_max, MODEL.h2h_max))


def _fatigue_nonlinear(games: float) -> float:
    if games <= MODEL.fatigue_threshold:
        return games * 0.5
    excess = games - MODEL.fatigue_threshold
    return MODEL.fatigue_threshold * 0.5 + excess ** MODEL.fatigue_exponent


def _compute_match_prob(setup: MatchSetup) -> float:
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


def _hold_probs(prob: float, setup: MatchSetup) -> Tuple[float, float]:
    cfg = setup.circuit_cfg
    mod = setup.surface_mod
    
    # 1. Base do Circuito ajustada pela superfície
    base_hold = float(np.clip(cfg.base_hold + mod.hold_delta, cfg.hold_min, cfg.hold_max))
    base_return = 1.0 - base_hold
    
    # 2. Vantagem de Matchup (Serviço vs Resposta)
    # Se o P1 segura 85% (média 80%) = +5%. Se o P2 quebra 25% (média 20%) = +5%. 
    # Vantagem final do P1 a servir = Base + 5% (Serviço) - 5% (Resposta do P2)
    p1_hold_edge = (setup.p1.hold_rate - base_hold) - (setup.p2.return_rate - base_return)
    p2_hold_edge = (setup.p2.hold_rate - base_hold) - (setup.p1.return_rate - base_return)
    
    # 3. Aplicar ajustamentos do Elo (probabilidade do encontro) em espaço logit
    z_base1 = _logit(base_hold + p1_hold_edge, cfg.hold_min, cfg.hold_max)
    z_base2 = _logit(base_hold + p2_hold_edge, cfg.hold_min, cfg.hold_max)
    delta = (prob - 0.5) * MODEL.game_prob_scale * MODEL.hold_logit_scale

    p1h = float(np.clip(_inv_logit(z_base1 + delta / 2, cfg.hold_min, cfg.hold_max), cfg.hold_min, cfg.hold_max))
    p2h = float(np.clip(_inv_logit(z_base2 - delta / 2, cfg.hold_min, cfg.hold_max), cfg.hold_min, cfg.hold_max))
    
    return p1h, p2h


def _apply_momentum(base_h: Union[float, np.ndarray], won_prev: np.ndarray, boost: float) -> np.ndarray:
    boosted = np.clip(base_h + boost * (1.0 - base_h), 0.0, 1.0)
    depressed = np.clip(base_h - boost * base_h, 0.0, 1.0)
    return np.where(won_prev, boosted, depressed)


def _sim_set(
    n: int,
    p1h: Union[float, np.ndarray],
    p2h: Union[float, np.ndarray],
    std1: float,           # NOVO
    std2: float,           # NOVO
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    g1 = np.zeros(n, dtype=np.int32)
    g2 = np.zeros(n, dtype=np.int32)
    active = np.ones(n, dtype=bool)

    for game_idx in range(13):  
        if not active.any():
            break
        base_p = p1h if (game_idx % 2 == 0) else (1.0 - p2h)
        
        # O ruído agora depende de quem está a servir!
        current_std = std1 if (game_idx % 2 == 0) else std2
        probs = np.clip(base_p + rng.normal(0, current_std, n), 0.0, 1.0)
        
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
    rng = np.random.default_rng(42)
    match_prob = _compute_match_prob(setup)
    p1h, p2h = _hold_probs(match_prob, setup)

    ace_base = setup.circuit_cfg.base_aces_per_game * setup.surface_mod.ace_multiplier

    if setup.p1.ace_rate_100 is not None:
        rate_a1 = max(0.02, (setup.p1.ace_rate_100 / 100.0) * PTS_PER_SERVICE_GAME)
    else:
        rate_a1 = max(0.05, ace_base + (p1h - setup.circuit_cfg.base_hold))

    if setup.p2.ace_rate_100 is not None:
        rate_a2 = max(0.02, (setup.p2.ace_rate_100 / 100.0) * PTS_PER_SERVICE_GAME)
    else:
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

        # O alinhamento correto (exatamente 8 espaços da margem)
        g1, g2 = _sim_set(n, cur_p1h, cur_p2h, setup.p1.hold_std, setup.p2.hold_std, rng)
        games_in_set = g1 + g2

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
        "p1_std": setup.p1.hold_std, 
        "p2_std": setup.p2.hold_std  
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
    market_fair_prob: float = 0.0  # NOVO: Probabilidade verdadeira estimada da casa

    @property
    def fair_odd(self) -> float:
        return 1.0 / self.prob if self.prob > 0 else float("inf")

    @property
    def stake_pct(self) -> float:
        return float(np.clip(self.kelly, MODEL.kelly_min, MODEL.kelly_max))


def _devig_probs(odd1: float, odd2: float) -> Tuple[float, float]:
    """Calcula a probabilidade justa da casa removendo o overround (margin)."""
    if odd1 <= 1.0 or odd2 <= 1.0:
        return (0.0, 0.0)
    implied1, implied2 = 1.0 / odd1, 1.0 / odd2
    margin = implied1 + implied2 - 1.0
    if margin > 0:
        return (implied1 / (1.0 + margin), implied2 / (1.0 + margin))
    return (implied1, implied2)


def optimize_portfolio(bets: List[Bet], max_global_exposure: float = 0.15) -> List[Bet]:
    """
    Filtra apostas correlacionadas e corta a stake caso a tua banca global 
    já esteja demasiado exposta (limite de 15% de banca pendente).
    """
    pendentes_atuais = get_exposed_bankroll()
    budget_disponivel = max(0.0, max_global_exposure - pendentes_atuais)
    
    sorted_bets = sorted([b for b in bets if b.ev > 0], key=lambda b: b.kelly, reverse=True)
    portfolio = []
    current_game_exposure = 0.0
    lados_cobertos = set()
    
    for b in sorted_bets:
        if any(x in b.market for x in ["P1", "Jogador 1"]): cat = "Lado_P1"
        elif any(x in b.market for x in ["P2", "Jogador 2"]): cat = "Lado_P2"
        else: cat = "Totais_O/U"
              
        if cat in lados_cobertos and cat != "Totais_O/U":
            continue 
            
        # Verifica se o jogo e a banca global aguentam mais esta aposta
        if (current_game_exposure + b.stake_pct <= (MODEL.kelly_max * 1.5)) and (b.stake_pct <= budget_disponivel):
            portfolio.append(b)
            current_game_exposure += b.stake_pct
            budget_disponivel -= b.stake_pct
            lados_cobertos.add(cat)
            
    return portfolio
def _kelly(prob: float, odd: float) -> float:
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

    # Correct Score (Resultado Exato em Sets)
    max_sets = np.max(p1s + p2s)
    cs_mkts = mkts.get("correct_score", {})
    
    if max_sets <= 3: 
        prob_2_0_p1 = float(np.mean((p1s == 2) & (p2s == 0)))
        prob_2_1_p1 = float(np.mean((p1s == 2) & (p2s == 1)))
        prob_2_0_p2 = float(np.mean((p2s == 2) & (p1s == 0)))
        prob_2_1_p2 = float(np.mean((p2s == 2) & (p1s == 1)))
        
        if "2-0 P1" in cs_mkts: bets.append(_bet(f"Resultado Exato 2-0 {p1}", prob_2_0_p1, cs_mkts["2-0 P1"]))
        if "2-1 P1" in cs_mkts: bets.append(_bet(f"Resultado Exato 2-1 {p1}", prob_2_1_p1, cs_mkts["2-1 P1"]))
        if "2-0 P2" in cs_mkts: bets.append(_bet(f"Resultado Exato 2-0 {p2}", prob_2_0_p2, cs_mkts["2-0 P2"]))
        if "2-1 P2" in cs_mkts: bets.append(_bet(f"Resultado Exato 2-1 {p2}", prob_2_1_p2, cs_mkts["2-1 P2"]))

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
        "correct_score": {},
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

@st.cache_data(ttl="1h", show_spinner=False)
def load_match_data() -> pd.DataFrame:
    with zipfile.ZipFile("dados_resumidos.zip", "r") as z:
        df = pd.read_csv(z.open("dados_resumidos.csv"))
    
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    if "winner" in df.columns and "loser" in df.columns and "player" not in df.columns:
        df["player"] = df["winner"]
        df["opponent"] = df["loser"]
        
    for col, norm in [("player", "_pn"), ("opponent", "_on"), ("winner", "_wn")]:
        if col in df.columns:
            df[norm] = df[col].astype(str).str.casefold().str.strip()
            
    return df


@st.cache_data(ttl="1h", show_spinner=False)
def load_elos(circuito: str) -> pd.DataFrame:
    f = "EloRankP.csv" if circuito == "WTA (Feminino)" else "PlayerElo.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        df["_nn"] = df["Player"].str.casefold().str.strip()
        return df
    return pd.DataFrame(columns=["Player", "Elo", "hElo", "cElo", "gElo", "_nn"])


@st.cache_data(ttl="1h", show_spinner=False)
def load_surface_profile() -> pd.DataFrame:
    f = "player_surface_profile.csv"
    if os.path.exists(f):
        prof = pd.read_csv(f)
        prof["_nn"] = prof["player"].astype(str).str.casefold().str.strip()
        return prof
    return pd.DataFrame(columns=[
        "player", "surface", "n_matches", "hold_rate_final", "hold_rate_source",
        "games_played_exact", "ace_rate_per_100_serve_pts", "_nn",
    ])


def load_agenda() -> pd.DataFrame:
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


def _fuzzy_match(name: str, choices: List[str], threshold: float = 85.0) -> str:
    if not choices:
        return name
    match = process.extractOne(name, choices, scorer=fuzz.WRatio)
    if match and match[1] >= threshold:
        return match[0]
    return name


def _lookup_elo(nn: str, superficie: str, df_elos: pd.DataFrame, default: float) -> float:
    if df_elos.empty or "_nn" not in df_elos.columns:
        logging.warning(f"Ficheiro de Elos vazio. A usar default ({default}) para {nn}.")
        return default
    
    row = df_elos[df_elos["_nn"] == nn]
    
    if row.empty:
        opcoes = df_elos["_nn"].tolist()
        best_match = _fuzzy_match(nn, opcoes)
        if best_match != nn:
            logging.info(f"Fuzzy match: '{nn}' corrigido para '{best_match}'")
            row = df_elos[df_elos["_nn"] == best_match]
        else:
            logging.warning(f"Jogador '{nn}' não encontrado. A usar Elo default ({default}).")
            return default

    col = {"Clay": "cElo", "Grass": "gElo", "Hard": "hElo"}.get(superficie, "Elo")
    if col in row.columns and pd.notna(row[col].iloc[0]):
        return float(row[col].iloc[0])
    
    return default

def _calculate_time_weights(dates_series: pd.Series, decay_rate: float = MODEL.time_decay_rate) -> np.ndarray:
    """Calcula pesos exponenciais com base nos dias até à data atual."""
    if dates_series.isna().all():
        return np.ones(len(dates_series))

    hoje = pd.Timestamp.now()
    dates = pd.to_datetime(dates_series, errors='coerce')
    
    # Diferença em dias (fallback de 5 anos para jogos sem data)
    dias_passados = (hoje - dates).dt.days.fillna(365 * 5)
    dias_passados = np.clip(dias_passados, 0, None)

    # Fórmula: W = e^(-lambda * dias)
    weights = np.exp(-decay_rate * dias_passados)
    return weights.to_numpy()
def get_stats(nome: str, superficie: str, circuito: str, df: pd.DataFrame, df_elos: pd.DataFrame) -> PlayerStats:
    cfg = CIRCUIT[circuito]
    nn = str(nome).casefold().strip()

    elo = _lookup_elo(nn, superficie, df_elos, cfg.default_elo)

    hold = cfg.base_hold
    if "_pn" in df.columns and "hold_percentage" in df.columns:
        m_all = df[df["_pn"] == nn]
        if not m_all.empty:
            # Novo: Cálculo de pesos se existir coluna de data (tourney_date ou date)
            date_col = "tourney_date" if "tourney_date" in m_all.columns else ("date" if "date" in m_all.columns else None)
            
            if date_col:
                pesos_all = _calculate_time_weights(m_all[date_col])
                hold_global = float(np.average(m_all["hold_percentage"], weights=pesos_all))
            else:
                hold_global = float(m_all["hold_percentage"].mean())

            if "surface" in df.columns:
                m_surf = m_all[m_all["surface"] == superficie]
            else:
                m_surf = m_all.iloc[0:0]
                
            if not m_surf.empty:
                if date_col:
                    pesos_surf = _calculate_time_weights(m_surf[date_col])
                    hold_surf = float(np.average(m_surf["hold_percentage"], weights=pesos_surf))
                else:
                    hold_surf = float(m_surf["hold_percentage"].mean())
                    
                n_surf = len(m_surf)
                w = n_surf / (n_surf + MODEL.hold_shrink_k)
                hold = w * hold_surf + (1 - w) * hold_global
            else:
                hold = hold_global

    ace_rate_100 = None
    df_profile = load_surface_profile()
    if not df_profile.empty:
        row = df_profile[(df_profile["_nn"] == nn) & (df_profile["surface"] == superficie)]
        if not row.empty:
            row = row.iloc[0]
            hold_real = float(row["hold_rate_final"])
            n_games_real = float(row.get("games_played_exact", 0) or 0)
            w_real = n_games_real / (n_games_real + MCP_HOLD_MIN_GAMES) if n_games_real > 0 else 0.5
            hold = w_real * hold_real + (1 - w_real) * hold
            if pd.notna(row.get("ace_rate_per_100_serve_pts")):
                ace_rate_100 = float(row["ace_rate_per_100_serve_pts"])

    fatigue = 0.0
    if "_pn" in df.columns and "games_played_last_week" in df.columns:
        m = df[df["_pn"] == nn]
        if not m.empty:
            fatigue = float(m["games_played_last_week"].iloc[-1])

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
# --- Modelação da Variância Individual (Hold STD) ---
    hold_std = MODEL.noise_std
    if "_pn" in df.columns and "hold_percentage" in df.columns and not m_all.empty:
        if len(m_all) >= 10:  
            if date_col:
                # Variância ponderada
                media_pond = np.average(m_all["hold_percentage"], weights=pesos_all)
                variancia_pond = np.average((m_all["hold_percentage"] - media_pond)**2, weights=pesos_all)
                std_real = float(np.sqrt(variancia_pond))
            else:
                std_real = float(m_all["hold_percentage"].std())
                
            if pd.notna(std_real):
                hold_std = float(np.clip(std_real, 0.01, 0.04))

    # --- Forma recente ponderada pelo Elo do adversário E pelo tempo ---
    form = 0.5
    if "_pn" in df.columns and "_wn" in df.columns and "_on" in df.columns:
        m_form = df[df["_pn"] == nn].tail(15) # Aumentamos para 15 jogos porque o tempo agora filtra o ruído
        if not m_form.empty:
            weights, wins = [], []
            
            # Gerar pesos de tempo se aplicável
            pesos_tempo = _calculate_time_weights(m_form[date_col]) if date_col else np.ones(len(m_form))
            
            for i, (_, row) in enumerate(m_form.iterrows()):
                opp_nn = row["_on"]
                opp_elo = _lookup_elo(opp_nn, superficie, df_elos, cfg.default_elo)
                
                # Peso Elo adversário
                w_elo = float(np.clip(
                    2.0 ** ((opp_elo - cfg.default_elo) / MODEL.form_elo_scale),
                    MODEL.form_weight_min, MODEL.form_weight_max,
                ))
                
                # Peso final = Peso do Tempo * Peso do Elo do Adversário
                weights.append(w_elo * pesos_tempo[i])
                wins.append(1.0 if row["_wn"] == nn else 0.0)
                
            weights_arr = np.array(weights)
            wins_arr = np.array(wins)
            if np.sum(weights_arr) > 0:
                form = float(np.sum(weights_arr * wins_arr) / np.sum(weights_arr))

    return_rate = 1.0 - cfg.base_hold

    return PlayerStats(
        elo=elo, 
        hold_rate=hold, 
        return_rate=return_rate,  # A linha que faltava!
        fatigue=fatigue, 
        recent_form=form, 
        ace_rate_100=ace_rate_100, 
        hold_std=hold_std
    )

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

@st.cache_data(ttl="1h", show_spinner=False)
def run_backtest(circuito: str, n_sample: int = 3000, limite_valor: float = 0.05) -> pd.DataFrame:
    ficheiro_backtest = "2023.csv"
    
    if not os.path.exists(ficheiro_backtest):
        return pd.DataFrame()
        
    df = pd.read_csv(ficheiro_backtest)
    df_elos = load_elos(circuito)
    cfg = CIRCUIT[circuito]
    
    df = df.dropna(subset=["Winner", "Loser", "AvgW", "AvgL"])
    
    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=42)

    resultados = []
    for _, r in df.iterrows():
        p1n, p2n = str(r["Winner"]).casefold().strip(), str(r["Loser"]).casefold().strip()
        surf = r["Surface"] if ("Surface" in r and pd.notna(r["Surface"])) else "Hard"
        
        e1 = _lookup_elo(p1n, surf, df_elos, cfg.default_elo)
        e2 = _lookup_elo(p2n, surf, df_elos, cfg.default_elo)
        
        prob_modelo_w = _elo_prob(e1 - e2)
        prob_modelo_l = 1.0 - prob_modelo_w
        
        prob_casa_w = 1.0 / r["AvgW"] if r["AvgW"] > 0 else 1.0
        prob_casa_l = 1.0 / r["AvgL"] if r["AvgL"] > 0 else 1.0
        
        ev_w = (r["AvgW"] * prob_modelo_w) - 1
        ev_l = (r["AvgL"] * prob_modelo_l) - 1
        
        aposta_feita = "Nenhuma"
        lucro = 0.0
        
        if ev_w > limite_valor and ev_w > ev_l:
            aposta_feita = "Vencedor"
            lucro = r["AvgW"] - 1.0  
        elif ev_l > limite_valor and ev_l > ev_w:
            aposta_feita = "Perdedor"
            lucro = -1.0  
            
        resultados.append({
            "prob": prob_modelo_w, 
            "actual": 1.0,  
            "aposta": aposta_feita,
            "lucro": lucro,
            "Surface": surf
        })

    return pd.DataFrame(resultados)


def brier_score(df_bt: pd.DataFrame) -> float:
    if df_bt.empty:
        return float("nan")
    return float(np.mean((df_bt["prob"] - df_bt["actual"]) ** 2))


def calibration_table(df_bt: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
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
# SECÇÃO 7 — RADAR DE MERCADO AO VIVO (The Odds API)
# ============================================================================

@st.cache_data(ttl="5m", show_spinner=False)
def fetch_live_odds(api_key: str, circuito: str) -> list:
    """Vai buscar todos os jogos e odds ao vivo da The Odds API."""
    
    # 1. Descobrir quais os torneios ativos neste momento
    sports_url = "https://api.the-odds-api.com/v4/sports/"
    try:
        sports_resp = requests.get(sports_url, params={"apiKey": api_key})
        sports_resp.raise_for_status()
        active_sports = sports_resp.json()
    except Exception as e:
        st.error(f"Erro ao obter a lista de torneios. Verifica a chave API. Detalhe: {e}")
        return []
    
    # 2. Procurar chaves que correspondam ao teu circuito selecionado (ATP ou WTA)
    prefix = "tennis_atp" if "ATP" in circuito else "tennis_wta"
    active_keys = [s["key"] for s in active_sports if str(s["key"]).startswith(prefix)]
    
    # 3. MODO FALLBACK: Se estiver vazio, tenta puxar QUALQUER ténis disponível
    if not active_keys:
        fallback_keys = [s["key"] for s in active_sports if "tennis" in str(s["key"]).lower()]
        
        if fallback_keys:
            st.info(f"ℹ️ Não há {prefix} hoje (a API foca-se em ATP/WTA 500 e superiores). Vamos carregar outros torneios disponíveis para testares a app!")
            active_keys = fallback_keys
        else:
            st.warning("Não há absolutamente nenhum torneio de ténis a decorrer na The Odds API neste momento.")
            return []

    # 4. Varrer cada torneio ativo e agregar todos os jogos numa única lista
    all_games = []
    for sport_key in active_keys:
        odds_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        params = {
            "apiKey": api_key,
            "regions": "eu",        # Casas europeias/globais (Betano, Pinnacle, etc.)
            "markets": "h2h",       # Mercado de vencedor da partida
            "oddsFormat": "decimal"
        }
        
        try:
            resp = requests.get(odds_url, params=params)
            resp.raise_for_status()
            all_games.extend(resp.json())
        except Exception:
            # Ignoramos se um torneio específico falhar
            pass
            
    return all_games

def match_api_names(api_name: str, escolhas_validas: list) -> str:
    """Corrige os nomes da API para baterem certo com a tua base de dados."""
    return _fuzzy_match(api_name, escolhas_validas, threshold=80.0)
# ============================================================================
# ============================================================================
# SECÇÃO 8 — GESTÃO DE BANCA E HISTÓRICO (SQLite)
# ============================================================================

def init_db():
    conn = sqlite3.connect("quantbet.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS bet_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_registo TEXT,
            encontro TEXT,
            mercado TEXT,
            odd REAL,
            stake_pct REAL,
            ev_projetado REAL,
            status TEXT DEFAULT 'Pendente',
            lucro_real REAL DEFAULT 0.0,
            closing_odd REAL DEFAULT 0.0
        )
    ''')
    # Tenta adicionar a coluna closing_odd caso a tabela já exista (atualização segura)
    try:
        c.execute("ALTER TABLE bet_history ADD COLUMN closing_odd REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass # A coluna já existe, ignorar
    conn.commit()
    conn.close()

def add_bet(encontro: str, mercado: str, odd: float, stake_pct: float, ev: float):
    conn = sqlite3.connect("quantbet.db")
    c = conn.cursor()
    hoje = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute('''
        INSERT INTO bet_history (data_registo, encontro, mercado, odd, stake_pct, ev_projetado)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (hoje, encontro, mercado, odd, stake_pct, ev))
    conn.commit()
    conn.close()

def get_bet_history() -> pd.DataFrame:
    conn = sqlite3.connect("quantbet.db")
    df_bets = pd.read_sql_query("SELECT * FROM bet_history ORDER BY id DESC", conn)
    conn.close()
    return df_bets

def get_exposed_bankroll() -> float:
    """Calcula quanta % da banca está neste momento pendente no mercado."""
    conn = sqlite3.connect("quantbet.db")
    c = conn.cursor()
    try:
        c.execute("SELECT SUM(stake_pct) FROM bet_history WHERE status = 'Pendente'")
        val = c.fetchone()[0]
        exposed = float(val) if val else 0.0
    except:
        exposed = 0.0
    conn.close()
    return exposed

def resolve_bet(bet_id: int, status: str, odd: float, stake_pct: float, closing_odd: float):
    lucro = 0.0
    if status == 'Ganha':
        lucro = stake_pct * (odd - 1.0)
    elif status == 'Perdida':
        lucro = -stake_pct

    conn = sqlite3.connect("quantbet.db")
    c = conn.cursor()
    c.execute('''
        UPDATE bet_history 
        SET status = ?, lucro_real = ?, closing_odd = ?
        WHERE id = ?
    ''', (status, lucro, closing_odd, bet_id))
    conn.commit()
    conn.close()

init_db()
def simulate_bankroll_ruin(win_rate: float, avg_odd: float, avg_stake: float, n_sims: int = 1000, n_bets: int = 500):
    """Simula o futuro da tua banca para prever o risco de ruína (falência)."""
    rng = np.random.default_rng()
    bancas_finais = []
    ruinas = 0
    trajetorias = []
    
    for _ in range(n_sims):
        banca = 100.0 # Começamos com 100 unidades (ou %)
        caminho = [banca]
        
        # Simula uma série de apostas
        vitorias = rng.random(n_bets) < win_rate
        
        for ganhou in vitorias:
            stake = banca * avg_stake
            if ganhou:
                banca += stake * (avg_odd - 1.0)
            else:
                banca -= stake
                
            caminho.append(banca)
            if banca <= 5.0: # Assumimos "ruína" se cair para 5% do inicial
                break
                
        if banca <= 5.0:
            ruinas += 1
            
        bancas_finais.append(banca)
        if len(trajetorias) < 50: # Guardar algumas trajetórias para desenhar o gráfico
            trajetorias.append(caminho)
            
    return ruinas / n_sims, np.median(bancas_finais), trajetorias
# ============================================================================
# SECÇÃO 6 — INTERFACE STREAMLIT
# ============================================================================

st.set_page_config(page_title="QuantBet OS", layout="wide")
st.title("🎾 QuantBet OS: Sistema Quantitativo ATP & WTA")

for k in ("agenda_p1", "agenda_p2"):
    if k not in st.session_state:
        st.session_state[k] = None

df       = load_match_data()
ml_model = load_ml_model()

st.sidebar.header("1. Configurações Globais")
circuito = st.sidebar.radio("Circuito", list(CIRCUIT.keys()))
if ml_model is None:
    st.sidebar.info("🤖 Motor: Elo Matemático (Fallback)")
else:
    st.sidebar.success("🤖 Motor: XGBoost Calibrado")

df_elos    = load_elos(circuito)
_profile_check = load_surface_profile()
if _profile_check.empty:
    st.sidebar.caption("📊 Perfil MCP (hold/ace reais): não encontrado — a usar apenas o proxy antigo.")
else:
    st.sidebar.caption(f"📊 Perfil MCP: {len(_profile_check)} combinações jogador/superfície carregadas.")
superficie = st.sidebar.selectbox("Superfície", sorted(df["surface"].dropna().unique()))
vel_campo = st.sidebar.selectbox("Velocidade do Campo", list(SURFACE_MOD.keys())) 
sets_padrao = [3] if circuito == "WTA (Feminino)" else [3, 5]
sets_input  = st.sidebar.radio("Sets do Encontro", sets_padrao)

jogadores = sorted(df_elos["Player"].dropna().unique()) if not df_elos.empty else ["Jogador A", "Jogador B"]
if len(jogadores) < 2:
    st.sidebar.error("⚠️ Ficheiro de Elos vazio ou não encontrado.")
# 1. Configurar a pasta física (Esta é a variável que estava a faltar!)
PASTA_DADOS = "dados_historicos"
if not os.path.exists(PASTA_DADOS):
    os.makedirs(PASTA_DADOS)

import os
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# SECÇÃO - PIPELINE DE DADOS (DATA CENTER LOCAL)
# ============================================================================

PASTA_DADOS = "dados_historicos"
if not os.path.exists(PASTA_DADOS):
    os.makedirs(PASTA_DADOS)

st.sidebar.header("🔄 Pipeline de Dados (Data Center)")
st.sidebar.markdown(f"Os teus ficheiros ficam guardados localmente na pasta `{PASTA_DADOS}`.")

opcoes_pipeline = st.sidebar.multiselect(
    "Opções de Cálculo a aplicar:",
    [
        "Normalizar Nomes (Fuzzy/Minúsculas)", 
        "Calcular Hold/Break Rates (Estatísticas de Serviço)",
        "Remover Jogos Incompletos (Retiradas/Walkovers)"
    ],
    default=[
        "Normalizar Nomes (Fuzzy/Minúsculas)", 
        "Calcular Hold/Break Rates (Estatísticas de Serviço)"
    ]
)

@st.cache_data(show_spinner=False)
def process_multiple_csvs(file_paths, opcoes) -> pd.DataFrame:
    dfs = []
    for path in file_paths:
        try:
            df_temp = pd.read_csv(path)
            df_temp.columns = [str(c).lower().strip() for c in df_temp.columns]
            
            if "Normalizar Nomes (Fuzzy/Minúsculas)" in opcoes:
                for col, norm in [("winner_name", "_wn"), ("loser_name", "_on")]:
                    if col in df_temp.columns:
                        df_temp[norm] = df_temp[col].astype(str).str.casefold().str.strip()
            
            if "Calcular Hold/Break Rates (Estatísticas de Serviço)" in opcoes:
                if "w_svgms" in df_temp.columns:
                    df_temp["w_svgms"] = df_temp["w_svgms"].replace(0, np.nan)
                    df_temp["l_svgms"] = df_temp["l_svgms"].replace(0, np.nan)
                    df_temp["w_hold_pct"] = (df_temp["w_svgms"] - df_temp.get("l_bpconverted", 0)) / df_temp["w_svgms"]
                    df_temp["l_hold_pct"] = (df_temp["l_svgms"] - df_temp.get("w_bpconverted", 0)) / df_temp["l_svgms"]
            
            if "Remover Jogos Incompletos (Retiradas/Walkovers)" in opcoes:
                if "score" in df_temp.columns:
                    df_temp = df_temp[~df_temp["score"].astype(str).str.contains("RET|W/O", na=False)]
            
            dfs.append(df_temp)
        except Exception as e:
            st.sidebar.error(f"Erro ao processar {path}: {e}")
            
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

ficheiros_upload = st.sidebar.file_uploader(
    "Adicionar NOVOS ficheiros ao arquivo", 
    type=["csv"], 
    accept_multiple_files=True
)

if ficheiros_upload:
    for f in ficheiros_upload:
        caminho_destino = os.path.join(PASTA_DADOS, f.name)
        with open(caminho_destino, "wb") as f_out:
            f_out.write(f.getbuffer())
    st.sidebar.success(f"✅ {len(ficheiros_upload)} ficheiros guardados permanentemente!")

ficheiros_locais = [os.path.join(PASTA_DADOS, f) for f in os.listdir(PASTA_DADOS) if f.endswith('.csv')]

if ficheiros_locais:
    st.sidebar.info(f"📂 Encontrados {len(ficheiros_locais)} ficheiros no teu arquivo local.")
    
    if st.sidebar.button("Carregar Dados do Arquivo 🚀", type="primary"):
        with st.spinner(f"A processar {len(ficheiros_locais)} ficheiro(s) do disco..."):
            df_live = process_multiple_csvs(ficheiros_locais, opcoes_pipeline)
            st.sidebar.success(f"✅ {len(df_live)} encontros injetados no motor com sucesso!")
else:
    df_live = pd.DataFrame()
    st.sidebar.warning("O teu arquivo local está vazio. Faz upload do primeiro CSV.")
st.sidebar.header("2. Filtros de Valor")
limite_ev      = st.sidebar.slider("EV Mínimo (%)", 1.0, 15.0, 5.0, 0.5) / 100
odd_minima_rec = st.sidebar.number_input("Odd Mínima Recomendada", value=1.50, step=0.05)

                                                                                  
def safe_idx(name, fallback=0) -> int:
    try:
        return jogadores.index(name) if name in jogadores else fallback
    except Exception:
        return fallback


def build_setup(p1: str, p2: str, vel_campo: str, h2h_override=None) -> MatchSetup:
    
    mod_seguro = SURFACE_MOD.get(vel_campo, SURFACE_MOD["Médio (Hard Normal)"])
    
    return MatchSetup(
        p1=get_stats(p1, superficie, circuito, df, df_elos),
        p2=get_stats(p2, superficie, circuito, df, df_elos),
        sets_to_win=sets_input // 2 + 1,
        circuit_cfg=CIRCUIT[circuito],
        surface_mod=mod_seguro,
        form_adj=0,
        fatigue_adj=0,
        context_adj=0,
        h2h=h2h_override if h2h_override is not None else get_h2h(p1, p2, df),
        ml_model=ml_model,
    )
    


def render_results(bets: List[Bet], p1: str, p2: str, sims: dict) -> None:
    if not bets:
        st.error("Não foram encontrados mercados. Verifica a formatação do texto.")
        return

    eligible_bets = [b for b in bets if b.ev >= limite_ev and b.odd >= odd_minima_rec]
    portfolio = optimize_portfolio(eligible_bets)

    # 1. Recomendação de Risco Gerido
    if portfolio:
        st.markdown("---")
        st.markdown("### 🏆 Portfólio de Jogo Recomendado (Risco Gerido)")
        
        cols = st.columns(len(portfolio))
        total_stake = 0.0
        
        for i, bet in enumerate(portfolio):
            total_stake += bet.stake_pct
            with cols[i]:
                st.success(
                    f"**{bet.market}**\n\n"
                    f"Odd: `{bet.odd:.2f}` | Prob: `{bet.prob:.1%}`\n\n"
                    f"📈 EV: `+{bet.ev:.1%}` | ⚖️ Stake: **{bet.stake_pct:.1%}**"
                )
                if st.button("Gravar Aposta 💾", key=f"gravar_{p1}_{p2}_{bet.market}"):
                    add_bet(f"{p1} vs {p2}", bet.market, bet.odd, bet.stake_pct, bet.ev)
                    st.toast("✅ Aposta registada com sucesso na tua Banca!")
            
        banca_livre = 0.15 - get_exposed_bankroll()
        st.caption(f"**Exposição no Jogo:** {total_stake:.1%} | **Banca Livre Restante:** {banca_livre:.1%}")
        st.markdown("---")

    # 2. Histogramas do Motor de Monte Carlo
    st.markdown("#### 📊 Distribuição Visual de Probabilidades (10.000 Simulações)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Histograma de Totais de Jogos
    ax1.hist(sims['total_games'], bins=range(15, 40), color='#4CAF50', edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(sims['total_games']), color='red', linestyle='dashed', linewidth=2, label='Média')
    ax1.set_title("Total de Jogos Previstos")
    ax1.legend()

    # Histograma de Handicaps
    ax2.hist(sims['game_diff'], bins=range(-12, 13), color='#2196F3', edgecolor='white', alpha=0.8)
    ax2.axvline(np.mean(sims['game_diff']), color='red', linestyle='dashed', linewidth=2, label='Média')
    ax2.set_title(f"Vantagem de Jogos ({p1} positivo)")
    ax2.legend()
    
    st.pyplot(fig)
    st.markdown("---")

    # 3. Auditoria
    rows = [{
        "Status":    "✅ Selecionada" if b in portfolio else ("🔄 Valor Redundante" if b.ev >= limite_ev else "❌ Evitar"),
        "Mercado":   b.market,
        "Odd":       f"{b.odd:.2f}",
        "Odd Justa": f"{b.fair_odd:.2f}",
        "Prob":      f"{b.prob:.2%}",
        "EV":        f"{'+' if b.ev > 0 else ''}{b.ev:.2%}",
    } for b in bets]
    
    st.markdown("#### 📋 Auditoria Completa")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

tab_agenda, tab_live, tab_scanner, tab_manual, tab_csv, tab_calib, tab_banca = st.tabs(
    ["📅 Agenda", "📡 Radar", "🤖 Auto-Scanner", "🔍 Manual", "🚀 CSV", "📉 Calibração", "💰 Gestão de Banca"]
)

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
with tab_live:
    st.header("📡 Radar de Valor em Tempo Real")
    st.markdown("O sistema varre dezenas de casas de apostas de uma vez, pega na **melhor odd do mercado** para cada jogador, e cruza-a com o teu modelo matemático.")
    
    api_key = st.text_input("A tua chave da The Odds API", type="password")
    
    if st.button("Varrer Mercado", type="primary"):
        if not api_key:
            st.warning("Por favor, insere a chave da API em cima.")
        else:
            with st.spinner("A extrair dados das casas de apostas e a simular..."):
                jogos_api = fetch_live_odds(api_key, circuito)
                encontrados = []
                
                for jogo in jogos_api:
                    p1_api, p2_api = jogo.get("home_team"), jogo.get("away_team")
                    if not p1_api or not p2_api: continue
                    
                    # Converte nomes da API para os teus nomes
                    p1_csv = match_api_names(p1_api, jogadores)
                    p2_csv = match_api_names(p2_api, jogadores)
                    
                    if p1_csv in jogadores and p2_csv in jogadores and p1_csv != p2_csv:
                        # Descobrir a odd mais alta em todo o mercado
                        best_odd_p1, best_odd_p2 = 0.0, 0.0
                        casa_p1, casa_p2 = "", ""
                        
                        for bookmaker in jogo.get("bookmakers", []):
                            for market in bookmaker.get("markets", []):
                                if market["key"] == "h2h":
                                    for outcome in market["outcomes"]:
                                        if outcome["name"] == p1_api and outcome["price"] > best_odd_p1:
                                            best_odd_p1 = outcome["price"]
                                            casa_p1 = bookmaker["title"]
                                        elif outcome["name"] == p2_api and outcome["price"] > best_odd_p2:
                                            best_odd_p2 = outcome["price"]
                                            casa_p2 = bookmaker["title"]
                        
                        # Injetar as melhores odds no modelo
                        sims = simulate(build_setup(p1_csv, p2_csv))
                        mkts = _empty_mkts()
                        if best_odd_p1 > 0: mkts["match_winner"]["P1"] = best_odd_p1
                        if best_odd_p2 > 0: mkts["match_winner"]["P2"] = best_odd_p2
                        
                        bets = compute_markets(sims, mkts, p1_csv, p2_csv)
                        top = best_bet(bets, limite_ev, odd_minima_rec)
                        
                        if top:
                            nome_casa = casa_p1 if "P1" in top.market or p1_csv in top.market else casa_p2
                            encontrados.append({
                                "Encontro": f"{p1_csv} vs {p2_csv}",
                                "Aposta": top.market.replace("P1", p1_csv).replace("P2", p2_csv),
                                "Melhor Odd": f"{top.odd:.2f}",
                                "Casa": nome_casa,
                                "Prob Real": f"{top.prob:.1%}",
                                "EV": f"+{top.ev:.1%}",
                                "Aposta Sugerida": f"{top.stake_pct:.1%} da Banca"
                            })
                
                if encontrados:
                    st.success(f"🚨 Bingo! Encontradas {len(encontrados)} oportunidades de lucro no mercado atual.")
                    df_live = pd.DataFrame(encontrados).sort_values("EV", ascending=False)
                    st.dataframe(df_live, use_container_width=True)
                else:
                    st.info("O mercado está eficiente neste momento. Nenhuma odd bate o EV mínimo que definiste.")
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
                sims = simulate(build_setup(mp1, mp2, vel_campo))
                mkts = _empty_mkts()
                mkts["match_winner"] = {"P1": odd1, "P2": odd2}
                bets = compute_markets(sims, mkts, mp1, mp2)
            render_results(bets, mp1, mp2, sims)

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

with tab_calib:
    st.header("📉 Backtest Histórico (2023)")
    st.markdown("Testa o teu modelo contra as odds reais de fecho (`AvgW` / `AvgL`) para veres se gera lucro a longo prazo.")

    c_amostra, c_ev = st.columns(2)
    n_amostra = c_amostra.slider("Nº de jogos a testar", 500, 5000, 2000, 500)
    filtro_ev = c_ev.slider("EV Mínimo para Apostar (%)", 1.0, 15.0, 5.0, 0.5) / 100

    if st.button("Correr Backtest", key="btn_backtest"):
        if not os.path.exists("2023.csv"):
            st.error("❌ O ficheiro `2023.csv` não foi encontrado na pasta principal.")
        else:
            with st.spinner("A cruzar previsões com odds de mercado..."):
                df_bt = run_backtest(circuito, n_amostra, filtro_ev)

            if df_bt.empty:
                st.error("Dados insuficientes ou ficheiro com formato incorreto.")
            else:
                bs = brier_score(df_bt)
                
                total_apostas = len(df_bt[df_bt["aposta"] != "Nenhuma"])
                lucro_unidades = df_bt["lucro"].sum()
                roi = (lucro_unidades / total_apostas) * 100 if total_apostas > 0 else 0
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Brier Score", f"{bs:.4f}")
                c2.metric("Jogos Analisados", f"{len(df_bt)}")
                c3.metric("Apostas Feitas", f"{total_apostas}")
                
                delta_color = "normal" if lucro_unidades >= 0 else "inverse"
                c4.metric("Lucro (Unidades)", f"{lucro_unidades:.2f} U", f"ROI: {roi:.2f}%", delta_color=delta_color)

                st.markdown("#### Performance por Superfície")
                if "Surface" in df_bt.columns:
                    perf_surf = df_bt[df_bt["aposta"] != "Nenhuma"].groupby("Surface").apply(
                        lambda x: pd.Series({
                            "Apostas": len(x),
                            "Lucro (U)": x["lucro"].sum(),
                            "ROI (%)": (x["lucro"].sum() / len(x)) * 100 if len(x) > 0 else 0
                        })
                    ).reset_index()
                    st.dataframe(perf_surf.style.format({"Lucro (U)": "{:.2f}", "ROI (%)": "{:.2f}%"}), use_container_width=True)

                tabela = calibration_table(df_bt)
                st.markdown("#### Curva de Calibração (Reliability Diagram)")
                st.dataframe(tabela, use_container_width=True)

                if not tabela.empty:
                    chart_df = tabela.set_index("Bucket")[["Prob. Média Prevista", "Taxa Real de Vitórias"]]
                    st.line_chart(chart_df)
with tab_banca:
    st.header("💰 Gestão de Banca e Tracking Real")
    st.markdown("Monitoriza a performance real das tuas recomendações e o teu **Closing Line Value (CLV)**.")
    
    df_history = get_bet_history()
    
    if df_history.empty:
        st.info("Ainda não registaste nenhuma aposta.")
    else:
        # Calcular CLV para apostas que tenham odd de fecho
        df_history['clv_pct'] = np.where(
            df_history['closing_odd'] > 0, 
            (df_history['odd'] / df_history['closing_odd']) - 1.0, 
            0.0
        )

        df_resolvidas = df_history[df_history["status"] != "Pendente"]
        total_apostas = len(df_resolvidas)
        lucro_total = df_resolvidas["lucro_real"].sum() * 100 
        volume_investido = df_resolvidas["stake_pct"].sum() * 100
        roi = (lucro_total / volume_investido) * 100 if volume_investido > 0 else 0.0
        
        # Média de CLV (apenas para apostas onde registaste o fecho)
        df_clv_validos = df_history[df_history['closing_odd'] > 0]
        clv_medio = df_clv_validos['clv_pct'].mean() * 100 if not df_clv_validos.empty else 0.0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume Investido", f"{volume_investido:.2f} U")
        cor_lucro = "normal" if lucro_total >= 0 else "inverse"
        c2.metric("Lucro Líquido", f"{lucro_total:.2f} U", delta_color=cor_lucro)
        c3.metric("ROI / Yield Real", f"{roi:.2f}%", delta_color=cor_lucro)
        cor_clv = "normal" if clv_medio >= 0 else "inverse"
        c4.metric("Avg. Closing Line Value", f"{clv_medio:.2f}%", delta_color=cor_clv, help="Bater a odd de fecho é a verdadeira prova matemática do teu modelo.")
        
        st.markdown("---")
        st.subheader("⏳ Resolver Apostas Pendentes")
        df_pendentes = df_history[df_history["status"] == "Pendente"]
        
        if not df_pendentes.empty:
            for _, row in df_pendentes.iterrows():
                with st.container():
                    c_info, c_input, c_win, c_loss = st.columns([4, 2, 1, 1])
                    c_info.markdown(f"**{row['encontro']}** | {row['mercado']}<br>Odd Comprada: `{row['odd']}` | Stake: `{row['stake_pct']:.2%}`", unsafe_allow_html=True)
                    
                    # Input para a Odd de Fecho
                    closing_odd_input = c_input.number_input(
                        "Odd Fecho (Opcional)", 
                        min_value=1.0, value=float(row['odd']), step=0.01, key=f"close_{row['id']}"
                    )
                    
                    if c_win.button("✅ Win", key=f"win_{row['id']}"):
                        resolve_bet(row['id'], "Ganha", row['odd'], row['stake_pct'], closing_odd_input)
                        st.rerun()
                    if c_loss.button("❌ Loss", key=f"loss_{row['id']}"):
                        resolve_bet(row['id'], "Perdida", row['odd'], row['stake_pct'], closing_odd_input)
                        st.rerun()
                st.write("") # espaçamento
        else:
            st.success("Não tens apostas pendentes!")
            
        st.markdown("---")
        st.subheader("📖 Histórico Completo")
        
        df_view = df_history.copy()
        df_view["stake_pct"] = (df_view["stake_pct"] * 100).map("{:.2f}%".format)
        df_view["ev_projetado"] = (df_view["ev_projetado"] * 100).map("+{:.1f}%".format)
        df_view["lucro_real"] = df_view["lucro_real"].map("{:.3f} U".format)
        df_view["clv_pct"] = (df_view["clv_pct"] * 100).map("{:+.2f}%".format)
        
        st.dataframe(
            df_view[["id", "encontro", "mercado", "odd", "closing_odd", "clv_pct", "stake_pct", "status", "lucro_real"]],
            use_container_width=True,
            hide_index=True
        )
        st.markdown("---")
        st.subheader("🔮 Teste de Stress ao Portfólio (Risco de Ruína)")
        
        if total_apostas >= 20: # Só corre se já tiveres alguma amostra real
            win_rate_real = len(df_resolvidas[df_resolvidas['status'] == 'Ganha']) / total_apostas
            odd_media = df_resolvidas['odd'].mean()
            stake_media = df_resolvidas['stake_pct'].mean()
            
            with st.spinner("A correr 1000 simulações do teu futuro financeiro..."):
                prob_ruina, mediana_final, trajetorias = simulate_bankroll_ruin(
                    win_rate=win_rate_real, avg_odd=odd_media, avg_stake=stake_media
                )
            
            cr1, cr2 = st.columns(2)
            cr1.metric("Probabilidade de Ruína (Falir)", f"{prob_ruina:.1%}", delta_color="inverse" if prob_ruina > 0.05 else "normal")
            cr2.metric("Banca Mediana após 500 Apostas", f"{mediana_final:.1f} U (Inicial: 100 U)")
            
            fig_ruin, ax_ruin = plt.subplots(figsize=(10, 4))
            for traj in trajetorias:
                ax_ruin.plot(traj, color='gray', alpha=0.1)
            ax_ruin.plot(trajetorias[0], color='blue', alpha=0.5, label='Exemplo de Caminho')
            ax_ruin.axhline(100, color='red', linestyle='--')
            ax_ruin.set_title("50 Simulações de Bankroll (Próximas 500 Apostas)")
            ax_ruin.set_ylabel("Unidades de Banca")
            ax_ruin.set_xlabel("Número de Apostas")
            st.pyplot(fig_ruin)
        else:
            st.info("⚠️ Regista pelo menos 20 apostas resolvidas para desbloquear a previsão de Risco de Ruína por Monte Carlo.")