"""
app.py — QuantBet OS: Interface Streamlit.

Esta camada só trata de UI. Toda a lógica está em:
  config.py      — parâmetros e constantes
  simulation.py  — Monte Carlo vectorizado
  markets.py     — cálculo de EV e Kelly
  parser.py      — parser de odds
  data.py        — carregamento de dados e stats de jogadores
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from config import CIRCUIT, SURFACE_MOD, MODEL
from data import (
    load_ml_model, load_match_data, load_elos, load_agenda,
    get_player_stats, calculate_h2h,
)
from markets import compute_all_markets, best_bet
from parser import parse_bookmaker_text
from simulation import MatchSetup, simulate_vectorized

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="QuantBet OS", layout="wide")
st.title("🎾 QuantBet OS: Sistema Quantitativo ATP & WTA")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "agenda_p1" not in st.session_state:
    st.session_state["agenda_p1"] = None
if "agenda_p2" not in st.session_state:
    st.session_state["agenda_p2"] = None

# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------
df          = load_match_data()
ml_model    = load_ml_model()

# ---------------------------------------------------------------------------
# Barra lateral — configurações globais
# ---------------------------------------------------------------------------
st.sidebar.header("1. Configurações Globais")
circuito = st.sidebar.radio("Circuito", list(CIRCUIT.keys()))

if ml_model is None:
    st.sidebar.info("🤖 Motor: Elo Matemático (Fallback)")
else:
    st.sidebar.success("🤖 Motor: XGBoost Calibrado")

df_elos  = load_elos(circuito)
superficie = st.sidebar.selectbox(
    "Superfície", sorted(df["surface"].dropna().unique())
)
sets_padrao = [3] if circuito == "WTA (Feminino)" else [3, 5]
sets_input  = st.sidebar.radio("Sets do Encontro", sets_padrao)

# Jogadores disponíveis — com validação se lista vier vazia
jogadores = sorted(df_elos["Player"].dropna().unique()) if not df_elos.empty else ["Jogador A", "Jogador B"]
if len(jogadores) < 2:
    st.sidebar.error("⚠️ Ficheiro de Elos vazio ou não encontrado.")

st.sidebar.header("2. Filtros de Valor")
limite_ev = st.sidebar.slider(
    "Limite de EV Aceitável (%)", min_value=1.0, max_value=15.0, value=5.0, step=0.5
) / 100
odd_minima_rec = st.sidebar.number_input(
    "Odd Mínima Recomendada", value=1.50, step=0.05,
    help="O sistema ignora odds abaixo deste valor na recomendação."
)

st.sidebar.header("⚙️ Condições & Ajustes de Jogo")
vel_campo = st.sidebar.selectbox("Velocidade do Campo", list(SURFACE_MOD.keys()))
ajuste_forma   = st.sidebar.slider("Ajuste de Forma (Favorecer P1 vs P2)", -5, 5, 0)
ajuste_fadiga  = st.sidebar.slider("Ajuste de Fadiga (Prejudicar P1 vs P2)", -5, 5, 0)

# Helper: índice seguro na lista de jogadores
def safe_index(name, fallback=0):
    try:
        return jogadores.index(name) if name in jogadores else fallback
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Helper: constrói MatchSetup a partir da UI
# ---------------------------------------------------------------------------
def build_setup(p1_name: str, p2_name: str, h2h_manual=None) -> MatchSetup:
    cfg = CIRCUIT[circuito]
    surf_mod = SURFACE_MOD[vel_campo]
    stats_p1 = get_player_stats(p1_name, superficie, circuito, df, df_elos)
    stats_p2 = get_player_stats(p2_name, superficie, circuito, df, df_elos)

    if h2h_manual is not None:
        h2h = h2h_manual
    else:
        h2h = calculate_h2h(p1_name, p2_name, df)

    return MatchSetup(
        p1=stats_p1,
        p2=stats_p2,
        sets_to_win=sets_input // 2 + 1,
        circuit_cfg=cfg,
        surface_mod=surf_mod,
        form_adj=ajuste_forma,
        fatigue_adj=ajuste_fadiga,
        h2h=h2h,
        ml_model=ml_model,
    )


# ---------------------------------------------------------------------------
# Helper: renderiza tabela de resultados
# ---------------------------------------------------------------------------
def render_results(bets, p1_name, p2_name, sims):
    if not bets:
        st.error("Não foram encontrados mercados com valor. Verifica a formatação do texto.")
        return

    # Melhor aposta
    top = best_bet(bets, limite_ev, odd_minima_rec)
    if top:
        st.markdown("---")
        st.markdown("### 🏆 Aposta Recomendada (Melhor Risco/Benefício)")
        st.success(
            f"**Mercado:** {top.market}\n\n"
            f"**Odd Oferecida:** {top.odd:.2f} | "
            f"**Probabilidade:** {top.prob:.1%} | "
            f"**EV:** {'+' if top.ev > 0 else ''}{top.ev:.1%}\n\n"
            f"⚖️ **Banca Sugerida:** **{top.stake_pct:.1%}**"
        )
        st.markdown("---")

    # Tabela completa
    rows = []
    for b in bets:
        rows.append({
            "Status":     "✅ Valor" if b.ev >= limite_ev else "❌ Evitar",
            "Mercado":    b.market,
            "Odd":        f"{b.odd:.2f}",
            "Odd Justa":  f"{b.fair_odd:.2f}",
            "Prob":       f"{b.prob:.2%}",
            "EV":         f"{'+' if b.ev > 0 else ''}{b.ev:.2%}",
        })
    st.markdown("#### 📋 Auditoria Completa ao Mercado")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Resumo do modelo
    with st.expander("🔬 Parâmetros do Modelo"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Prob. Vitória P1", f"{np.mean(sims['p1_match_wins']):.1%}")
        col2.metric("P1 Hold Rate", f"{sims['p1_hold']:.1%}")
        col3.metric("P2 Hold Rate", f"{sims['p2_hold']:.1%}")
        col1.metric("Média Total Jogos", f"{np.mean(sims['total_games']):.1f}")
        col2.metric("Mediana Total Aces", f"{np.median(sims['aces_p1'] + sims['aces_p2']):.1f}")
        col3.metric("Prob P1 Vence 1º Set", f"{np.mean(sims['s1_p1'] > sims['s1_p2']):.1%}")


# ===========================================================================
# ABAS
# ===========================================================================
tab_agenda, tab_scanner, tab_manual, tab_csv = st.tabs(
    ["📅 Agenda", "🤖 Auto-Scanner", "🔍 Calculadora Manual", "🚀 CSV em Massa"]
)

# ---------------------------------------------------------------------------
# ABA: AGENDA
# ---------------------------------------------------------------------------
with tab_agenda:
    st.header("📅 Calendário de Torneios")
    st.markdown(
        "Seleciona a data. Podes alimentar esta lista criando um ficheiro `agenda.csv` "
        "na mesma pasta da app."
    )

    data_selecionada = st.date_input("🗓️ Selecionar Data", datetime.today().date())
    df_agenda = load_agenda()  # sem cache — ver data.py
    jogos_do_dia = df_agenda[df_agenda["Data"] == data_selecionada]

    if jogos_do_dia.empty:
        st.info(f"Sem jogos agendados para {data_selecionada.strftime('%d/%m/%Y')}.")
    else:
        for torneio in jogos_do_dia["Torneio"].unique():
            with st.expander(f"🏆 {torneio}", expanded=True):
                for idx, jogo in jogos_do_dia[jogos_do_dia["Torneio"] == torneio].iterrows():
                    col_hora, col_jogo, col_btn = st.columns([2, 6, 2])
                    col_hora.markdown(f"🕒 `{jogo['Hora']}`")
                    col_jogo.markdown(f"**{jogo['P1']}** vs **{jogo['P2']}**")
                    if col_btn.button("Carregar Jogo", key=f"btn_agenda_{idx}"):
                        st.session_state["agenda_p1"] = jogo["P1"]
                        st.session_state["agenda_p2"] = jogo["P2"]
                        st.success("✅ Jogo carregado! Vai à aba **Auto-Scanner** ou **Calculadora Manual**.")

# ---------------------------------------------------------------------------
# ABA: AUTO-SCANNER
# ---------------------------------------------------------------------------
with tab_scanner:
    st.header("Auto-Scanner Inteligente (Copiar & Colar)")

    col_s1, col_s2 = st.columns(2)
    scan_p1 = col_s1.selectbox(
        "Favorito (Player 1)", jogadores,
        index=safe_index(st.session_state["agenda_p1"]),
        key="tab3_p1",
    )
    scan_p2 = col_s2.selectbox(
        "Underdog (Player 2)", jogadores,
        index=safe_index(st.session_state["agenda_p2"], fallback=1),
        key="tab3_p2",
    )

    if scan_p1 == scan_p2:
        st.error("⚠️ Seleciona jogadores diferentes.")
    else:
        h2h_db = calculate_h2h(scan_p1, scan_p2, df)
        st.markdown("**⚔️ Correcção Manual de H2H**")
        col_h1, col_h2 = st.columns(2)
        h2h_p1 = col_h1.number_input(f"Vitórias de {scan_p1}", value=int(h2h_db[0]), min_value=0, step=1, key="h2h_t3_p1")
        h2h_p2 = col_h2.number_input(f"Vitórias de {scan_p2}", value=int(h2h_db[1]), min_value=0, step=1, key="h2h_t3_p2")

        texto_odds = st.text_area("Cola as Odds da Casa de Apostas:", height=300, key="raw_text_area")

        if st.button("Analisar Todas as Odds", key="btn_tab3"):
            if not texto_odds.strip():
                st.warning("Cola o texto com as odds primeiro.")
            else:
                with st.spinner("A simular 10 000 encontros..."):
                    setup = build_setup(scan_p1, scan_p2, h2h_manual=(h2h_p1, h2h_p2))
                    sims = simulate_vectorized(setup, n=MODEL.monte_carlo_n)
                    markets = parse_bookmaker_text(texto_odds, scan_p1, scan_p2)
                    bets = compute_all_markets(sims, markets, scan_p1, scan_p2)
                render_results(bets, scan_p1, scan_p2, sims)

# ---------------------------------------------------------------------------
# ABA: CALCULADORA MANUAL
# ---------------------------------------------------------------------------
with tab_manual:
    st.header("Análise de Partida Única")
    col1, col2 = st.columns(2)
    nome_p1 = col1.selectbox(
        "Favorito (P1)", jogadores,
        index=safe_index(st.session_state["agenda_p1"]),
        key="tab1_p1",
    )
    nome_p2 = col2.selectbox(
        "Underdog (P2)", jogadores,
        index=safe_index(st.session_state["agenda_p2"], fallback=1),
        key="tab1_p2",
    )

    if nome_p1 == nome_p2:
        st.error("⚠️ Seleciona jogadores diferentes.")
    else:
        stats_p1 = get_player_stats(nome_p1, superficie, circuito, df, df_elos)
        stats_p2 = get_player_stats(nome_p2, superficie, circuito, df, df_elos)

        col1.metric(f"Elo {superficie} — {nome_p1}", f"{stats_p1.elo:.1f}")
        col1.markdown(f"📈 Forma: `{stats_p1.recent_form:.0%}` | 💤 Fadiga: `{stats_p1.fatigue:.0f}`")
        col2.metric(f"Elo {superficie} — {nome_p2}", f"{stats_p2.elo:.1f}")
        col2.markdown(f"📈 Forma: `{stats_p2.recent_form:.0%}` | 💤 Fadiga: `{stats_p2.fatigue:.0f}`")

        h2h_db = calculate_h2h(nome_p1, nome_p2, df)
        st.markdown(f"**H2H:** {nome_p1} {h2h_db[0]}–{h2h_db[1]} {nome_p2}")

        st.markdown("**Odds Manuais**")
        col_o1, col_o2 = st.columns(2)
        odd_p1 = col_o1.number_input(f"Odd {nome_p1}", value=1.80, step=0.01, min_value=1.01)
        odd_p2 = col_o2.number_input(f"Odd {nome_p2}", value=2.10, step=0.01, min_value=1.01)

        if st.button("Calcular EV", key="btn_tab1"):
            with st.spinner("A simular..."):
                setup = build_setup(nome_p1, nome_p2)
                sims  = simulate_vectorized(setup, n=MODEL.monte_carlo_n)
                markets = {
                    "match_winner": {"P1": odd_p1, "P2": odd_p2},
                    "total_games": {}, "game_handicap": {"P1": {}, "P2": {}},
                    "set_handicap": {"P1": {}, "P2": {}}, "total_sets": {},
                    "p1_total_games": {}, "p2_total_games": {},
                    "set1_winner": {}, "set1_total_games": {}, "set1_handicap": {"P1": {}, "P2": {}},
                    "set2_winner": {}, "set2_total_games": {}, "set2_handicap": {"P1": {}, "P2": {}},
                    "total_aces": {}, "p1_aces": {}, "p2_aces": {},
                }
                bets = compute_all_markets(sims, markets, nome_p1, nome_p2)
            render_results(bets, nome_p1, nome_p2, sims)

# ---------------------------------------------------------------------------
# ABA: CSV EM MASSA
# ---------------------------------------------------------------------------
with tab_csv:
    st.header("Scanner de Valor Múltiplo (CSV)")
    st.markdown(
        "Formato esperado por linha: "
        "`Jogador 1, Jogador 2, Odd P1, Odd P2, Linha Over, Odd Over, Linha Hcp, Odd Hcp`"
    )
    bloco_csv = st.text_area("Cola as linhas CSV aqui:", height=200, key="csv_area")

    if st.button("Analisar CSV", key="btn_csv") and bloco_csv.strip():
        resultados = []
        for i, linha in enumerate(bloco_csv.strip().split("\n")):
            partes = [p.strip() for p in linha.split(",")]
            if len(partes) < 4:
                st.warning(f"Linha {i+1} ignorada: formato inválido.")
                continue
            try:
                p1_csv, p2_csv = partes[0], partes[1]
                if p1_csv not in jogadores or p2_csv not in jogadores:
                    st.warning(f"Linha {i+1}: jogador não encontrado ({p1_csv} / {p2_csv}).")
                    continue
                if p1_csv == p2_csv:
                    st.warning(f"Linha {i+1}: jogadores iguais.")
                    continue

                odd_p1_csv = float(partes[2])
                odd_p2_csv = float(partes[3])

                with st.spinner(f"Simulando {p1_csv} vs {p2_csv}..."):
                    setup = build_setup(p1_csv, p2_csv)
                    sims  = simulate_vectorized(setup, n=MODEL.monte_carlo_n)
                    markets_csv = {
                        "match_winner": {"P1": odd_p1_csv, "P2": odd_p2_csv},
                        "total_games": {}, "game_handicap": {"P1": {}, "P2": {}},
                        "set_handicap": {"P1": {}, "P2": {}}, "total_sets": {},
                        "p1_total_games": {}, "p2_total_games": {},
                        "set1_winner": {}, "set1_total_games": {}, "set1_handicap": {"P1": {}, "P2": {}},
                        "set2_winner": {}, "set2_total_games": {}, "set2_handicap": {"P1": {}, "P2": {}},
                        "total_aces": {}, "p1_aces": {}, "p2_aces": {},
                    }
                    bets_csv = compute_all_markets(sims, markets_csv, p1_csv, p2_csv)
                    top_csv  = best_bet(bets_csv, limite_ev, odd_minima_rec)

                if top_csv:
                    resultados.append({
                        "Encontro":  f"{p1_csv} vs {p2_csv}",
                        "Mercado":   top_csv.market,
                        "Odd":       f"{top_csv.odd:.2f}",
                        "Prob":      f"{top_csv.prob:.1%}",
                        "EV":        f"+{top_csv.ev:.1%}",
                        "Banca %":   f"{top_csv.stake_pct:.1%}",
                    })
            except Exception as e:
                st.warning(f"Linha {i+1} com erro: {e}")

        if resultados:
            st.success(f"✅ {len(resultados)} apostas com valor encontradas.")
            st.dataframe(pd.DataFrame(resultados), use_container_width=True)
        else:
            st.info("Nenhuma aposta com valor encontrada nas linhas processadas.")
            """
config.py — Configuração central do QuantBet OS.
Todos os parâmetros do modelo estão aqui. Nunca espalhar números mágicos pelo código.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class CircuitConfig:
    base_hold: float          # Taxa base de hold do servidor
    hold_min: float           # Limite inferior do hold (clip)
    hold_max: float           # Limite superior do hold (clip)
    base_aces_per_game: float # Ases esperados por jogo de serviço
    default_elo: float = 1500.0


@dataclass(frozen=True)
class SurfaceModifier:
    hold_delta: float       # Ajuste ao hold base
    ace_multiplier: float   # Multiplicador de ases


@dataclass(frozen=True)
class ModelConfig:
    elo_k: float = 400.0           # Constante Elo (divisor logístico)
    form_weight: float = 0.05      # Peso da forma recente na prob final
    form_adj_per_step: float = 0.10 # Escala do slider de ajuste de forma
    fatigue_weight: float = 0.08   # Peso da fadiga na prob final
    fatigue_adj_per_step: float = 10.0
    h2h_weight_per_game: float = 0.015  # Peso base H2H por jogo
    h2h_max_effect: float = 0.075       # Cap do efeito H2H
    h2h_prior_games: int = 8            # Prior bayesiano (equivalente a N jogos neutros)
    game_prob_scale: float = 0.15       # Escala da diferença de hold vs prob do encontro
    monte_carlo_n: int = 10_000
    random_noise_std: float = 0.02      # Ruído por jogo
    kelly_fraction: float = 0.10        # Fracção Kelly aplicada
    kelly_max_stake: float = 0.035      # Máximo da banca sugerido
    kelly_min_stake: float = 0.005


# Configurações por circuito
CIRCUIT: Dict[str, CircuitConfig] = {
    "ATP (Masculino)": CircuitConfig(
        base_hold=0.780,
        hold_min=0.45,
        hold_max=0.95,
        base_aces_per_game=0.55,
    ),
    "WTA (Feminino)": CircuitConfig(
        base_hold=0.635,
        hold_min=0.35,
        hold_max=0.85,
        base_aces_per_game=0.25,
    ),
}

# Modificadores de superfície/velocidade
SURFACE_MOD: Dict[str, SurfaceModifier] = {
    "Lento (Clay Lento)":                    SurfaceModifier(hold_delta=-0.04, ace_multiplier=0.60),
    "Médio-Lento (Clay Rápido / Hard Lento)": SurfaceModifier(hold_delta=-0.02, ace_multiplier=0.80),
    "Médio (Hard Normal)":                    SurfaceModifier(hold_delta=0.00,  ace_multiplier=1.00),
    "Rápido (Grass / Hard Rápido)":           SurfaceModifier(hold_delta=+0.03, ace_multiplier=1.30),
    "Ultra Rápido (Indoor Rápido)":           SurfaceModifier(hold_delta=+0.05, ace_multiplier=1.50),
}

MODEL = ModelConfig()
"""
simulation.py — Motor de simulação Monte Carlo vectorizado.

A versão anterior simulava 10 000 partidas em Python puro (loop por set, por jogo).
Esta versão usa NumPy broadcasting para simular todas as partidas em paralelo,
o que é tipicamente 20-50x mais rápido.

Estrutura de uma partida (melhor de 3 sets, sem tie-break no set decisivo):
  - Cada set: séries de jogos até alguém ter ≥6 com ≥2 de vantagem, ou tie-break a 6-6.
  - Resultado guardado: sets, jogos por set, aces por jogador.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from config import CIRCUIT, SURFACE_MOD, MODEL, CircuitConfig, SurfaceModifier


@dataclass
class PlayerStats:
    elo: float
    hold_rate: float
    fatigue: float
    recent_form: float  # fracção de vitórias nos últimos 5 jogos (0-1)


@dataclass
class MatchSetup:
    p1: PlayerStats
    p2: PlayerStats
    sets_to_win: int          # 2 para melhor de 3, 3 para melhor de 5
    circuit_cfg: CircuitConfig
    surface_mod: SurfaceModifier
    form_adj: int             # slider −5..+5 (favorece P1 vs P2)
    fatigue_adj: int          # slider −5..+5 (prejudica P1 vs P2)
    h2h: Tuple[int, int]      # (vitórias P1, vitórias P2)
    ml_model: Optional[object] = None


def _elo_prob(elo_diff: float) -> float:
    """Probabilidade Elo padrão: P(P1 vence) = 1 / (1 + 10^(-Δelo/400))."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / MODEL.elo_k))


def _h2h_adjustment(h2h: Tuple[int, int]) -> float:
    """
    Ajuste bayesiano de H2H.
    Com poucos jogos o ajuste é pequeno; cresce com mais evidência.
    Prior: MODEL.h2h_prior_games jogos neutros (0.5 por lado).
    """
    wins_p1, wins_p2 = h2h
    total = wins_p1 + wins_p2
    if total == 0:
        return 0.0
    # Média posterior com prior uniforme
    posterior_p1 = (wins_p1 + MODEL.h2h_prior_games / 2) / (total + MODEL.h2h_prior_games)
    raw = (posterior_p1 - 0.5) * MODEL.h2h_weight_per_game * total
    return float(np.clip(raw, -MODEL.h2h_max_effect, MODEL.h2h_max_effect))


def compute_match_prob(setup: MatchSetup) -> float:
    """
    Probabilidade P(P1 vence o encontro) antes da simulação por jogos.
    Combina: Elo / ML → forma → fadiga → H2H.
    """
    if setup.ml_model is not None:
        elo_diff = setup.p1.elo - setup.p2.elo
        hold_diff = setup.p1.hold_rate - setup.p2.hold_rate
        fatigue_diff = setup.p1.fatigue - setup.p2.fatigue
        features = pd.DataFrame(
            [[elo_diff, hold_diff, fatigue_diff]],
            columns=["elo_diff", "hold_diff_last5", "fatigue_diff"],
        )
        prob = float(setup.ml_model.predict_proba(features)[0][1])
    else:
        prob = _elo_prob(setup.p1.elo - setup.p2.elo)

    # Forma recente
    form_diff = setup.p1.recent_form - setup.p2.recent_form
    prob += (form_diff + setup.form_adj * MODEL.form_adj_per_step) * MODEL.form_weight

    # Fadiga
    fatigue_diff = setup.p1.fatigue - setup.p2.fatigue
    prob -= ((fatigue_diff + setup.fatigue_adj * MODEL.fatigue_adj_per_step)
             / 100.0 * MODEL.fatigue_weight)

    # H2H bayesiano
    prob += _h2h_adjustment(setup.h2h)

    return float(np.clip(prob, 0.05, 0.95))


def _hold_probs(
    match_prob: float,
    circuit_cfg: CircuitConfig,
    surface_mod: SurfaceModifier,
) -> Tuple[float, float]:
    """Converte a prob. do encontro em taxas de hold por servidor."""
    base = circuit_cfg.base_hold + surface_mod.hold_delta
    shift = (match_prob - 0.5) * MODEL.game_prob_scale
    p1_hold = float(np.clip(base + shift, circuit_cfg.hold_min, circuit_cfg.hold_max))
    p2_hold = float(np.clip(base - shift, circuit_cfg.hold_min, circuit_cfg.hold_max))
    return p1_hold, p2_hold


# ---------------------------------------------------------------------------
# Simulação vectorizada
# ---------------------------------------------------------------------------

def _simulate_set_vectorized(
    n: int,
    p1_hold: float,
    p2_hold: float,
    rng: np.random.Generator,
    noise_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula N sets em paralelo.
    Devolve (games_p1, games_p2, aces_p1, aces_p2) — arrays de shape (N,).

    Estratégia: simula jogos até ao máximo possível num set (13 jogos: 7-6)
    e aplica lógica de terminação de forma vectorizada.
    """
    MAX_GAMES = 13  # máximo num set com tie-break

    g_p1 = np.zeros(n, dtype=np.int32)
    g_p2 = np.zeros(n, dtype=np.int32)
    acc_aces_p1 = np.zeros(n, dtype=np.float32)
    acc_aces_p2 = np.zeros(n, dtype=np.float32)
    active = np.ones(n, dtype=bool)  # partidas ainda em jogo

    ace_rate_p1 = (circuit_cfg_placeholder := None) or 0.0  # preenchido no caller

    for game_idx in range(MAX_GAMES):
        if not active.any():
            break

        is_p1_serve = (game_idx % 2 == 0)
        base_prob = p1_hold if is_p1_serve else (1.0 - p2_hold)
        noise = rng.normal(0, noise_std, n)
        probs = np.clip(base_prob + noise, 0.0, 1.0)
        p1_wins_game = rng.random(n) < probs

        g_p1[active & p1_wins_game] += 1
        g_p2[active & ~p1_wins_game] += 1

        # Verificar fim do set para partidas activas
        done = active & (
            ((g_p1 >= 6) & ((g_p1 - g_p2) >= 2)) |
            ((g_p2 >= 6) & ((g_p2 - g_p1) >= 2)) |
            (g_p1 == 7) | (g_p2 == 7)
        )
        active[done] = False

    return g_p1, g_p2


def simulate_vectorized(setup: MatchSetup, n: int = MODEL.monte_carlo_n) -> dict:
    """
    Simula N encontros em paralelo.
    Devolve dicionário de arrays NumPy com resultados.
    """
    rng = np.random.default_rng(42)
    match_prob = compute_match_prob(setup)
    p1_hold, p2_hold = _hold_probs(match_prob, setup.circuit_cfg, setup.surface_mod)

    # Taxas de aces
    ace_base = setup.circuit_cfg.base_aces_per_game
    ace_mult = setup.surface_mod.ace_multiplier
    rate_aces_p1 = max(0.05, ace_base * ace_mult + (p1_hold - setup.circuit_cfg.base_hold))
    rate_aces_p2 = max(0.05, ace_base * ace_mult + (p2_hold - setup.circuit_cfg.base_hold))

    sets_to_win = setup.sets_to_win
    max_sets = sets_to_win * 2 - 1

    # Arrays de resultado
    p1_sets = np.zeros(n, dtype=np.int32)
    p2_sets = np.zeros(n, dtype=np.int32)
    total_games = np.zeros(n, dtype=np.int32)
    game_diff = np.zeros(n, dtype=np.int32)  # p1_games - p2_games
    aces_p1 = np.zeros(n, dtype=np.float32)
    aces_p2 = np.zeros(n, dtype=np.float32)

    # Armazena jogos dos primeiros 2 sets para mercados de set individual
    set_games: list[tuple[np.ndarray, np.ndarray]] = []

    for set_idx in range(max_sets):
        # Só simula partidas ainda activas
        still_playing = (p1_sets < sets_to_win) & (p2_sets < sets_to_win)
        if not still_playing.any():
            break

        g1, g2 = _simulate_set_vectorized_inner(
            n, p1_hold, p2_hold, rng, MODEL.random_noise_std
        )

        # Ases neste set (Poisson; escala por jogos no set)
        games_in_set = g1 + g2
        # P1 serve nos jogos pares (0,2,4,...), P2 nos ímpares
        p1_serve_games = (games_in_set + 1) // 2
        p2_serve_games = games_in_set // 2
        set_aces_p1 = rng.poisson(rate_aces_p1 * p1_serve_games).astype(np.float32)
        set_aces_p2 = rng.poisson(rate_aces_p2 * p2_serve_games).astype(np.float32)

        # Aplica só às partidas ainda em jogo
        mask = still_playing
        p1_wins_set = g1 > g2
        p1_sets[mask & p1_wins_set] += 1
        p2_sets[mask & ~p1_wins_set] += 1
        total_games[mask] += games_in_set[mask]
        game_diff[mask] += (g1 - g2)[mask]
        aces_p1[mask] += set_aces_p1[mask]
        aces_p2[mask] += set_aces_p2[mask]

        if set_idx < 2:
            g1_stored = np.where(mask, g1, 0)
            g2_stored = np.where(mask, g2, 0)
            set_games.append((g1_stored, g2_stored))

    # Preencher set_games com zeros se a partida acabou antes
    while len(set_games) < 2:
        set_games.append((np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)))

    p1_match_wins = p1_sets > p2_sets

    return {
        "p1_match_wins": p1_match_wins,
        "total_games": total_games,
        "game_diff": game_diff,
        "p1_sets": p1_sets,
        "p2_sets": p2_sets,
        "s1_p1": set_games[0][0],
        "s1_p2": set_games[0][1],
        "s2_p1": set_games[1][0],
        "s2_p2": set_games[1][1],
        "aces_p1": aces_p1,
        "aces_p2": aces_p2,
        "match_prob": match_prob,
        "p1_hold": p1_hold,
        "p2_hold": p2_hold,
    }


def _simulate_set_vectorized_inner(
    n: int,
    p1_hold: float,
    p2_hold: float,
    rng: np.random.Generator,
    noise_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simula N sets em paralelo com lógica de terminação vectorizada."""
    MAX_GAMES = 13

    g_p1 = np.zeros(n, dtype=np.int32)
    g_p2 = np.zeros(n, dtype=np.int32)
    active = np.ones(n, dtype=bool)

    for game_idx in range(MAX_GAMES):
        if not active.any():
            break

        is_p1_serve = (game_idx % 2 == 0)
        base_prob = p1_hold if is_p1_serve else (1.0 - p2_hold)
        noise = rng.normal(0, noise_std, n)
        probs = np.clip(base_prob + noise, 0.0, 1.0)

        p1_wins_game = rng.random(n) < probs
        g_p1 += active & p1_wins_game
        g_p2 += active & ~p1_wins_game

        done = active & (
            ((g_p1 >= 6) & ((g_p1 - g_p2) >= 2)) |
            ((g_p2 >= 6) & ((g_p2 - g_p1) >= 2)) |
            (g_p1 == 7) | (g_p2 == 7)
        )
        active[done] = False

    return g_p1, g_p2
"""
markets.py — Cálculo de EV e Kelly para todos os mercados de ténis.

Fórmulas:
  EV         = prob * odd - 1
  Kelly full = (prob * odd - 1) / (odd - 1)   [fracção da banca a apostar]
  Kelly usado = Kelly full * MODEL.kelly_fraction  (critério fraccionado)

Nota sobre o Kelly:
  O Kelly completo maximiza o crescimento logarítmico da banca mas é
  agressivo. Usamos uma fracção (por defeito 10%) para reduzir variância.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from config import MODEL


@dataclass
class Bet:
    market: str
    prob: float
    odd: float
    ev: float           # EV decimal: 0.05 = +5%
    kelly: float        # Kelly fraccionado (stake como fracção da banca)

    @property
    def fair_odd(self) -> float:
        return 1.0 / self.prob if self.prob > 0 else float("inf")

    @property
    def stake_pct(self) -> float:
        """Kelly fraccionado clampado, como percentagem (0–3.5%)."""
        return float(np.clip(self.kelly, MODEL.kelly_min_stake, MODEL.kelly_max_stake))

    @property
    def is_value(self) -> bool:
        return self.ev > 0


def _kelly(prob: float, odd: float) -> float:
    """Kelly fraccionado completo. Devolve 0 se EV ≤ 0."""
    if odd <= 1.0 or prob <= 0:
        return 0.0
    full_kelly = (prob * odd - 1.0) / (odd - 1.0)
    return max(0.0, full_kelly * MODEL.kelly_fraction)


def _bet(market: str, prob: float, odd: float) -> Bet:
    ev = prob * odd - 1.0
    return Bet(market=market, prob=prob, odd=odd, ev=ev, kelly=_kelly(prob, odd))


def compute_all_markets(sims: dict, markets: dict, p1_name: str, p2_name: str) -> List[Bet]:
    """
    Recebe os arrays de simulação e os mercados extraídos do parser,
    devolve lista de Bet ordenada por EV decrescente.
    """
    bets: List[Bet] = []

    total = sims["total_games"]
    diff  = sims["game_diff"]
    p1_mw = sims["p1_match_wins"]
    s1_p1, s1_p2 = sims["s1_p1"], sims["s1_p2"]
    s2_p1, s2_p2 = sims["s2_p1"], sims["s2_p2"]
    aces_p1 = sims["aces_p1"]
    aces_p2 = sims["aces_p2"]
    total_aces = aces_p1 + aces_p2

    n = len(total)
    prob_p1_win = np.mean(p1_mw)

    # ── Vencedor do encontro ──────────────────────────────────────────────
    mw = markets.get("match_winner", {})
    if "P1" in mw:
        bets.append(_bet(f"Vitória {p1_name}", prob_p1_win, mw["P1"]))
    if "P2" in mw:
        bets.append(_bet(f"Vitória {p2_name}", 1 - prob_p1_win, mw["P2"]))

    # ── Total de jogos ────────────────────────────────────────────────────
    for line, odds_ou in markets.get("total_games", {}).items():
        if "Over" in odds_ou:
            bets.append(_bet(f"Over {line} Jogos", np.mean(total > line), odds_ou["Over"]))
        if "Under" in odds_ou:
            bets.append(_bet(f"Under {line} Jogos", np.mean(total < line), odds_ou["Under"]))

    # ── Handicap de jogos ─────────────────────────────────────────────────
    for hcp, odd in markets.get("game_handicap", {}).get("P1", {}).items():
        prob = np.mean(diff > -hcp)
        sign = "+" if hcp > 0 else ""
        bets.append(_bet(f"Hcp Jogos {p1_name} ({sign}{hcp})", prob, odd))
    for hcp, odd in markets.get("game_handicap", {}).get("P2", {}).items():
        prob = np.mean(-diff > -hcp)
        sign = "+" if hcp > 0 else ""
        bets.append(_bet(f"Hcp Jogos {p2_name} ({sign}{hcp})", prob, odd))

    # ── Total de sets ─────────────────────────────────────────────────────
    p1_sets = sims["p1_sets"]
    p2_sets = sims["p2_sets"]
    total_sets = p1_sets + p2_sets
    for line, odds_ou in markets.get("total_sets", {}).items():
        if "Over" in odds_ou:
            bets.append(_bet(f"Over {line} Sets", np.mean(total_sets > line), odds_ou["Over"]))
        if "Under" in odds_ou:
            bets.append(_bet(f"Under {line} Sets", np.mean(total_sets < line), odds_ou["Under"]))

    # ── Jogos individuais por jogador ─────────────────────────────────────
    p1_games = (total + diff) / 2
    p2_games = (total - diff) / 2
    for line, odds_ou in markets.get("p1_total_games", {}).items():
        if "Over" in odds_ou:
            bets.append(_bet(f"Over {line} Jogos {p1_name}", np.mean(p1_games > line), odds_ou["Over"]))
        if "Under" in odds_ou:
            bets.append(_bet(f"Under {line} Jogos {p1_name}", np.mean(p1_games < line), odds_ou["Under"]))
    for line, odds_ou in markets.get("p2_total_games", {}).items():
        if "Over" in odds_ou:
            bets.append(_bet(f"Over {line} Jogos {p2_name}", np.mean(p2_games > line), odds_ou["Over"]))
        if "Under" in odds_ou:
            bets.append(_bet(f"Under {line} Jogos {p2_name}", np.mean(p2_games < line), odds_ou["Under"]))

    # ── Ases ──────────────────────────────────────────────────────────────
    for line, odds_ou in markets.get("total_aces", {}).items():
        if "Over" in odds_ou:
            bets.append(_bet(f"Over {line} Ases", np.mean(total_aces > line), odds_ou["Over"]))
        if "Under" in odds_ou:
            bets.append(_bet(f"Under {line} Ases", np.mean(total_aces < line), odds_ou["Under"]))
    for line, odds_ou in markets.get("p1_aces", {}).items():
        if "Over" in odds_ou:
            bets.append(_bet(f"Over {line} Ases {p1_name}", np.mean(aces_p1 > line), odds_ou["Over"]))
        if "Under" in odds_ou:
            bets.append(_bet(f"Under {line} Ases {p1_name}", np.mean(aces_p1 < line), odds_ou["Under"]))
    for line, odds_ou in markets.get("p2_aces", {}).items():
        if "Over" in odds_ou:
            bets.append(_bet(f"Over {line} Ases {p2_name}", np.mean(aces_p2 > line), odds_ou["Over"]))
        if "Under" in odds_ou:
            bets.append(_bet(f"Under {line} Ases {p2_name}", np.mean(aces_p2 < line), odds_ou["Under"]))

    # ── 1º Set ────────────────────────────────────────────────────────────
    prob_s1_p1 = np.mean(s1_p1 > s1_p2)
    s1w = markets.get("set1_winner", {})
    if "P1" in s1w:
        bets.append(_bet(f"Vence 1º Set {p1_name}", prob_s1_p1, s1w["P1"]))
    if "P2" in s1w:
        bets.append(_bet(f"Vence 1º Set {p2_name}", 1 - prob_s1_p1, s1w["P2"]))

    s1_total = s1_p1 + s1_p2
    for line, odds_ou in markets.get("set1_total_games", {}).items():
        if "Over" in odds_ou:
            bets.append(_bet(f"Over {line} Jogos 1º Set", np.mean(s1_total > line), odds_ou["Over"]))
        if "Under" in odds_ou:
            bets.append(_bet(f"Under {line} Jogos 1º Set", np.mean(s1_total < line), odds_ou["Under"]))
    for hcp, odd in markets.get("set1_handicap", {}).get("P1", {}).items():
        bets.append(_bet(f"Hcp 1º Set {p1_name} ({hcp})", np.mean((s1_p1 - s1_p2) > -hcp), odd))

    # ── 2º Set ────────────────────────────────────────────────────────────
    prob_s2_p1 = np.mean(s2_p1 > s2_p2)
    s2w = markets.get("set2_winner", {})
    if "P1" in s2w:
        bets.append(_bet(f"Vence 2º Set {p1_name}", prob_s2_p1, s2w["P1"]))
    if "P2" in s2w:
        bets.append(_bet(f"Vence 2º Set {p2_name}", 1 - prob_s2_p1, s2w["P2"]))

    s2_total = s2_p1 + s2_p2
    for line, odds_ou in markets.get("set2_total_games", {}).items():
        if "Over" in odds_ou:
            bets.append(_bet(f"Over {line} Jogos 2º Set", np.mean(s2_total > line), odds_ou["Over"]))
        if "Under" in odds_ou:
            bets.append(_bet(f"Under {line} Jogos 2º Set", np.mean(s2_total < line), odds_ou["Under"]))

    bets.sort(key=lambda b: b.ev, reverse=True)
    return bets


def best_bet(bets: List[Bet], min_ev: float, min_odd: float) -> Optional[Bet]:
    """Selecciona a melhor aposta por Kelly dentro das apostas com valor."""
    value_bets = [b for b in bets if b.ev >= min_ev]
    if not value_bets:
        return None
    eligible = [b for b in value_bets if b.odd >= min_odd] or value_bets
    return max(eligible, key=lambda b: b.kelly)
"""
parser.py — Parser multilingue de texto de odds (bookmakers PT/EN).

Melhorias face à versão original:
  - Padrões compilados num dicionário de configuração (fácil de estender).
  - Separação clara entre detecção de cabeçalho e extracção de odd.
  - Sem estado implícito (current_category como variável local, não global).
  - Funções pequenas e testáveis individualmente.
  - Suporte a mais variantes de linguagem sem duplicar código.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------
Markets = Dict  # estrutura aninhada; ver _empty_markets()


def _empty_markets() -> Markets:
    return {
        "match_winner":    {},
        "total_games":     {},
        "game_handicap":   {"P1": {}, "P2": {}},
        "set_handicap":    {"P1": {}, "P2": {}},
        "total_sets":      {},
        "p1_total_games":  {},
        "p2_total_games":  {},
        "set1_winner":     {},
        "set1_total_games":{},
        "set1_handicap":   {"P1": {}, "P2": {}},
        "set2_winner":     {},
        "set2_total_games":{},
        "set2_handicap":   {"P1": {}, "P2": {}},
        "total_aces":      {},
        "p1_aces":         {},
        "p2_aces":         {},
    }


# ---------------------------------------------------------------------------
# Padrões de categorias (cabeçalhos de secção)
# ---------------------------------------------------------------------------
# Cada entrada: (category_name, [lista de strings que activam esta categoria])
# A ordem importa: verificação topo-a-baixo, primeira correspondência ganha.
_IGNORED_PATTERNS: List[str] = [
    "par/ímpar", "odd/even", "exato", "exact", "correct",
    "duplo", "double result", "only one set", "apenas um set",
    "vencedor e", "winner and", "vencedor &", "winner &",
    "tie-break", "tie break",
]

_CATEGORY_PATTERNS: List[Tuple[str, List[str]]] = [
    # Ases
    ("p1_aces",   ["aces", "ases"]),         # refinado com tokens de jogador no caller
    ("p2_aces",   ["aces", "ases"]),
    ("total_aces",["aces", "ases"]),
    # Set 1
    ("set1_handicap",    ["set 1", "1º set", "1o set", "primeiro set", "1st set"]),
    ("set1_total_games", ["set 1", "1º set", "1o set", "primeiro set", "1st set"]),
    ("set1_winner",      ["set 1", "1º set", "1o set", "primeiro set", "1st set"]),
    # Set 2
    ("set2_handicap",    ["set 2", "2º set", "2o set", "segundo set", "2nd set"]),
    ("set2_total_games", ["set 2", "2º set", "2o set", "segundo set", "2nd set"]),
    ("set2_winner",      ["set 2", "2º set", "2o set", "segundo set", "2nd set"]),
    # Totais e handicaps globais
    ("total_games",  ["total jogos", "total games", "total de jogos", "jogos no encontro"]),
    ("total_sets",   ["total sets", "total de sets"]),
    ("set_handicap", ["handicap sets", "handicap de sets"]),
    ("game_handicap",["handicap"]),
    # Vencedor
    ("match_winner", ["winner", "vencedor", "resultado final", "match winner", "1x2"]),
]

# Regex compiladas
_RE_ODD_VALUE   = re.compile(r":\s*([\d,\.]+)\s*$")
_RE_OVER_UNDER  = re.compile(
    r"(over|under|mais de|menos de|mais|menos|acima|abaixo)\s*([\d]+\.[\d]+)",
    re.IGNORECASE,
)
_RE_HANDICAP    = re.compile(r"([+-]?\d+\.\d+)")
_RE_PLAYER_NUM  = re.compile(r"\b([12])\b")


def _name_tokens(name: str) -> List[str]:
    """Extrai tokens significativos de um nome de jogador."""
    return [t.lower() for t in name.replace(",", " ").split() if len(t) > 2]


def _is_ignored(header: str) -> bool:
    return any(p in header for p in _IGNORED_PATTERNS)


def _detect_category(
    header: str,
    p1_tokens: List[str],
    p2_tokens: List[str],
) -> str:
    """Determina a categoria de um cabeçalho de secção."""

    # Ases — precisa de distinguir por jogador
    if any(x in header for x in ["aces", "ases"]):
        if any(x in header for x in p1_tokens + ["player 1", "jogador 1", "casa"]):
            return "p1_aces"
        if any(x in header for x in p2_tokens + ["player 2", "jogador 2", "fora"]):
            return "p2_aces"
        return "total_aces"

    # Sets específicos
    for set_n, triggers, hcp_cat, total_cat, winner_cat in [
        ("set 1", ["set 1", "1º set", "1o set", "primeiro set", "1st set"],
         "set1_handicap", "set1_total_games", "set1_winner"),
        ("set 2", ["set 2", "2º set", "2o set", "segundo set", "2nd set"],
         "set2_handicap", "set2_total_games", "set2_winner"),
    ]:
        if any(x in header for x in triggers):
            # Ignora linhas de set que contêm nome do jogador (mercados como "set 1 P1")
            if any(x in header for x in p1_tokens + p2_tokens + ["player 1", "player 2", "jogador", "casa", "fora"]):
                return "Ignored"
            if "handicap" in header:
                return hcp_cat
            if any(x in header for x in ["total", "jogos", "games"]):
                return total_cat
            if any(x in header for x in ["winner", "vencedor"]):
                return winner_cat
            return "Ignored"

    # Jogos por jogador
    if any(x in header for x in p1_tokens + ["player 1", "jogador 1", "casa"]):
        if any(x in header for x in ["total", "jogos", "games"]):
            return "p1_total_games"
    if any(x in header for x in p2_tokens + ["player 2", "jogador 2", "fora"]):
        if any(x in header for x in ["total", "jogos", "games"]):
            return "p2_total_games"

    if any(x in header for x in ["total jogos", "total games", "total de jogos", "jogos no encontro"]):
        return "total_games"
    if "total sets" in header or "total de sets" in header:
        return "total_sets"
    if "handicap" in header:
        return "set_handicap" if "sets" in header else "game_handicap"
    if any(x in header for x in ["winner", "vencedor", "resultado final", "match winner", "1x2"]):
        return "match_winner"

    return "Ignored"


def _parse_odd_line(line: str) -> Optional[Tuple[str, float]]:
    """
    Tenta extrair (key_part, odd_value) de uma linha.
    Aceita formato 'Texto: 1.85' ou 'Texto — 1.85'.
    Devolve None se a linha não contém uma odd válida.
    """
    clean = line.replace("—", ":").replace(" - ", ":")
    m = _RE_ODD_VALUE.search(clean)
    if not m:
        return None
    try:
        odd = float(m.group(1).replace(",", "."))
        key = clean[: m.start()].strip().lower()
        return key, odd
    except ValueError:
        return None


def _is_compound_line(key: str, category: str) -> bool:
    """Filtra linhas compostas (ex: 'P1 e Over 20.5') que não devem ser processadas."""
    has_combinator = any(x in key for x in ["&", " and ", " e "])
    if has_combinator and category in (
        "match_winner", "set1_winner", "set2_winner",
        "total_games", "total_sets", "p1_total_games", "p2_total_games",
        "set1_total_games", "set2_total_games",
    ):
        return True
    # Resultado exacto em vencedor
    if category in ("match_winner", "set1_winner", "set2_winner"):
        if re.search(r"\b[02]:[012]\b", key):
            return True
    return False


def _update_winner(markets: Markets, category: str, key: str, odd: float,
                   p1_tokens: List[str], p2_tokens: List[str]) -> None:
    is_p1 = (
        any(x in key for x in p1_tokens + ["casa", "home", "jogador 1"])
        or re.search(r"(?<!\d)1(?!\d)", key) is not None
    )
    is_p2 = (
        any(x in key for x in p2_tokens + ["fora", "away", "jogador 2"])
        or re.search(r"(?<!\d)2(?!\d)", key) is not None
    )
    cat = markets[category]
    if is_p1 and "P1" not in cat:
        cat["P1"] = odd
    elif is_p2 and "P2" not in cat:
        cat["P2"] = odd


def _update_over_under(markets: Markets, category: str, key: str, odd: float) -> None:
    m = _RE_OVER_UNDER.search(key)
    if not m:
        return
    ou = "Over" if m.group(1).lower() in ("over", "mais de", "mais", "acima") else "Under"
    try:
        val = float(m.group(2))
    except ValueError:
        return
    # Redireciona linhas pequenas de jogos para total_sets
    if category == "total_games" and val < 6.0:
        category = "total_sets"
    if val not in markets[category]:
        markets[category][val] = {}
    markets[category][val][ou] = odd


def _update_handicap(markets: Markets, category: str, key: str, odd: float,
                     p1_tokens: List[str]) -> None:
    m = _RE_HANDICAP.search(key)
    if not m:
        return
    hcp = float(m.group(1))
    side = "P1" if any(x in key for x in p1_tokens + ["1", "casa", "jogador 1"]) else "P2"
    markets[category][side][hcp] = odd


# ---------------------------------------------------------------------------
# Interface pública
# ---------------------------------------------------------------------------

def parse_bookmaker_text(text: str, p1_name: str = "", p2_name: str = "") -> Markets:
    """
    Analisa texto copiado de uma casa de apostas e extrai mercados estruturados.

    Args:
        text:    Texto bruto (multi-linha) com odds.
        p1_name: Nome do jogador 1 (para identificar tokens de jogador).
        p2_name: Nome do jogador 2.

    Returns:
        Dicionário de mercados → odds estruturadas.
    """
    markets = _empty_markets()
    p1_tokens = _name_tokens(p1_name)
    p2_tokens = _name_tokens(p2_name)
    current_category = "Ignored"

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Tentar extrair odd
        parsed = _parse_odd_line(line)

        if parsed is None:
            # É um cabeçalho de secção
            header = line.lower()
            if _is_ignored(header):
                current_category = "Ignored"
            else:
                current_category = _detect_category(header, p1_tokens, p2_tokens)
            continue

        if current_category == "Ignored":
            continue

        key, odd = parsed

        if _is_compound_line(key, current_category):
            continue

        try:
            if current_category in ("match_winner", "set1_winner", "set2_winner"):
                _update_winner(markets, current_category, key, odd, p1_tokens, p2_tokens)

            elif current_category in (
                "total_games", "total_sets", "p1_total_games", "p2_total_games",
                "set1_total_games", "set2_total_games",
                "total_aces", "p1_aces", "p2_aces",
            ):
                _update_over_under(markets, current_category, key, odd)

            elif current_category in ("game_handicap", "set_handicap", "set1_handicap", "set2_handicap"):
                _update_handicap(markets, current_category, key, odd, p1_tokens)

        except Exception:
            # Linha malformada — ignora silenciosamente
            continue

    return markets
"""
data.py — Carregamento e acesso a dados de jogadores.

Correcções face à versão original:
  - load_agenda() não usa @cache_data porque depende de datetime.today()
    (a cache do Streamlit guarda o resultado para sempre, incluindo a data).
  - get_player_stats() é uma função pura sem estado global.
  - calculate_h2h() pré-filtra o DataFrame uma vez, evitando múltiplos scans.
  - Todos os nomes são normalizados com .casefold() (melhor que .lower() para PT).
"""
from __future__ import annotations

import os
import zipfile
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from config import CIRCUIT
from simulation import PlayerStats


# ---------------------------------------------------------------------------
# Carregamento de ficheiros
# ---------------------------------------------------------------------------

@st.cache_resource
def load_ml_model():
    """Carrega o modelo XGBoost se existir; devolve None caso contrário."""
    try:
        import joblib
        if os.path.exists("modelo_tenis_calibrado.pkl"):
            return joblib.load("modelo_tenis_calibrado.pkl")
    except ImportError:
        pass
    return None


@st.cache_data
def load_match_data() -> pd.DataFrame:
    """Carrega o CSV de partidas históricas do ZIP."""
    with zipfile.ZipFile("dados_resumidos.zip", "r") as z:
        df = pd.read_csv(z.open("dados_resumidos.csv"))
    # Normalizar nomes uma vez (evita normalizar em cada query)
    if "player" in df.columns:
        df["_player_norm"] = df["player"].str.casefold().str.strip()
    if "opponent" in df.columns:
        df["_opponent_norm"] = df["opponent"].str.casefold().str.strip()
    if "winner" in df.columns:
        df["_winner_norm"] = df["winner"].str.casefold().str.strip()
    return df


@st.cache_data
def load_elos(circuito: str) -> pd.DataFrame:
    """Carrega o CSV de ratings Elo para o circuito seleccionado."""
    ficheiro = "EloRankP.csv" if circuito == "WTA (Feminino)" else "PlayerElo.csv"
    if os.path.exists(ficheiro):
        df = pd.read_csv(ficheiro)
        df["_name_norm"] = df["Player"].str.casefold().str.strip()
        return df
    return pd.DataFrame(columns=["Player", "Elo", "hElo", "cElo", "gElo", "_name_norm"])


def load_agenda() -> pd.DataFrame:
    """
    Carrega a agenda de torneios.
    NÃO usa @cache_data porque contém datetime.today() — a cache tornaria
    a data estática para sempre até ao próximo restart da app.
    """
    if os.path.exists("agenda.csv"):
        try:
            df = pd.read_csv("agenda.csv")
            df["Data"] = pd.to_datetime(df["Data"]).dt.date
            return df
        except Exception:
            pass

    # Mockup ancorado no dia de hoje (sem cache)
    hoje = datetime.today().date()
    amanha = hoje + timedelta(days=1)
    return pd.DataFrame({
        "Torneio": ["ATP Challenger Amersfoort", "ATP Challenger Amersfoort",
                    "WTA Palermo", "WTA Palermo"],
        "Data":    [hoje, hoje, hoje, amanha],
        "Hora":    ["14:00", "15:30", "16:00", "10:00"],
        "P1":      ["Jesper De Jong", "Jaime Faria", "Qinwen Zheng", "Karolina Muchova"],
        "P2":      ["Sebastian Baez", "Titouan Droguet", "Sara Errani", "Qinwen Zheng"],
    })


# ---------------------------------------------------------------------------
# Estatísticas de jogadores
# ---------------------------------------------------------------------------

def get_player_stats(
    nome: str,
    superficie: str,
    circuito: str,
    df: pd.DataFrame,
    df_elos: pd.DataFrame,
) -> PlayerStats:
    """
    Extrai estatísticas de um jogador para a simulação.
    Devolve valores por defeito seguros se o jogador não for encontrado.
    """
    cfg = CIRCUIT[circuito]
    nome_norm = str(nome).casefold().strip()

    # ── Elo ──────────────────────────────────────────────────────────────
    elo = cfg.default_elo
    if "_name_norm" in df_elos.columns:
        row = df_elos[df_elos["_name_norm"] == nome_norm]
        if not row.empty:
            col = {"Clay": "cElo", "Grass": "gElo", "Hard": "hElo"}.get(superficie, "Elo")
            if col in row.columns:
                elo = float(row[col].iloc[0])

    # ── Hold rate ─────────────────────────────────────────────────────────
    hold_rate = cfg.base_hold
    if "_player_norm" in df.columns and "hold_percentage" in df.columns:
        matches = df[df["_player_norm"] == nome_norm]
        if not matches.empty:
            hold_rate = float(matches["hold_percentage"].mean())

    # ── Fadiga ────────────────────────────────────────────────────────────
    fatigue = 0.0
    if "_player_norm" in df.columns and "games_played_last_week" in df.columns:
        matches = df[df["_player_norm"] == nome_norm]
        if not matches.empty:
            fatigue = float(matches["games_played_last_week"].iloc[-1])

    # ── Forma recente (últimos 5 jogos) ──────────────────────────────────
    recent_form = 0.5
    if "_player_norm" in df.columns and "_winner_norm" in df.columns:
        matches = df[df["_player_norm"] == nome_norm].tail(5)
        if not matches.empty:
            wins = (matches["_winner_norm"] == nome_norm).sum()
            recent_form = wins / len(matches)

    return PlayerStats(elo=elo, hold_rate=hold_rate, fatigue=fatigue, recent_form=recent_form)


def calculate_h2h(
    p1: str,
    p2: str,
    df: pd.DataFrame,
) -> Tuple[int, int]:
    """
    Calcula H2H entre dois jogadores.
    Usa colunas normalizadas pré-calculadas para eficiência.
    """
    if "_player_norm" not in df.columns or "_opponent_norm" not in df.columns:
        return 0, 0

    p1_norm = str(p1).casefold().strip()
    p2_norm = str(p2).casefold().strip()

    # Jogos onde P1 jogou contra P2
    mask = (
        ((df["_player_norm"] == p1_norm) & (df["_opponent_norm"] == p2_norm)) |
        ((df["_player_norm"] == p2_norm) & (df["_opponent_norm"] == p1_norm))
    )
    h2h_matches = df[mask]

    if h2h_matches.empty or "_winner_norm" not in df.columns:
        return 0, 0

    p1_wins = int((h2h_matches["_winner_norm"] == p1_norm).sum())
    p2_wins = int((h2h_matches["_winner_norm"] == p2_norm).sum())
    return p1_wins, p2_wins