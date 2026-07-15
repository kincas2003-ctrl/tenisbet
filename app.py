import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

# Configuração
st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Motor Ponto-a-Ponto")

# --- CARREGAMENTO ---
@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos():
    return pd.read_csv("PlayerElo.csv") if os.path.exists("PlayerElo.csv") else pd.DataFrame()

df, df_elos = load_data(), load_elos()

# --- FUNÇÕES ---
def get_elo(nome_jogador, superficie):
    if not nome_jogador: return 1500
    nome_norm = str(nome_jogador).lower().strip()
    match = df_elos[df_elos['Player'].str.lower().str.strip() == nome_norm]
    if match.empty: return 1500
    col = {'Clay': 'cElo', 'Grass': 'gElo', 'Hard': 'hElo'}.get(superficie, 'Elo')
    return int(match[col].values[0])

def play_point(prob_p1):
    return 1 if np.random.random() < prob_p1 else 0

def simulate_match(elo_p1, elo_p2, sets_to_win):
    # Probabilidade de ganhar um ponto baseada na diferença de Elo
    # (Fórmula calibrada para performance de pontos em ATP)
    prob_p1 = 0.5 + ((elo_p1 - elo_p2) / 2000)
    
    p1_sets, p2_sets = 0, 0
    total_g, diff_g = 0, 0
    
    while p1_sets < sets_to_win and p2_sets < sets_to_win:
        p1_g, p2_g = 0, 0
        while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
            # Simula jogo (simples: o melhor ponto vence)
            p1_games_in_set = 0
            p2_games_in_set = 0
            # Vence o jogo quem ganhar 4 pontos primeiro (com vantagem)
            p1_p, p2_p = 0, 0
            while (p1_p < 4 and p2_p < 4) or abs(p1_p - p2_p) < 2:
                if play_point(prob_p1): p1_p += 1
                else: p2_p += 1
            if p1_p > p2_p: p1_g += 1
            else: p2_g += 1
        
        total_g += (p1_g + p2_g)
        diff_g += (p1_g - p2_g)
        if p1_g > p2_g: p1_sets += 1
        else: p2_sets += 1
        
    return total_g, diff_g, (1 if p1_sets > p2_sets else 0)

# --- INTERFACE ---
superficie = st.sidebar.selectbox("Superfície", sorted(df['surface'].dropna().unique()))
df_filtrado = df[df['surface'] == superficie]
jogadores = sorted(df_filtrado['player'].unique())
nome_p1, nome_p2 = st.columns(2)[0].selectbox("Favorito", jogadores), st.columns(2)[1].selectbox("Adversário", jogadores)
sets_input = st.sidebar.radio("Formato", [3, 5])

if st.button("Executar Simulação de Pontos"):
    elo1, elo2 = get_elo(nome_p1, superficie), get_elo(nome_p2, superficie)
    sims = [simulate_match(elo1, elo2, (sets_input//2 + 1)) for _ in range(2000)]
    
    totais = np.array([s[0] for s in sims])
    diffs = np.array([s[1] for s in sims])
    vitorias = np.array([s[2] for s in sims])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Prob. Vitória (P1)", f"{np.mean(vitorias):.1%}")
    h = c2.number_input("Handicap Jogos", value=-3.5)
    c2.metric("Prob. Handicap", f"{np.mean(diffs > abs(h) if h < 0 else diffs < -h):.1%}")
    linha = c3.number_input("Linha de Jogos", value=21.5)
    c3.metric("Prob. Over", f"{np.mean(totais > linha):.1%}")