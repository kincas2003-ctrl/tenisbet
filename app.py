import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Analisador Profissional")

# --- 1. CARREGAMENTO ---
@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos():
    return pd.read_csv("PlayerElo.csv") if os.path.exists("PlayerElo.csv") else pd.DataFrame()

df, df_elos = load_data(), load_elos()

# --- 2. FUNÇÕES ---
def get_elo(nome_jogador, superficie):
    if not nome_jogador: return 1500
    nome_norm = str(nome_jogador).lower().strip()
    mask = df_elos['Player'].str.lower().str.strip() == nome_norm
    match = df_elos[mask]
    if match.empty: return 1500
    col = {'Clay': 'cElo', 'Grass': 'gElo', 'Hard': 'hElo'}.get(superficie, 'Elo')
    return int(match[col].values[0])

def monte_carlo_simulation(elo_p1, elo_p2, sets_to_win, n_simulations=5000):
    prob_p1 = 1 / (1 + 10**((elo_p2 - elo_p1) / 400))
    # Armazenar resultados: Jogos totais, Diferença de Jogos
    resultados_totais = []
    diff_jogos = []
    vitorias_p1 = 0
    
    for _ in range(n_simulations):
        p1_sets, p2_sets = 0, 0
        total_g, diff_g = 0, 0
        
        while p1_sets < sets_to_win and p2_sets < sets_to_win:
            p1_g, p2_g = 0, 0
            while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
                if np.random.random() < prob_p1: p1_g += 1
                else: p2_g += 1
                if p1_g == 7 or p2_g == 7: break
            
            total_g += (p1_g + p2_g)
            diff_g += (p1_g - p2_g)
            
            if p1_g > p2_g: p1_sets += 1
            else: p2_sets += 1
            
        resultados_totais.append(total_g)
        diff_jogos.append(diff_g)
        if p1_sets > p2_sets: vitorias_p1 += 1
            
    return np.array(resultados_totais), np.array(diff_jogos), (vitorias_p1 / n_simulations)

# --- 3. INTERFACE ---
superficie = st.sidebar.selectbox("Superfície", sorted(df['surface'].dropna().unique()))
df_filtrado = df[df['surface'] == superficie]
jogadores = sorted(df_filtrado['player'].unique())

c1, c2 = st.columns(2)
nome_p1 = c1.selectbox("Favorito", jogadores, key="p1")
nome_p2 = c2.selectbox("Adversário", jogadores, key="p2")
sets_input = st.sidebar.radio("Sets", [3, 5])

if st.button("Simular Mercados"):
    elo1, elo2 = get_elo(nome_p1, superficie), get_elo(nome_p2, superficie)
    totais, diffs, prob_vitoria = monte_carlo_simulation(elo1, elo2, sets_to_win=(sets_input//2 + 1))
    
    col1, col2, col3 = st.columns(3)
    # Vencedor
    col1.metric("Prob. Vencedor (P1)", f"{prob_vitoria:.1%}")
    # Handicap
    h = col2.number_input("Handicap Jogos", value=-3.5)
    prob_h = np.mean(diffs > abs(h)) if h < 0 else np.mean(diffs < -h)
    col2.metric("Prob. Handicap", f"{prob_h:.1%}")
    # Total Jogos
    linha = col3.number_input("Linha de Jogos", value=21.5)
    col3.metric("Prob. Over", f"{np.mean(totais > linha):.1%}")