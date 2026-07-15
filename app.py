import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

# Configuração
st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Analisador Direto")

# --- 1. CARREGAMENTO ---
@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos():
    return pd.read_csv("PlayerElo.csv") if os.path.exists("PlayerElo.csv") else pd.DataFrame(columns=['Player', 'Elo', 'hElo', 'cElo', 'gElo'])

df = load_data()
df_elos = load_elos()

# --- 2. FUNÇÕES ---
def normalize_name(name):
    if pd.isna(name): return ""
    return str(name).lower().strip()

def get_elo(nome_jogador, superficie):
    if not nome_jogador: return 1500
    nome_alvo = normalize_name(nome_jogador)
    
    # Busca com normalização
    mask = df_elos['Player'].apply(normalize_name) == nome_alvo
    match = df_elos[mask]
    
    if match.empty: return 1500
    col = {'Clay': 'cElo', 'Grass': 'gElo', 'Hard': 'hElo'}.get(superficie, 'Elo')
    return int(match[col].values[0])

def monte_carlo_simulation(elo_p1, elo_p2, sets_to_win, n_simulations=10000):
    prob_p1 = 1 / (1 + 10**((elo_p2 - elo_p1) / 400))
    total_jogos = []
    for _ in range(n_simulations):
        p1_sets, p2_sets, total_g = 0, 0, 0
        while p1_sets < sets_to_win and p2_sets < sets_to_win:
            p1_g, p2_g = 0, 0
            while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
                if np.random.random() < prob_p1: p1_g += 1
                else: p2_g += 1
                if p1_g == 7 or p2_g == 7: break
            total_g += (p1_g + p2_g)
            if p1_g > p2_g: p1_sets += 1
            else: p2_sets += 1
        total_jogos.append(total_g)
    return np.array(total_jogos)

# --- 3. INTERFACE ---
superficies = sorted(df['surface'].dropna().unique())
superficie = st.sidebar.selectbox("Superfície", superficies)

# Filtro apenas por superfície (sem torneios)
df_filtrado = df[df['surface'] == superficie]
jogadores = sorted(df_filtrado['player'].unique())

c1, c2 = st.columns(2)
nome_p1 = c1.selectbox("Favorito", jogadores, key="p1")
nome_p2 = c2.selectbox("Adversário", jogadores, key="p2")

sets_input = st.sidebar.radio("Formato", [3, 5])

# Mostrar Elos
elo1 = get_elo(nome_p1, superficie)
elo2 = get_elo(nome_p2, superficie)
c1.metric(f"Elo {nome_p1}", elo1)
c2.metric(f"Elo {nome_p2}", elo2)

if st.button("Simular"):
    if nome_p1 == nome_p2:
        st.error("Escolhe jogadores diferentes.")
    else:
        totais = monte_carlo_simulation(elo1, elo2, sets_to_win=(sets_input//2 + 1))
        st.metric("Média de Jogos Previstos", f"{np.mean(totais):.1f}")
        st.metric("Probabilidade Over 21.5", f"{np.mean(totais > 21.5):.1%}")
