import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

# Configuração
st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Motor de Simulação Avançado")

# --- CARREGAMENTO ---
@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos():
    return pd.read_csv("PlayerElo.csv") if os.path.exists("PlayerElo.csv") else pd.DataFrame()

df = load_data()
df_elos = load_elos()

# --- FUNÇÕES ---
def get_elo(nome_jogador, superficie):
    row = df_elos[df_elos['Player'].str.lower() == nome_jogador.lower()]
    if row.empty: return 1500
    col = {'Clay': 'cElo', 'Grass': 'gElo', 'Hard': 'hElo'}.get(superficie, 'Elo')
    return int(row[col].values[0])

def monte_carlo_simulation(elo_p1, elo_p2, sets_to_win, n_simulations=10000):
    prob_p1 = 1 / (1 + 10**((elo_p2 - elo_p1) / 400))
    total_jogos_lista = []
    diff_lista = []
    
    for _ in range(n_simulations):
        p1_sets, p2_sets = 0, 0
        total_g = 0
        while p1_sets < sets_to_win and p2_sets < sets_to_win:
            # Simula um set
            p1_g, p2_g = 0, 0
            while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
                if np.random.random() < prob_p1: p1_g += 1
                else: p2_g += 1
                if p1_g == 7 or p2_g == 7: break
            total_g += (p1_g + p2_g)
            if p1_g > p2_g: p1_sets += 1
            else: p2_sets += 1
        total_jogos_lista.append(total_g)
        diff_lista.append(p1_sets - p2_sets) # Simplificado para diferença de sets
    return np.array(total_jogos_lista), np.array(diff_lista)

# --- INTERFACE ---
superficie = st.sidebar.selectbox("Superfície", sorted(df['surface'].dropna().unique()))
torneios = sorted(df[df['surface'] == superficie]['tournament'].unique()) # Garante que tournament existe no teu CSV
torneio_escolhido = st.sidebar.multiselect("Torneio", torneios, default=torneios[0] if torneios else [])
sets_input = st.sidebar.radio("Formato do Encontro", [3, 5])

# Seleção de jogadores
jogadores = sorted(df[df['surface'] == superficie]['player'].unique())
c1, c2 = st.columns(2)
nome_p1 = c1.selectbox("Favorito", jogadores)
nome_p2 = c2.selectbox("Adversário", jogadores)

if st.button("Executar Simulação"):
    elo1, elo2 = get_elo(nome_p1, superficie), get_elo(nome_p2, superficie)
    totais, diffs = monte_carlo_simulation(elo1, elo2, sets_to_win=(sets_input//2 + 1))
    
    st.subheader("Resultados")
    c_res1, c_res2 = st.columns(2)
    c_res1.metric("Média de Jogos", f"{np.mean(totais):.1f}")
    c_res2.metric("Probabilidade Over 21.5", f"{np.mean(totais > 21.5):.1%}")