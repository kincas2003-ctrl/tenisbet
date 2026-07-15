import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

# Configuração da página
st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Analisador Quantitativo")

# --- FUNÇÕES DE CARGA E CÁLCULO ---

@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos():
    if os.path.exists("elos_jogadores.csv"):
        return pd.read_csv("elos_jogadores.csv")
    return pd.DataFrame(columns=['player', 'elo'])

# Carregar dados
df = load_data()
df_elos = load_elos()

def get_elo(nome_jogador):
    resultado = df_elos[df_elos['player'] == nome_jogador]
    return int(resultado['elo'].values[0]) if not resultado.empty else 1500

def monte_carlo_simulation(elo_p1, elo_p2, n_simulations=10000):
    prob_p1 = 1 / (1 + 10**((elo_p2 - elo_p1) / 400))
    resultados = []
    for _ in range(n_simulations):
        p1_games, p2_games = 0, 0
        # Simula um set (primeiro a 6)
        while (p1_games < 6 and p2_games < 6) or abs(p1_games - p2_games) < 2:
            if np.random.random() < prob_p1: p1_games += 1
            else: p2_games += 1
            if p1_games == 7 or p2_games == 7: break
        resultados.append(p1_games - p2_games)
    return np.array(resultados)

# --- INTERFACE ---

# Sidebar: Filtro
superficies = [s for s in df['surface'].unique() if pd.notna(s)]
superficie_escolhida = st.sidebar.selectbox("Escolhe a Superfície", superficies)
df_filtrado = df[df['surface'] == superficie_escolhida]

# Seleção de Jogadores
jogadores = sorted(df_filtrado['player'].unique())
col1, col2 = st.columns(2)

nome_p1 = col1.selectbox("Favorito", jogadores, key="p1_unique")
elo_p1 = get_elo(nome_p1)
col1.metric(label=f"Elo: {nome_p1}", value=elo_p1)

nome_p2 = col2.selectbox("Adversário", jogadores, key="p2_unique")
elo_p2 = get_elo(nome_p2)
col2.metric(label=f"Elo: {nome_p2}", value=elo_p2)

# Painel de Análise
st.divider()
m1, m2 = st.columns(2)

with m1:
    st.subheader("Análise: Vencedor")
    prob_p1 = 1 / (1 + 10**((elo_p2 - elo_p1) / 400))
    st.write(f"Probabilidade de vitória: **{prob_p1:.2%}**")

with m2:
    st.subheader("Simulação Monte Carlo")
    h_valor = st.number_input("Handicap de Jogos (ex: -2.5)", value=-2.5)
    
    if st.button("Executar Simulação"):
        simulacoes = monte_carlo_simulation(elo_p1, elo_p2)
        # Calcula sucesso do Handicap
        if h_valor < 0:
            prob_handicap = np.mean(simulacoes > abs(h_valor))
        else:
            prob_handicap = np.mean(simulacoes < -h_valor)
            
        st.write(f"Probabilidade de cumprir o Handicap: **{prob_handicap:.2%}**")
        if prob_handicap > 0:
            st.write(f"Odd Justa sugerida: **{1/prob_handicap:.2f}**")