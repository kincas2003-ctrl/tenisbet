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
            import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

# --- FUNÇÕES DE CARGA ---
@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos():
    return pd.read_csv("elos_jogadores.csv") if os.path.exists("elos_jogadores.csv") else pd.DataFrame(columns=['player', 'elo'])

df = load_data()
df_elos = load_elos()

# --- LÓGICA DE SIMULAÇÃO ---
def monte_carlo_simulation(elo_p1, elo_p2, n_simulations=10000):
    prob_p1 = 1 / (1 + 10**((elo_p2 - elo_p1) / 400))
    diff_resultados = []
    total_jogos = []
    
    for _ in range(n_simulations):
        p1_g, p2_g = 0, 0
        while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
            if np.random.random() < prob_p1: p1_g += 1
            else: p2_g += 1
            if p1_g == 7 or p2_g == 7: break
        diff_resultados.append(p1_g - p2_g)
        total_jogos.append(p1_g + p2_g)
        
    return np.array(diff_resultados), np.array(total_jogos)

# --- INTERFACE ---
st.title("🎾 QuantBet Pro: Analisador Avançado")

# Filtro de Superfície (Todas as opções)
superficies = sorted([s for s in df['surface'].unique() if pd.notna(s)])
superficie_escolhida = st.sidebar.selectbox("Superfície", ["Todas"] + superficies)

if superficie_escolhida != "Todas":
    df_filtrado = df[df['surface'] == superficie_escolhida]
else:
    df_filtrado = df

# Seleção Jogadores
jogadores = sorted(df_filtrado['player'].unique())
c1, c2 = st.columns(2)
nome_p1 = c1.selectbox("Favorito", jogadores, key="p1")
nome_p2 = c2.selectbox("Adversário", jogadores, key="p2")

# Cálculo e Botão
if st.button("Executar Simulação Completa"):
    elo1 = df_elos[df_elos['player'] == nome_p1]['elo'].values[0] if nome_p1 in df_elos['player'].values else 1500
    elo2 = df_elos[df_elos['player'] == nome_p2]['elo'].values[0] if nome_p2 in df_elos['player'].values else 1500
    
    diffs, totais = monte_carlo_simulation(elo1, elo2)
    
    # Resultados
    st.subheader("Resultados da Simulação")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Mercado Handicap**")
        h = st.number_input("Handicap (-2.5)", value=-2.5)
        prob_h = np.mean(diffs > abs(h)) if h < 0 else np.mean(diffs < -h)
        st.write(f"Prob: {prob_h:.2%}")
        
    with col_b:
        st.markdown("**Mercado Over/Under**")
        total_linha = st.number_input("Linha de Jogos (ex: 21.5)", value=21.5)
        prob_over = np.mean(totais > total_linha)
        st.write(f"Probabilidade Over: {prob_over:.2%}")
        st.write(f"Probabilidade Under: {1-prob_over:.2%}")