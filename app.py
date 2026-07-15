import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

# Configuração da página
st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Analisador Quantitativo")

# --- 1. CARREGAMENTO DE DADOS ---
@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos():
    if os.path.exists("elos_jogadores.csv"):
        return pd.read_csv("elos_jogadores.csv")
    return pd.DataFrame(columns=['player', 'elo'])

df = load_data()
df_elos = load_elos()

# --- 2. FUNÇÕES DE CÁLCULO ---
def get_elo(nome_jogador):
    resultado = df_elos[df_elos['player'] == nome_jogador]
    return int(resultado['elo'].values[0]) if not resultado.empty else 1500

def monte_carlo_simulation(elo_p1, elo_p2, n_simulations=10000):
    prob_p1 = 1 / (1 + 10**((elo_p2 - elo_p1) / 400))
    diff_resultados = []
    total_jogos = []
    
    for _ in range(n_simulations):
        p1_g, p2_g = 0, 0
        # Simula um set (primeiro a 6, tie-break se 6-6)
        while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
            if np.random.random() < prob_p1: p1_g += 1
            else: p2_g += 1
            if p1_g == 7 or p2_g == 7: break
        diff_resultados.append(p1_g - p2_g)
        total_jogos.append(p1_g + p2_g)
    return np.array(diff_resultados), np.array(total_jogos)

# --- 3. INTERFACE ---
# Filtro de Superfície (Obrigatório)
superficies = sorted([s for s in df['surface'].unique() if pd.notna(s)])
superficie_escolhida = st.sidebar.selectbox("Escolhe a Superfície", superficies)
df_filtrado = df[df['surface'] == superficie_escolhida]

# Seleção Jogadores
jogadores = sorted(df_filtrado['player'].unique())
c1, c2 = st.columns(2)
nome_p1 = c1.selectbox("Favorito", jogadores, key="p1")
nome_p2 = c2.selectbox("Adversário", jogadores, key="p2")

# Mostrar Elos
elo1, elo2 = get_elo(nome_p1), get_elo(nome_p2)
c1.metric(f"Elo {nome_p1}", elo1)
c2.metric(f"Elo {nome_p2}", elo2)

st.divider()

# Simulação
if st.button("Executar Simulação de Monte Carlo"):
    diffs, totais = monte_carlo_simulation(elo1, elo2)
    
    st.subheader("Resultados da Simulação (10.000 cenários)")
    
    # Coluna 1: Vencedor e Handicap
    with st.columns(2)[0]:
        st.markdown("### Mercado: Vencedor/Handicap")
        prob_vitoria = np.mean(diffs > 0)
        st.write(f"Prob. de Vitória ({nome_p1}): **{prob_vitoria:.2%}**")
        
        h = st.number_input("Handicap de Jogos", value=-2.5)
        prob_h = np.mean(diffs > abs(h)) if h < 0 else np.mean(diffs < -h)
        st.write(f"Prob. de cumprir Handicap: **{prob_h:.2%}**")
        
    # Coluna 2: Over/Under
    with st.columns(2)[1]:
        st.markdown("### Mercado: Total de Jogos")
        linha = st.number_input("Linha de Jogos", value=21.5)
        prob_over = np.mean(totais > linha)
        st.write(f"Probabilidade Over: **{prob_over:.2%}**")
        st.write(f"Probabilidade Under: **{1-prob_over:.2%}**")