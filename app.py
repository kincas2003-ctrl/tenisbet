import streamlit as st
import pandas as pd
import zipfile
import os

st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Analisador de Valor")

# 1. Carregar Dados
@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

df = load_data()

# 2. Lógica de Cálculo (Probabilidade e Handicap)
def calcular_probabilidade(p1, p2, df):
    stats = df[df['player'].isin([p1, p2])]
    if stats.empty: return 0.5
    s1 = stats[stats['player'] == p1][['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()
    s2 = stats[stats['player'] == p2][['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()
    return (s1**1.5) / ((s1**1.5) + (s2**1.5))

# 3. Interface: Input de Jogadores
jogadores = sorted(df['player'].unique())
c1, c2 = st.columns(2)
nome_p1 = c1.selectbox("Favorito", jogadores, key="p1")
nome_p2 = c2.selectbox("Adversário", jogadores, key="p2")

prob_p1 = calcular_probabilidade(nome_p1, nome_p2, df)

# 4. Painel de Mercados (Onde inseres as odds da casa)
st.subheader("Inserir Odds da Casa")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("### Mercado: Vencedor")
    odd_win = st.number_input("Odd Vencedor", min_value=1.01, value=1.50)
    edge_win = (prob_p1 * odd_win) - 1
    if edge_win > 0:
        st.success(f"VALOR DETETADO! Edge: {edge_win*100:.2f}%")
    else:
        st.warning("Mercado eficiente.")

with col_m2:
    st.markdown("### Mercado: Handicap")
    h_valor = st.number_input("Handicap (ex: -2.5)", value=-2.5)
    odd_h = st.number_input("Odd do Handicap", min_value=1.01, value=1.90)
    
    # Conversão simples de probabilidade para handicap de jogos
    # Se a diferença de habilidade for X, projetamos uma vantagem de jogos
    diff_skill = (prob_p1 - 0.5) * 10
    if abs(h_valor) < diff_skill:
        st.success(f"VALOR NO HANDICAP! Projeção: {diff_skill:.1f} jogos.")
    else:
        st.info("Handicap justo.")

st.sidebar.info("QuantBet Pro: A tua ferramenta de decisão baseada em dados reais de serviço.")