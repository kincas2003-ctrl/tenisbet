import streamlit as st
import pandas as pd
import zipfile
import os

# Configuração da página
st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Analisador de Apostas ATP")

# 1. Carregar dados com segurança
@st.cache_data
def load_data():
    # Verifica se o ficheiro existe antes de tentar abrir
    if not os.path.exists("dados_resumidos.zip"):
        st.error("Ficheiro 'dados_resumidos.zip' não encontrado na raiz do repositório.")
        st.stop()
    
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao processar ficheiros: {e}")
    st.stop()

# 2. Funções de Cálculo
def calcular_probabilidade(jogador1, jogador2, df):
    def get_stats(nome):
        dados = df[df['player'] == nome]
        if dados.empty: return 0.5
        # Soma a eficácia total das direções de serviço
        return dados[['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()

    s1 = get_stats(jogador1)
    s2 = get_stats(jogador2)
    
    if (s1 + s2) == 0: return 0.5
    
    # Fórmula de Elo-Service ajustada
    prob_p1 = (s1**1.5) / ((s1**1.5) + (s2**1.5))
    return prob_p1

# 3. Interface Principal
st.subheader("Análise de Performance de Serviço")

# Seleção de jogadores com keys únicas para evitar erros de duplicidade
jogadores = sorted(df['player'].unique())
col1, col2 = st.columns(2)

with col1:
    nome_p1 = st.selectbox("Escolhe o Favorito", jogadores, key="p1")
with col2:
    nome_p2 = st.selectbox("Escolhe o Adversário", jogadores, key="p2")

if st.button("Gerar Projeção Automática"):
    prob = calcular_probabilidade(nome_p1, nome_p2, df)
    
    # Exibir resultado
    st.metric(label=f"Probabilidade de vitória: {nome_p1}", value=f"{prob:.2%}")
    
    # Análise de valor (Edge)
    odd = st.number_input("Odd oferecida pela Casa", min_value=1.01, value=1.50)
    edge = (prob * odd) - 1
    
    if edge > 0:
        st.success(f"VALOR DETETADO! Edge de {edge*100:.2f}%")
    else:
        st.info("Mercado eficiente. Sem valor estatístico nesta odd.")

# Footer informativo
st.write("---")
st.caption("QuantBet Pro - Dados extraídos do MatchChartingProject")