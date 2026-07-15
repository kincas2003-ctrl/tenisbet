import streamlit as st
import pandas as pd
import zipfile

@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

df = load_data()

st.title("🎾 QuantBet Pro: Analisador")

# Em vez de filtrar por superfície (que não temos), filtramos apenas pelos jogadores
nome_p1 = st.selectbox("Favorito", df['player'].unique())
nome_p2 = st.selectbox("Adversário", df['player'].unique())

def calcular_probabilidade(jogador1, jogador2, df):
    def get_stats(nome):
        dados = df[df['player'] == nome]
        if dados.empty: return 0.5
        # Somamos a eficácia total de serviço (T e Wide)
        return dados[['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()

    s1 = get_stats(jogador1)
    s2 = get_stats(jogador2)
    
    if (s1 + s2) == 0: return 0.5
    return (s1**1.5) / ((s1**1.5) + (s2**1.5))

if st.button("Gerar Projeção Automática"):
    prob = calcular_probabilidade(nome_p1, nome_p2, df)
    st.write(f"### Projeção para {nome_p1}: **{prob:.2%}**")
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

df = load_data()

st.title("🎾 QuantBet Pro: Analisador")

# Em vez de filtrar por superfície (que não temos), filtramos apenas pelos jogadores
nome_p1 = st.selectbox("Favorito", df['player'].unique())
nome_p2 = st.selectbox("Adversário", df['player'].unique())

def calcular_probabilidade(jogador1, jogador2, df):
    def get_stats(nome):
        dados = df[df['player'] == nome]
        if dados.empty: return 0.5
        # Somamos a eficácia total de serviço (T e Wide)
        return dados[['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()

    s1 = get_stats(jogador1)
    s2 = get_stats(jogador2)
    
    if (s1 + s2) == 0: return 0.5
    return (s1**1.5) / ((s1**1.5) + (s2**1.5))

if st.button("Gerar Projeção Automática"):
    prob = calcular_probabilidade(nome_p1, nome_p2, df)
    st.write(f"### Projeção para {nome_p1}: **{prob:.2%}**")
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

df = load_data()

# 2. Definir a função de cálculo (COM A IDENTAÇÃO CORRETA)
def calcular_probabilidade(jogador1, jogador2, df_filtrado):
    def get_stats(nome):
        # Filtra pelo jogador
        dados = df_filtrado[df_filtrado['player'] == nome]
        if dados.empty: return 0.5
        return dados[['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()

    s1 = get_stats(jogador1)
    s2 = get_stats(jogador2)
    
    # Evita divisão por zero
    if (s1 + s2) == 0: return 0.5
    
    prob_p1 = (s1**1.5) / ((s1**1.5) + (s2**1.5))
    return prob_p1

# 3. Interface Streamlit
st.title("🎾 QuantBet Pro: Analisador")

# Filtro de Superfície
superficies = df['surface'].unique()
superficie_escolhida = st.selectbox("Escolhe a Superfície", superficies)
df_filtrado = df[df['surface'] == superficie_escolhida]

# Inputs de Jogadores
nome_p1 = st.selectbox("Favorito", df_filtrado['player'].unique())
nome_p2 = st.selectbox("Adversário", df_filtrado['player'].unique())

if st.button("Gerar Projeção Automática"):
    prob = calcular_probabilidade(nome_p1, nome_p2, df_filtrado)
    st.write(f"### Projeção para {nome_p1}: **{prob:.2%}**")