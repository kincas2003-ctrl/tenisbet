import streamlit as st

st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Analisador de Apostas ATP")

# Sidebar para escolha de mercado
mercado = st.sidebar.selectbox("Escolhe o Mercado", ["Vencedor (Match Winner)", "Handicap de Jogos"])

st.subheader(f"Análise de: {mercado}")

if mercado == "Vencedor (Match Winner)":
    odd = st.number_input("Odd do Favorito", min_value=1.01, value=1.45)
    prob = st.slider("Probabilidade do Modelo (0-1)", 0.0, 1.0, 0.74)
    banca = st.number_input("Tamanho da Banca (€)", value=1000.0)
    
    if st.button("Calcular Aposta"):
        edge = (prob * odd) - 1
        # Kelly Fracionado (0.25)
        kelly = ((odd - 1) * prob - (1 - prob)) / (odd - 1)
        stake = banca * (max(0, kelly) * 0.25)
        
        if edge > 0:
            st.success(f"VALOR DETETADO! Edge: {edge*100:.2f}%")
            st.metric("Stake Sugerida", f"€{stake:.2f}")
        else:
            st.error("Sem valor estatístico.")

elif mercado == "Handicap de Jogos":
    handicap_casa = st.number_input("Handicap da Casa (ex: -2.5)", value=-2.5)
    projecao = st.slider("Projeção do Modelo (Dif. de Jogos)", -6.0, 6.0, 0.0)
    
    if st.button("Analisar Handicap"):
        # Se a projeção for mais favorável que o handicap da casa, há edge
        diferenca = projecao - abs(handicap_casa)
        if diferenca > 0:
            st.success(f"Valor no Handicap! A tua projeção supera a casa em {diferenca:.1f} jogos.")
        else:
            st.info("O mercado está eficiente. Sem valor claro no Handicap.") 
            import pandas as pd

def projetar_resultado(jogador1, jogador2, df):
    # Filtra dados históricos dos jogadores
    stats_j1 = df[df['Player'] == jogador1]['win_rate_clay'].mean()
    stats_j2 = df[df['Player'] == jogador2]['win_rate_clay'].mean()
    
    # Projeção simples: diferença de performance como base da probabilidade
    # (Podes tornar esta fórmula tão complexa quanto quiseres)
    prob_base = 0.5 + (stats_j1 - stats_j2)
    return prob_base

# No Streamlit, agora fazes isto:
nome_p1 = st.text_input("Nome do Favorito")
nome_p2 = st.text_input("Nome do Adversário")

if st.button("Gerar Projeção Automática"):
    df = pd.read_csv("atp_odds.csv") # O teu ficheiro com histórico
    prob_proj = projetar_resultado(nome_p1, nome_p2, df)
    st.write(f"O modelo projeta {prob_proj:.2%} de vitória para {nome_p1}")
    import pandas as pd
import glob
import os

def obter_stats_jogador(nome_jogador, pasta_path):
    # Procura todos os ficheiros CSV na pasta
    ficheiros = glob.glob(os.path.join(pasta_path, "*.csv"))
    
    # Lista para guardar os dados filtrados
    dados_jogador = []
    
    for f in ficheiros:
        df = pd.read_csv(f, low_memory=False)
        # Filtra jogos onde o jogador participou
        stats = df[(df['match_winner'] == nome_jogador) | (df['match_loser'] == nome_jogador)]
        dados_jogador.append(stats)
        
    return pd.concat(dados_jogador) 
import streamlit as st
import pandas as pd
import zipfile

@st.cache_data
def load_data():
    # Abre o zip e lê o CSV que está lá dentro
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

df = load_data()