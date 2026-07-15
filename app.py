import streamlit as st
import pandas as pd
import zipfile

# 1. Configuração da página
st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Analisador de Apostas ATP")

# 2. Carregar dados (o nosso novo ficheiro zip)
@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# 3. Sidebar para escolha de mercado
mercado = st.sidebar.selectbox("Escolhe o Mercado", ["Vencedor (Match Winner)", "Handicap de Jogos"])

st.subheader(f"Análise de: {mercado}")

# 4. Lógica de Mercado
if mercado == "Vencedor (Match Winner)":
    # Aqui podes usar o 'df' para consultar estatísticas reais dos jogadores
    jogador_selecionado = st.selectbox("Escolhe o jogador para ver stats:", df['player'].unique())
    st.write(df[df['player'] == jogador_selecionado])
    
    odd = st.number_input("Odd do Favorito", min_value=1.01, value=1.45)
    prob = st.slider("Probabilidade do Modelo (0-1)", 0.0, 1.0, 0.74)
    banca = st.number_input("Tamanho da Banca (€)", value=1000.0)
    
    if st.button("Calcular Aposta"):
        kelly = ((odd - 1) * prob - (1 - prob)) / (odd - 1)
        stake = banca * (max(0, kelly) * 0.25)
        if (prob * odd) > 1:
            st.success(f"VALOR DETETADO! Stake Sugerida: €{stake:.2f}")
        else:
            st.error("Sem valor estatístico.")

elif mercado == "Handicap de Jogos":
    handicap_casa = st.number_input("Handicap da Casa (ex: -2.5)", value=-2.5)
    projecao = st.slider("Projeção do Modelo (Dif. de Jogos)", -6.0, 6.0, 0.0)
    
    if st.button("Analisar Handicap"):
        if projecao > abs(handicap_casa):
            st.success("Valor no Handicap!")
        else:
            st.info("Sem valor claro.")
# Adiciona esta função de cálculo ao teu app.py
def calcular_probabilidade(jogador1, jogador2, df):
    # Vamos criar um "Score de Força" baseado no serviço (as colunas que temos)
    # Exemplo: Média de eficácia no T e Wide
    stats1 = df[df['player'] == jogador1][['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()
    stats2 = df[df['player'] == jogador2][['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()
    
    # Probabilidade simples: Força do P1 / (Força do P1 + Força do P2)
    prob_p1 = stats1 / (stats1 + stats2)
    return prob_p1

# Na tua interface:
nome_p1 = st.selectbox("Favorito", df['player'].unique())
nome_p2 = st.selectbox("Adversário", df['player'].unique())

if st.button("Gerar Projeção Automática"):
    prob = calcular_probabilidade(nome_p1, nome_p2, df)
    st.write(f"Probabilidade calculada pelo modelo para {nome_p1}: {prob:.2%}")
    def calcular_probabilidade(jogador1, jogador2, df):
    # 1. Obter stats (se não houver dados, assumimos um valor neutro de 0.5)
    def get_stats(nome):
        dados = df[df['player'] == nome]
        if dados.empty: return 0.5
        # Peso maior para os pontos que ocorrem mais vezes (T e Wide)
        return dados[['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()

    s1 = get_stats(jogador1)
    s2 = get_stats(jogador2)
    
    # 2. Fórmula de Elo-Service (mais agressiva na diferenciação)
    # A fórmula (S1^1.5) / (S1^1.5 + S2^1.5) exagera a diferença entre bons e maus 
    # servidores, tornando o modelo mais decisivo e menos "em cima do muro"
    prob_p1 = (s1**1.5) / ((s1**1.5) + (s2**1.5))
    
    return prob_p1
# Agora que o CSV tem a coluna 'surface', isto já não será vazio:
superficies_disponiveis = df['surface'].unique()
superficie_jogo = st.selectbox("Superfície do Torneio", superficies_disponiveis)