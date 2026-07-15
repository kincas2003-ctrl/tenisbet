import streamlit as st
import pandas as pd
import zipfile
import os

# Configuração da página
st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Analisador de Valor ATP")

# 1. Carregar dados de forma segura
@st.cache_data
def load_data():
    if not os.path.exists("dados_resumidos.zip"):
        st.error("Ficheiro 'dados_resumidos.zip' não encontrado.")
        st.stop()
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao carregar: {e}")
    st.stop()

# 2. Configuração do filtro de superfície na Sidebar
superficies = [s for s in df['surface'].unique() if pd.notna(s)]
superficie_escolhida = st.sidebar.selectbox("Escolhe a Superfície", superficies)
df_filtrado = df[df['surface'] == superficie_escolhida]

# 3. Definição da lógica de predição
def calcular_probabilidade(p1, p2, df_sub):
    stats1 = df_sub[df_sub['player'] == p1][['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()
    stats2 = df_sub[df_sub['player'] == p2][['deuce_t', 'ad_t', 'deuce_wide', 'ad_wide']].mean().sum()
    
    if (stats1 + stats2) == 0: return 0.5
    # Fórmula de Elo-Service para determinar vantagem competitiva
    return (stats1**1.5) / ((stats1**1.5) + (stats2**1.5))

# 4. Interface Principal
jogadores = sorted(df_filtrado['player'].unique())
col1, col2 = st.columns(2)

nome_p1 = col1.selectbox("Escolhe o Favorito", jogadores, key="player_one_unique")
nome_p2 = col2.selectbox("Escolhe o Adversário", jogadores, key="player_two_unique")

# Cálculo automático
prob_p1 = calcular_probabilidade(nome_p1, nome_p2, df_filtrado)

# 5. Painel de Análise de Valor
st.subheader("Análise de Mercados")
m1, m2 = st.columns(2)

with m1:
    st.markdown("### Mercado: Vencedor")
    odd_win = st.number_input("Odd Vencedor", min_value=1.01, value=1.50)
    edge_win = (prob_p1 * odd_win) - 1
    
    st.write(f"Probabilidade calculada: **{prob_p1:.2%}**")
    if edge_win > 0:
        st.success(f"VALOR DETETADO! Edge: {edge_win*100:.2f}%")
    else:
        st.warning("Mercado eficiente.")

with m2:
    st.markdown("### Mercado: Handicap de Jogos")
    h_valor = st.number_input("Handicap da Casa (ex: -2.5)", value=-2.5)
    odd_h = st.number_input("Odd do Handicap", min_value=1.01, value=1.90)
    
    # Projeção de jogos (baseada na diferença de habilidade)
    diff_skill = (prob_p1 - 0.5) * 12
    st.write(f"Vantagem projetada pelo modelo: **{diff_skill:.1f} jogos**")
    
    if abs(h_valor) < diff_skill:
        st.success("VALOR NO HANDICAP!")
    else:
        st.info("O mercado está justo.")

st.sidebar.info("QuantBet Pro: Analisando performance real de serviço.")
def calcular_elo_historico(df_matches):
    # Dicionário para guardar o Elo atual de cada jogador
    elos = {}
    k_factor = 32
    
    # Ordenar por data para processar os jogos cronologicamente
    df_matches = df_matches.sort_values('Date')
    
    for index, row in df_matches.iterrows():
        p1, p2 = row['Player 1'], row['Player 2']
        # Elo inicial 1500
        elo1 = elos.get(p1, 1500)
        elo2 = elos.get(p2, 1500)
        
        # Probabilidade esperada
        e1 = 1 / (1 + 10**((elo2 - elo1) / 400))
        
        # Resultado (1 se Player 1 venceu, 0 caso contrário)
        # Nota: Precisas de uma coluna que indique o vencedor (ex: 'Winner')
        resultado = 1 if row['Winner'] == p1 else 0
        
        # Atualização
        elos[p1] = elo1 + k_factor * (resultado - e1)
        elos[p2] = elo2 + k_factor * ((1 - resultado) - (1 - e1))
        
    return elos