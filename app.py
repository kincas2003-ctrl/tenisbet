import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

# Configuração da página
st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Motor de Simulação Avançado")

# --- 1. CARREGAMENTO DE DADOS ---
@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos():
    if os.path.exists("PlayerElo.csv"):
        return pd.read_csv("PlayerElo.csv")
    return pd.DataFrame(columns=['Player', 'Elo', 'hElo', 'cElo', 'gElo'])

df = load_data()
df_elos = load_elos()

# --- 2. FUNÇÕES DE CÁLCULO ---
def get_elo(nome_jogador, superficie):
    row = df_elos[df_elos['Player'].str.lower() == nome_jogador.lower()]
    if row.empty: return 1500
    # Mapeamento de colunas do PlayerElo.csv
    col = {'Clay': 'cElo', 'Grass': 'gElo', 'Hard': 'hElo'}.get(superficie, 'Elo')
    return int(row[col].values[0])

def monte_carlo_simulation(elo_p1, elo_p2, sets_to_win, n_simulations=10000):
    prob_p1 = 1 / (1 + 10**((elo_p2 - elo_p1) / 400))
    total_jogos_lista = []
    diff_sets = []
    
    for _ in range(n_simulations):
        p1_sets, p2_sets = 0, 0
        total_g = 0
        while p1_sets < sets_to_win and p2_sets < sets_to_win:
            p1_g, p2_g = 0, 0
            while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
                if np.random.random() < prob_p1: p1_g += 1
                else: p2_g += 1
                if p1_g == 7 or p2_g == 7: break
            total_g += (p1_g + p2_g)
            if p1_g > p2_g: p1_sets += 1
            else: p2_sets += 1
        total_jogos_lista.append(total_g)
        diff_sets.append(p1_sets - p2_sets)
    return np.array(total_jogos_lista), np.array(diff_sets)

# --- 3. INTERFACE ---
# --- FILTROS DE INTERFACE (Seguros e Corretos) ---

# 1. Superfície
superficies = sorted(df['surface'].dropna().unique())
superficie = st.sidebar.selectbox("Escolhe a Superfície", superficies)

# 2. Torneio (Filtrado apenas pela superfície selecionada)
# Como a coluna 'tournament' agora existe, isto não dará mais o KeyError
torneios_disponiveis = sorted(df[df['surface'] == superficie]['tournament'].unique())
torneio_escolhido = st.sidebar.multiselect("Escolhe o(s) Torneio(s)", torneios_disponiveis, default=torneios_disponiveis[0] if torneios_disponiveis else [])

# 3. Filtragem final do DataFrame
df_filtrado = df[(df['surface'] == superficie) & (df['tournament'].isin(torneio_escolhido))]

# Extração de jogadores do dataset já filtrado
jogadores = sorted(df_filtrado['player'].unique())
# --- AJUSTE: Filtro de Torneio (À prova de erro) ---

# Define os filtros de forma segura
if 'surface' in df.columns:
    superficies = sorted([s for s in df['surface'].unique() if pd.notna(s)])
    superficie = st.sidebar.selectbox("Superfície", superficies)
    
    # Verifica se a coluna tournament existe antes de filtrar
    if 'tournament' in df.columns:
        torneios_disponiveis = sorted(df[df['surface'] == superficie]['tournament'].unique())
        torneio_escolhido = st.sidebar.multiselect("Torneio", torneios_disponiveis, default=torneios_disponiveis[:1])
        df_filtrado = df[(df['surface'] == superficie) & (df['tournament'].isin(torneio_escolhido))]
    else:
        st.sidebar.warning("Coluna 'tournament' não encontrada no ficheiro.")
        df_filtrado = df[df['surface'] == superficie]
else:
    st.error("Coluna 'surface' não encontrada no ficheiro.")
    st.stop()

c1, c2 = st.columns(2)
nome_p1 = c1.selectbox("Favorito", jogadores, key="p1")
nome_p2 = c2.selectbox("Adversário", jogadores, key="p2")

# Mostrar Elos
elo1 = get_elo(nome_p1, superficie)
elo2 = get_elo(nome_p2, superficie)
c1.metric(f"Elo {nome_p1}", elo1)
c2.metric(f"Elo {nome_p2}", elo2)

st.divider()

# Simulação
if st.button("Executar Simulação de Monte Carlo"):
    totais, diffs = monte_carlo_simulation(elo1, elo2, sets_to_win=(sets_input//2 + 1))
    
    st.subheader("Resultados da Simulação (10.000 cenários)")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### Handicap")
        h = st.number_input("Handicap de Jogos", value=-2.5)
        # Nota: diffs aqui é diferença de sets, para Handicap de Jogos precisaríamos 
        # guardar a diferença de games na função de simulação.
        st.write("A simulação de Handicap de Jogos requer rastreio de games na função.")
        
    with col_b:
        st.markdown("### Mercado: Total de Jogos")
        linha = st.number_input("Linha de Jogos", value=21.5 if sets_input == 3 else 35.5)
        prob_over = np.mean(totais > linha)
        st.metric("Probabilidade Over", f"{prob_over:.1%}")
        st.metric("Média de Jogos Previstos", f"{np.mean(totais):.1f}")
        # Verificação de Debug (Coloca isto logo após o load_data())
df = load_data()

st.sidebar.write(f"Colunas detetadas: {df.columns.tolist()}") # Isto vai mostrar-te o que a app está a ver

if 'tournament' not in df.columns:
    st.error("ERRO: A coluna 'tournament' não existe no ficheiro carregado!")
    st.stop() # Para a execução aqui