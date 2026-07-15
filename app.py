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
    if not nome_jogador or pd.isna(nome_jogador):
        return 1500
    
    nome_norm = str(nome_jogador).lower().strip()
    match = df_elos[df_elos['Player'].str.lower().str.strip() == nome_norm]
    
    if match.empty: 
        return 1500
    
    col = {'Clay': 'cElo', 'Grass': 'gElo', 'Hard': 'hElo'}.get(superficie, 'Elo')
    return float(match[col].values[0])

def simulate_match(elo_p1, elo_p2, sets_to_win):
    elo_diff = elo_p1 - elo_p2
    
    # 100 pontos de Elo de diferença representam um aumento de ~3.3% na 
    # probabilidade de ganhar um game de serviço (fator de escala = 3000).
    game_prob_shift = elo_diff / 3000 
    
    # Probabilidade de cada jogador confirmar o seu próprio serviço (Hold %)
    # Base ATP é ~78% (0.78)
    p1_hold_prob = np.clip(0.78 + game_prob_shift, 0.40, 0.95)
    p2_hold_prob = np.clip(0.78 - game_prob_shift, 0.40, 0.95)
    
    p1_sets, p2_sets = 0, 0
    total_g, diff_g = 0, 0
    
    while p1_sets < sets_to_win and p2_sets < sets_to_win:
        p1_g, p2_g = 0, 0
        while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
            # Lógica de Serviço: Os jogadores alternam o serviço a cada game
            # P1 serve nos games pares (0, 2, 4...) do set
            if (p1_g + p2_g) % 2 == 0:
                prob_p1_wins_game = p1_hold_prob
            else:
                # Se o P2 serve, a probabilidade do P1 ganhar (fazer break) é o inverso
                prob_p1_wins_game = 1 - p2_hold_prob
            
            # Adiciona um pequeno ruído de 2% para simular a variância do encontro
            prob_p1_wins_game += np.random.normal(0, 0.02)
            
            if np.random.random() < prob_p1_wins_game:
                p1_g += 1
            else:
                p2_g += 1
                
            if p1_g == 7 or p2_g == 7: break
            
        total_g += (p1_g + p2_g)
        diff_g += (p1_g - p2_g)
        if p1_g > p2_g: p1_sets += 1
        else: p2_sets += 1
        
    return total_g, diff_g, (1 if p1_sets > p2_sets else 0)

# --- 3. INTERFACE ---
st.sidebar.header("Filtros")

# Filtro Superfície
superficies = sorted([s for s in df['surface'].dropna().unique()])
superficie = st.sidebar.selectbox("Superfície", superficies)

# Filtro apenas por superfície (Estabilidade total de nomes)
df_filtrado = df[df['surface'] == superficie]
jogadores = sorted(df_filtrado['player'].unique())

c1, c2 = st.columns(2)
nome_p1 = c1.selectbox("Favorito", jogadores, key="p1")
nome_p2 = c2.selectbox("Adversário", jogadores, key="p2")

# Seleção Formato (3 ou 5 sets)
sets_input = st.sidebar.radio("Formato do Encontro (Sets)", [3, 5])

# Mostrar Elos
elo1 = get_elo(nome_p1, superficie)
elo2 = get_elo(nome_p2, superficie)
c1.metric(f"Elo {superficie}", f"{elo1:.1f}")
c2.metric(f"Elo {superficie}", f"{elo2:.1f}")

st.divider()

# Simulação
if st.button("Executar Simulação de Monte Carlo"):
    if nome_p1 == nome_p2:
        st.error("Por favor, seleciona dois jogadores diferentes.")
    else:
        # Aumentamos para 5000 simulações para máxima precisão estatística rápida
        sims = [simulate_match(elo1, elo2, (sets_input//2 + 1)) for _ in range(5000)]
        totais = np.array([s[0] for s in sims])
        diffs = np.array([s[1] for s in sims])
        vitorias = np.array([s[2] for s in sims])
        
        st.subheader("Resultados (5.000 cenários)")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            prob_p1_win = np.mean(vitorias)
            col_a.metric("Probabilidade Vitória (P1)", f"{prob_p1_win:.1%}")
            if prob_p1_win > 0:
                col_a.write(f"Odd Justa: **{1/prob_p1_win:.2f}**")
        
        with col_b:
            h = st.number_input("Handicap de Jogos", value=-2.5)
            prob_h = np.mean(diffs > abs(h)) if h < 0 else np.mean(diffs < -h)
            col_b.metric("Probabilidade Handicap", f"{prob_h:.1%}")
            if prob_h > 0:
                col_b.write(f"Odd Justa H: **{1/prob_h:.2f}**")
                
        with col_c:
            linha = st.number_input("Linha de Jogos", value=21.5 if sets_input == 3 else 35.5)
            prob_over = np.mean(totais > linha)
            col_c.metric("Probabilidade Over", f"{prob_over:.1%}")
            col_c.write(f"Média de Jogos Previstos: **{np.mean(totais):.1f}**")