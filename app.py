import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Motor Quantitativo")

# --- CARREGAMENTO ---
@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos():
    return pd.read_csv("PlayerElo.csv") if os.path.exists("PlayerElo.csv") else pd.DataFrame()

df, df_elos = load_data(), load_elos()

# --- FUNÇÕES ---
def get_elo(nome_jogador, superficie):
    if not nome_jogador: return 1500
    nome_norm = str(nome_jogador).lower().strip()
    match = df_elos[df_elos['Player'].str.lower().str.strip() == nome_norm]
    if match.empty: return 1500
    col = {'Clay': 'cElo', 'Grass': 'gElo', 'Hard': 'hElo'}.get(superficie, 'Elo')
    return float(match[col].values[0])

def simulate_match(elo_p1, elo_p2, sets_to_win):
    # 1. Conversão de Elo para Probabilidade (Logistic Curve)
    diff = (elo_p1 - elo_p2) / 400
    prob_p1_base = 1 / (1 + 10**(-diff))
    
    # 2. CALIBRAÇÃO DE REALISMO (O segredo do mercado)
    # Em vez de 100% de precisão no Elo, introduzimos a 'Incerteza do Jogador'
    # O ténis tem uma volatilidade alta; um favorito nunca ganha mais de 85-90% das vezes.
    ruido = np.random.normal(0, 0.08)  # Desvio padrão de 8% nos pontos
    prob_p1 = np.clip(prob_p1_base + ruido, 0.05, 0.95)
    
    p1_sets, p2_sets = 0, 0
    total_g, diff_g = 0, 0
    
    while p1_sets < sets_to_win and p2_sets < sets_to_win:
        # Simulação de cada SET
        p1_g, p2_g = 0, 0
        while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
            # Probabilidade de ganhar um game no set
            # A vantagem do serviço é importante: ~55-60% para o servidor
            prob_game = prob_p1 + (0.05 if (p1_g + p2_g) % 2 == 0 else -0.05)
            if np.random.random() < prob_game:
                p1_g += 1
            else:
                p2_g += 1
            if p1_g == 7 or p2_g == 7: break
            
        total_g += (p1_g + p2_g)
        diff_g += (p1_g - p2_g)
        if p1_g > p2_g: p1_sets += 1
        else: p2_sets += 1
        
    return total_g, diff_g, (1 if p1_sets > p2_sets else 0)

# --- INTERFACE ---
superficie = st.sidebar.selectbox("Superfície", sorted(df['surface'].dropna().unique()))
df_filtrado = df[df['surface'] == superficie]
jogadores = sorted(df_filtrado['player'].unique())

c1, c2 = st.columns(2)
nome_p1 = c1.selectbox("Favorito", jogadores, key="p1")
nome_p2 = c2.selectbox("Adversário", jogadores, key="p2")
sets_input = st.sidebar.radio("Formato", [3, 5])

# Mostrar Elo Específico
elo1 = get_elo(nome_p1, superficie)
elo2 = get_elo(nome_p2, superficie)
c1.metric(f"Elo {superficie}", f"{elo1:.0f}")
c2.metric(f"Elo {superficie}", f"{elo2:.0f}")

if st.button("Simular Mercados"):
    if nome_p1 == nome_p2:
        st.error("Selecione jogadores diferentes.")
    else:
        # Aumentamos para 5000 simulações para maior precisão estatística
        sims = [simulate_match(elo1, elo2, (sets_input//2 + 1)) for _ in range(5000)]
        totais = np.array([s[0] for s in sims])
        diffs = np.array([s[1] for s in sims])
        vitorias = np.array([s[2] for s in sims])
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Prob. Vencedor (P1)", f"{np.mean(vitorias):.1%}")
        
        h = col2.number_input("Handicap de Jogos", value=-2.5)
        prob_h = np.mean(diffs > abs(h)) if h < 0 else np.mean(diffs < -h)
        col2.metric("Prob. Handicap", f"{prob_h:.1%}")
        
        linha = col3.number_input("Linha de Jogos", value=21.5 if sets_input==3 else 35.5)
        col3.metric("Prob. Over", f"{np.mean(totais > linha):.1%}")