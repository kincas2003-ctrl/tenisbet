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
    game_prob_shift = elo_diff / 3000 
    
    p1_hold_prob = np.clip(0.78 + game_prob_shift, 0.40, 0.95)
    p2_hold_prob = np.clip(0.78 - game_prob_shift, 0.40, 0.95)
    
    p1_sets, p2_sets = 0, 0
    total_g, diff_g = 0, 0
    
    while p1_sets < sets_to_win and p2_sets < sets_to_win:
        p1_g, p2_g = 0, 0
        while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
            if (p1_g + p2_g) % 2 == 0:
                prob_p1_wins_game = p1_hold_prob
            else:
                prob_p1_wins_game = 1 - p2_hold_prob
            
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
        
    return total_g, diff_g, p1_sets, p2_sets

# --- 3. INTERFACE ---
st.sidebar.header("1. Configurações")
superficie = st.sidebar.selectbox("Superfície", sorted(df['surface'].dropna().unique()))
sets_input = st.sidebar.radio("Formato do Encontro (Sets)", [3, 5])

# Filtro por superfície
df_filtrado = df[df['surface'] == superficie]
jogadores = sorted(df_filtrado['player'].unique())

c1, c2 = st.columns(2)
nome_p1 = c1.selectbox("Favorito (P1)", jogadores, key="p1")
nome_p2 = c2.selectbox("Underdog (P2)", jogadores, key="p2")

# Mostrar Elos
elo1 = get_elo(nome_p1, superficie)
elo2 = get_elo(nome_p2, superficie)
c1.metric(f"Elo {superficie}", f"{elo1:.1f}")
c2.metric(f"Elo {superficie}", f"{elo2:.1f}")

# --- INPUTS DAS ODDS DA CASA DE APOSTAS ---
st.sidebar.header("2. Odds da Casa de Apostas")
odd_p1_casa = st.sidebar.number_input(f"Odd Vitória {nome_p1}", value=1.50, step=0.01)
odd_p2_casa = st.sidebar.number_input(f"Odd Vitória {nome_p2}", value=2.50, step=0.01)
odd_over_casa = st.sidebar.number_input("Odd Over Jogos", value=1.85, step=0.01)
odd_hcp_casa = st.sidebar.number_input("Odd Handicap P1", value=1.90, step=0.01)

st.divider()

# Simulação
if st.button("Executar Simulação e Procurar Valor (+EV)"):
    if nome_p1 == nome_p2:
        st.error("Por favor, seleciona dois jogadores diferentes.")
    else:
        sims = [simulate_match(elo1, elo2, (sets_input//2 + 1)) for _ in range(5000)]
        totais = np.array([s[0] for s in sims])
        diffs = np.array([s[1] for s in sims])
        p1_sets_ganhos = np.array([s[2] for s in sims])
        p2_sets_ganhos = np.array([s[3] for s in sims])
        
        # Probabilidades do Modelo
        prob_p1_win = np.mean(p1_sets_ganhos > p2_sets_ganhos)
        prob_p2_win = 1 - prob_p1_win
        
        linha = 21.5 if sets_input == 3 else 35.5
        prob_over = np.mean(totais > linha)
        
        h = -2.5
        prob_h = np.mean(diffs > abs(h)) if h < 0 else np.mean(diffs < -h)
        
        # --- CÁLCULO DO VALOR ESPERADO (EV) ---
        ev_p1 = (odd_p1_casa * prob_p1_win) - 1
        ev_p2 = (odd_p2_casa * prob_p2_win) - 1
        ev_over = (odd_over_casa * prob_over) - 1
        ev_hcp = (odd_hcp_casa * prob_h) - 1
        
        # Dicionário com todas as apostas analisadas
        analise_apostas = [
            {"Aposta": f"Vitória {nome_p1}", "EV": ev_p1, "Odd Casa": odd_p1_casa, "Prob Modelo": prob_p1_win},
            {"Aposta": f"Vitória {nome_p2}", "EV": ev_p2, "Odd Casa": odd_p2_casa, "Prob Modelo": prob_p2_win},
            {"Aposta": f"Over {linha} Jogos", "EV": ev_over, "Odd Casa": odd_over_casa, "Prob Modelo": prob_over},
            {"Aposta": f"Handicap P1 ({h})", "EV": ev_hcp, "Odd Casa": odd_hcp_casa, "Prob Modelo": prob_h}
        ]
        
        df_ev = pd.DataFrame(analise_apostas)
        df_ev = df_ev.sort_values(by="EV", ascending=False)
        
        st.subheader("📊 Relatório de Valor Esperado (Value Bets)")
        
        # Mostrar a melhor aposta
        melhor_aposta = df_ev.iloc[0]
        if melhor_aposta["EV"] > 0:
            st.success(f"🔥 **Aposta Recomendada:** {melhor_aposta['Aposta']} | EV: **+{melhor_aposta['EV']:.1%}** | Odd Justa: **{1/melhor_aposta['Prob Modelo']:.2f}** (Odd Casa: {melhor_aposta['Odd Casa']})")
        else:
            st.warning("⚠️ **Sem Aposta de Valor:** Nenhuma das odds oferecidas pela casa tem valor matemático positivo em relação ao teu modelo.")
            
        st.divider()
        
        # Mostrar tabela de comparação completa
        st.write("### Comparação de Mercados")
        for index, row in df_ev.iterrows():
            cor_ev = "green" if row["EV"] > 0 else "red"
            st.markdown(
                f"- **{row['Aposta']}**:"
                f" Odd Casa: `{row['Odd Casa']:.2f}` |"
                f" Probabilidade Modelo: `{row['Prob Modelo']:.1%}` |"
                f" EV: <span style='color:{cor_ev}; font-weight:bold;'>{row['EV']:.1%}</span>", 
                unsafe_allow_html=True
            )