import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import joblib  
import re

# Configuração da página
st.set_page_config(page_title="QuantBet OS", layout="wide")
st.title("🎾 QuantBet OS: Sistema Quantitativo ATP & WTA")

# --- 1. CARREGAMENTO DE MODELOS E DADOS ---
@st.cache_resource
def load_ml_model():
    if os.path.exists("modelo_tenis_calibrado.pkl"):
        return joblib.load("modelo_tenis_calibrado.pkl")
    return None

@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos(circuito):
    if circuito == "WTA (Feminino)":
        ficheiro = "EloRankP.csv"
    else:
        ficheiro = "PlayerElo.csv"
        
    if os.path.exists(ficheiro):
        return pd.read_csv(ficheiro)
    return pd.DataFrame(columns=['Player', 'Elo', 'hElo', 'cElo', 'gElo'])

df = load_data()
ml_model = load_ml_model()

# --- 2. INTERFACE GLOBAL (BARRA LATERAL) ---
st.sidebar.header("1. Configurações Globais")
circuito = st.sidebar.radio("Circuito", ["ATP (Masculino)", "WTA (Feminino)"])

if ml_model is None:
    st.sidebar.info("🤖 Motor: Elo Matemático (Fallback)")
else:
    st.sidebar.success("🤖 Motor: XGBoost Calibrado")

df_elos = load_elos(circuito)
superficie = st.sidebar.selectbox("Superfície", sorted(df['surface'].dropna().unique()))
sets_padrao = [3] if circuito == "WTA (Feminino)" else [3, 5]
sets_input = st.sidebar.radio("Sets do Encontro", sets_padrao)

jogadores = sorted(df_elos['Player'].dropna().unique())

st.sidebar.header("2. Filtros de Valor")
limite_ev = st.sidebar.slider("Limite de EV Aceitável (%)", min_value=1.0, max_value=15.0, value=5.0, step=0.5) / 100
odd_minima_rec = st.sidebar.number_input("Odd Mínima Recomendada", value=1.50, step=0.05, help="O sistema ignora odds abaixo deste valor na recomendação.")

st.sidebar.header("⚙️ Condições & Ajustes de Jogo")
vel_campo = st.sidebar.selectbox(
    "Velocidade do Campo", 
    ["Médio (Hard Normal)", "Lento (Clay Lento)", "Médio-Lento (Clay Rápido / Hard Lento)", "Rápido (Grass / Hard Rápido)", "Ultra Rápido (Indoor Rápido)"]
)
ajuste_forma = st.sidebar.slider("Ajuste de Forma (Favorecer P1 vs P2)", -5, 5, 0)
ajuste_fadiga = st.sidebar.slider("Ajuste de Fadiga (Prejudicar P1 vs P2)", -5, 5, 0)

# --- 3. ENGENHARIA DE FEATURES ---
def get_player_stats(nome_jogador, superficie, circuito):
    if not nome_jogador or pd.isna(nome_jogador):
        return {"elo": 1500, "hold_rate": 0.78 if circuito == "ATP (Masculino)" else 0.635, "fatigue": 0, "recent_form": 0.5}
    
    nome_norm = str(nome_jogador).lower().strip()
    match_elo = df_elos[df_elos['Player'].str.lower().str.strip() == nome_norm]
    elo = 1500
    if not match_elo.empty:
        col = {'Clay': 'cElo', 'Grass': 'gElo', 'Hard': 'hElo'}.get(superficie, 'Elo')
        elo = float(match_elo[col].values[0])
        
    partidas_jogador = df[df['player'].str.lower().str.strip() == nome_norm]
    hold_rate = 0.78 if circuito == "ATP (Masculino)" else 0.635
    if not partidas_jogador.empty and 'hold_percentage' in partidas_jogador.columns:
        hold_rate = partidas_jogador['hold_percentage'].mean()
        
    fatigue = 0
    if not partidas_jogador.empty and 'games_played_last_week' in partidas_jogador.columns:
        fatigue = partidas_jogador['games_played_last_week'].iloc[-1]
        
    recent_form = 0.5
    if not partidas_jogador.empty and 'winner' in df.columns:
        recentes = partidas_jogador.tail(5)
        vitorias = sum(str(row['winner']).lower().strip() == nome_norm for _, row in recentes.iterrows())
        recent_form = vitorias / len(recentes) if len(recentes) > 0 else 0.5
        
    return {"elo": elo, "hold_rate": hold_rate, "fatigue": fatigue, "recent_form": recent_form}

def calculate_h2h(p1, p2):
    p1_norm = str(p1).lower().strip()
    p2_norm = str(p2).lower().strip()
    p1_wins, p2_wins = 0, 0
    
    if 'opponent' in df.columns and 'winner' in df.columns:
        m1 = df[(df['player'].str.lower().str.strip() == p1_norm) & (df['opponent'].str.lower().str.strip() == p2_norm)]
        m2 = df[(df['player'].str.lower().str.strip() == p2_norm) & (df['opponent'].str.lower().str.strip() == p1_norm)]
        for _, row in m1.iterrows():
            if str(row['winner']).lower().strip() == p1_norm: p1_wins += 1
            else: p2_wins += 1
        for _, row in m2.iterrows():
            if str(row['winner']).lower().strip() == p2_norm: p2_wins += 1
            else: p1_wins += 1
    return p1_wins, p2_wins

# --- 4. SIMULAÇÃO MONTE CARLO ---
def simulate_match_ml(stats_p1, stats_p2, sets_to_win, ml_model, circuito, h2h_stats):
    if circuito == "WTA (Feminino)":
        base_hold = 0.635  
        limite_inf, limite_sup = 0.35, 0.85
        base_aces = 0.25
    else:
        base_hold = 0.780  
        limite_inf, limite_sup = 0.45, 0.95
        base_aces = 0.55

    ace_multiplier = 1.0
    if vel_campo == "Lento (Clay Lento)": base_hold -= 0.04; ace_multiplier = 0.6
    elif vel_campo == "Médio-Lento (Clay Rápido / Hard Lento)": base_hold -= 0.02; ace_multiplier = 0.8
    elif vel_campo == "Rápido (Grass / Hard Rápido)": base_hold += 0.03; ace_multiplier = 1.3
    elif vel_campo == "Ultra Rápido (Indoor Rápido)": base_hold += 0.05; ace_multiplier = 1.5

    rate_aces_p1 = max(0.05, (base_aces + (stats_p1['hold_rate'] - base_hold) * 2)) * ace_multiplier
    rate_aces_p2 = max(0.05, (base_aces + (stats_p2['hold_rate'] - base_hold) * 2)) * ace_multiplier

    if ml_model is not None:
        elo_diff = stats_p1['elo'] - stats_p2['elo']
        hold_diff = stats_p1['hold_rate'] - stats_p2['hold_rate']
        fatigue_diff = stats_p1['fatigue'] - stats_p2['fatigue']
        features = pd.DataFrame([[elo_diff, hold_diff, fatigue_diff]], columns=['elo_diff', 'hold_diff_last5', 'fatigue_diff'])
        prob_p1_match = ml_model.predict_proba(features)[0][1]
    else:
        elo_diff = stats_p1['elo'] - stats_p2['elo']
        prob_p1_match = 1 / (1 + 10**(-elo_diff / 400))
    
    prob_p1_match += ((stats_p1['recent_form'] - stats_p2['recent_form']) + (ajuste_forma * 0.10)) * 0.05 
    prob_p1_match -= ((stats_p1['fatigue'] - stats_p2['fatigue']) + (ajuste_fadiga * 10)) / 100.0 * 0.08 
    prob_p1_match += np.clip((h2h_stats[0] - h2h_stats[1]) * 0.015, -0.075, 0.075)

    prob_p1_match = np.clip(prob_p1_match, 0.05, 0.95)
    game_prob_shift = (prob_p1_match - 0.5) * 0.15
    p1_hold_prob = np.clip(base_hold + game_prob_shift, limite_inf, limite_sup)
    p2_hold_prob = np.clip(base_hold - game_prob_shift, limite_inf, limite_sup)
    
    p1_sets, p2_sets = 0, 0
    total_g, diff_g = 0, 0
    aces_p1, aces_p2 = 0, 0
    
    set1_p1_g, set1_p2_g = 0, 0
    set2_p1_g, set2_p2_g = 0, 0
    
    while p1_sets < sets_to_win and p2_sets < sets_to_win:
        p1_g, p2_g = 0, 0
        while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
            is_p1_serve = (p1_g + p2_g) % 2 == 0
            if is_p1_serve:
                prob_p1_wins_game = p1_hold_prob
                aces_p1 += np.random.poisson(rate_aces_p1) 
            else:
                prob_p1_wins_game = 1 - p2_hold_prob
                aces_p2 += np.random.poisson(rate_aces_p2) 
                
            prob_p1_wins_game += np.random.normal(0, 0.02)
            
            if np.random.random() < prob_p1_wins_game: p1_g += 1
            else: p2_g += 1
            if p1_g == 7 or p2_g == 7: break
            
        total_sets_played = p1_sets + p2_sets
        if total_sets_played == 0: set1_p1_g, set1_p2_g = p1_g, p2_g
        elif total_sets_played == 1: set2_p1_g, set2_p2_g = p1_g, p2_g
            
        total_g += (p1_g + p2_g)
        diff_g += (p1_g - p2_g)
        if p1_g > p2_g: p1_sets += 1
        else: p2_sets += 1
        
    return total_g, diff_g, p1_sets, p2_sets, set1_p1_g, set1_p2_g, set2_p1_g, set2_p2_g, aces_p1, aces_p2

# --- 5. PARSER MULTILINGUE DE ODDS ---
def parse_bookmaker_text(text, p1_name="", p2_name=""):
    markets = {
        'match_winner': {}, 'total_games': {}, 
        'game_handicap': {'P1': {}, 'P2': {}}, 'set_handicap': {'P1': {}, 'P2': {}},
        'total_sets': {}, 'p1_set': None, 'p2_set': None,
        'p1_total_games': {}, 'p2_total_games': {},
        
        'set1_winner': {}, 'set1_total_games': {}, 'set1_handicap': {'P1': {}, 'P2': {}},
        'set2_winner': {}, 'set2_total_games': {}, 'set2_handicap': {'P1': {}, 'P2': {}},
        'total_aces': {}, 'p1_aces': {}, 'p2_aces': {}
    }
    
    p1_tokens = [t.lower() for t in str(p1_name).replace(",", " ").split() if len(t) > 2]
    p2_tokens = [t.lower() for t in str(p2_name).replace(",", " ").split() if len(t) > 2]
    
    current_category = "Ignored"
    
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        
        clean_line = line.replace("—", ":").replace(" - ", ":")
        is_odds_line = False
        key_part, odd_val = "", 0.0
        
        if ":" in clean_line:
            parts = clean_line.rsplit(":", 1)
            try:
                odd_val = float(parts[1].strip().replace(",", "."))
                key_part = parts[0].strip().lower()
                is_odds_line = True
            except ValueError: pass
        
        if not is_odds_line:
            header = line.lower()
            
            if any(x in header for x in ["par/ímpar", "odd/even", "exato", "exact", "correct", "duplo", "double result", "only one set", "apenas um set", "vencedor e", "winner and", "vencedor &", "winner &", "tie-break", "tie break"]): 
                current_category = "Ignored"
                
            elif any(x in header for x in ["aces", "ases"]):
                if any(x in header for x in p1_tokens + ["player 1", "jogador 1", "casa"]): current_category = "p1_aces"
                elif any(x in header for x in p2_tokens + ["player 2", "jogador 2", "fora"]): current_category = "p2_aces"
                else: current_category = "total_aces"
                
            elif any(x in header for x in ["set 1", "1º set", "1o set", "primeiro set", "1st set"]):
                if any(x in header for x in p1_tokens + p2_tokens + ["player 1", "player 2", "jogador", "casa", "fora"]): current_category = "Ignored"
                elif "handicap" in header: current_category = "set1_handicap"
                elif any(x in header for x in ["total", "jogos", "games"]): current_category = "set1_total_games"
                elif any(x in header for x in ["winner", "vencedor"]): current_category = "set1_winner"
                else: current_category = "Ignored"
                
            elif any(x in header for x in ["set 2", "2º set", "2o set", "segundo set", "2nd set"]):
                if any(x in header for x in p1_tokens + p2_tokens + ["player 1", "player 2", "jogador", "casa", "fora"]): current_category = "Ignored"
                elif "handicap" in header: current_category = "set2_handicap"
                elif any(x in header for x in ["total", "jogos", "games"]): current_category = "set2_total_games"
                elif any(x in header for x in ["winner", "vencedor"]): current_category = "set2_winner"
                else: current_category = "Ignored"
                
            elif any(x in header for x in p1_tokens + ["player 1", "jogador 1", "casa"]) and any(x in header for x in ["total", "jogos", "games"]):
                current_category = "p1_total_games"
            elif any(x in header for x in p2_tokens + ["player 2", "jogador 2", "fora"]) and any(x in header for x in ["total", "jogos", "games"]):
                current_category = "p2_total_games"
            elif any(x in header for x in ["total jogos", "total games", "total de jogos", "jogos no encontro"]):
                current_category = "total_games"
            elif "total sets" in header or "total de sets" in header:
                current_category = "total_sets"
            elif "handicap" in header:
                current_category = "set_handicap" if "sets" in header else "game_handicap"
            elif any(x in header for x in ["winner", "vencedor", "resultado final", "match winner", "1x2"]):
                current_category = "match_winner"
            else:
                current_category = "Ignored"
            continue
            
        if current_category == "Ignored": continue
        
        try:
            if any(x in key_part for x in ["&", " and ", " e ", "+", "over", "under", "mais", "menos", "-", "/", "2:0", "0:2", "2:1", "1:2"]) and current_category in ["match_winner", "set1_winner", "set2_winner"]: continue
            if ("&" in key_part or " and " in key_part or " e " in key_part) and current_category in ["total_games", "total_sets", "p1_total_games", "p2_total_games", "set1_total_games", "set2_total_games"]: continue

            if current_category in ["match_winner", "set1_winner", "set2_winner"]:
                is_p1 = any(x in key_part for x in p1_tokens + ["1", "casa", "home", "jogador 1"]) or key_part == "1"
                is_p2 = any(x in key_part for x in p2_tokens + ["2", "fora", "away", "jogador 2"]) or key_part == "2"
                if is_p1 and 'P1' not in markets[current_category]: markets[current_category]['P1'] = odd_val
                elif is_p2 and 'P2' not in markets[current_category]: markets[current_category]['P2'] = odd_val
            
            elif current_category in ["total_games", "p1_total_games", "p2_total_games", "total_sets", "set1_total_games", "set2_total_games", "total_aces", "p1_aces", "p2_aces"]:
                m = re.search(r"(over|under|mais de|menos de|mais|menos|acima|abaixo)\s*(\d+\.\d+)", key_part)
                if m:
                    ou = "Over" if m.group(1) in ["over", "mais de", "mais", "acima"] else "Under"
                    val = float(m.group(2))
                    target_cat = current_category
                    if target_cat == "total_games" and val < 6.0: target_cat = "total_sets"
                    if val not in markets[target_cat]: markets[target_cat][val] = {}
                    markets[target_cat][val][ou] = odd_val
                    
            elif current_category in ["game_handicap", "set_handicap", "set1_handicap", "set2_handicap"]:
                m = re.search(r"([+-]?\d+\.\d+)", key_part)
                if m:
                    hcp = float(m.group(1))
                    if any(x in key_part for x in p1_tokens + ["1", "casa", "jogador 1"]): markets[current_category]['P1'][hcp] = odd_val
                    else: markets[current_category]['P2'][hcp] = odd_val
        except: continue
            
    return markets

# --- 6. ABAS DE TRABALHO ---
tab1, tab2, tab3 = st.tabs(["🔍 Calculadora Manual", "🚀 CSV em Massa", "🤖 Auto-Scanner (Colar Texto)"])

# ==========================================
# ABA 1: CALCULADORA MANUAL
# ==========================================
with tab1:
    st.header("Análise de Partida Única")
    c1, c2 = st.columns(2)
    nome_p1 = c1.selectbox("Favorito (P1)", jogadores, key="tab1_p1")
    nome_p2 = c2.selectbox("Underdog (P2)", jogadores, key="tab1_p2")

    stats_p1 = get_player_stats(nome_p1, superficie, circuito)
    stats_p2 = get_player_stats(nome_p2, superficie, circuito)
    h2h_p1_bd, h2h_p2_bd = calculate_h2h(nome_p1, nome_p2)

    c1.metric(f"Elo {superficie} {nome_p1}", f"{stats_p1['elo']:.1f}")
    c1.markdown(f"📈 Forma Recente: `{stats_p1['recent_form']:.0%}` | 💤 Fadiga: `{stats_p1['fatigue']}`")
    
    c2.metric(f"Elo {superficie} {nome_p2}", f"{stats_p2['elo']:.1f}")
    c2.markdown(f"📈 Forma Recente: `{stats_p2['recent_form']:.0%}` | 💤 Fadiga: `{stats_p2['fatigue']}`")
    
    st.markdown("**⚔️ Correção Manual de H2H** (BD Automática preenchida)")
    ch1, ch2 = st.columns(2)
    h2h_p1_manual_t1 = ch1.number_input(f"Vitórias reais de {nome_p1}", value=int(h2h_p1_bd), min_value=0, step=1, key="h2h_t1_p1")
    h2h_p2_manual_t1 = ch2.number_input(f"Vitórias reais de {nome_p2}", value=int(h2h_p2_bd), min_value=0, step=1, key="h2h_t1_p2")

    st.subheader("Odds Disponíveis na Casa de Apostas")
    col_o1, col_o2, col_o3, col_o4 = st.columns(4)
    odd_p1_casa = col_o1.number_input(f"Odd {nome_p1}", value=1.70, step=0.01, key="odd_manual_p1")
    odd_p2_casa = col_o2.number_input(f"Odd {nome_p2}", value=2.15, step=0.01, key="odd_manual_p2")
    odd_over_casa = col_o3.number_input("Odd Over Jogos", value=1.85, step=0.01, key="odd_manual_over")
    odd_hcp_casa = col_o4.number_input("Odd Handicap P1", value=1.90, step=0.01, key="odd_manual_hcp")

    st.divider()

    if st.button("Executar Sistema Quantitativo", key="btn_tab1"):
        if nome_p1 == nome_p2:
            st.error("Seleciona jogadoras/jogadores diferentes.")
        else:
            np.random.seed(42)
            sims = [simulate_match_ml(stats_p1, stats_p2, (sets_input//2 + 1), ml_model, circuito, (h2h_p1_manual_t1, h2h_p2_manual_t1)) for _ in range(10000)]
            
            totais = np.array([s[0] for s in sims])
            diffs = np.array([s[1] for s in sims])
            p1_sets_ganhos = np.array([s[2] for s in sims])
            p2_sets_ganhos = np.array([s[3] for s in sims])
            
            prob_p1_win = np.mean(p1_sets_ganhos > p2_sets_ganhos)
            prob_p2_win = 1 - prob_p1_win
            
            linha = 21.5 if sets_input == 3 else 35.5
            prob_over = np.mean(totais > linha)
            
            h = -2.5
            prob_h = np.mean(diffs > -h)  
            prob_p2_set = np.mean(p2_sets_ganhos >= 1)
            odd_justa_p2_set = 1 / prob_p2_set if prob_p2_set > 0 else 999.0
            
            ev_p1 = (odd_p1_casa * prob_p1_win) - 1
            ev_p2 = (odd_p2_casa * prob_p2_win) - 1
            ev_over = (odd_over_casa * prob_over) - 1
            ev_hcp = (odd_hcp_casa * prob_h) - 1
            
            dados_mercados = [
                {"Mercado": f"Vitória {nome_p1}", "EV": ev_p1, "Odd Casa": odd_p1_casa, "Prob": prob_p1_win},
                {"Mercado": f"Vitória {nome_p2}", "EV": ev_p2, "Odd Casa": odd_p2_casa, "Prob": prob_p2_win},
                {"Mercado": f"Over {linha} Jogos", "EV": ev_over, "Odd Casa": odd_over_casa, "Prob": prob_over},
                {"Mercado": f"Handicap P1 ({h})", "EV": ev_hcp, "Odd Casa": odd_hcp_casa, "Prob": prob_h}
            ]
            
            df_resultados = pd.DataFrame(dados_mercados).sort_values(by="EV", ascending=False)
            st.subheader("📊 Relatório de Oportunidades")
            oportunidades_validas = df_resultados[df_resultados['EV'] >= limite_ev]
            
            if not oportunidades_validas.empty:
                for idx, op in oportunidades_validas.iterrows():
                    st.success(f"🎯 **{op['Mercado']}** | EV: **+{op['EV']:.2%}** | Odd Justa: **{1/op['Prob']:.2f}** (Casa: {op['Odd Casa']:.2f})")
            else:
                st.warning(f"❌ Nenhuma aposta encontrou valor suficiente acima de +{limite_ev:.1%}.")

# ==========================================
# ABA 2: CSV EM MASSA
# ==========================================
with tab2:
    st.header("Scanner de Valor Múltiplo")
    st.markdown("`Jogador 1, Jogador 2, Odd P1, Odd P2, Linha Over, Odd Over, Linha Hcp, Odd Hcp`")
    bloco_texto_csv = st.text_area("Cola as Odds CSV aqui:", height=200, key="csv_area")

    if st.button("Varrer Mercado (Scan CSV)", key="btn_tab2"):
        if not bloco_texto_csv.strip():
            st.error("Cola alguns dados primeiro.")
        else:
            linhas = bloco_texto_csv.strip().split('\n')
            todas_apostas_valor = []
            with st.spinner('A simular todos os encontros...'):
                for linha_texto in linhas:
                    try:
                        partes = [p.strip() for p in linha_texto.split(',')]
                        if len(partes) < 8: continue
                        j1, j2 = partes[0], partes[1]
                        odd_j1, odd_j2 = float(partes[2]), float(partes[3])
                        linha_ov, odd_ov = float(partes[4]), float(partes[5])
                        linha_hcp, odd_hcp = float(partes[6]), float(partes[7])
                        
                        s_p1 = get_player_stats(j1, superficie, circuito)
                        s_p2 = get_player_stats(j2, superficie, circuito)
                        h2h_vals = calculate_h2h(j1, j2)
                        
                        np.random.seed(42)
                        sims = [simulate_match_ml(s_p1, s_p2, (sets_input//2 + 1), ml_model, circuito, h2h_vals) for _ in range(4000)]
                        
                        totais = np.array([s[0] for s in sims])
                        diffs = np.array([s[1] for s in sims])
                        p1_sets_ganhos = np.array([s[2] for s in sims])
                        p2_sets_ganhos = np.array([s[3] for s in sims])
                        
                        prob_p1 = np.mean(p1_sets_ganhos > p2_sets_ganhos)
                        prob_p2 = 1 - prob_p1
                        prob_over = np.mean(totais > linha_ov)
                        prob_hcp = np.mean(diffs > -linha_hcp)  
                        
                        evs = {
                            f"Vitória {j1}": (odd_j1 * prob_p1) - 1,
                            f"Vitória {j2}": (odd_j2 * prob_p2) - 1,
                            f"Over {linha_ov}": (odd_ov * prob_over) - 1,
                            f"Hcp {j1} ({linha_hcp})": (odd_hcp * prob_hcp) - 1
                        }
                        for mercado, ev in evs.items():
                            if ev >= limite_ev:
                                prob_mod = prob_p1 if "Vitória" in mercado and j1 in mercado else prob_p2 if "Vitória" in mercado and j2 in mercado else prob_over if "Over" in mercado else prob_hcp
                                odd_casa = odd_j1 if "Vitória" in mercado and j1 in mercado else odd_j2 if "Vitória" in mercado and j2 in mercado else odd_ov if "Over" in mercado else odd_hcp
                                todas_apostas_valor.append({
                                    "Jogo": f"{j1} vs {j2}", "Aposta": mercado, "EV": ev,
                                    "Odd Casa": odd_casa, "Odd Justa": 1 / prob_mod if prob_mod > 0 else 999.0
                                })
                    except Exception:
                        continue
            
            if todas_apostas_valor:
                df_scanner = pd.DataFrame(todas_apostas_valor).sort_values(by="EV", ascending=False)
                st.success(f"✅ Varrimento concluído! Encontradas {len(df_scanner)} apostas de valor.")
                df_visual = df_scanner.copy()
                df_visual['EV'] = df_visual['EV'].apply(lambda x: f"+{x:.2%}")
                df_visual['Odd Casa'] = df_visual['Odd Casa'].apply(lambda x: f"{x:.2f}")
                df_visual['Odd Justa'] = df_visual['Odd Justa'].apply(lambda x: f"{x:.2f}")
                st.dataframe(df_visual, use_container_width=True)
            else:
                st.warning("O scanner não encontrou nenhuma aposta de valor.")

# ==========================================
# ABA 3: AUTO-SCANNER DE TEXTO BRUTO
# ==========================================
with tab3:
    st.header("Auto-Scanner Inteligente (Copiar & Colar)")
    st.markdown("Seleciona os jogadores do texto colado para o modelo processar os respetivos Elos e calcular os mercados.")
    
    c_scan1, c_scan2 = st.columns(2)
    scan_p1 = c_scan1.selectbox("Favorito (Player 1 no texto)", jogadores, key="tab3_p1")
    scan_p2 = c_scan2.selectbox("Underdog (Player 2 no texto)", jogadores, key="tab3_p2")
    
    stats_scan_p1 = get_player_stats(scan_p1, superficie, circuito)
    stats_scan_p2 = get_player_stats(scan_p2, superficie, circuito)
    h2h_scan_vals = calculate_h2h(scan_p1, scan_p2)

    st.markdown("**⚔️ Correção Manual de H2H** (Ajusta vitórias caso falte info no CSV)")
    col_h1, col_h2 = st.columns(2)
    h2h_p1_manual_t3 = col_h1.number_input(f"Vitórias reais de {scan_p1}", value=int(h2h_scan_vals[0]), min_value=0, step=1, key="h2h_t3_p1")
    h2h_p2_manual_t3 = col_h2.number_input(f"Vitórias reais de {scan_p2}", value=int(h2h_scan_vals[1]), min_value=0, step=1, key="h2h_t3_p2")

    texto_odds = st.text_area("Cola as Odds da Casa de Apostas:", height=300, key="raw_text_area")
    
    if st.button("Analisar Todas as Odds (Scan Texto)", key="btn_tab3"):
        if scan_p1 == scan_p2:
            st.error("Seleciona jogadoras/jogadores diferentes.")
        elif not texto_odds.strip():
            st.warning("Cola o texto com as odds primeiro.")
        else:
            with st.spinner("A simular e a analisar todas as linhas detetadas..."):
                mercados_extraidos = parse_bookmaker_text(texto_odds, scan_p1, scan_p2)
                
                np.random.seed(42)
                sims = [simulate_match_ml(stats_scan_p1, stats_scan_p2, (sets_input//2 + 1), ml_model, circuito, (h2h_p1_manual_t3, h2h_p2_manual_t3)) for _ in range(10000)]
                
                totais = np.array([s[0] for s in sims])
                diffs = np.array([s[1] for s in sims])
                p1_sets_ganhos = np.array([s[2] for s in sims])
                p2_sets_ganhos = np.array([s[3] for s in sims])
                s1_p1 = np.array([s[4] for s in sims])
                s1_p2 = np.array([s[5] for s in sims])
                s2_p1 = np.array([s[6] for s in sims])
                s2_p2 = np.array([s[7] for s in sims])
                aces_p1 = np.array([s[8] for s in sims])
                aces_p2 = np.array([s[9] for s in sims])
                total_aces = aces_p1 + aces_p2
                
                p1_games_ganhos = (totais + diffs) / 2
                p2_games_ganhos = (totais - diffs) / 2
                
                lista_ev = []
                
                # --- MERCADOS DO ENCONTRO ---
                prob_p1_win = np.mean(p1_sets_ganhos > p2_sets_ganhos)
                if 'P1' in mercados_extraidos['match_winner']:
                    odd = mercados_extraidos['match_winner']['P1']
                    lista_ev.append({"Mercado": f"Vitória {scan_p1}", "Prob": prob_p1_win, "Odd": odd, "EV": (odd * prob_p1_win) - 1})
                if 'P2' in mercados_extraidos['match_winner']:
                    odd = mercados_extraidos['match_winner']['P2']
                    lista_ev.append({"Mercado": f"Vitória {scan_p2}", "Prob": 1-prob_p1_win, "Odd": odd, "EV": (odd * (1-prob_p1_win)) - 1})
                
                for linha_g, odds_ou in mercados_extraidos['total_games'].items():
                    if 'Over' in odds_ou: lista_ev.append({"Mercado": f"Over {linha_g} Jogos", "Prob": np.mean(totais > linha_g), "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * np.mean(totais > linha_g)) - 1})
                    if 'Under' in odds_ou: lista_ev.append({"Mercado": f"Under {linha_g} Jogos", "Prob": np.mean(totais < linha_g), "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * np.mean(totais < linha_g)) - 1})

                for linha_g, odds_ou in mercados_extraidos['p1_total_games'].items():
                    if 'Over' in odds_ou: lista_ev.append({"Mercado": f"Over {linha_g} Jogos ({scan_p1})", "Prob": np.mean(p1_games_ganhos > linha_g), "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * np.mean(p1_games_ganhos > linha_g)) - 1})
                    if 'Under' in odds_ou: lista_ev.append({"Mercado": f"Under {linha_g} Jogos ({scan_p1})", "Prob": np.mean(p1_games_ganhos < linha_g), "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * np.mean(p1_games_ganhos < linha_g)) - 1})

                for linha_g, odds_ou in mercados_extraidos['p2_total_games'].items():
                    if 'Over' in odds_ou: lista_ev.append({"Mercado": f"Over {linha_g} Jogos ({scan_p2})", "Prob": np.mean(p2_games_ganhos > linha_g), "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * np.mean(p2_games_ganhos > linha_g)) - 1})
                    if 'Under' in odds_ou: lista_ev.append({"Mercado": f"Under {linha_g} Jogos ({scan_p2})", "Prob": np.mean(p2_games_ganhos < linha_g), "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * np.mean(p2_games_ganhos < linha_g)) - 1})

                for linha_s, odds_ou in mercados_extraidos['total_sets'].items():
                    if 'Over' in odds_ou: lista_ev.append({"Mercado": f"Over {linha_s} Sets", "Prob": np.mean((p1_sets_ganhos + p2_sets_ganhos) > linha_s), "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * np.mean((p1_sets_ganhos + p2_sets_ganhos) > linha_s)) - 1})
                    if 'Under' in odds_ou: lista_ev.append({"Mercado": f"Under {linha_s} Sets", "Prob": np.mean((p1_sets_ganhos + p2_sets_ganhos) < linha_s), "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * np.mean((p1_sets_ganhos + p2_sets_ganhos) < linha_s)) - 1})

                for hcp_linha, odd in mercados_extraidos['game_handicap']['P1'].items():
                    prob = np.mean(diffs > -hcp_linha)  
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap Games {scan_p1} ({linha_str})", "Prob": prob, "Odd": odd, "EV": (odd * prob) - 1})

                for hcp_linha, odd in mercados_extraidos['game_handicap']['P2'].items():
                    prob = np.mean(diffs < hcp_linha)  
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap Games {scan_p2} ({linha_str})", "Prob": prob, "Odd": odd, "EV": (odd * prob) - 1})

                for hcp_linha, odd in mercados_extraidos['set_handicap']['P1'].items():
                    prob_set = np.mean((p1_sets_ganhos - p2_sets_ganhos) > -hcp_linha)
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap Sets {scan_p1} ({linha_str})", "Prob": prob_set, "Odd": odd, "EV": (odd * prob_set) - 1})

                for hcp_linha, odd in mercados_extraidos['set_handicap']['P2'].items():
                    prob_set = np.mean((p1_sets_ganhos - p2_sets_ganhos) < hcp_linha)
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap Sets {scan_p2} ({linha_str})", "Prob": prob_set, "Odd": odd, "EV": (odd * prob_set) - 1})

                # --- MERCADOS DE ASES ---
                for linha_g, odds_ou in mercados_extraidos['total_aces'].items():
                    if 'Over' in odds_ou: lista_ev.append({"Mercado": f"Over {linha_g} Ases", "Prob": np.mean(total_aces > linha_g), "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * np.mean(total_aces > linha_g)) - 1})
                    if 'Under' in odds_ou: lista_ev.append({"Mercado": f"Under {linha_g} Ases", "Prob": np.mean(total_aces < linha_g), "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * np.mean(total_aces < linha_g)) - 1})
                
                for linha_g, odds_ou in mercados_extraidos['p1_aces'].items():
                    if 'Over' in odds_ou: lista_ev.append({"Mercado": f"Over {linha_g} Ases ({scan_p1})", "Prob": np.mean(aces_p1 > linha_g), "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * np.mean(aces_p1 > linha_g)) - 1})
                    if 'Under' in odds_ou: lista_ev.append({"Mercado": f"Under {linha_g} Ases ({scan_p1})", "Prob": np.mean(aces_p1 < linha_g), "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * np.mean(aces_p1 < linha_g)) - 1})

                for linha_g, odds_ou in mercados_extraidos['p2_aces'].items():
                    if 'Over' in odds_ou: lista_ev.append({"Mercado": f"Over {linha_g} Ases ({scan_p2})", "Prob": np.mean(aces_p2 > linha_g), "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * np.mean(aces_p2 > linha_g)) - 1})
                    if 'Under' in odds_ou: lista_ev.append({"Mercado": f"Under {linha_g} Ases ({scan_p2})", "Prob": np.mean(aces_p2 < linha_g), "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * np.mean(aces_p2 < linha_g)) - 1})

                # --- MERCADOS DO 1º SET ---
                prob_s1_p1 = np.mean(s1_p1 > s1_p2)
                if 'P1' in mercados_extraidos['set1_winner']: lista_ev.append({"Mercado": f"Vence 1º Set {scan_p1}", "Prob": prob_s1_p1, "Odd": mercados_extraidos['set1_winner']['P1'], "EV": (mercados_extraidos['set1_winner']['P1'] * prob_s1_p1) - 1})
                if 'P2' in mercados_extraidos['set1_winner']: lista_ev.append({"Mercado": f"Vence 1º Set {scan_p2}", "Prob": 1-prob_s1_p1, "Odd": mercados_extraidos['set1_winner']['P2'], "EV": (mercados_extraidos['set1_winner']['P2'] * (1-prob_s1_p1)) - 1})
                
                for linha_g, odds_ou in mercados_extraidos['set1_total_games'].items():
                    if 'Over' in odds_ou: lista_ev.append({"Mercado": f"Over {linha_g} Jogos no 1º Set", "Prob": np.mean((s1_p1 + s1_p2) > linha_g), "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * np.mean((s1_p1 + s1_p2) > linha_g)) - 1})
                    if 'Under' in odds_ou: lista_ev.append({"Mercado": f"Under {linha_g} Jogos no 1º Set", "Prob": np.mean((s1_p1 + s1_p2) < linha_g), "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * np.mean((s1_p1 + s1_p2) < linha_g)) - 1})

                for hcp_linha, odd in mercados_extraidos['set1_handicap']['P1'].items():
                    prob = np.mean((s1_p1 - s1_p2) > -hcp_linha)  
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap 1º Set {scan_p1} ({linha_str})", "Prob": prob, "Odd": odd, "EV": (odd * prob) - 1})
                    
                for hcp_linha, odd in mercados_extraidos['set1_handicap']['P2'].items():
                    prob = np.mean((s1_p1 - s1_p2) < hcp_linha)  
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap 1º Set {scan_p2} ({linha_str})", "Prob": prob, "Odd": odd, "EV": (odd * prob) - 1})

                # --- APRESENTAÇÃO ---
                if lista_ev:
                    df_scan = pd.DataFrame(lista_ev).sort_values(by="EV", ascending=False)
                    df_scan['Status'] = df_scan['EV'].apply(lambda x: "✅ Valor" if x >= limite_ev else "❌ Evitar")
                    
                    df_scan_valor = df_scan[df_scan['EV'] >= limite_ev].copy()
                    
                    if not df_scan_valor.empty:
                        df_scan_valor['Kelly_Score'] = df_scan_valor['EV'] / (df_scan_valor['Odd'] - 1)
                        df_elegiveis = df_scan_valor[df_scan_valor['Odd'] >= odd_minima_rec]
                        
                        if not df_elegiveis.empty: melhor_aposta = df_elegiveis.loc[df_elegiveis['Kelly_Score'].idxmax()]
                        else: melhor_aposta = df_scan_valor.loc[df_scan_valor['Kelly_Score'].idxmax()]
                        
                        sugestao_banca_decimal = np.clip(float(melhor_aposta['Kelly_Score'] * 0.10), 0.005, 0.035)
                        
                        st.markdown("---")
                        st.markdown("### 🏆 Aposta Recomendada (Melhor Risco/Benefício)")
                        st.success(f"**Mercado:** {melhor_aposta['Mercado']}\n\n**Odd Oferecida:** {melhor_aposta['Odd']:.2f} | **Probabilidade:** {melhor_aposta['Prob']:.1%} | **EV:** +{melhor_aposta['EV']:.1%}\n\n⚖️ **Banca Sugerida:** **{sugestao_banca_decimal:.1%}**.")
                        st.markdown("---")
                        
                    st.markdown("#### 📋 Auditoria Completa ao Mercado")
                    df_visual_all = df_scan.copy()
                    df_visual_all['EV'] = df_visual_all['EV'].apply(lambda x: f"{'+' if x>0 else ''}{x:.2%}")
                    df_visual_all['Odd Justa'] = df_visual_all['Prob'].apply(lambda x: f"{1/x:.2f}" if x > 0 else "N/A")
                    df_visual_all['Prob'] = df_visual_all['Prob'].apply(lambda x: f"{x:.2%}")
                    df_visual_all['Odd'] = df_visual_all['Odd'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(df_visual_all[['Status', 'Mercado', 'Odd', 'Odd Justa', 'Prob', 'EV']], use_container_width=True)
                else:
                    st.error("Não foram encontrados mercados com valor. Verifica a formatação do texto.")