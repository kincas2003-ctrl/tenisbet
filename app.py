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
odd_minima_rec = st.sidebar.number_input("Odd Mínima Recomendada", value=1.50, step=0.05, help="O sistema vai ignorar odds abaixo deste valor ao escolher a Melhor Aposta.")

st.sidebar.header("⚙️ Condições & Ajustes de Jogo")
vel_campo = st.sidebar.selectbox(
    "Velocidade do Campo", 
    ["Médio (Hard Normal)", "Lento (Clay Lento)", "Médio-Lento (Clay Rápido / Hard Lento)", "Rápido (Grass / Hard Rápido)", "Ultra Rápido (Indoor Rápido)"]
)
ajuste_forma = st.sidebar.slider("Ajuste de Forma (Favorecer P1 vs P2)", -5, 5, 0, help="Desvia a forma recente. Positivo ajuda P1.")
ajuste_fadiga = st.sidebar.slider("Ajuste de Fadiga (Prejudicar P1 vs P2)", -5, 5, 0, help="Positivo cansa mais o P1, negativo cansa mais o P2.")

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
    else:
        base_hold = 0.780  
        limite_inf, limite_sup = 0.45, 0.95

    if vel_campo == "Lento (Clay Lento)": base_hold -= 0.04
    elif vel_campo == "Médio-Lento (Clay Rápido / Hard Lento)": base_hold -= 0.02
    elif vel_campo == "Rápido (Grass / Hard Rápido)": base_hold += 0.03
    elif vel_campo == "Ultra Rápido (Indoor Rápido)": base_hold += 0.05

    if ml_model is not None:
        elo_diff = stats_p1['elo'] - stats_p2['elo']
        hold_diff = stats_p1['hold_rate'] - stats_p2['hold_rate']
        fatigue_diff = stats_p1['fatigue'] - stats_p2['fatigue']
        features = pd.DataFrame([[elo_diff, hold_diff, fatigue_diff]], columns=['elo_diff', 'hold_diff_last5', 'fatigue_diff'])
        prob_p1_match = ml_model.predict_proba(features)[0][1]
    else:
        elo_diff = stats_p1['elo'] - stats_p2['elo']
        prob_p1_match = 1 / (1 + 10**(-elo_diff / 400))
    
    diff_forma = (stats_p1['recent_form'] - stats_p2['recent_form']) + (ajuste_forma * 0.10)
    prob_p1_match += (diff_forma * 0.05) 

    diff_fadiga = (stats_p1['fatigue'] - stats_p2['fatigue']) + (ajuste_fadiga * 10)
    prob_p1_match -= (diff_fadiga / 100.0) * 0.08 

    diff_h2h = h2h_stats[0] - h2h_stats[1]
    prob_p1_match += np.clip(diff_h2h * 0.015, -0.075, 0.075)

    prob_p1_match = np.clip(prob_p1_match, 0.05, 0.95)
    game_prob_shift = (prob_p1_match - 0.5) * 0.15
    p1_hold_prob = np.clip(base_hold + game_prob_shift, limite_inf, limite_sup)
    p2_hold_prob = np.clip(base_hold - game_prob_shift, limite_inf, limite_sup)
    
    p1_sets, p2_sets = 0, 0
    total_g, diff_g = 0, 0
    
    while p1_sets < sets_to_win and p2_sets < sets_to_win:
        p1_g, p2_g = 0, 0
        while (p1_g < 6 and p2_g < 6) or abs(p1_g - p2_g) < 2:
            if (p1_g + p2_g) % 2 == 0: prob_p1_wins_game = p1_hold_prob
            else: prob_p1_wins_game = 1 - p2_hold_prob
            prob_p1_wins_game += np.random.normal(0, 0.02)
            
            if np.random.random() < prob_p1_wins_game: p1_g += 1
            else: p2_g += 1
            if p1_g == 7 or p2_g == 7: break
            
        total_g += (p1_g + p2_g)
        diff_g += (p1_g - p2_g)
        if p1_g > p2_g: p1_sets += 1
        else: p2_sets += 1
        
    return total_g, diff_g, p1_sets, p2_sets

# --- 5. PARSER DE TEXTO BRUTO DAS ODDS (MÉTODO BLINDADO COM SUPORTE A SET HCP) ---
def parse_bookmaker_text(text):
    markets = {
        'match_winner': {}, 'total_games': {}, 
        'game_handicap': {'P1': {}, 'P2': {}}, 
        'set_handicap': {'P1': {}, 'P2': {}}, # NOVO DICIONÁRIO DE HANDICAP DE SETS
        'total_sets': {}, 'p1_set': None, 'p2_set': None,
        'p1_total_games': {}, 'p2_total_games': {}
    }
    current_category = "Ignored"
    
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        
        line_lower = line.lower()
        
        # Testar se a linha é uma Odd
        line_clean = line.replace("—", ":").replace(" - ", ":")
        is_odds_line = False
        key_part = ""
        odd_val = 0.0
        
        if ":" in line_clean:
            parts = line_clean.rsplit(":", 1)
            if len(parts) == 2:
                potential_odd = parts[1].strip().replace(",", ".")
                try:
                    odd_val = float(potential_odd)
                    key_part = parts[0].strip().lower()
                    is_odds_line = True
                except ValueError:
                    pass
        
        # Identificar Cabeçalho com Hierarquia Avançada
        if not is_odds_line:
            clean_header = line_lower.replace(":", "").strip()
            
            # SALVAGUARDA DE SET HANDICAP: Se contiver 'handicap' e 'set/sets', muda logo de categoria
            if "handicap" in clean_header and ("sets" in clean_header or "set" in clean_header):
                current_category = "set_handicap"
            elif "set 1" in clean_header or "set 2" in clean_header or "odd/even" in clean_header or "exact score" in clean_header or "correct score" in clean_header or "double result" in clean_header or "only one set" in clean_header:
                current_category = "Ignored"
            elif "player 1" in clean_header and "total games" in clean_header:
                current_category = "p1_total_games"
            elif "player 2" in clean_header and "total games" in clean_header:
                current_category = "p2_total_games"
            elif "total games" in clean_header or "total de jogos" in clean_header:
                current_category = "total_games"
            elif "total sets" in clean_header or "total de sets" in clean_header:
                current_category = "total_sets"
            elif "handicap" in clean_header:
                current_category = "game_handicap"
            elif "player 1 to win at least one set" in clean_header:
                current_category = "p1_set"
            elif "player 2 to win at least one set" in clean_header:
                current_category = "p2_set"
            elif "winner" in clean_header or "vencedor" in clean_header:
                current_category = "match_winner"
            else:
                current_category = "Ignored"
            continue
            
        if current_category == "Ignored":
            continue
            
        # Processamento das Odds
        try:
            if current_category == "match_winner":
                if key_part in ["player 1", "1"]: markets['match_winner']['P1'] = odd_val
                elif key_part in ["player 2", "2"]: markets['match_winner']['P2'] = odd_val
                
            elif current_category in ["total_games", "total_sets", "p1_total_games", "p2_total_games"]:
                m = re.match(r"(over|under)\s+(\d+\.\d+)", key_part)
                if m:
                    ou, val = m.group(1).capitalize(), float(m.group(2))
                    if current_category == "total_games":
                        target = 'total_sets' if val < 6.0 else 'total_games'
                    else:
                        target = current_category
                    if val not in markets[target]: markets[target][val] = {}
                    markets[target][val][ou] = odd_val
                    
            elif current_category in ["game_handicap", "set_handicap"]:
                m = re.match(r"(?:player )?(1|2)\s*\(?([+-]?\d+\.\d+)\)?", key_part)
                if m:
                    p_num, hcp = m.group(1), float(m.group(2))
                    if p_num == "1": markets[current_category]['P1'][hcp] = odd_val
                    else: markets[current_category]['P2'][hcp] = odd_val
                    
            elif current_category == "p1_set" and "yes" in key_part: markets['p1_set'] = odd_val
            elif current_category == "p2_set" and "yes" in key_part: markets['p2_set'] = odd_val
        except Exception:
            continue
            
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
                
            st.divider()
            
            col_sec1, col_sec2 = st.columns(2)
            col_sec1.metric("Média de Jogos Previstos", f"{np.mean(totais):.1f}")
            col_sec2.metric(f"Probabilidade {nome_p2} Ganhar +1 Set", f"{prob_p2_set:.1%}", help=f"Odd Justa: {odd_justa_p2_set:.2f}")
            
            df_formatado = df_resultados.copy()
            df_formatado['EV'] = df_formatado['EV'].apply(lambda x: f"{x:.2%}")
            df_formatado['Odd Casa'] = df_formatado['Odd Casa'].apply(lambda x: f"{x:.2f}")
            df_formatado['Prob'] = df_formatado['Prob'].apply(lambda x: f"{x:.2%}")
            st.dataframe(df_formatado)

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
                                prob_mod = prob_p1 if "Vitória" in mercado and j1 in mercado else \
                                           prob_p2 if "Vitória" in mercado and j2 in mercado else \
                                           prob_over if "Over" in mercado else prob_hcp
                                odd_casa = odd_j1 if "Vitória" in mercado and j1 in mercado else \
                                           odd_j2 if "Vitória" in mercado and j2 in mercado else \
                                           odd_ov if "Over" in mercado else odd_hcp
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
    st.markdown("Seleciona os jogadores do texto colado para o modelo processar os respetivos Elos.")
    
    c_scan1, c_scan2 = st.columns(2)
    scan_p1 = c_scan1.selectbox("Favorito (Player 1 no texto)", jogadores, key="tab3_p1")
    scan_p2 = c_scan2.selectbox("Underdog (Player 2 no texto)", jogadores, key="tab3_p2")
    
    stats_scan_p1 = get_player_stats(scan_p1, superficie, circuito)
    stats_scan_p2 = get_player_stats(scan_p2, superficie, circuito)
    h2h_scan_vals = calculate_h2h(scan_p1, scan_p2)

    st.markdown("**⚔️ Correção Manual de H2H** (Se a base de dados falhar torneios menores, ajusta aqui na hora)")
    col_h1, col_h2 = st.columns(2)
    h2h_p1_manual_t3 = col_h1.number_input(f"Vitórias reais de {scan_p1}", value=int(h2h_scan_vals[0]), min_value=0, step=1, key="h2h_t3_p1")
    h2h_p2_manual_t3 = col_h2.number_input(f"Vitórias reais de {scan_p2}", value=int(h2h_scan_vals[1]), min_value=0, step=1, key="h2h_t3_p2")

    texto_odds = st.text_area("Cola as Odds da Casa de Apostas:", height=300, key="raw_text_area",
                              placeholder="Match winner\n1 — 1.34\n2 — 3.20\n...\nTotal games\nOver 21.5 — 1.66\n...")
    
    if st.button("Analisar Todas as Odds (Scan Texto)", key="btn_tab3"):
        if scan_p1 == scan_p2:
            st.error("Seleciona jogadoras/jogadores diferentes.")
        elif not texto_odds.strip():
            st.warning("Cola o texto com as odds primeiro.")
        else:
            with st.spinner("A simular e a analisar todas as linhas detetadas..."):
                mercados_extraidos = parse_bookmaker_text(texto_odds)
                
                np.random.seed(42)
                sims = [simulate_match_ml(stats_scan_p1, stats_scan_p2, (sets_input//2 + 1), ml_model, circuito, (h2h_p1_manual_t3, h2h_p2_manual_t3)) for _ in range(10000)]
                
                totais = np.array([s[0] for s in sims])
                diffs = np.array([s[1] for s in sims])
                p1_sets_ganhos = np.array([s[2] for s in sims])
                p2_sets_ganhos = np.array([s[3] for s in sims])
                
                # Extração correta de jogos individuais
                p1_games_ganhos = (totais + diffs) / 2
                p2_games_ganhos = (totais - diffs) / 2
                
                prob_p1_win = np.mean(p1_sets_ganhos > p2_sets_ganhos)
                prob_p2_win = 1 - prob_p1_win
                
                lista_ev = []
                
                # 1. Vitória Seca
                if 'P1' in mercados_extraidos['match_winner']:
                    odd = mercados_extraidos['match_winner']['P1']
                    lista_ev.append({"Mercado": f"Vitória {scan_p1}", "Prob": prob_p1_win, "Odd": odd, "EV": (odd * prob_p1_win) - 1})
                if 'P2' in mercados_extraidos['match_winner']:
                    odd = mercados_extraidos['match_winner']['P2']
                    lista_ev.append({"Mercado": f"Vitória {scan_p2}", "Prob": prob_p2_win, "Odd": odd, "EV": (odd * prob_p2_win) - 1})
                
                # 2. Ganhar pelo menos 1 Set
                if mercados_extraidos['p1_set']:
                    prob = np.mean(p1_sets_ganhos >= 1)
                    lista_ev.append({"Mercado": f"{scan_p1} ganha +1 Set", "Prob": prob, "Odd": mercados_extraidos['p1_set'], "EV": (mercados_extraidos['p1_set'] * prob) - 1})
                if mercados_extraidos['p2_set']:
                    prob = np.mean(p2_sets_ganhos >= 1)
                    lista_ev.append({"Mercado": f"{scan_p2} ganha +1 Set", "Prob": prob, "Odd": mercados_extraidos['p2_set'], "EV": (mercados_extraidos['p2_set'] * prob) - 1})
                    
                # 3. Over/Under de Games (Jogos Totais)
                for linha_g, odds_ou in mercados_extraidos['total_games'].items():
                    if 'Over' in odds_ou:
                        prob = np.mean(totais > linha_g)
                        lista_ev.append({"Mercado": f"Over {linha_g} Jogos", "Prob": prob, "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * prob) - 1})
                    if 'Under' in odds_ou:
                        prob = np.mean(totais < linha_g)
                        lista_ev.append({"Mercado": f"Under {linha_g} Jogos", "Prob": prob, "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * prob) - 1})
                
                # 3.1 Over/Under de Jogos Individuais
                for linha_g, odds_ou in mercados_extraidos['p1_total_games'].items():
                    if 'Over' in odds_ou:
                        prob = np.mean(p1_games_ganhos > linha_g)
                        lista_ev.append({"Mercado": f"Over {linha_g} Jogos ({scan_p1})", "Prob": prob, "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * prob) - 1})
                    if 'Under' in odds_ou:
                        prob = np.mean(p1_games_ganhos < linha_g)
                        lista_ev.append({"Mercado": f"Under {linha_g} Jogos ({scan_p1})", "Prob": prob, "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * prob) - 1})
                        
                for line_g, odds_ou in mercados_extraidos['p2_total_games'].items():
                    if 'Over' in odds_ou:
                        prob = np.mean(p2_games_ganhos > line_g)
                        lista_ev.append({"Mercado": f"Over {line_g} Jogos ({scan_p2})", "Prob": prob, "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * prob) - 1})
                    if 'Under' in odds_ou:
                        prob = np.mean(p2_games_ganhos < line_g)
                        lista_ev.append({"Mercado": f"Under {line_g} Jogos ({scan_p2})", "Prob": prob, "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * prob) - 1})
                
                # 4. Over/Under de Sets Totais
                total_sets_jogados = p1_sets_ganhos + p2_sets_ganhos
                for linha_s, odds_ou in mercados_extraidos['total_sets'].items():
                    if 'Over' in odds_ou:
                        prob = np.mean(total_sets_jogados > linha_s)
                        lista_ev.append({"Mercado": f"Over {linha_s} Sets", "Prob": prob, "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * prob) - 1})
                    if 'Under' in odds_ou:
                        prob = np.mean(total_sets_jogados < linha_s)
                        lista_ev.append({"Mercado": f"Under {linha_s} Sets", "Prob": prob, "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * prob) - 1})
                        
                # 5. Handicaps de Games do Player 1 (Favorito)
                for hcp_linha, odd in mercados_extraidos['game_handicap']['P1'].items():
                    prob = np.mean(diffs > -hcp_linha)  
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap Games {scan_p1} ({linha_str})", "Prob": prob, "Odd": odd, "EV": (odd * prob) - 1})

                # 6. Handicaps de Games do Player 2 (Underdog)
                for hcp_linha, odd in mercados_extraidos['game_handicap']['P2'].items():
                    prob = np.mean(diffs < hcp_linha)  
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap Games {scan_p2} ({linha_str})", "Prob": prob, "Odd": odd, "EV": (odd * prob) - 1})
                    
                # 7. MATEMÁTICA CORRETA: Handicaps de SETS do Player 1 (Ex: -1.5 Sets significa ganhar por 2-0)
                for hcp_linha, odd in mercados_extraidos['set_handicap']['P1'].items():
                    prob_set = np.mean((p1_sets_ganhos - p2_sets_ganhos) > -hcp_linha)
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap Sets {scan_p1} ({linha_str})", "Prob": prob_set, "Odd": odd, "EV": (odd * prob_set) - 1})

                # 8. MATEMÁTICA CORRETA: Handicaps de SETS do Player 2 (Ex: +1.5 Sets significa ganhar pelo menos um set)
                for hcp_linha, odd in mercados_extraidos['set_handicap']['P2'].items():
                    prob_set = np.mean((p1_sets_ganhos - p2_sets_ganhos) < hcp_linha)
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap Sets {scan_p2} ({linha_str})", "Prob": prob_set, "Odd": odd, "EV": (odd * prob_set) - 1})
                    
                # --- APRESENTAÇÃO E AUDITORIA COMPLETA ---
                if lista_ev:
                    df_scan = pd.DataFrame(lista_ev).sort_values(by="EV", ascending=False)
                    df_scan['Status'] = df_scan['EV'].apply(lambda x: "✅ Valor" if x >= limite_ev else "❌ Evitar")
                    
                    df_scan_valor = df_scan[df_scan['EV'] >= limite_ev].copy()
                    
                    if not df_scan_valor.empty:
                        df_scan_valor['Kelly_Score'] = df_scan_valor['EV'] / (df_scan_valor['Odd'] - 1)
                        df_elegiveis = df_scan_valor[df_scan_valor['Odd'] >= odd_minima_rec]
                        
                        if not df_elegiveis.empty:
                            melhor_aposta = df_elegiveis.loc[df_elegiveis['Kelly_Score'].idxmax()]
                        else:
                            melhor_aposta = df_scan_valor.loc[df_scan_valor['Kelly_Score'].idxmax()]
                            st.info(f"Nota: Nenhuma odd ultrapassou a odd mínima de {odd_minima_rec:.2f}. Abaixo encontras a melhor opção de segurança.")
                        
                        sugestao_banca = np.clip(float(melhor_aposta['Kelly_Score'] * 10.0), 0.5, 3.5)
                        
                        st.markdown("---")
                        st.markdown("### 🏆 Aposta Recomendada (Melhor Risco/Benefício)")
                        st.success(
                            f"**Mercado:** {melhor_aposta['Mercado']}\n\n"
                            f"**Odd Oferecida:** {melhor_aposta['Odd']:.2f} | "
                            f"**Probabilidade Simulada:** {melhor_aposta['Prob']:.1%} | "
                            f"**Valor Esperado (EV):** +{melhor_aposta['EV']:.1%}\n\n"
                            f"⚖️ **Banca Sugerida:** **{sugestao_banca:.1%}** (Cálculo de Kelly Fracionário para maximização de capital)."
                        )
                        st.markdown("---")
                    else:
                        st.error(f"❌ Nenhuma linha neste jogo oferece rentabilidade (EV acima de +{limite_ev:.1%}). A aposta recomendada é **NÃO APOSTAR**.")
                        
                    st.markdown("#### 📋 Auditoria Completa ao Mercado")
                    st.markdown("Avaliação de todas as linhas e odds extraídas do teu texto:")
                    
                    df_visual_all = df_scan.copy()
                    df_visual_all['EV'] = df_visual_all['EV'].apply(lambda x: f"{'+' if x>0 else ''}{x:.2%}")
                    df_visual_all['Odd Justa'] = df_visual_all['Prob'].apply(lambda x: f"{1/x:.2f}" if x > 0 else "N/A")
                    df_visual_all['Prob'] = df_visual_all['Prob'].apply(lambda x: f"{x:.2%}")
                    df_visual_all['Odd'] = df_visual_all['Odd'].apply(lambda x: f"{x:.2f}")
                    
                    st.dataframe(df_visual_all[['Status', 'Mercado', 'Odd', 'Odd Justa', 'Prob', 'EV']], use_container_width=True)
                else:
                    st.error("Não foi possível identificar mercados válidos no texto colado. Verifica a formatação.")