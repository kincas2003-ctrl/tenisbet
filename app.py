import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import joblib  # Para carregar o modelo de Machine Learning (XGBoost/LightGBM)
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

# --- 3. ENGENHARIA DE FEATURES "ON-THE-FLY" ---
def get_player_stats(nome_jogador, superficie, circuito):
    if not nome_jogador or pd.isna(nome_jogador):
        return {"elo": 1500, "hold_rate": 0.78 if circuito == "ATP (Masculino)" else 0.635, "fatigue": 0}
    
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
        
    return {"elo": elo, "hold_rate": hold_rate, "fatigue": fatigue}

# --- 4. SIMULAÇÃO MONTE CARLO ---
def simulate_match_ml(stats_p1, stats_p2, sets_to_win, ml_model, circuito):
    if circuito == "WTA (Feminino)":
        base_hold = 0.635  
        limite_inf, limite_sup = 0.35, 0.85
    else:
        base_hold = 0.780  
        limite_inf, limite_sup = 0.45, 0.95

    if ml_model is not None:
        elo_diff = stats_p1['elo'] - stats_p2['elo']
        hold_diff = stats_p1['hold_rate'] - stats_p2['hold_rate']
        fatigue_diff = stats_p1['fatigue'] - stats_p2['fatigue']
        
        features = pd.DataFrame([[elo_diff, hold_diff, fatigue_diff]], 
                                columns=['elo_diff', 'hold_diff_last5', 'fatigue_diff'])
        prob_p1_match = ml_model.predict_proba(features)[0][1]
    else:
        elo_diff = stats_p1['elo'] - stats_p2['elo']
        prob_p1_match = 1 / (1 + 10**(-elo_diff / 400))
    
    game_prob_shift = (prob_p1_match - 0.5) * 0.15
    
    p1_hold_prob = np.clip(base_hold + game_prob_shift, limite_inf, limite_sup)
    p2_hold_prob = np.clip(base_hold - game_prob_shift, limite_inf, limite_sup)
    
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
            
            if np.random.random() < prob_p1_wins_game: p1_g += 1
            else: p2_g += 1
            if p1_g == 7 or p2_g == 7: break
            
        total_g += (p1_g + p2_g)
        diff_g += (p1_g - p2_g)
        if p1_g > p2_g: p1_sets += 1
        else: p2_sets += 1
        
    return total_g, diff_g, p1_sets, p2_sets

# --- 5. PARSER DE TEXTO BRUTO DAS ODDS (REGEX ULTRA-ROBUSTO) ---
def parse_bookmaker_text(text):
    """Lê o texto bruto de qualquer casa de apostas e extrai as linhas e odds de forma imune a erros."""
    markets = {
        'match_winner': {},
        'total_games': {}, 
        'game_handicap': {'P1': {}, 'P2': {}},  # Suporta agora a separação por jogador
        'total_sets': {},
        'p1_set': None,
        'p2_set': None
    }
    
    current_category = "Ignored"
    
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        line_lower = line.lower()
        
        # Detetar separadores de odds comuns (—, : ou -)
        idx_em = line.rfind("—")
        idx_col = line.rfind(":")
        idx_hyp = line.rfind(" - ")
        
        last_sep = None
        last_idx = -1
        for sep, idx in [("—", idx_em), (":", idx_col), (" - ", idx_hyp)]:
            if idx > last_idx:
                last_idx = idx
                last_sep = sep
                
        is_odd_line = False
        key_part = ""
        odd = 0.0
        
        if last_idx != -1:
            key_part = line[:last_idx].strip()
            odd_part = line[last_idx + len(last_sep):].strip().replace(",", ".")
            try:
                odd = float(odd_part)
                is_odd_line = True
            except ValueError:
                pass
                
        # Se for um Cabeçalho de Categoria
        if not is_odd_line:
            if "set 1" in line_lower or "set 2" in line_lower or "odd/even" in line_lower or ("player" in line_lower and "total games" in line_lower):
                current_category = "Ignored"
            elif line_lower in ["match winner", "winner", "vencedor"]:
                current_category = "match_winner"
            elif line_lower in ["total games", "total de jogos"]:
                current_category = "total_games"
            elif line_lower in ["total sets", "total de sets"]:
                current_category = "total_sets"
            elif line_lower in ["game handicap", "handicap of games", "handicap de jogos"]:
                current_category = "game_handicap"
            elif "player 1 to win at least one set" in line_lower:
                current_category = "p1_set"
            elif "player 2 to win at least one set" in line_lower:
                current_category = "p2_set"
            else:
                current_category = "Ignored"
            continue
            
        if current_category == "Ignored": 
            continue
            
        # Limpar chave para lidar com formatos em português e decimais (ex: '2,5' -> '2.5')
        key_lower = key_part.lower().replace(",", ".")
        
        if current_category == "match_winner":
            if key_lower in ["player 1", "1"]: markets['match_winner']['P1'] = odd
            elif key_lower in ["player 2", "2"]: markets['match_winner']['P2'] = odd
                
        elif current_category in ["total_games", "total_sets"]:
            m = re.match(r"(over|under)\s+(\d+\.\d+)", key_lower)
            if m:
                ou = m.group(1).capitalize()
                line_val = float(m.group(2))
                
                # Salvaguarda: Se a linha de Over/Under for menor que 6.0, refere-se garantidamente a Sets!
                target_dict = 'total_sets' if line_val < 6.0 else 'total_games'
                if line_val not in markets[target_dict]: 
                    markets[target_dict][line_val] = {}
                markets[target_dict][line_val][ou] = odd
                
        elif current_category == "game_handicap":
            m = re.match(r"(?:player )?(1|2)\s*\(([+-]?\d+\.\d+)\)", key_lower)
            if m:
                player_num = m.group(1)
                hcp_val = float(m.group(2))
                if player_num == "1": 
                    markets['game_handicap']['P1'][hcp_val] = odd
                elif player_num == "2":
                    markets['game_handicap']['P2'][hcp_val] = odd
                    
        elif current_category == "p1_set":
            if key_lower == "yes": markets['p1_set'] = odd
                
        elif current_category == "p2_set":
            if key_lower == "yes": markets['p2_set'] = odd
            
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

    c1.metric(f"Elo {superficie} {nome_p1}", f"{stats_p1['elo']:.1f}")
    c2.metric(f"Elo {superficie} {nome_p2}", f"{stats_p2['elo']:.1f}")

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
            sims = [simulate_match_ml(stats_p1, stats_p2, (sets_input//2 + 1), ml_model, circuito) for _ in range(10000)]
            
            totais = np.array([s[0] for s in sims])
            diffs = np.array([s[1] for s in sims])
            p1_sets_ganhos = np.array([s[2] for s in sims])
            p2_sets_ganhos = np.array([s[3] for s in sims])
            
            prob_p1_win = np.mean(p1_sets_ganhos > p2_sets_ganhos)
            prob_p2_win = 1 - prob_p1_win
            
            linha = 21.5 if sets_input == 3 else 35.5
            prob_over = np.mean(totais > linha)
            
            h = -2.5
            prob_h = np.mean(diffs > -h)  # Ajuste universal de Handicap
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
    st.header("Scanner de Valor Múltiplo (Separado por vírgula)")
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
                        
                        np.random.seed(42)
                        sims = [simulate_match_ml(s_p1, s_p2, (sets_input//2 + 1), ml_model, circuito) for _ in range(4000)]
                        
                        totais = np.array([s[0] for s in sims])
                        diffs = np.array([s[1] for s in sims])
                        p1_sets_ganhos = np.array([s[2] for s in sims])
                        p2_sets_ganhos = np.array([s[3] for s in sims])
                        
                        prob_p1 = np.mean(p1_sets_ganhos > p2_sets_ganhos)
                        prob_p2 = 1 - prob_p1
                        prob_over = np.mean(totais > linha_ov)
                        prob_hcp = np.mean(diffs > -linha_hcp)  # Ajuste universal de Handicap
                        
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
    st.markdown("Seleciona os jogadores do texto colado para o modelo saber os respetivos Elos.")
    
    # Seleções de jogadores
    c_scan1, c_scan2 = st.columns(2)
    scan_p1 = c_scan1.selectbox("Favorito (Player 1 no texto)", jogadores, key="tab3_p1")
    scan_p2 = c_scan2.selectbox("Underdog (Player 2 no texto)", jogadores, key="tab3_p2")
    
    stats_scan_p1 = get_player_stats(scan_p1, superficie, circuito)
    stats_scan_p2 = get_player_stats(scan_p2, superficie, circuito)

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
                sims = [simulate_match_ml(stats_scan_p1, stats_scan_p2, (sets_input//2 + 1), ml_model, circuito) for _ in range(10000)]
                
                totais = np.array([s[0] for s in sims])
                diffs = np.array([s[1] for s in sims])
                p1_sets_ganhos = np.array([s[2] for s in sims])
                p2_sets_ganhos = np.array([s[3] for s in sims])
                
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
                
                # 2. Ganhar pelo menos 1 Set (+1.5 Set hcp)
                if mercados_extraidos['p1_set']:
                    prob = np.mean(p1_sets_ganhos >= 1)
                    lista_ev.append({"Mercado": f"{scan_p1} ganha +1 Set", "Prob": prob, "Odd": mercados_extraidos['p1_set'], "EV": (mercados_extraidos['p1_set'] * prob) - 1})
                if mercados_extraidos['p2_set']:
                    prob = np.mean(p2_sets_ganhos >= 1)
                    lista_ev.append({"Mercado": f"{scan_p2} ganha +1 Set", "Prob": prob, "Odd": mercados_extraidos['p2_set'], "EV": (mercados_extraidos['p2_set'] * prob) - 1})
                    
                # 3. Over/Under de Games (Jogos)
                for linha_g, odds_ou in mercados_extraidos['total_games'].items():
                    if 'Over' in odds_ou:
                        prob = np.mean(totais > linha_g)
                        lista_ev.append({"Mercado": f"Over {linha_g} Jogos", "Prob": prob, "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * prob) - 1})
                    if 'Under' in odds_ou:
                        prob = np.mean(totais < linha_g)
                        lista_ev.append({"Mercado": f"Under {linha_g} Jogos", "Prob": prob, "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * prob) - 1})
                
                # 4. Over/Under de Sets (Ex: Over 2.5 Sets)
                total_sets_jogados = p1_sets_ganhos + p2_sets_ganhos
                for linha_s, odds_ou in mercados_extraidos['total_sets'].items():
                    if 'Over' in odds_ou:
                        prob = np.mean(total_sets_jogados > linha_s)
                        lista_ev.append({"Mercado": f"Over {linha_s} Sets", "Prob": prob, "Odd": odds_ou['Over'], "EV": (odds_ou['Over'] * prob) - 1})
                    if 'Under' in odds_ou:
                        prob = np.mean(total_sets_jogados < linha_s)
                        lista_ev.append({"Mercado": f"Under {linha_s} Sets", "Prob": prob, "Odd": odds_ou['Under'], "EV": (odds_ou['Under'] * prob) - 1})
                        
                # 5. Handicaps do Player 1 (Favorito) - Positivos e Negativos
                for hcp_linha, odd in mercados_extraidos['game_handicap']['P1'].items():
                    prob = np.mean(diffs > -hcp_linha)  # Fórmula universal P1
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap {scan_p1} ({linha_str})", "Prob": prob, "Odd": odd, "EV": (odd * prob) - 1})

                # 6. Handicaps do Player 2 (Underdog) - Positivos e Negativos
                for hcp_linha, odd in mercados_extraidos['game_handicap']['P2'].items():
                    prob = np.mean(diffs < hcp_linha)  # Fórmula universal P2
                    linha_str = f"+{hcp_linha}" if hcp_linha > 0 else f"{hcp_linha}"
                    lista_ev.append({"Mercado": f"Handicap {scan_p2} ({linha_str})", "Prob": prob, "Odd": odd, "EV": (odd * prob) - 1})
                    
                # --- APRESENTAÇÃO E AUDITORIA COMPLETA ---
                if lista_ev:
                    df_scan = pd.DataFrame(lista_ev).sort_values(by="EV", ascending=False)
                    df_scan['Status'] = df_scan['EV'].apply(lambda x: "✅ Valor" if x >= limite_ev else "❌ Evitar")
                    
                    df_scan_valor = df_scan[df_scan['EV'] >= limite_ev].copy()
                    
                    if not df_scan_valor.empty:
                        df_scan_valor['Kelly_Score'] = df_scan_valor['EV'] / (df_scan_valor['Odd'] - 1)
                        
                        # Filtrar apenas apostas com odds consideradas aceitáveis (evita odds esmagadas)
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