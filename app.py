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

# --- 2. INTERFACE E SIDEBAR (ESCOLHA DO CIRCUITO) ---
st.sidebar.header("1. Configurações Globais")
circuito = st.sidebar.radio("Circuito", ["ATP (Masculino)", "WTA (Feminino)"])

# Mostrar o aviso do modelo de forma discreta na barra lateral
if ml_model is None:
    st.sidebar.info("🤖 Motor: Elo Matemático (Fallback)")
else:
    st.sidebar.success("🤖 Motor: XGBoost Calibrado")

# Carregar os Elos específicos do circuito escolhido
df_elos = load_elos(circuito)

# --- 3. ENGENHARIA DE FEATURES "ON-THE-FLY" ---
def get_player_stats(nome_jogador, superficie):
    if not nome_jogador or pd.isna(nome_jogador):
        return {"elo": 1500, "hold_rate": 0.78 if circuito == "ATP (Masculino)" else 0.635, "fatigue": 0}
    
    # Obter Elo
    nome_norm = str(nome_jogador).lower().strip()
    match_elo = df_elos[df_elos['Player'].str.lower().str.strip() == nome_norm]
    elo = 1500
    if not match_elo.empty:
        col = {'Clay': 'cElo', 'Grass': 'gElo', 'Hard': 'hElo'}.get(superficie, 'Elo')
        elo = float(match_elo[col].values[0])
        
    # Extrair estatísticas recentes dos dados históricos (Match Charting Project)
    partidas_jogador = df[df['player'].str.lower().str.strip() == nome_norm]
    
    # Média de Hold % (padrão conforme circuito)
    hold_rate = 0.78 if circuito == "ATP (Masculino)" else 0.635
    if not partidas_jogador.empty and 'hold_percentage' in partidas_jogador.columns:
        hold_rate = partidas_jogador['hold_percentage'].mean()
        
    fatigue = 0
    if not partidas_jogador.empty and 'games_played_last_week' in partidas_jogador.columns:
        fatigue = partidas_jogador['games_played_last_week'].iloc[-1]
        
    return {"elo": elo, "hold_rate": hold_rate, "fatigue": fatigue}

# --- 4. SIMULAÇÃO MONTE CARLO INTEGRADA COM ML E CIRCUITO ---
def simulate_match_ml(stats_p1, stats_p2, sets_to_win, ml_model, circuito):
    if circuito == "WTA (Feminino)":
        base_hold = 0.635  # Média WTA
        limite_inf, limite_sup = 0.35, 0.85
    else:
        base_hold = 0.780  # Média ATP
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

# --- 5. PARSER DE TEXTO BRUTO DAS ODDS (REGEX) ---
def parse_bookmaker_text(text):
    """Lê o texto bruto da casa de apostas e extrai as linhas e odds de todos os mercados."""
    markets = {
        'match_winner': {},
        'total_games': {}, 
        'game_handicap': {},
        'p1_set': None,
        'p2_set': None
    }
    
    current_category = None
    
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        
        # Detetar categorias principais
        if line in ["Match winner", "Total games", "Game handicap", 
                    "Player 1 to win at least one set?", "Player 2 to win at least one set?"]:
            current_category = line
            continue
            
        # Ignorar categorias de Set individual ou Ímpar/Par para o EV do jogo completo
        if line.startswith("Set 1") or line.startswith("Set 2") or line == "Odd/Even games":
            current_category = "Ignored"
            continue
            
        if current_category == "Match winner":
            if line.startswith("Player 1:"): 
                markets['match_winner']['P1'] = float(line.split(":")[1].strip())
            elif line.startswith("Player 2:"): 
                markets['match_winner']['P2'] = float(line.split(":")[1].strip())
                
        elif current_category == "Total games":
            m = re.match(r"(Over|Under) (\d+\.\d+): (\d+\.\d+)", line)
            if m:
                ou, line_val, odd = m.groups()
                line_val, odd = float(line_val), float(odd)
                if line_val not in markets['total_games']: 
                    markets['total_games'][line_val] = {}
                markets['total_games'][line_val][ou] = odd
                
        elif current_category == "Game handicap":
            m = re.match(r"(Player 1|Player 2) \(([+-]?\d+\.\d+)\): (\d+\.\d+)", line)
            if m:
                player, hcp_val, odd = m.groups()
                hcp_val, odd = float(hcp_val), float(odd)
                # Guardamos a linha do Handicap de Jogo na ótica do Jogador 1 (Favorito)
                if player == "Player 1":
                    markets['game_handicap'][hcp_val] = odd
                    
        elif current_category == "Player 1 to win at least one set?":
            if line.startswith("Yes:"): 
                markets['p1_set'] = float(line.split(":")[1].strip())
                
        elif current_category == "Player 2 to win at least one set?":
            if line.startswith("Yes:"): 
                markets['p2_set'] = float(line.split(":")[1].strip())
                
    return markets

# --- 6. INTERFACE DO UTILIZADOR (ABAS) ---
superficie = st.sidebar.selectbox("Superfície", sorted(df['surface'].dropna().unique()))

sets_padrao = [3] if circuito == "WTA (Feminino)" else [3, 5]
sets_input = st.sidebar.radio("Sets do Encontro", sets_padrao)

# Filtra a lista dos jogadores apenas do circuito atualmente ativo
jogadores = sorted(df_elos['Player'].dropna().unique())

limite_ev = st.sidebar.slider("Limite de EV Aceitável (%)", min_value=1.0, max_value=15.0, value=5.0, step=0.5) / 100

# Construção das 3 Abas de Trabalho
tab1, tab2, tab3 = st.tabs(["🔍 Calculadora Manual", "🚀 CSV em Massa", "🤖 Auto-Scanner (Colar Texto)"])

# ==========================================
# ABA 1: CALCULADORA MANUAL
# ==========================================
with tab1:
    st.header("Análise de Partida Única")
    c1, c2 = st.columns(2)
    nome_p1 = c1.selectbox("Favorito (P1)", jogadores, key="p1")
    nome_p2 = c2.selectbox("Underdog (P2)", jogadores, key="p2")

    stats_p1 = get_player_stats(nome_p1, superficie)
    stats_p2 = get_player_stats(nome_p2, superficie)

    c1.metric(f"Elo {superficie} {nome_p1}", f"{stats_p1['elo']:.1f}")
    c2.metric(f"Elo {superficie} {nome_p2}", f"{stats_p2['elo']:.1f}")

    st.subheader("Odds Disponíveis na Casa de Apostas")
    col_o1, col_o2, col_o3, col_o4 = st.columns(4)
    odd_p1_casa = col_o1.number_input(f"Odd {nome_p1}", value=1.70, step=0.01, key="odd_manual_p1")
    odd_p2_casa = col_o2.number_input(f"Odd {nome_p2}", value=2.15, step=0.01, key="odd_manual_p2")
    odd_over_casa = col_o3.number_input("Odd Over Jogos", value=1.85, step=0.01, key="odd_manual_over")
    odd_hcp_casa = col_o4.number_input("Odd Handicap P1", value=1.90, step=0.01, key="odd_manual_hcp")

    st.divider()

    if st.button("Executar Sistema Quantitativo"):
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
            prob_h = np.mean(diffs > abs(h)) if h < 0 else np.mean(diffs < -h)
            
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
            
            st.write("### Detalhes de Auditoria de Odds")
            
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
    st.markdown("""
    **Instruções de Colagem:** Copia os dados (por exemplo do Excel) e cola na caixa abaixo. O formato deve ser separado por vírgulas, uma partida por linha:  
    `Jogador 1, Jogador 2, Odd P1, Odd P2, Linha Over, Odd Over, Linha Hcp, Odd Hcp`
    
    *Exemplo:* `Aryna Sabalenka, Iga Swiatek, 1.80, 2.05, 21.5, 1.90, -1.5, 1.85`  
    `Coco Gauff, Jessica Pegula, 1.55, 2.40, 20.5, 1.85, -2.5, 1.95`
    """)

    bloco_texto = st.text_area("Cola as Odds CSV aqui:", height=200, key="csv_area")

    if st.button("Varrer Mercado (Scan CSV)"):
        if not bloco_texto.strip():
            st.error("Cola alguns dados primeiro.")
        else:
            linhas = bloco_texto.strip().split('\n')
            todas_apostas_valor = []
            
            with st.spinner('A simular todos os encontros...'):
                for linha_texto in linhas:
                    try:
                        partes = [p.strip() for p in linha_texto.split(',')]
                        if len(partes) < 8:
                            st.warning(f"Linha ignorada por formatação incompleta: {linha_texto}")
                            continue
                            
                        j1, j2 = partes[0], partes[1]
                        odd_j1, odd_j2 = float(partes[2]), float(partes[3])
                        linha_ov, odd_ov = float(partes[4]), float(partes[5])
                        linha_hcp, odd_hcp = float(partes[6]), float(partes[7])
                        
                        s_p1 = get_player_stats(j1, superficie)
                        s_p2 = get_player_stats(j2, superficie)
                        
                        np.random.seed(42)
                        sims = [simulate_match_ml(s_p1, s_p2, (sets_input//2 + 1), ml_model, circuito) for _ in range(4000)]
                        
                        totais = np.array([s[0] for s in sims])
                        diffs = np.array([s[1] for s in sims])
                        p1_sets_ganhos = np.array([s[2] for s in sims])
                        p2_sets_ganhos = np.array([s[3] for s in sims])
                        
                        prob_p1 = np.mean(p1_sets_ganhos > p2_sets_ganhos)
                        prob_p2 = 1 - prob_p1
                        prob_over = np.mean(totais > linha_ov)
                        prob_hcp = np.mean(diffs > abs(linha_hcp)) if linha_hcp < 0 else np.mean(diffs < -linha_hcp)
                        
                        evs = {
                            f"Vitória {j1}": (odd_j1 * prob_p1) - 1,
                            f"Vitória {j2}": (odd_j2 * prob_p2) - 1,
                            f"Over {linha_ov}": (odd_ov * prob_over) - 1,
                            f"Hcp {j1} ({linha_hcp})": (odd_hcp * prob_hcp) - 1
                        }
                        
                        for mercado, ev in evs.items():
                            if ev >= limite_ev:
                                odd_casa = odd_j1 if "Vitória" in mercado and j1 in mercado else \
                                           odd_j2 if "Vitória" in mercado and j2 in mercado else \
                                           odd_ov if "Over" in mercado else odd_hcp
                                prob_mod = prob_p1 if "Vitória" in mercado and j1 in mercado else \
                                           prob_p2 if "Vitória" in mercado and j2 in mercado else \
                                           prob_over if "Over" in mercado else prob_hcp
                                           
                                todas_apostas_valor.append({
                                    "Jogo": f"{j1} vs {j2}",
                                    "Aposta": mercado,
                                    "EV": ev,
                                    "Odd Casa": odd_casa,
                                    "Odd Justa": 1 / prob_mod if prob_mod > 0 else 999.0
                                })
                    except Exception as e:
                        st.error(f"Erro ao processar a linha: '{linha_texto}'. Erro: {e}")
            
            if todas_apostas_valor:
                df_scanner = pd.DataFrame(todas_apostas_valor).sort_values(by="EV", ascending=False)
                st.success(f"✅ Varrimento concluído! Encontradas {len(df_scanner)} apostas de valor.")
                
                df_visual = df_scanner.copy()
                df_visual['EV'] = df_visual['EV'].apply(lambda x: f"+{x:.2%}")
                df_visual['Odd Casa'] = df_visual['Odd Casa'].apply(lambda x: f"{x:.2f}")
                df_visual['Odd Justa'] = df_visual['Odd Justa'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(df_visual, use_container_width=True)
            else:
                st.warning(f"