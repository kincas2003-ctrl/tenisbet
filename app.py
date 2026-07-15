import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import joblib  
import re
import json
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(page_title="QuantBet OS", layout="wide")
st.title("🎾 QuantBet OS: Sistema Quantitativo Profissional")

# --- 1. MEMÓRIA E CARREGAMENTO ---
if 'agenda_p1' not in st.session_state: st.session_state['agenda_p1'] = None
if 'agenda_p2' not in st.session_state: st.session_state['agenda_p2'] = None

@st.cache_resource
def load_ml_model():
    return joblib.load("modelo_tenis_calibrado.pkl") if os.path.exists("modelo_tenis_calibrado.pkl") else None

@st.cache_data
def load_data():
    with zipfile.ZipFile("dados_resumidos.zip", 'r') as z:
        return pd.read_csv(z.open("dados_resumidos.csv"))

@st.cache_data
def load_elos(circuito):
    ficheiro = "EloRankP.csv" if circuito == "WTA (Feminino)" else "PlayerElo.csv"
    return pd.read_csv(ficheiro) if os.path.exists(ficheiro) else pd.DataFrame(columns=['Player', 'Elo', 'hElo', 'cElo', 'gElo'])

def load_agenda_json():
    if os.path.exists("agenda.json"):
        with open("agenda.json", "r", encoding="utf-8") as f: return json.load(f)
    return {}

df = load_data()
ml_model = load_ml_model()
df_elos = load_elos("ATP (Masculino)")

# --- 2. LÓGICA DE SIMULAÇÃO (MONTE CARLO) ---
def simulate_match_ml(stats_p1, stats_p2, sets_to_win, ml_model, circuito, h2h_stats):
    # (Inserir aqui a função simulate_match_ml completa da mensagem anterior)
    # Deixei o retorno simplificado abaixo para o código correr, mas deves manter a tua versão robusta
    return 22, 0, 1, 0, 6, 4, 6, 4, 5, 5 

# --- 3. PARSER BLINDADO ---
def parse_bookmaker_text(text, p1_name="", p2_name=""):
    # (Inserir aqui a função parse_bookmaker_text da mensagem anterior)
    return {'match_winner': {}, 'total_games': {}, 'game_handicap': {'P1': {}, 'P2': {}}, 'set_handicap': {'P1': {}, 'P2': {}}, 'total_sets': {}, 'p1_set': None, 'p2_set': None, 'p1_total_games': {}, 'p2_total_games': {}, 'set1_winner': {}, 'set1_total_games': {}, 'set1_handicap': {'P1': {}, 'P2': {}}, 'set2_winner': {}, 'set2_total_games': {}, 'set2_handicap': {'P1': {}, 'P2': {}}, 'total_aces': {}, 'p1_aces': {}, 'p2_aces': {}}

# --- 4. ABAS ---
tab_admin, tab4, tab3, tab1 = st.tabs(["⚙️ Admin", "📅 Agenda", "🤖 Auto-Scanner", "🔍 Calc"])

with tab_admin:
    st.header("⚙️ Painel de Admin")
    d = st.date_input("Data")
    txt = st.text_area("Formato: Torneio, P1, P2, Hora")
    if st.button("Salvar Agenda"):
        agenda = load_agenda_json()
        if str(d) not in agenda: agenda[str(d)] = []
        for l in txt.strip().split('\n'):
            p = [x.strip() for x in l.split(',')]
            if len(p)==4: agenda[str(d)].append({"torneio": p[0], "p1": p[1], "p2": p[2], "hora": p[3]})
        with open("agenda.json", "w", encoding="utf-8") as f: json.dump(agenda, f, ensure_ascii=False, indent=4)
        st.success("Guardado!")

with tab4:
    st.header("📅 Agenda")
    d_sel = st.date_input("Data para ver jogos", datetime.today().date())
    agenda = load_agenda_json()
    if str(d_sel) in agenda:
        for j in agenda[str(d_sel)]:
            if st.button(f"{j['hora']} - {j['p1']} vs {j['p2']}"):
                st.session_state['agenda_p1'] = j['p1']
                st.session_state['agenda_p2'] = j['p2']
                st.success("Jogo carregado!")

with tab3:
    st.header("🤖 Auto-Scanner")
    # (Inserir aqui a lógica da Aba 3 que criámos anteriormente)
    st.info("Scanner pronto para processar odds.")

with tab1:
    st.header("🔍 Calculadora Manual")
    # (Inserir aqui a lógica da Aba 1)