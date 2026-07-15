import streamlit as st

st.set_page_config(page_title="QuantBet Pro", layout="wide")
st.title("🎾 QuantBet Pro: Analisador de Apostas ATP")

# Sidebar para escolha de mercado
mercado = st.sidebar.selectbox("Escolhe o Mercado", ["Vencedor (Match Winner)", "Handicap de Jogos"])

st.subheader(f"Análise de: {mercado}")

if mercado == "Vencedor (Match Winner)":
    odd = st.number_input("Odd do Favorito", min_value=1.01, value=1.45)
    prob = st.slider("Probabilidade do Modelo (0-1)", 0.0, 1.0, 0.74)
    banca = st.number_input("Tamanho da Banca (€)", value=1000.0)
    
    if st.button("Calcular Aposta"):
        edge = (prob * odd) - 1
        # Kelly Fracionado (0.25)
        kelly = ((odd - 1) * prob - (1 - prob)) / (odd - 1)
        stake = banca * (max(0, kelly) * 0.25)
        
        if edge > 0:
            st.success(f"VALOR DETETADO! Edge: {edge*100:.2f}%")
            st.metric("Stake Sugerida", f"€{stake:.2f}")
        else:
            st.error("Sem valor estatístico.")

elif mercado == "Handicap de Jogos":
    handicap_casa = st.number_input("Handicap da Casa (ex: -2.5)", value=-2.5)
    projecao = st.slider("Projeção do Modelo (Dif. de Jogos)", -6.0, 6.0, 0.0)
    
    if st.button("Analisar Handicap"):
        # Se a projeção for mais favorável que o handicap da casa, há edge
        diferenca = projecao - abs(handicap_casa)
        if diferenca > 0:
            st.success(f"Valor no Handicap! A tua projeção supera a casa em {diferenca:.1f} jogos.")
        else:
            st.info("O mercado está eficiente. Sem valor claro no Handicap.")