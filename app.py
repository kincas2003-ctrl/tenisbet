import streamlit as st
st.title("QuantBet: Calculadora ATP")
odd = st.number_input("Odd do Favorito", min_value=1.0)
prob = st.slider("Probabilidade Estimada", 0.0, 1.0, 0.70)
banca = st.number_input("Banca (€)", value=1000.0)
if st.button("Calcular Veredito"):
    edge = (prob * odd) - 1
    if edge > 0:
        st.success(f"Aposta com Valor! Edge: {edge*100:.2f}%")
    else:
        st.error("Sem valor estatístico.")
import streamlit as st

st.title("QuantBet Pro: Analisador Multi-Mercado")

# 1. Seletor de Mercado
mercado = st.selectbox("Escolhe o Mercado", 
                       ["Vencedor (Match Winner)", "Handicap de Jogos", "Total de Sets"])

# 2. Inputs variáveis
odd = st.number_input("Odd do mercado", min_value=1.01, value=1.45)
prob = st.slider("Probabilidade do teu modelo", 0.0, 1.0, 0.70)

# 3. Lógica de cálculo específica
if st.button("Calcular Veredito"):
    edge = (prob * odd) - 1
    
    st.subheader(f"Análise para: {mercado}")
    
    if edge > 0.05:
        st.success(f"VALOR DETETADO! Edge: {edge*100:.2f}%")
        st.write(">> Recomendação: Aposta com unidade padrão.")
    elif edge > 0:
        st.warning(f"Margem mínima. Edge: {edge*100:.2f}%")
        st.write(">> Recomendação: Aposta reduzida.")
    else:
        st.error(f"Sem valor. Edge: {edge*100:.2f}%")
        
