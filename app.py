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
