
import streamlit as st
import pandas as pd
import numpy as np
import re

# --- LÓGICA DO SCRIPT REFINADA ---

@st.cache_data
def carregar_e_processar_dados(caminho_arquivo):
    try:
        df = pd.read_excel(caminho_arquivo)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        return None

    motores_padrao = np.array([
        15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300,
        350, 400, 450, 500, 550, 600
    ])
    def encontrar_motor_final(potencia_real):
        candidatos = motores_padrao[motores_padrao >= potencia_real]
        return candidatos.min() if len(candidatos) > 0 else np.nan

    df["Motor (HP)"] = df["Potência (HP)"].apply(encontrar_motor_final)

    def extrair_rotor_num(rotor_str):
        match = re.match(r"(\d+)(?:\s*\((\d+)°\))?", str(rotor_str))
        if match:
            base = int(match.group(1))
            grau = int(match.group(2)) if match.group(2) else 0
            return base + grau / 100
        return np.nan

    df["RotorNum"] = df["Rotor"].apply(extrair_rotor_num)

    df["rotor_min_modelo"] = df.groupby("Modelo")["RotorNum"].transform("min")
    df["rotor_max_modelo"] = df.groupby("Modelo")["RotorNum"].transform("max")
    df["pressao_max_modelo"] = df.groupby("Modelo")["Pressão (mca)"].transform("max")

    intervalos_vazao = df.groupby(["Modelo", "Rotor"])["Vazão (m³/h)"].agg(["min", "max"]).reset_index()
    df = pd.merge(df, intervalos_vazao, on=["Modelo", "Rotor"], how="left", suffixes=("", "_range"))
    df["vazao_centro"] = (df["min"] + df["max"]) / 2
    df["erro_relativo"] = np.abs(df["Vazão (m³/h)"] - df["vazao_centro"]) / (df["max"] - df["min"]).replace(0, np.nan)

    return df

def filtrar_e_classificar(df, vazao, pressao, top_n=5):
    if df is None:
        return pd.DataFrame()

    condicoes = [
        df['RotorNum'] == df['rotor_max_modelo'],
        df['RotorNum'] == df['rotor_min_modelo']
    ]
    margens_cima = [df['pressao_max_modelo'] * 0.015, df['pressao_max_modelo'] * 0.05]
    margens_baixo = [df['pressao_max_modelo'] * 0.05, df['pressao_max_modelo'] * 0.015]
    df['margem_cima'] = np.select(condicoes, margens_cima, default=df['pressao_max_modelo'] * 0.05)
    df['margem_baixo'] = np.select(condicoes, margens_baixo, default=df['pressao_max_modelo'] * 0.05)

    pressao_min_aceita = pressao - df['margem_baixo']
    pressao_max_aceita = pressao + df['margem_cima']
    df_filtrado = df[
        (df["Vazão (m³/h)"] == vazao) &
        (df["Pressão (mca)"] >= pressao_min_aceita) &
        (df["Pressão (mca)"] <= pressao_max_aceita)
    ].copy()

    if df_filtrado.empty:
        return pd.DataFrame()

    df_filtrado["erro_pressao"] = abs(df_filtrado["Pressão (mca)"] - pressao)

    df_resultado = df_filtrado.sort_values(
        by=["Motor (HP)", "Rendimento (%)", "erro_relativo", "erro_pressao"],
        ascending=[True, False, True, True]
    )

    return df_resultado[['Modelo', 'Rotor', 'Vazão (m³/h)', 'Pressão (mca)', 'Rendimento (%)',
                         'erro_pressao', 'erro_relativo', 'Potência (HP)', 'Motor (HP)']].head(top_n)

def selecionar_bombas(df, vazao_desejada, pressao_desejada, top_n=5):
    resultado_unico = filtrar_e_classificar(df, vazao_desejada, pressao_desejada, top_n)
    if not resultado_unico.empty and resultado_unico.iloc[0]["Rendimento (%)"] > 60:
        return resultado_unico, "unica"

    resultado_paralelo = filtrar_e_classificar(df, vazao_desejada / 2, pressao_desejada, top_n)
    if not resultado_paralelo.empty:
        return resultado_paralelo, "paralelo"

    resultado_serie = filtrar_e_classificar(df, vazao_desejada, pressao_desejada / 2, top_n)
    if not resultado_serie.empty:
        return resultado_serie, "serie"

    return pd.DataFrame(), "nenhuma"

# INTERFACE STREAMLIT

st.set_page_config(layout="wide")
st.title("🛠️ Seletor de Bombas Hidráulicas (Versão PRO)")

df_processado = carregar_e_processar_dados("Todos os dados.xlsx")

if df_processado is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Parâmetros de Entrada")
        vazao_input = st.number_input("Vazão Desejada (m³/h):", min_value=0.1, value=500.0, step=10.0)
        pressao_input = st.number_input("Pressão Desejada (mca):", min_value=0.1, value=50.0, step=5.0)

    buscar = st.button("Buscar Melhor Opção", type="primary", use_container_width=True)
    st.divider()

    if buscar:
        with st.spinner("Calculando as melhores opções..."):
            resultado, tipo = selecionar_bombas(df_processado, vazao_input, pressao_input)

        st.header("Resultados da Busca")
        if tipo == "unica":
            st.success("✅ Solução encontrada com **BOMBA ÚNICA**:")
        elif tipo == "paralelo":
            st.warning("⚠️ Nenhuma bomba única com bom rendimento. Alternativa: **DUAS BOMBAS EM PARALELO**:")
            st.info("A vazão e potência abaixo são POR BOMBA. Vazão total = 2x.")
        elif tipo == "serie":
            st.warning("⚠️ Nenhuma opção única ou paralela. Alternativa: **DUAS BOMBAS EM SÉRIE**:")
            st.info("A pressão abaixo é POR BOMBA. Pressão total = 2x.")
        else:
            st.error("❌ Nenhuma bomba encontrada. Tente outros valores.")
            st.stop()

        st.dataframe(resultado, hide_index=True, use_container_width=True)
