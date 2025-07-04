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
    df["erro_relativo"] = ((df["Vazão (m³/h)"] - df["vazao_centro"]) / (df["max"] - df["min"])) * 100
    df["abs_erro_relativo"] = df["erro_relativo"].abs()

    return df

def filtrar_e_classificar(df, vazao, pressao, top_n=5):
    """
    Filtra as bombas e aplica a ordenação final usando a abordagem de
    "coluna-chave", que é mais robusta e definitiva.
    """
    if df is None:
        return pd.DataFrame()

    # ===================================================================
    # ETAPA 1: FILTRAGEM (Seu código original, 100% preservado)
    # ===================================================================
    cond_max = df['RotorNum'] == df['rotor_max_modelo']
    cond_min = df['RotorNum'] == df['rotor_min_modelo']

    df['margem_cima'] = np.select(
        [cond_max, cond_min],
        [df['pressao_max_modelo'] * 0.03, df['pressao_max_modelo'] * 0.075],
        default=df['pressao_max_modelo'] * 0.075
    )
    df['margem_baixo'] = np.select(
        [cond_max, cond_min],
        [df['pressao_max_modelo'] * 0.075, df['pressao_max_modelo'] * 0.03],
        default=df['pressao_max_modelo'] * 0.075
    )

    pressao_min_aceita = pressao - df['margem_baixo']
    pressao_max_aceita = pressao + df['margem_cima']

    df_filtrado = df[
        (df["Vazão (m³/h)"] == vazao) &
        (df["Pressão (mca)"] >= pressao_min_aceita) &
        (df["Pressão (mca)"] <= pressao_max_aceita)
    ].copy()

    if not df_filtrado.empty:
        df_filtrado = df_filtrado[
            ~((df_filtrado['RotorNum'] == df_filtrado['rotor_min_modelo']) &
              (pressao < df_filtrado["Pressão (mca)"] - df_filtrado['pressao_max_modelo'] * 0.03)) &
            ~((df_filtrado['RotorNum'] == df_filtrado['rotor_max_modelo']) &
              (pressao > df_filtrado["Pressão (mca)"] + df_filtrado['pressao_max_modelo'] * 0.03))
        ]

    if df_filtrado.empty:
        return pd.DataFrame()

    # ===================================================================
    # ETAPA 2: ORDENAÇÃO COM COLUNA-CHAVE (A SOLUÇÃO DEFINITIVA)
    # ===================================================================
    
    # Adiciona uma coluna com o erro absoluto da pressão
    df_filtrado["erro_pressao_abs"] = (df_filtrado["Pressão (mca)"] - pressao).abs()

# --- NOVA LÓGICA DE DESEMPATE ---
# 1. Calcula a menor diferença de rendimento entre bombas do mesmo motor
    df_filtrado['diff_rendimento_vs_grupo'] = df_filtrado.groupby('Motor (HP)')['Rendimento (%)'].transform(
        lambda x: x.apply(lambda y: (x - y).abs().min())
    )

# 2. Chave de desempate: prioriza erro relativo apenas se houver bombas com mesmo motor e rendimento ≤5% diferente
    df_filtrado['chave_desempate'] = np.where(
        df_filtrado['diff_rendimento_vs_grupo'] <= 5,  # Condição corrigida
        df_filtrado['abs_erro_relativo'],
        np.inf
    )

# 3. Mantém a chave padrão (pressão)
    df_filtrado['chave_padrao'] = df_filtrado['erro_pressao_abs']

    # ORDENAÇÃO FINAL E SIMPLES USANDO AS CHAVES
    df_resultado = df_filtrado.sort_values(
        by=["Motor (HP)", "chave_desempate", "chave_padrao"],
        ascending=[True, True, True]
    )
    
    # Prepara as colunas finais para exibição.
    df_resultado["erro_pressao"] = df_resultado["Pressão (mca)"] - pressao
    
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
