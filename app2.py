import streamlit as st
import pandas as pd
import numpy as np

# --- LÓGICA ORIGINAL DO SEU SCRIPT (COM AJUSTES) ---

# Tabela de motores movida para dentro da função que a usa para melhor organização
motores_padrao = np.array([
    15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300,
    350, 400, 450, 500, 550, 600
])

def encontrar_motor_final(potencia_real):
    candidatos = motores_padrao[motores_padrao >= potencia_real]
    return candidatos.min() if len(candidatos) > 0 else np.nan

# A anotação @st.cache_data é um "superpoder" do Streamlit.
# Ele faz com que o Excel seja lido e processado APENAS UMA VEZ.
# Isso deixa o aplicativo extremamente rápido.
@st.cache_data
def carregar_e_processar_dados(caminho_arquivo):
    """Lê o arquivo excel e faz o pré-processamento inicial."""
    try:
        df_bombas = pd.read_excel(caminho_arquivo)
    except FileNotFoundError:
        # Se o arquivo não for encontrado, mostra um erro claro na tela.
        st.error(f"Erro: Arquivo '{caminho_arquivo}' não encontrado. "
                 "Certifique-se de que ele está na mesma pasta que o script.")
        return None

    intervalos_por_rotor = (
        df_bombas.groupby(["Modelo", "Rotor"])["Vazão (m³/h)"]
        .agg(["min", "max"])
        .rename(columns={"min": "vazao_min", "max": "vazao_max"})
        .reset_index()
    )
    df = df_bombas.merge(intervalos_por_rotor, on=["Modelo", "Rotor"], how="left")
    df["Motor (HP)"] = df["Potência (HP)"].apply(encontrar_motor_final)
    df["vazao_centro"] = (df["vazao_min"] + df["vazao_max"]) / 2
    # Evita divisão por zero se vazao_max == vazao_min
    df["erro_relativo"] = abs(df["Vazão (m³/h)"] - df["vazao_centro"]) / \
                           (df["vazao_max"] - df["vazao_min"]).replace(0, np.nan)
    return df

def filtrar_e_classificar(df, vazao, pressao, top_n=5):
    """Filtra as bombas e aplica a lógica de desempate e classificação."""
    if df is None:
        return pd.DataFrame()

    df_filtrado = df[
        (df["Vazão (m³/h)"] >= vazao) & (df["Pressão (mca)"] >= pressao)
    ].copy()

    if df_filtrado.empty:
        return pd.DataFrame()

    def desempate(grupo):
        if len(grupo) <= 1:
            return grupo
        grupo = grupo.sort_values(by="Rendimento (%)", ascending=False)
        melhor = grupo.iloc[0]
        candidatos = grupo[
            abs(grupo["Rendimento (%)"] - melhor["Rendimento (%)"]) <= 4
        ]
        if len(candidatos) > 1:
            return candidatos.sort_values("erro_relativo").head(1)
        else:
            return melhor.to_frame().T

    df_resultado = (
        df_filtrado
        .groupby(["Modelo", "Motor (HP)"], group_keys=False)
        .apply(desempate)
        .reset_index(drop=True)
        .sort_values(by=["Motor (HP)", "Rendimento (%)"], ascending=[True, False])
    )
    return df_resultado.head(top_n)

def selecionar_bombas(df, vazao_desejada, pressao_desejada, top_n=5):
    """Função principal que orquestra a busca por bombas."""
    # 1. Tenta encontrar uma bomba única
    resultado_unico = filtrar_e_classificar(df, vazao_desejada, pressao_desejada, top_n)

    if not resultado_unico.empty and resultado_unico["Rendimento (%)"].max() > 60:
        return resultado_unico, "unica"

    # 2. Se não encontrou ou o rendimento é baixo, tenta em paralelo
    resultado_paralelo = filtrar_e_classificar(df, vazao_desejada / 2, pressao_desejada, top_n)
    if not resultado_paralelo.empty:
        return resultado_paralelo, "paralelo"

    # 3. Se ainda não encontrou, tenta em série
    resultado_serie = filtrar_e_classificar(df, vazao_desejada, pressao_desejada / 2, top_n)
    if not resultado_serie.empty:
        return resultado_serie, "serie"

    # 4. Se nada funcionou
    return pd.DataFrame(), "nenhuma"

# --- INTERFACE GRÁFICA DA APLICAÇÃO WEB (Streamlit) ---

st.set_page_config(layout="wide")
st.title("🛠️ Seletor de Bombas Hidráulicas")

# Carrega os dados (usando o cache)
df_processado = carregar_e_processar_dados("Todos os dados.xlsx")

if df_processado is not None:
    # Cria as colunas para organizar a interface
    col1, col2 = st.columns(2)

    with col1:
        st.header("Parâmetros de Entrada")
        vazao_input = st.number_input("Vazão Desejada (m³/h):", min_value=0.1, value=500.0, step=10.0)
        pressao_input = st.number_input("Pressão Desejada (mca):", min_value=0.1, value=100.0, step=5.0)

    # Botão para iniciar a busca
    buscar = st.button("Buscar Melhor Opção", type="primary", use_container_width=True)

    st.divider() # Uma linha divisória

    if buscar:
        with st.spinner("Calculando as melhores opções..."):
            resultado, tipo = selecionar_bombas(df_processado, vazao_input, pressao_input)
        
        st.header("Resultados da Busca")

        # Colunas que queremos exibir no resultado final
        colunas_display = ["Modelo", "Rotor", "Vazão (m³/h)", "Pressão (mca)", "Rendimento (%)", "Potência (HP)", "Motor (HP)"]

        if tipo == "unica":
            st.success("✅ Solução encontrada com **BOMBA ÚNICA**:")
            st.dataframe(resultado[colunas_display], hide_index=True)
        
        elif tipo == "paralelo":
            st.warning("⚠️ Nenhuma bomba única com bom rendimento. Alternativa encontrada com **DUAS BOMBAS EM PARALELO**:")
            st.info("A vazão e potência abaixo são POR BOMBA. A vazão total será 2x o valor da tabela.")
            st.dataframe(resultado[colunas_display], hide_index=True)
        
        elif tipo == "serie":
            st.warning("⚠️ Nenhuma opção única ou paralela. Alternativa encontrada com **DUAS BOMBAS EM SÉRIE**:")
            st.info("A pressão abaixo é POR BOMBA. A pressão total será 2x o valor da tabela.")
            st.dataframe(resultado[colunas_display], hide_index=True)

        elif tipo == "nenhuma":
            st.error("❌ Nenhuma bomba (única, em série ou paralela) foi encontrada para os critérios informados. Por favor, consulte a engenharia ou tente outros valores.")
