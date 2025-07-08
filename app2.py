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
    # ETAPA 1: FILTRAGEM  código original, 100% preservado
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
    # ETAPA 2: ORDENAÇÃO COM COLUNA-CHAVE 
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
    if not resultado_unico.empty and resultado_unico.iloc[0]["Rendimento (%)"] > 50:
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
st.title("🛠️ Seletor de Bombas Hidráulicas")

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
        
# ======================= PRECIFICADOR ============================

st.divider()
st.header("💸 Precificador de Bombas")

# --- PARTE 1: MAPEAMENTO DE OPÇÕES ---
modelos = ["R1", "R2", "R3", "R4", "R5", "M1", "M2"]
diametros_por_modelo = {
    "R1": [195, 260, 265, 310, 320, 360, 365, 390, 394, 430],
    "R2": [155, 265, 320, 360, 365, 390],
    "R3": [155, 265, 320, 360, 365],
    "R4": [155, 265, 320, 360, 365],
    "R5": [265, 320],
    "M1": [240, 305, 335, 345, 400, 420, 550],
    "M2": [345, 370]
}
potencias_por_modelo_diametro = {
    "R1-195": [15, 20, 25, 30], "R1-260": [25, 30, 40], "R1-265": [20, 25, 30], "R1-310": [40, 50, 60, 75],
    "R1-320": [40, 50, 60], "R1-360": [75, 100, 125, 150], "R1-365": [125, 150, 175],
    "R1-390": [150, 175, 200, 250, 300], "R1-394": [150, 175, 200, 250, 300], "R1-430": [350, 400],
    "R2-155": [15, 20], "R2-265": [40, 50, 60], "R2-320": [75, 100, 125],
    "R2-360": [150, 175, 200, 250, 300], "R2-365": [200, 250, 300, 350], "R2-390": [350, 400, 450, 500, 550, 600],
    "R3-155": [20, 25], "R3-265": [50, 60, 75], "R3-320": [125, 150], "R3-360": [300, 350, 400], "R3-365": [400, 450, 500, 550],
    "R4-155": [25, 30], "R4-265": [75, 100, 125], "R4-320": [150, 175, 200, 250], "R4-360": [350, 400, 450, 500, 550, 600], "R4-365": [550, 600],
    "R5-265": [75, 100, 125], "R5-320": [175, 200, 250, 300],
    "M1-240": [25, 30, 40, 50, 60], "M1-305": [75, 100, 125, 150], "M1-335": [150, 175, 200], "M1-345": [150, 175, 200, 250, 300],
    "M1-400": [350, 400, 450, 500, 550], "M1-420": [150, 175, 200, 250, 300], "M1-550": [450, 500, 550, 600],
    "M2-345": [350, 400, 450, 500, 600], "M2-370": [400, 450, 500, 600]
}

rotor_opcoes = ["FOFO", "CA40", "INOX304"]
difusor_opcoes_total = ["FOFO", "CA40", "INOX304"]


# -------------------------- CONFIGURAÇÃO DO APP --------------------------
st.set_page_config(page_title="Precificador de Bombas", layout="wide")
st.title("💰 Precificador e Simulador de Bombas")
st.write("Selecione uma configuração, clique em 'Calcular' e use o simulador para testar cenários em tempo real.")

# -------------------------- FUNÇÕES E CARREGAMENTO DE DADOS --------------------------

@st.cache_data
def carregar_dados():
    try:
        df_bombas = pd.read_excel("Dados ID valor.xlsx", sheet_name="Id com valor")
        df_markups = pd.read_excel("Dados ID valor.xlsx", sheet_name="MARKUPS")
        for df in [df_bombas, df_markups]:
            df.columns = df.columns.str.strip().str.upper()
        for col in df_bombas.select_dtypes(include=['object']).columns:
            df_bombas[col] = df_bombas[col].str.strip().str.upper()
        for col in df_markups.select_dtypes(include=['object']).columns:
            df_markups[col] = df_markups[col].str.strip().str.upper()
        if 'CHAVE_BUSCA' in df_markups.columns:
            df_markups['CHAVE_BUSCA'] = df_markups['CHAVE_BUSCA'].astype(str).str.strip().str.upper().str.replace(' ', '')
        df_bombas.dropna(how='all', inplace=True)
        df_markups.dropna(how='all', inplace=True)
        df_bombas.dropna(subset=['POTÊNCIA'], inplace=True)
        df_bombas['POTÊNCIA'] = df_bombas['POTÊNCIA'].astype(int)
        return df_bombas, df_markups
    except FileNotFoundError:
        st.error("ERRO: Arquivo 'Dados ID valor.xlsx' não encontrado.")
        return None, None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o Excel: {e}")
        return None, None

df_bombas, df_markups = carregar_dados()

# Inicializa o "post-it" (session_state) se ele não existir
if 'calculo_iniciado' not in st.session_state:
    st.session_state.calculo_iniciado = False
if 'dados_calculo' not in st.session_state:
    st.session_state.dados_calculo = {}

# -------------------------- INTERFACE DO USUÁRIO --------------------------
if df_bombas is not None:
    st.header("1. Selecione a Configuração da Bomba")
    
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        modelo = st.selectbox("Modelo", sorted(df_bombas["MODELO"].dropna().unique()))
    with col_sel2:
        diametro = st.selectbox("Diâmetro", sorted(df_bombas[df_bombas["MODELO"] == modelo]["DIAMETRO"].dropna().unique()))
    with col_sel3:
        potencia = st.selectbox("Potência", sorted(df_bombas[(df_bombas["MODELO"] == modelo) & (df_bombas["DIAMETRO"] == diametro)]["POTÊNCIA"].dropna().unique()))

    col_sel4, col_sel5 = st.columns(2)
    with col_sel4: rotor = st.selectbox("Material Rotor", df_bombas["MATERIAL ROTOR"].dropna().unique())
    with col_sel5: difusor = st.selectbox("Material Difusor", df_bombas["MATERIAL DIFUSOR"].dropna().unique())

    st.write("Selecione os demais opcionais:")
    col_sel6, col_sel7, col_sel8 = st.columns(3)
    with col_sel6: crivo = st.selectbox("Crivo", df_bombas["CRIVO"].dropna().unique(), key='crivo')
    with col_sel7: equalizador = st.selectbox("Equalizador de Pressão", df_bombas["EQUALIZADOR DE PRESSÃO"].dropna().unique(), key='eq')
    with col_sel8: sensor_motor = st.selectbox("Sensor Temp Motor", df_bombas["SENSOR TEMP MOTOR"].dropna().unique(), key='sm')
    
    col_sel9, col_sel10, col_sel11 = st.columns(3)
    with col_sel9: sensor_nivel = st.selectbox("Sensor Nível", df_bombas["SENSOR DE NIVEL"].dropna().unique(), key='sn')
    with col_sel10: sensor_mancal = st.selectbox("Sensor Temp Mancal", df_bombas["SENSOR TEMP MANCAL"].dropna().unique(), key='stm')
    with col_sel11: sensor_vibracao = st.selectbox("Sensor Vibração", df_bombas["SENSOR VIBRAÇÃO"].dropna().unique(), key='sv')
    
    st.divider()
    
    # Botão principal que INICIA o cálculo e liga a "memória"
    if st.button("Calcular Preço e Simular", type="primary", use_container_width=True):
        st.session_state.calculo_iniciado = True
        
        try:
            # --- LÓGICA DE CÁLCULO INICIAL (só roda quando o botão é clicado) ---
            filtro_atual = (
                (df_bombas["MODELO"] == modelo) & (df_bombas["DIAMETRO"] == diametro) & (df_bombas["POTÊNCIA"] == potencia) &
                (df_bombas["MATERIAL ROTOR"] == rotor) & (df_bombas["MATERIAL DIFUSOR"] == difusor) &
                (df_bombas["EQUALIZADOR DE PRESSÃO"] == equalizador) & (df_bombas["SENSOR TEMP MOTOR"] == sensor_motor) &
                (df_bombas["SENSOR DE NIVEL"] == sensor_nivel) & (df_bombas["SENSOR TEMP MANCAL"] == sensor_mancal) &
                (df_bombas["SENSOR VIBRAÇÃO"] == sensor_vibracao) & (df_bombas["CRIVO"] == crivo)
            )
            linha_bomba_atual = df_bombas[filtro_atual]
            potencia_max = df_bombas[(df_bombas["MODELO"] == modelo) & (df_bombas["DIAMETRO"] == diametro)]["POTÊNCIA"].max()
            
            filtro_ref = (
                (df_bombas["MODELO"] == modelo) & (df_bombas["DIAMETRO"] == diametro) & (df_bombas["POTÊNCIA"] == potencia_max) &
                (df_bombas["MATERIAL ROTOR"] == rotor) & (df_bombas["MATERIAL DIFUSOR"] == difusor) &
                (df_bombas["EQUALIZADOR DE PRESSÃO"] == equalizador) & (df_bombas["SENSOR TEMP MOTOR"] == sensor_motor) &
                (df_bombas["SENSOR DE NIVEL"] == sensor_nivel) & (df_bombas["SENSOR TEMP MANCAL"] == sensor_mancal) &
                (df_bombas["SENSOR VIBRAÇÃO"] == sensor_vibracao) & (df_bombas["CRIVO"] == crivo)
            )
            linha_ref = df_bombas[filtro_ref]
            
            if linha_ref.empty:
                st.error(f"ERRO CRÍTICO: Custo da bomba de referência (Potência {int(potencia_max)} HP) não encontrado.")
                st.session_state.calculo_iniciado = False
                st.stop()
            
            def criar_chave(m, d, p, r, f):
                return (str(m) + str(int(d)) + str(int(p)) + str(r) + str(f)).upper().replace(' ', '')
            
            chave_markup_busca = criar_chave(modelo, diametro, potencia_max, rotor, difusor)
            linha_markup = df_markups[df_markups["CHAVE_BUSCA"] == chave_markup_busca]

            if linha_markup.empty:
                st.error(f"ERRO CRÍTICO: Markup não encontrado para a bomba de referência. Chave: `{chave_markup_busca}`")
                st.session_state.calculo_iniciado = False
                st.stop()

            # Guarda os dados essenciais na "memória"
            st.session_state.dados_calculo = {
                "custo_atual": linha_bomba_atual["VALOR"].values[0] if not linha_bomba_atual.empty else 0,
                "id_bomba": linha_bomba_atual["ID"].values[0] if not linha_bomba_atual.empty else "Não Cadastrado",
                "custo_referencia": linha_ref["VALOR"].values[0],
                "markup_excel": linha_markup["MARKUP"].values[0],
                "potencia": potencia,
                "potencia_max": potencia_max,
                "modelo": modelo, "diametro": diametro, "rotor": rotor, "difusor": difusor
            }
            
            reducao_excel = 0.0
            if potencia < potencia_max:
                chave_reducao_busca = criar_chave(modelo, diametro, potencia, 'nan', 'nan')
                linha_reducao = df_markups[df_markups["CHAVE_BUSCA"] == chave_reducao_busca]
                if not linha_reducao.empty:
                    reducao_excel = linha_reducao["REDUÇÃO(%)"].values[0]
            st.session_state.dados_calculo["reducao_excel"] = reducao_excel

        except Exception as e:
            st.error(f"Ocorreu um erro durante o cálculo: {e}")
            st.session_state.calculo_iniciado = False

# --- LÓGICA DE EXIBIÇÃO (só roda se a "memória" estiver ligada) ---
if st.session_state.calculo_iniciado:
    try:
        # Pega os dados da "memória"
        dados = st.session_state.dados_calculo
        
        # --- ETAPA DE CÁLCULO OFICIAL ---
        preco_referencia = dados["custo_referencia"] * dados["markup_excel"]
        
        # Lógica de Redução Oficial Corrigida
        reducao_excel = 0.0 # Começa com 0
        if dados["potencia"] < dados["potencia_max"]:
            # Recria a chave de busca para a redução
            def criar_chave(m, d, p, r, f):
                s_r = '' if str(r).upper() == 'NAN' else str(r)
                s_f = '' if str(f).upper() == 'NAN' else str(f)
                return (str(m) + str(int(d)) + str(int(p)) + s_r + s_f).upper().replace(' ', '')
            
            chave_reducao_busca = criar_chave(dados['modelo'], dados['diametro'], dados['potencia'], 'nan', 'nan')
            
            # Filtra a linha de redução no dataframe de markups
            linha_reducao = df_markups[df_markups["CHAVE_BUSCA"] == chave_reducao_busca]
            
            # --- PONTO DE DEPURAÇÃO DA REDUÇÃO ---
            with st.expander("🔍 Detalhes da Busca por Redução (Depuração)"):
                st.write(f"**Chave de Redução que o Python está procurando:** `{chave_reducao_busca}`")
                st.write("**Chaves de Redução encontradas no seu Excel (onde o Markup é vazio):**")
                st.dataframe(df_markups[df_markups['MARKUP'].isna()][['MODELO', 'DIAMETRO', 'POTÊNCIA', 'CHAVE_BUSCA']])

            if not linha_reducao.empty:
                reducao_excel = linha_reducao["REDUÇÃO(%)"].values[0]
        
        preco_final_oficial = preco_referencia * (1 - reducao_excel / 100)
        
        # ----------------- RESULTADO OFICIAL -----------------
        st.header("2. Resultado do Preço Oficial (Baseado no Excel)")
        lucro_oficial = preco_final_oficial - dados["custo_atual"] if dados["custo_atual"] > 0 else 0
        lucro_pct_oficial = (lucro_oficial / dados["custo_atual"]) * 100 if dados["custo_atual"] > 0 else 0
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Preço Final de Venda", f"R$ {preco_final_oficial:,.2f}")
        res_col2.metric("Lucro", f"R$ {lucro_oficial:,.2f}", delta_color="off")
        res_col3.metric("Margem de Lucro", f"{lucro_pct_oficial:.1f}%", delta_color="off")

        # ----------------- CAIXA DE INFORMAÇÃO -----------------
        with st.container(border=True):
            st.subheader("📝 Fatores Utilizados no Cálculo Oficial")
            info_col1, info_col2 = st.columns(2)
            info_col1.metric("Markup da Referência", f"{dados['markup_excel']:.2f}x")
            if dados["potencia"] < dados["potencia_max"]:
                info_col2.metric("Redução Aplicada", f"{reducao_excel:.1f}%")
            else:
                info_col2.info("Esta é a bomba de referência, sem redução.")
        
        st.divider()
        
        # ----------------- CAIXA DE SIMULAÇÃO (AGORA INTERATIVA) -----------------
        st.header("3. Simulador de Markups")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            markup_simulado = st.number_input("Testar novo Markup", min_value=1.0, value=dados["markup_excel"], step=0.05, format="%.2f")
        with sim_col2:
            reducao_simulada = st.number_input("Testar nova Redução (%)", min_value=0.0, max_value=100.0, value=reducao_excel, step=0.5, format="%.1f")

        # Recalcula com os valores do simulador (LÓGICA CORRIGIDA)
        preco_ref_simulado = dados["custo_referencia"] * markup_simulado
        
        if dados["potencia"] < dados["potencia_max"]:
            preco_final_simulado = preco_ref_simulado * (1 - reducao_simulada / 100)
        else:
            # Se for a bomba de referência, não aplica redução
            preco_final_simulado = preco_ref_simulado
        
        lucro_simulado = preco_final_simulado - dados["custo_atual"] if dados["custo_atual"] > 0 else 0
        lucro_pct_simulado = (lucro_simulado / dados["custo_atual"]) * 100 if dados["custo_atual"] > 0 else 0
        
        with st.container(border=True):
            st.subheader("📈 Resultado da Simulação")
            sim_res_col1, sim_res_col2, sim_res_col3 = st.columns(3)
            sim_res_col1.metric("Novo Preço de Venda", f"R$ {preco_final_simulado:,.2f}")
            sim_res_col2.metric("Novo Lucro", f"R$ {lucro_simulado:,.2f}", delta_color="off")
            sim_res_col3.metric("Nova Margem", f"{lucro_pct_simulado:.1f}%", delta_color="off")

            # --- BOTÃO DE COPIAR ---
            st.divider()
            texto_resumo = f"""
=================================
Resumo da Simulação de Preço
=================================
Bomba: {dados['modelo']} - {dados['diametro']} - {dados['potencia']} HP
Materiais: {dados['rotor']} / {dados['difusor']}
Custo de Produção: R$ {dados['custo_atual']:,.2f}

-- Parâmetros Testados --
Markup Multiplicador: {markup_simulado:.2f}x
Redução Aplicada: {reducao_simulada:.1f}%

-- Resultado da Simulação --
Novo Preço de Venda: R$ {preco_final_simulado:,.2f}
Novo Lucro: R$ {lucro_simulado:,.2f} ({lucro_pct_simulado:.1f}%)
"""
            st.code(texto_resumo, language="text")
    
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante a exibição dos resultados: {e}")
