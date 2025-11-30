import streamlit as st
from supabase import create_client, Client
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import google.generativeai as gemini
import unicodedata
import random
import numpy as np

# --- 1. CONFIGURAÇÕES INICIAIS ---

st.set_page_config(page_title="NutriBot Inteligente", page_icon="🥗")

# Configuração SUPABASE
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error("Erro ao configurar Supabase: Verifique as Secrets.")
    st.stop()

# Configuração GEMINI (Pegando das Secrets)
# Certifique-se de configurar "GOOGLE_API_KEY" no Streamlit Cloud
try:
    gemini_api_key = st.secrets["GOOGLE_API_KEY"]
    gemini.configure(api_key=gemini_api_key)
    modelo_gemini = gemini.GenerativeModel("models/gemini-flash-latest")
except Exception as e:
    st.error("Erro ao configurar Gemini: Verifique a Secret GOOGLE_API_KEY.")
    st.stop()

# Inicialização de Variáveis de Estado (Session State)
if 'user' not in st.session_state:
    st.session_state.user = None
if 'fase' not in st.session_state:
    st.session_state.fase = 0
if 'dados_usuario' not in st.session_state:
    st.session_state.dados_usuario = {"diabetes": "", "refeicao": "", "ingrediente": ""}
if 'receita_atual' not in st.session_state:
    st.session_state.receita_atual = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'gemini_chat_instance' not in st.session_state:
    st.session_state.gemini_chat_instance = modelo_gemini.start_chat(history=[])

# --- 2. SISTEMA DE LOGIN ---

def login(email, password):
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state.user = user
        return True
    except Exception as e:
        st.error("Erro no login: " + str(e))
        return False

def signup(email, password):
    try:
        supabase.auth.sign_up({"email": email, "password": password})
        st.success("Cadastro realizado! Verifique seu email.")
    except Exception as e:
        st.error("Erro no cadastro: " + str(e))

def login_page():
    st.title("🔐 NutriBot - Acesso")
    menu = st.radio("Selecione:", ("Login", "Cadastro"), horizontal=True)
    
    email = st.text_input("Email")
    password = st.text_input("Senha", type="password")
    
    if menu == "Login":
        if st.button("Entrar"):
            if login(email, password):
                st.rerun()
    elif menu == "Cadastro":
        if st.button("Cadastrar"):
            signup(email, password)

if st.session_state.user is None:
    login_page()
    st.stop()

# --- 3. LÓGICA DE DADOS E IA ---

def normalizar_texto(texto):
    if not isinstance(texto, str): return str(texto)
    texto = texto.lower()
    nfkd_form = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# Base de Dados
data = {
    'nome': ['Omelete de claras', 'Creme de abacate com cacau', 'Iogurte com granola caseira', 'Tapioca com queijo branco', 'Panqueca integral', 'Nhoque ao molho mediterrâneo', 'Maminha ao molho de ervas', 'Risoto do mar', 'Ravioli de cúrcuma com alho-poro', 'Charutinho caipira', 'Estrogonofe de frango com berinjela', 'Sopa de cebola especial', 'Tilápia grelhada', 'Arroz de couve-flor', 'Sopa de legumes', 'Quiche de presunto', 'Refresco de melancia', 'Salada de frutas ao forno', 'Mix de castanhas', 'Frutas com chia'],
    'tipo_refeicao': ['cafe', 'cafe', 'cafe', 'cafe', 'cafe', 'principal', 'principal', 'principal', 'principal', 'principal', 'principal', 'principal', 'principal', 'principal', 'principal', 'lanche', 'lanche', 'lanche', 'lanche', 'lanche'],
    'tipo_diabetes': ['tipo2', 'ambos', 'ambos', 'tipo2', 'tipo1', 'tipo1', 'tipo2', 'ambos', 'tipo1', 'ambos', 'ambos', 'tipo2', 'tipo2', 'tipo2', 'ambos', 'tipo2', 'tipo1', 'tipo1', 'tipo2', 'ambos'],
    'ingredientes': ['Ovos, tomate, sal, azeite, ovo', 'Abacate, cacau em pó, adoçante natural, leite vegetal, castanhas, fruta', 'Iogurte natural, aveia, mel, castanhas, iogurte', 'Goma de tapioca, queijo branco, sal, queijo', 'Farinha integral, ovo, leite, açucar mascavo', 'Mandioquinha, clara, margarina, farinha, azeite, berinjela, abobrinha, tomate, vegetal', 'Maminha, sal, alho, azeite, manjericão, salsa, tomilho, creme de leite light', 'Badejo, cebola, arroz integral, vinho branco, brócolis, creme de leite light, açafrão', 'Farinha de trigo, ovo, azeite, curcuma, queijo de minas light, alho-poró, manjericão, azeitona, queijo', 'Couve-manteiga, frango desfiado, cebola, milho verde, salsa, caldo de legumes, frango', 'Frango, pimenta, azeite, cebola, berinjela, tomate, mostarda, catchup light, creme de leite light', 'Alho-poró, cebola, cebolinha, azeite, farinha, mostarda, caldo de galinha, leite em pó desnatado', 'Tilápia, sal, limão, alho', 'Couve-flor, azeite, sal, alho, vegetal', 'Abobrinha, cenoura, alho, tomate, água, vegetal', 'Iogurte, margarina, farinha, presunto magro, acelga, queijo parmesão light', 'Melancia, água, iogurte desnatado, gengibre, fruta', 'Pêssego diet, banana, manga, morango, fruta', 'Castanhas, nozes, amêndoas, castanha', 'Morango, banana, chia, fruta']
}
df = pd.DataFrame(data)
df['descricao'] = [
    'Opção focada em proteína, com baixo teor de carboidratos.', 'Rico em gorduras saudáveis e fibras.', 'Mistura equilibrada de proteínas e fibras.', 'Carboidrato simples combinado com proteína.', 'Carboidrato de absorção mais lenta.', 'Prato de carboidrato complexo e legumes.', 'Carne magra com baixo teor de gordura.', 'Rico em fibras e proteínas.', 'Massa caseira com legumes e ingredientes naturais.', 'Prato completo com vegetais e proteína magra.', 'Versão adaptada, baixo carboidrato.', 'Sopa nutritiva com baixo teor calórico.', 'Opção leve e proteica.', 'Excelente substituto do arroz tradicional.', 'Sopa leve, rica em fibras.', 'Lanche salgado equilibrado.', 'Lanche refrescante.', 'Frutas assadas com creme dietético.', 'Fonte de gorduras boas e fibras.', 'Lanche rico em fibras e antioxidantes.'
]

# Treinamento da Árvore de Decisão
if 'modelo_arvore' not in st.session_state:
    df_treino = df.copy()

    # 1. Pré-processamento e Encoding
    df_treino['tipo_diabetes'] = df_treino['tipo_diabetes'].apply(normalizar_texto)

    cols_to_encode = ['tipo_diabetes']
    df_encoded = pd.get_dummies(df_treino, columns=cols_to_encode, prefix=cols_to_encode)

    ingredientes_list = [[normalizar_texto(i.strip()) for i in row.split(',')] for row in df['ingredientes']]
    mlb = MultiLabelBinarizer()
    ingredientes_encoded = mlb.fit_transform(ingredientes_list)
    ingredientes_df = pd.DataFrame(ingredientes_encoded, columns=[f'ingrediente_{i}' for i in mlb.classes_])

    # Remove colunas não necessárias para o treino de X
    df_final = pd.concat([df_encoded.drop(['ingredientes', 'tipo_refeicao'], axis=1), ingredientes_df], axis=1)

    colunas_x_treino = df_final.drop(['nome', 'descricao'], axis=1).columns
    X = df_final[colunas_x_treino]
    Y = df_final['nome']

    # 2. Geração dos Pesos da Amostra (Sample Weights)
    df_treino['num_ingredientes'] = df_treino['ingredientes'].apply(lambda x: len(x.split(',')))
    pesos = df_treino['num_ingredientes'].values / df_treino['num_ingredientes'].sum() * len(df_treino)
    pesos = np.maximum(pesos, 1) # Garante que o peso mínimo é 1

    # 3. Treinamento do Modelo com Pesos
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X, Y, sample_weight=pesos)

    # Salva no estado da sessão
    st.session_state.modelo_arvore = modelo
    st.session_state.mlb = mlb
    st.session_state.colunas_x_treino = colunas_x_treino
    st.session_state.df_completo = df


# Função de Recomendação com ML

def recomendar_com_ml(refeicao, diabetes, ingrediente_usuario):

    colunas_local = st.session_state.colunas_x_treino
    modelo_local = st.session_state.modelo_arvore
    df_receitas = st.session_state.df_completo # Puxa do Session State

    # 1. Pré-processamento e Normalização
    ref_norm = normalizar_texto(refeicao)
    # Mapeamento para os tipos de refeição no DataFrame ('cafe', 'principal', 'lanche')
    user_ref_check = 'principal' if ref_norm in ['almoco', 'jantar'] else ref_norm
    dia_norm = normalizar_texto(diabetes).replace(" ", "")
    ing_norm = normalizar_texto(ingrediente_usuario)

    tipos_validos_diabetes = ['tipo1', 'ambos'] if "tipo1" in dia_norm else ['tipo2', 'ambos']

    # 2.Sub-DataFrame com receitas do tipo de refeição desejado
    df_filtrado = df_receitas[df_receitas['tipo_refeicao'] == user_ref_check].copy()

    if df_filtrado.empty:
        return None, None, None

    # 3. Preparação do Input para o ML
    X_novo = pd.DataFrame(0, index=[0], columns=colunas_local)

    # Seta coluna de Diabetes
    dia_col = f'tipo_diabetes_{dia_norm}'
    if dia_col in X_novo.columns:
        X_novo.loc[0, dia_col] = 1
    elif 'ambos' in dia_norm and f'tipo_diabetes_ambos' in X_novo.columns:
        X_novo.loc[0, f'tipo_diabetes_ambos'] = 1

    # Seta coluna de Ingrediente
    ing_col_name = f'ingrediente_{ing_norm}'
    if ing_norm and ing_norm not in ['nao', 'na', ''] and ing_col_name in X_novo.columns:
          X_novo.loc[0, ing_col_name] = 1

    receita_obj = None

    try:
        # 4. Predição do ML
        receita_prevista_nome = modelo_local.predict(X_novo)[0]
        receita_obj_ml = df_receitas[df_receitas['nome'] == receita_prevista_nome].iloc[0]

        # 5. Validação da Predição (Pós-Processamento)

        # 5a. O ML ACERTOU na refeição E é compatível com o diabetes?
        ml_ref_match = (normalizar_texto(receita_obj_ml['tipo_refeicao']) == user_ref_check)
        ml_dia_match = (normalizar_texto(receita_obj_ml['tipo_diabetes']) in tipos_validos_diabetes)

        if ml_ref_match and ml_dia_match:
             # ML ACERTOU TUDO. Usamos a sugestão dele.
             receita_obj = receita_obj_ml
        else:
             # ML NÃO ACERTOU na refeição ou no diabetes. Precisa de Fallback
             raise Exception("ML Inconsistente")

    except Exception as e:
        # 6. FALLBACK HEURÍSTICO CORRIGIDO (Prioridade Absoluta ao Ingrediente + Refeição)

        # Tenta achar Ingrediente + Refeição, ignorando o Diabetes inicialmente
        if ing_norm and ing_norm not in ['nao', 'na', '']:
            df_match_ing = df_filtrado[df_filtrado['ingredientes'].apply(lambda x: ing_norm in normalizar_texto(x))]

            if not df_match_ing.empty:
                 # PRIORIDADE 1.1: Encontrou Ingrediente + Refeição.
                 # Retorna a sugestão, mesmo com mismatch de Diabetes (será avisado ao usuário).
                 # Usando random.randint para amostragem aleatória no Streamlit.
                 receita_obj = df_match_ing.sample(n=1, random_state=random.randint(1, 1000)).iloc[0]

            else:
                 # PRIORIDADE 1.2: Ingrediente não encontrado. Volta a focar em Refeição + Diabetes.
                 df_prioridade_dia = df_filtrado[df_filtrado['tipo_diabetes'].apply(lambda x: normalizar_texto(x) in tipos_validos_diabetes)]

                 if not df_prioridade_dia.empty:
                     # Encontrou Refeição + Diabetes (sem o ingrediente).
                     receita_obj = df_prioridade_dia.sample(n=1, random_state=random.randint(1, 1000)).iloc[0]
                 else:
                     # Último recurso: pega qualquer receita do tipo de refeição solicitado
                     receita_obj = df_filtrado.sample(n=1, random_state=random.randint(1, 1000)).iloc[0]
        else:
            # Não houve ingrediente. Foca em Refeição + Diabetes.
            df_prioridade_dia = df_filtrado[df_filtrado['tipo_diabetes'].apply(lambda x: normalizar_texto(x) in tipos_validos_diabetes)]

            if not df_prioridade_dia.empty:
                receita_obj = df_prioridade_dia.sample(n=1, random_state=random.randint(1, 1000)).iloc[0]
            else:
                 # Último recurso: pega qualquer receita do tipo de refeição solicitado
                receita_obj = df_filtrado.sample(n=1, random_state=random.randint(1, 1000)).iloc[0]


    # 7. Cálculo de Mismatch (Apenas para o Tipo de Diabetes)
    mismatch = None
    if receita_obj is not None:
        tipo_user_str = "Tipo 1" if "tipo1" in dia_norm else "Tipo 2"
        # Verifica se o tipo de diabetes da receita (mesmo a do fallback) é incompatível com o usuário
        if normalizar_texto(receita_obj['tipo_diabetes']) not in tipos_validos_diabetes:
             mismatch = (receita_obj['tipo_diabetes'], tipo_user_str)

    return receita_obj, None, mismatch


def verificar_assunto_com_gemini(texto):
    try:
        response = modelo_gemini.generate_content(
            f"O texto '{texto}' tem relacao com alimentacao, saude, diabetes, culinaria ou receitas? Responda APENAS 'SIM' ou 'NAO'."
        )
        return "SIM" in response.text.upper()
    except:
        return True
    
# --- 4. INTERFACE STREAMLIT (Lógica de Fases) ---

# Cabeçalho e Logout
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🥗 NutriBot Inteligente")
with col2:
    if st.button("Sair"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.session_state.fase = 0
        st.session_state.chat_history = []
        st.session_state.receita_atual = None
        st.rerun()

st.write(f"Olá, **{st.session_state.user.user.email}**! Vamos encontrar a melhor refeição para você.")
st.markdown("---")

# FASE 0: Tipo de Diabetes
if st.session_state.fase == 0:
    st.subheader("1. Qual é o seu tipo de diabetes?")
    tipo = st.radio("Selecione:", ("Tipo 1", "Tipo 2"), horizontal=True, key="input_diabetes")
    
    if st.button("Próximo ➡️"):
        st.session_state.dados_usuario['diabetes'] = tipo
        st.session_state.fase = 1
        st.rerun()

# FASE 1: Tipo de Refeição
elif st.session_state.fase == 1:
    st.subheader("2. Para qual refeição é a sugestão?")
    refeicao = st.selectbox("Selecione:", ["Café da Manhã", "Almoço", "Jantar", "Lanche"], key="input_refeicao")
    
    col_voltar, col_prox = st.columns([0.2, 0.8])
    if col_voltar.button("⬅️ Voltar"):
        st.session_state.fase = 0
        st.rerun()
    if col_prox.button("Próximo ➡️"):
        # Mapeamento para o banco de dados
        mapa_ref = {"Café da Manhã": "cafe", "Almoço": "almoco", "Jantar": "jantar", "Lanche": "lanche"}
        st.session_state.dados_usuario['refeicao'] = mapa_ref[refeicao]
        st.session_state.fase = 2
        st.rerun()

# FASE 2: Ingrediente
elif st.session_state.fase == 2:
    st.subheader("3. Tem algum ingrediente preferido?")
    ingrediente = st.text_input("Digite um ingrediente (ou deixe em branco para ignorar):", key="input_ingrediente")
    
    col_voltar, col_buscar = st.columns([0.2, 0.8])
    if col_voltar.button("⬅️ Voltar"):
        st.session_state.fase = 1
        st.rerun()
    
    if col_buscar.button("🔍 Buscar Receita"):
        # Executa a recomendação
        with st.spinner("Consultando nossa base de dados inteligente..."):
            rec_obj, aviso, mismatch = recomendar_com_ml(
                st.session_state.dados_usuario['refeicao'],
                st.session_state.dados_usuario['diabetes'],
                ingrediente
            )
            
            if rec_obj is not None:
                st.session_state.receita_atual = {
                    "data": rec_obj,
                    "aviso": aviso,
                    "mismatch": mismatch
                }
                # Adiciona mensagem inicial do bot ao chat
                st.session_state.chat_history = [] # Limpa chat anterior
                msg_inicial = f"Encontrei uma receita para você: **{rec_obj['nome']}**! \n\nPosso tirar dúvidas sobre ela ou sobre nutrição."
                st.session_state.chat_history.append({"role": "assistant", "content": msg_inicial})
                
                st.session_state.fase = 3
                st.rerun()
            else:
                st.error("Não encontrei receitas compatíveis. Tente mudar os filtros.")

# FASE 3: Resultado e Chat
elif st.session_state.fase == 3:
    rec = st.session_state.receita_atual['data']
    aviso = st.session_state.receita_atual['aviso']
    mismatch = st.session_state.receita_atual['mismatch']

    # --- Área de Exibição da Receita ---
    st.success(f"🍽️ **Sugestão:** {rec['nome']}")
    
    if aviso:
        st.warning(aviso)
    
    if mismatch:
        original, user = mismatch
        st.error(f"⚠️ **Atenção:** Esta receita é ideal para {original}, mas você indicou ser {user}. Veja as ressalvas no chat abaixo!")

    with st.expander("📖 Ver Detalhes da Receita", expanded=True):
        st.write(f"**Descrição:** {rec['descricao']}")
        st.write(f"**Ingredientes:** {rec['ingredientes']}")
        st.caption(f"Categoria de Diabetes da Receita: {rec['tipo_diabetes']}")

    if st.button("🔄 Começar Nova Consulta"):
        st.session_state.fase = 0
        st.session_state.receita_atual = None
        st.rerun()

    st.markdown("---")
    st.subheader("💬 Tire suas dúvidas com a IA")

    # Exibe Histórico do Chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input do Chat
    if prompt := st.chat_input("Pergunte sobre a receita, substituições ou nutrição..."):
        # 1. Adiciona user msg ao histórico
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Verifica assunto
        if not verificar_assunto_com_gemini(prompt):
            resposta_erro = "Desculpe, só posso responder sobre nutrição, diabetes e receitas."
            st.session_state.chat_history.append({"role": "assistant", "content": resposta_erro})
            with st.chat_message("assistant"):
                st.markdown(resposta_erro)
        else:
            # 3. Monta o contexto inteligente
            contexto_base = f"Você é um assistente virtual nutricional. O usuário recebeu a recomendação da receita: {rec['nome']} ({rec['descricao']}). Ingredientes: {rec['ingredientes']}."
            
            ressalva_txt = ""
            if mismatch:
                original_type, user_type = mismatch
                ressalva_txt = f"""
                ATENÇÃO CRÍTICA: Esta receita é classificada para {original_type}, mas o usuário é {user_type}.
                Sua resposta DEVE começar obrigatoriamente com uma ressalva de segurança nutricional sobre isso.
                """
            
            prompt_final = f"Contexto: {contexto_base}. {ressalva_txt} Pergunta do usuário: {prompt}"

            # 4. Chama Gemini
            try:
                with st.spinner("Pensando..."):
                    resposta = st.session_state.gemini_chat_instance.send_message(prompt_final)
                    bot_reply = resposta.text
            except Exception as e:
                bot_reply = f"ERRO REAL: {str(e)}"

            # 5. Adiciona bot msg ao histórico
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)


#mudanca