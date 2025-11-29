import streamlit as st
from supabase import create_client, Client
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import google.generativeai as gemini
import unicodedata
import random

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
    modelo_gemini = gemini.GenerativeModel("gemini-1.5-flash")
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
    'nome': ['Omelete de claras', 'Creme de abacate com cacau', 'Iogurte com granola caseira', 'Tapioca com queijo branco', 'Panqueca integral', 'Nhoque ao molho mediterraneo', 'Maminha ao molho de ervas', 'Risoto do mar', 'Ravioli de curcuma com alho-poro', 'Charutinho caipira', 'Estrogonofe de frango com berinjela', 'Sopa de cebola especial', 'Tilapia grelhada', 'Arroz de couve-flor', 'Sopa de legumes', 'Quiche de presunto', 'Refresco de melancia', 'Salada de frutas ao forno', 'Mix de castanhas', 'Frutas com chia'],
    'tipo_refeicao': ['cafe', 'cafe', 'cafe', 'cafe', 'cafe', 'principal', 'principal', 'principal', 'principal', 'principal', 'principal', 'principal', 'principal', 'principal', 'principal', 'lanche', 'lanche', 'lanche', 'lanche', 'lanche'],
    'tipo_diabetes': ['tipo2', 'ambos', 'ambos', 'tipo2', 'tipo1', 'tipo1', 'tipo2', 'ambos', 'tipo1', 'ambos', 'ambos', 'tipo2', 'tipo2', 'tipo2', 'ambos', 'tipo2', 'tipo1', 'tipo1', 'tipo2', 'ambos'],
    'ingredientes': ['Ovos, tomate, sal, azeite, ovo', 'Abacate, cacau em po, adocante natural, leite vegetal, castanhas, fruta', 'Iogurte natural, aveia, mel, castanhas, iogurte', 'Goma de tapioca, queijo branco, sal, queijo', 'Farinha integral, ovo, leite, acucar mascavo, ovo', 'Mandioquinha, clara, margarina, farinha, azeite, berinjela, abobrinha, tomate, vegetal', 'Maminha, sal, alho, azeite, manjericao, salsa, tomilho, creme de leite light, carne', 'Badejo, cebola, arroz integral, vinho branco, brocolis, creme de leite light, acafrao, peixe', 'Farinha de trigo, ovo, azeite, curcuma, queijo de minas light, alho-poro, manjericao, azeitona, queijo', 'Couve-manteiga, frango desfiado, cebola, milho verde, salsa, caldo de legumes, frango', 'Frango, pimenta, azeite, cebola, berinjela, tomate, mostarda, catchup light, creme de leite light, frango', 'Alho-poro, cebola, cebolinha, azeite, farinha, mostarda, caldo de galinha, leite em po desnatado, cebola', 'Tilapia, sal, limao, alho, peixe', 'Couve-flor, azeite, sal, alho, vegetal', 'Abobrinha, cenoura, alho, tomate, agua, vegetal', 'Iogurte, margarina, farinha, presunto magro, acelga, queijo parmesao light, queijo', 'Melancia, agua, iogurte desnatado, gengibre, fruta', 'Pessego diet, banana, manga, morango, fruta', 'Castanhas, nozes, amendoas, castanha', 'Morango, banana, chia, fruta']
}
df = pd.DataFrame(data)
df['descricao'] = [
    'Opção focada em proteína, com baixo teor de carboidratos.', 'Rico em gorduras saudáveis e fibras.', 'Mistura equilibrada de proteínas e fibras.', 'Carboidrato simples combinado com proteína.', 'Carboidrato de absorção mais lenta.', 'Prato de carboidrato complexo e legumes.', 'Carne magra com baixo teor de gordura.', 'Rico em fibras e proteínas.', 'Massa caseira com legumes e ingredientes naturais.', 'Prato completo com vegetais e proteína magra.', 'Versão adaptada, baixo carboidrato.', 'Sopa nutritiva com baixo teor calórico.', 'Opção leve e proteica.', 'Excelente substituto do arroz tradicional.', 'Sopa leve, rica em fibras.', 'Lanche salgado equilibrado.', 'Lanche refrescante.', 'Frutas assadas com creme dietético.', 'Fonte de gorduras boas e fibras.', 'Lanche rico em fibras e antioxidantes.'
]

# Função Principal de Recomendação
def recomendar_sistema_especialista(refeicao, diabetes, ingrediente_usuario):
    # Usando o DF carregado na memória
    db_local = df.copy()

    ref_norm = normalizar_texto(refeicao)
    dia_norm = normalizar_texto(diabetes).replace(" ", "")
    ing_norm = normalizar_texto(ingrediente_usuario)

    db_local['ref_norm'] = db_local['tipo_refeicao'].apply(normalizar_texto)
    db_local['dia_norm'] = db_local['tipo_diabetes'].apply(normalizar_texto)

    ref_filtro = ref_norm
    if ref_norm in ['almoco', 'jantar']:
        ref_filtro = 'principal'

    if "tipo1" in dia_norm:
        tipos_validos = ['tipo1', 'ambos']
        tipo_user = "Tipo 1"
    elif "tipo2" in dia_norm:
        tipos_validos = ['tipo2', 'ambos']
        tipo_user = "Tipo 2"
    else:
        tipos_validos = ['tipo1', 'tipo2', 'ambos']
        tipo_user = "Desconhecido"

    receita_escolhida_nome = None
    aviso_ingrediente = None
    mismatch_info = None 

    candidatos_compativeis = db_local[(db_local['ref_norm'] == ref_filtro) & (db_local['dia_norm'].isin(tipos_validos))]
    candidatos_all_types = db_local[(db_local['ref_norm'] == ref_filtro)]

    # P1/P2: Busca por Ingrediente
    if ing_norm and ing_norm not in ['nao', 'na', '']:
        candidatos_P1 = []
        for index, row in candidatos_compativeis.iterrows():
            if ing_norm in normalizar_texto(row['ingredientes']):
                candidatos_P1.append(row['nome'])
        
        if candidatos_P1:
            receita_escolhida_nome = random.choice(candidatos_P1)
        else:
            # P2: Mismatch search
            candidatos_P2 = []
            for index, row in candidatos_all_types.iterrows():
                if ing_norm in normalizar_texto(row['ingredientes']):
                    candidatos_P2.append(row['nome'])
            
            if candidatos_P2:
                receita_escolhida_nome = random.choice(candidatos_P2)
                receita_obj_mismatch = db_local[db_local['nome'] == receita_escolhida_nome].iloc[0]
                if receita_obj_mismatch['dia_norm'] not in tipos_validos:
                    mismatch_info = (receita_obj_mismatch['tipo_diabetes'], tipo_user)
            else:
                aviso_ingrediente = f"⚠️ Não encontrei nenhuma receita com '{ingrediente_usuario}' para {refeicao}. Selecionei uma alternativa compatível:"

    # P3: Fallback Aleatório
    if receita_escolhida_nome is None and not candidatos_compativeis.empty:
        receita_escolhida_nome = candidatos_compativeis.sample(1)['nome'].values[0]
    elif receita_escolhida_nome is None:
        return None, None, None

    receita_final = db_local[db_local['nome'] == receita_escolhida_nome].iloc[0]
    return receita_final, aviso_ingrediente, mismatch_info

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
            rec_obj, aviso, mismatch = recomendar_sistema_especialista(
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
                bot_reply = "Tive um problema de conexão. Tente novamente."

            # 5. Adiciona bot msg ao histórico
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
