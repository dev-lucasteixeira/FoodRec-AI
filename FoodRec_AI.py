import sqlite3
import uuid
import requests
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

load_dotenv()

# Banco de dados  - Histórico de Pedidos
DB_NAME = "foodrec_memory.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pedidos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            nome_usuario TEXT,
            cpf_usuario TEXT,
            restaurante TEXT,
            categoria TEXT,
            data_hora TEXT
        )
    ''')
    conn.commit()
    conn.close()

def salvar_pedido(user_id, nome, cpf, restaurante, categoria):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    data_hoje = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO pedidos (user_id, nome_usuario, cpf_usuario, restaurante, categoria, data_hora) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, nome, cpf, restaurante, categoria, data_hoje)
    )
    conn.commit()
    conn.close()
    print(f"Pedido salvo: {restaurante} ({categoria})")

def recuperar_historico(cpf_usuario):
    """
    Tenta achar o usuário pelo nome.
    Retorna: (user_id, lista_de_pedidos)
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM pedidos WHERE cpf_usuario = ?", (cpf_usuario,))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return str(uuid.uuid4()), []
    
    user_id = rows[0]['user_id']
    historico_formatado = []
    for row in rows:
        historico_formatado.append({
            "restaurante": row['restaurante'],
            "categoria": row['categoria'],
            "prato": "Desconhecido",
            "data_hora": row['data_hora']
        })
        
    return user_id, historico_formatado

# Localização do Usuário (via IP)

def get_localizacao_real():
    print("Rastreando sua localização via IP...")
    
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        response.raise_for_status()
        
        dados = response.json()

        cidade = dados["city"]
        estado = dados["region"]
        pais = dados["countryCode"]
        
        local_detectado = f"{cidade}, {estado} ({pais})"
        print(f"Localização confirmada: {local_detectado}")
        
        return local_detectado

    except Exception as e:
        print(f"Não consegui detectar automaticamente (Erro: {e}).")
        local_manual = input("Por favor, digite sua Cidade e Estado (Ex: Curitiba, PR): ").strip()
        
        if not local_manual:
            return "São Paulo, SP"
        
        return local_manual

# Criando a IA

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
tool_busca = TavilySearchResults(max_results=5)

#Modelagem do State
class Pedido(TypedDict):
    categoria: str    
    prato: str        
    data_hora: str    

class AgentState(TypedDict):
    user_id: str
    nome: str
    cpf: str
    localizacao: str
    historico_pedidos: List[Pedido]
    perfil_resumo: Optional[str]
    query_busca: Optional[str]
    restaurantes_encontrados: List[dict]
    decisao_usuario: Optional[str]
    url_restaurante: Optional[str]
    restaurante_escolhido: Optional[dict]
    detalhes_finais: Optional[str]
    erro_scraping: Optional[bool]
    mensagem_final: Optional[str]
    resultados_brutos: List[dict] 
    feedback_validacao: Optional[str] 
    tentativas_busca: int

# Funções dos Nós

def node_entrevistador(state):
    nome = state['nome']
    
    if state.get("restaurantes_encontrados"):
        print(f"\nENTREVISTADOR: Entendi, hoje você quer inovar!")
        pergunta = f"{nome}, esqueça o histórico então. O que você está com vontade de comer AGORA?"
        
    else:
        print(f"\nENTREVISTADOR: Olá {nome}, seja bem-vindo!")
        chain_pergunta = ChatPromptTemplate.from_template(
            "Faça uma pergunta curta e simpática para descobrir o que {nome} quer comer em {localizacao}."
        ) | llm | StrOutputParser()
        pergunta = chain_pergunta.invoke(state)

    print(f"FoodRec AI: {pergunta}")
    resposta = input("Você: ")
    
    analise_prompt = ChatPromptTemplate.from_template(
        "O usuário disse: '{resposta}'. Local: '{localizacao}'. Gere uma query de busca Google Maps. Apenas a query."
    )
    chain_analise = analise_prompt | llm | StrOutputParser()
    query = chain_analise.invoke({"resposta": resposta, "localizacao": state["localizacao"]})
    
    return {
        "query_busca": query, 
        "perfil_resumo": resposta, 
        "restaurantes_encontrados": [],
        "tentativas_busca": 0 
    }

def node_analista(state):
    historico = state["historico_pedidos"]
    ultimos = historico[-5:]
    
    resumo_textual = ", ".join([f"{p['categoria']} no {p['restaurante']}" for p in ultimos])
    
    print(f"   -> O usuário costuma pedir: {resumo_textual}")
    
    prompt_analise = """
    Você é um assistente pessoal focado em REPETIÇÃO DE PADRÃO.
    
    O usuário está em: {localizacao}
    
    HISTÓRICO DE PEDIDOS RECENTES:
    [{historico}]
    
    SUA MISSÃO:
    Identifique a categoria de comida que ele MAIS pede (Ex: se pediu pizza 3 vezes, ele quer pizza).
    Gere uma query de busca para o Google Maps focada nos "Melhores [CATEGORIA FAVORITA]" na cidade.
    
    Exemplo: Se ele pede muito Hambúrguer, busque "Melhor hamburgueria artesanal em {localizacao}".
    
    Responda APENAS a query de busca.
    """
    
    chain = ChatPromptTemplate.from_template(prompt_analise) | llm | StrOutputParser()
    query = chain.invoke({
        "localizacao": state['localizacao'], 
        "historico": resumo_textual
    })
    
    return {"query_busca": query, "perfil_resumo": f"Fã de {ultimos[-1]['categoria']}"}

def node_busca(state):
    query = state["query_busca"]
    resultados = tool_busca.invoke(query)
    return {"resultados_brutos": resultados, "tentativas_busca": state.get("tentativas_busca", 0)}

def node_web_scraping(state):
    url = state["url_restaurante"]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        loader = WebBaseLoader(url, header_template=headers)
        docs = loader.load()
        conteudo = docs[0].page_content[:4000]
        
        termos_bloqueio = [
            "enable javascript", 
            "access denied", 
            "verify you are human", 
            "captcha", 
            "bloqueador de anúncios",
            "pardon our interruption"
        ]
        
        if any(termo in conteudo.lower() for termo in termos_bloqueio):
            print(f"Site bloqueou o acesso (Detectou robô).")
            raise Exception("Conteúdo bloqueado por WAF")
            
        if len(conteudo) < 200:
             print(f"Conteúdo muito curto (possível erro).")
             raise Exception("Conteúdo insuficiente")

        print("Scraping realizado com sucesso!")
        return {"detalhes_finais": conteudo, "erro_scraping": False}
        
    except Exception as e:
        print(f"Falha no scraping: {e}. Ativando Plano B (Usar dados do Google).")
        return {"detalhes_finais": None, "erro_scraping": True}

def node_vendedor(state):
    print("\nEscrevendo recomendação...")
    
    perfil = state.get("perfil_resumo")
    restaurante = state.get("restaurante_escolhido")
    
    decisao = state.get("decisao_usuario", "")
    
    if decisao.isdigit() and restaurante:
        salvar_pedido(
            user_id=state["user_id"],
            nome=state["nome"],
            cpf=state["cpf"],
            restaurante=restaurante.get("nome", "Desconhecido"),
            categoria=state.get("perfil_resumo", "Geral")
        )
    
    if state.get("erro_scraping") or not state.get("detalhes_finais"):
        prompt = f"""
        Atue como um amigo local confiável. O usuário quer: {perfil}.
        O site do restaurante não carregou, mas recomende: {restaurante.get('nome')} ({restaurante.get('endereco')}).
        Diga que é uma "Aposta Segura".
        """
    else:
        detalhes = state["detalhes_finais"]
        prompt = f"""
        Atue como um Sommelier Gastronômico.
        Analise o site: {detalhes}
        Recomende pratos para o gosto: "{perfil}".
        Finalize com o endereço: {restaurante.get('endereco')}.
        """
        
    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
    resposta = chain.invoke({})
    
    print(f"\nRECOMENDAÇÃO FINAL:\n{resposta}")
    return {"mensagem_final": resposta}

def node_validador(state):
    resultados = state["resultados_brutos"]
    query_atual = state["query_busca"]
    tentativas = state.get("tentativas_busca", 0)

    if tentativas >= 3:
        pass 

    prompt_validacao = """
    Você é um validador de busca.
    
    Termo buscado: "{query}"
    Resultados encontrados:
    ---
    {dados}
    ---
    
    Responda apenas APROVADO se os resultados contiverem restaurantes reais.
    Responda REPROVADO se forem apenas links quebrados ou irrelevantes.
    """
    
    chain_validacao = ChatPromptTemplate.from_template(prompt_validacao) | llm | StrOutputParser()
    
    parecer = chain_validacao.invoke({
        "query": query_atual, 
        "dados": str(resultados) 
    })

    if "REPROVADO" in parecer and tentativas < 3:
        return {
            "feedback_validacao": "REPROVADO", 
            "tentativas_busca": tentativas + 1,
            "query_busca": query_atual + " endereço horário"
        }

    prompt_formatacao = """
    Você é um assistente de dados.
    Analise os textos brutos abaixo e extraia informações estruturadas.
    
    ENTRADA BRUTA:
    {dados_brutos}
    
    SAÍDA ESPERADA (JSON Lista):
    Retorne uma lista de objetos JSON. Para cada restaurante, tente extrair:
    - "nome": O nome do local (limpe emojis)
    - "endereco": O endereço encontrado (se não achar, coloque "Endereço não informado")
    - "horario": O horário de funcionamento (se não achar, coloque "Ver no site")
    - "url": O link original (Copie exatamente da entrada)
    
    Importante: Retorne APENAS o JSON, sem markdown (```json).
    """
    
    chain_fmt = ChatPromptTemplate.from_template(prompt_formatacao) | llm | JsonOutputParser()
    
    try:
        lista_limpa = chain_fmt.invoke({"dados_brutos": str(resultados)})
        
        return {
            "feedback_validacao": "APROVADO", 
            "restaurantes_encontrados": lista_limpa 
        }
    except Exception as e:
        print(f"{e}. Usando dados brutos.")
        return {"feedback_validacao": "APROVADO", "restaurantes_encontrados": resultados}

def node_apresentacao(state):
    opcoes = state["restaurantes_encontrados"]
    print("\n" + "="*30)
    print("MENU SELECIONADO")
    print("="*30)
    
    opcoes_formatadas = []
    
    for i, res in enumerate(opcoes):
        nome = res.get("nome", res.get("content", "")[:30])
        end = res.get("endereco", "Endereço não detectado")
        hora = res.get("horario", "Horário não detectado")
        
        print(f"\n[{i+1}] {nome}")
        print(f"Endereço: {end}")
        print(f"Horário: {hora}")
        
        opcoes_formatadas.append(res)

    print("\n[0]Nenhuma dessas (Falar com Entrevistador)")
    print("="*40)

    escolha = input("\nDigite o número da sua escolha: ").strip()
    
    update = {"decisao_usuario": escolha}
    
    if escolha.isdigit() and 0 < int(escolha) <= len(opcoes_formatadas):
        idx = int(escolha) - 1
        escolhido = opcoes_formatadas[idx]
        
        update["url_restaurante"] = escolhido.get('url')
        update["restaurante_escolhido"] = {
            "nome": escolhido.get("nome"),
            "nota": "?",
            "categoria": state.get("perfil_resumo")
        }
    else:
        update["url_restaurante"] = None
        
    return update

# --- ROTEAMENTO (Porteiros) ---
def route_user(state):
    if state.get("historico_pedidos"):
        return "analista_perfil"
    return "entrevistador"

def router_decisao(state):
    decisao = state.get("decisao_usuario", "").strip()
    
    if decisao.isdigit():
        numero = int(decisao)
        if numero == 0:
            return "voltar_entrevista"
        return "ir_scraping"
    
    return "voltar_entrevista"
    
def router_validacao(state):
    feedback = state.get("feedback_validacao", "")
    
    if "APROVADO" in feedback:
        return "mostrar_opcoes"
    else:
        return "refazer_busca"

# --- MONTAGEM DO GRAFO ---
workflow = StateGraph(AgentState)

workflow.add_node("entrevistador", node_entrevistador)
workflow.add_node("analista_perfil", node_analista)
workflow.add_node("buscador_maps", node_busca)
workflow.add_node("validador_busca", node_validador)      
workflow.add_node("apresentador", node_apresentacao)      
workflow.add_node("web_scraper", node_web_scraping)
workflow.add_node("vendedor_rag", node_vendedor)

workflow.set_conditional_entry_point(route_user, {"analista_perfil": "analista_perfil", "entrevistador": "entrevistador"})

workflow.add_edge("entrevistador", "buscador_maps")
workflow.add_edge("analista_perfil", "buscador_maps")
workflow.add_edge("buscador_maps", "validador_busca") 

workflow.add_conditional_edges(
    "validador_busca",
    router_validacao,
    {
        "refazer_busca": "buscador_maps",
        "mostrar_opcoes": "apresentador"
    }
)

workflow.add_conditional_edges(
    "apresentador",
    router_decisao,
    {
        "ir_scraping": "web_scraper",
        "voltar_entrevista": "entrevistador"
    }
)

workflow.add_edge("web_scraper", "vendedor_rag")
workflow.add_edge("vendedor_rag", END)

app = workflow.compile()

# Execução Principal
if __name__ == "__main__":
    # 1. Inicializa Banco de Dados
    init_db()

    print("\n" + "="*40)
    print("---- BEM-VINDO AO FOODREC AI ----")
    print("="*40)
        
    # 2. Input do Nome
    nome_usuario = input("Para começar, qual é o seu nome? ").strip()
    cpf_usuario = input("E qual é o seu CPF? ").strip()
        
    # 3. Recupera ID
    user_id, historico = recuperar_historico(cpf_usuario)
        
    if historico:
        print(f"Bem-vindo de volta, {nome_usuario}!")
    else:
        print(f"Prazer, {nome_usuario}! Criamos seu perfil.")
            
        # 4. Localização
    local_atual = get_localizacao_real()
        
        # 5. Monta o Input Inicial
    input_inicial = {
        "user_id": user_id,
        "nome": nome_usuario,
        "cpf": cpf_usuario,
        "localizacao": local_atual,
        "historico_pedidos": historico
    }
        
    # O app.invoke roda até chegar no nó END e depois o código continua abaixo
    app.invoke(input_inicial)
        
    # 7. MENSAGEM FINAL DE ENCERRAMENTO
    print("\n" + "="*40)
    print(" Obrigado por usar o FoodRec AI!")
    print("="*40)




