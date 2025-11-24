<div align="center">

  # 🍔 FoodRec AI: Sistema de Recomendação Agêntico
  
  **Uma IA Autônoma que descobre, valida e recomenda restaurantes baseado no seu perfil gastronômico.**

  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange?style=for-the-badge)](https://langchain.com)
  [![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
  [![SQLite](https://img.shields.io/badge/SQLite-Persistence-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
  [![Tavily](https://img.shields.io/badge/Tavily-Search_Tool-red?style=for-the-badge)](https://tavily.com)

  [Ver Demo](#-demo) • [Arquitetura](#-arquitetura) • [Como Rodar](#-como-rodar) • [Tecnologias](#-tech-stack)

</div>

---

## 💡 Sobre o Projeto

O **FoodRec AI** não é apenas um chatbot. É um **Sistema Multi-Agente** capaz de atuar como um *Sommelier Gastronômico Pessoal*.

Diferente de sistemas de recomendação estáticos, ele utiliza **LangGraph** para criar um fluxo de pensamento cíclico e adaptativo. Ele lida com usuários novos (*Cold Start*), aprende com o histórico, valida dados da web em tempo real e utiliza técnicas de **Self-Correction** para garantir a qualidade da resposta.

> 🚀 **Objetivo:** Projeto desenvolvido para demonstrar proficiência em Arquitetura de IA Generativa, RAG (Retrieval-Augmented Generation) e Engenharia de Software.

---

## ⚙️ Arquitetura do Sistema (LangGraph)

O cérebro do projeto é uma máquina de estados finitos (State Graph). Abaixo está o fluxo de decisão autônomo dos agentes:

```mermaid
graph TD
    Start([Início]) --> Router{"Tem Histórico?"}
    
    %% Fluxo Cold Start vs Usuário Recorrente
    Router -- Não --> Entrevistador["🤖 Entrevistador"]
    Router -- Sim --> Analista["🧠 Analista de Perfil"]
    
    Entrevistador --> Busca["🌍 Buscador (Tavily)"]
    Analista --> Busca
    
    %% Loop de Validação e Auto-Correção
    Busca --> Validador{"🧐 Validador"}
    Validador -- Reprovado (Loop) --> Busca
    Validador -- Aprovado --> Apresentador["🍽️ Apresentador de Menu"]
    
    %% Decisão Humana
    Apresentador --> Decisao{"Usuário Gostou?"}
    Decisao -- "Não (0)" --> Entrevistador
    Decisao -- "Sim (Escolha)" --> Scraper["🕷️ Web Scraper"]
    
    %% RAG Final
    Scraper --> Vendedor["🤵 Vendedor RAG"]
    Vendedor --> End([Fim / Persistência])

    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style Validador fill:#bbf,stroke:#333,stroke-width:2px
