# Synthia Foundation

Sistema de **memória de contexto persistente** para LLMs locais (Ollama/LM Studio). Funciona como uma **fundação de módulos magnéticos** que mantêm boa saúde através de ciclos de uso: cada interação fortalece a conexão entre informações relevantes, permitindo que a memória **aprenda a se curar** e reorganizar seu mapa de contexto conforme a demanda aumenta.

A busca por similaridade via TF-IDF recupera não apenas as últimas memórias, mas os fragmentos mais relevantes para cada query — funcionando como memória a longo prazo para agentes offline.

---

## O que faz

1. **Adicionar** memórias de texto (user, assistant, documentos, notas, entrevistas)
2. **Persistir** em JSON — não perde nada entre sessões
3. **Buscar** por similaridade com query — recupera *relevantes*, não os últimos
4. **Injetar** contexto automaticamente em cada interação com LLM local

---

## Requisitos

- Python 3.11+
- Ollama rodando localmente

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Adicionar um documento à memoria
python context_memory.py --add "Resultados da pesquisa sobre nanotecnologia..." --memory pesquisa.json

# Buscar relevancia
python context_memory.py --search "nanotecnologia e aplicacoes" --memory pesquisa.json

# Chat com memoria (modo interativo)
python context_memory.py --chat --memory pesquisa.json

# Na sessao chat: cada pergunta do usuario busca contexto relevante
# e injeta-o automaticamente no prompt do LLM
```

## Como funciona a recuperação

| Passo | Descrição |
|-------|-----------|
| 1. Tokenização | Extrai palavras-chave (>=3 chars) da query e de cada memória |
| 2. TF vetor | Constrói vetor de frequência para query e memórias sobre vocabulário unificado |
| 3. Cosine similarity | Score entre 0 e 1 entre query e cada entry |
| 4. Rank + cutoff | Retorna top-5 com score > 0.01 |
| 5. Injeção | Concatena memórias recuperadas como `system` message antes do prompt |

## Estrutura do JSON

```json
[
  {
    "id": 1,
    "content": "Resultados da pesquisa...",
    "source": "paper_2024_1",
    "tags": ["biotecnologia"],
    "metadata": {"autor": "Dr. Silva"},
    "timestamp": "2026-05-26T03:00:00"
  }
]
```

## Por que nao usar vector DB?

- **100% local**: nenhuma dependência externa (nada de Pinecone/Weaviate/Chroma)
- **JSON simples**: backup, versionamento e inspeção humana trivial
- **Leve**: com <1000 documents a busca gasta <100ms
- **Zero infra**: nao precisa de servicos, containers ou GPU extra

Ideal para pesquisa e comunicação cientifica de pequena/média escala onde **controle total dos dados** e **compliance LGPD** sao prioridade.

---

**Stack:** Python, JSON, TF-IDF, Ollama
