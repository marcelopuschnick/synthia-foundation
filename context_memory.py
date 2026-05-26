#!/usr/bin/env python3
"""
context_memory.py
Sistema de memoria de contexto para agentes de IA locais (Ollama/LM Studio).
Persiste conversas e contexto em JSON, com busca por similaridade simples (TF-IDF)
para recuperar memórias relevantes em tempo real.

Ideal para:
- Projetos de pesquisa com multiplos papers e entrevistas
- Documentação de processos internos (onde o historico importa)
- Memoria longa de comunicacao cientifica
"""
import json, math, os, re, time
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
import requests

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL       = os.getenv("OLLAMA_MODEL", "mistral:latest")


def _tokenize(text: str) -> list:
    """Tokenizacao simples: minusculas, remove pontuacao menores."""
    return re.findall(r"[a-zA-Z\u00C0-\u00FF]{3,}", text.lower())


def _vectorize(doc: str, vocab: list) -> list:
    """TF vetor para um documento sobre vocabulario."""
    tokens = _tokenize(doc)
    if not tokens:
        return [0.0] * len(vocab)
    tf = Counter(tokens)
    return [tf.get(w, 0) / len(tokens) for w in vocab]


class ContextMemory:
    """Memoria de contexto persistente com busca por similaridade."""

    def __init__(self, memory_file: str = "memory.json"):
        self.memory_file = Path(memory_file)
        self.entries: List[Dict] = []
        self._load()

    def _load(self):
        if self.memory_file.exists():
            try:
                self.entries = json.loads(self.memory_file.read_text(encoding="utf-8"))
            except Exception:
                self.entries = []

    def save(self):
        """Persistir memoria em JSON."""
        self.memory_file.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, content: str, source: str = "user", tags: list = None, metadata: dict = None):
        """Adiciona uma entrada na memoria."""
        entry = {
            "id": len(self.entries) + 1,
            "content": content,
            "source": source,
            "tags": tags or [],
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.entries.append(entry)
        self.save()
        return entry["id"]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Busca entradas relevantes por similaridade de cosseno."""
        if not self.entries:
            return []

        vocab = list(set(_tokenize(query)))
        for e in self.entries:
            vocab.extend(_tokenize(e["content"]))
        vocab = list(set(vocab))

        q_vec = _vectorize(query, vocab)
        scores = []
        for e in self.entries:
            e_vec = _vectorize(e["content"], vocab)
            score = _cosine_similarity(q_vec, e_vec)
            scores.append({"entry": e, "score": score})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return [s["entry"] for s in scores[:top_k] if s["score"] > 0.01]

    def get_context_for_prompt(self, query: str, max_tokens: int = 1500) -> str:
        """Retorna bloco de contexto concatenado para injetar em prompt LLM."""
        results = self.search(query, top_k=5)
        context = []
        total_len = 0
        for r in results:
            chunk = f"[{r['source']}] {r['content'][:300]}"
            if total_len + len(chunk) > max_tokens:
                break
            context.append(chunk)
            total_len += len(chunk)
        return "\n\n".join(context)

    def summary(self) -> Dict:
        """Resumo da memoria."""
        return {
            "total_entries": len(self.entries),
            "sources": list(set(e["source"] for e in self.entries)),
            "last_update": self.entries[-1]["timestamp"] if self.entries else None,
        }


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chat_with_memory(memory: ContextMemory, user_input: str, system_prompt: str = None) -> str:
    """Chat com memoria de contexto injetada automaticamente."""
    contexto = memory.get_context_for_prompt(user_input)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if contexto:
        messages.append({"role": "system", "content": f"Contexto relevante da memória:\n{contexto}"})

    messages.append({"role": "user", "content": user_input})

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": MODEL, "messages": messages,
                  "options": {"temperature": 0.3, "num_predict": 2000, "top_p": 0.8},
                  "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        return f"[ERRO] {e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memoria de Contexto para IA Local")
    parser.add_argument("--add", metavar="TEXTO", help="Adicionar entrada")
    parser.add_argument("--search", metavar="QUERY", help="Buscar por similaridade")
    parser.add_argument("--chat", action="store_true", help="Modo chat com memoria")
    parser.add_argument("--memory", default="memory.json", help="Arquivo de memoria")
    args = parser.parse_args()

    mem = ContextMemory(args.memory)

    if args.add:
        idx = mem.add(args.add, source="CLI")
        print(f"[{idx}] Entrada adicionada.")
    elif args.search:
        resultados = mem.search(args.search)
        for r in resultados:
            print(f"\n[{r['source']}] Score: {r.get('_score','?')}\n{r['content'][:300]}")
    elif args.chat:
        print("Modo chat. Digite /sair para encerrar.")
        while True:
            q = input("\nVoce: ").strip()
            if q.lower() in ("/sair", "exit", "quit"):
                break
            resposta = chat_with_memory(mem, q)
            print(f"\n[IA]: {resposta}")
    else:
        print(f"Memoria: {args.memory}")
        print(f"Entradas: {mem.summary()}")
