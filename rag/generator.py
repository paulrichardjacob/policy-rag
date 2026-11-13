"""Turn retrieved chunks into an answer.

Three ways to answer:
1. Ollama  - an LLM running on the local PC. Free and private.
2. API     - a hosted LLM behind an OpenAI-compatible endpoint (OpenAI, Groq,
             OpenRouter and others use the same request format).
3. Extractive fallback - quotes the most relevant passage as it is written in
   the documents. Needs no model, so the app still works with no LLM at all.

Options 1 and 2 get the same prompt, so the only thing that changes is where
the model runs.
"""

import re

import requests

from .bm25 import tokenize

NOT_FOUND = "I couldn't find this in the policy documents."

SYSTEM_PROMPT = (
    "You answer questions about an insurance policy. Use only the numbered context "
    "passages. Cite the passages you used like [1] or [2]. Keep the answer short and "
    f'factual. If the context does not contain the answer, reply exactly: "{NOT_FOUND}"'
)

THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_prompt(question: str, hits: list) -> str:
    context = "\n\n".join(
        f"[{i}] ({h.chunk.source} - {h.chunk.section})\n{h.chunk.text}" for i, h in enumerate(hits, 1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def build_messages(question: str, hits: list) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits)},
    ]


def clean(text: str) -> str:
    # Reasoning models such as deepseek-r1 wrap their scratch work in <think> tags.
    return THINK.sub("", text).strip()


# ---------- Option 1: Ollama on the local PC ----------

def ollama_available(host: str, timeout: float = 1.5) -> bool:
    try:
        return requests.get(f"{host}/api/tags", timeout=timeout).ok
    except requests.RequestException:
        return False


def generate_with_ollama(question: str, hits: list, model: str, host: str, timeout: float = 120) -> str:
    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": build_messages(question, hits),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return clean(response.json()["message"]["content"])


# ---------- Option 2: OpenAI-compatible API ----------

def generate_with_api(question: str, hits: list, model: str, api_base: str, api_key: str, timeout: float = 60) -> str:
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "temperature": 0, "messages": build_messages(question, hits)},
        timeout=timeout,
    )
    response.raise_for_status()
    return clean(response.json()["choices"][0]["message"]["content"])


# ---------- Option 3: no LLM ----------

def extractive_answer(question: str, hits: list) -> str:
    """Quote the best passage that shares at least one term with the question.

    I first tried picking single sentences by word overlap, but the sentence
    with the answer often uses different words from the question, so the
    fallback quotes the whole passage instead.
    """
    terms = set(tokenize(question))
    for number, hit in enumerate(hits, 1):
        if terms & set(tokenize(hit.chunk.indexed_text)):
            return f"{hit.chunk.text} [{number}]"
    return NOT_FOUND
