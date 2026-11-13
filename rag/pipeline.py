"""Glue: build an index from documents, then retrieve and answer."""

from dataclasses import dataclass, field

import requests

from .chunker import chunk_document
from .config import MIN_SCORE, PROVIDERS, Settings
from .embeddings import make_embedder
from .generator import (
    NOT_FOUND,
    extractive_answer,
    generate_with_api,
    generate_with_ollama,
    ollama_available,
)
from .loader import load_folder
from .store import VectorIndex


def build_index(docs_folder, index_folder, settings: Settings = None) -> VectorIndex:
    settings = settings or Settings()
    chunks = []
    for name, text in load_folder(docs_folder).items():
        chunks += chunk_document(
            text, name, settings.chunk_words, settings.chunk_overlap, start_id=len(chunks)
        )
    index = VectorIndex.build(chunks, make_embedder(settings.embedder, settings.embed_model))
    index.save(index_folder)
    return index


@dataclass
class Answer:
    question: str
    text: str
    hits: list = field(default_factory=list)
    generator: str = "extractive"  # "ollama:<model>", "api:<model>", "extractive" or "none"
    found: bool = True
    note: str = ""  # explains why the LLM was not used, if it was asked for


class RAGPipeline:
    def __init__(self, index: VectorIndex, settings: Settings = None):
        self.index = index
        self.settings = settings or Settings()
        self.min_score = MIN_SCORE.get(index.embedder.name, 0.1)

    @classmethod
    def from_folder(cls, index_folder, settings: Settings = None):
        return cls(VectorIndex.load(index_folder), settings)

    def ask(self, question: str, k: int = None, mode: str = "hybrid", llm: str = "auto") -> Answer:
        """llm: "ollama", "api", "off", or "auto" to use LLM_PROVIDER from the settings."""
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")
        provider = self.settings.llm_provider if llm == "auto" else llm
        if provider not in PROVIDERS:
            raise ValueError(f"LLM provider must be one of {PROVIDERS}, got '{provider}'")

        hits = self.index.search(question, k or self.settings.top_k, mode)

        # Guardrail: if nothing retrieved is similar enough, do not answer.
        if not hits or max(h.similarity for h in hits) < self.min_score:
            return Answer(question, NOT_FOUND, hits, "none", found=False)

        text, label, note = self._generate(provider, question, hits)
        if text is None:  # no LLM, or the LLM call failed
            text, label = extractive_answer(question, hits), "extractive"
        found = NOT_FOUND.lower() not in text.lower()
        return Answer(question, text, hits, label, found, note)

    def _generate(self, provider: str, question: str, hits: list):
        """Return (text, label, note). text is None when no LLM answer was produced."""
        s = self.settings
        if provider == "off":
            return None, "", ""
        try:
            if provider == "ollama":
                if not ollama_available(s.ollama_host):
                    return None, "", f"Ollama is not running at {s.ollama_host}, so the passage is quoted instead."
                text = generate_with_ollama(question, hits, s.ollama_model, s.ollama_host)
                return text, f"ollama:{s.ollama_model}", ""
            if not s.api_key:
                return None, "", "No API key set (LLM_API_KEY), so the passage is quoted instead."
            text = generate_with_api(question, hits, s.api_model, s.api_base, s.api_key)
            return text, f"api:{s.api_model}", ""
        except (requests.RequestException, KeyError, IndexError, ValueError) as error:
            return None, "", f"The {provider} call failed ({type(error).__name__}), so the passage is quoted instead."
