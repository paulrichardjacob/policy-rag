"""Settings. Everything is read from environment variables or a .env file,
so no keys or machine-specific values are hard-coded."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(path=".env"):
    """Read KEY=VALUE lines from a .env file into os.environ.
    Variables that are already set in the shell are not overwritten."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


def env(name, default):
    return field(default_factory=lambda: os.getenv(name, default))


@dataclass
class Settings:
    # --- retrieval ---
    # "auto" uses sentence-transformers if it is installed, otherwise TF-IDF.
    embedder: str = env("RAG_EMBEDDER", "auto")
    embed_model: str = env("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_words: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_WORDS", "120")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "30")))
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "4")))

    # --- answering ---
    # "ollama" = model running on this PC, "api" = hosted model, "off" = no LLM.
    llm_provider: str = env("LLM_PROVIDER", "ollama")

    # Option 1: Ollama on the local machine
    ollama_host: str = env("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = env("OLLAMA_MODEL", "deepseek-r1:1.5b")

    # Option 2: any OpenAI-compatible API (OpenAI, Groq, OpenRouter, ...)
    api_base: str = env("LLM_API_BASE", "https://api.openai.com/v1")
    api_model: str = env("LLM_API_MODEL", "gpt-4o-mini")
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""))


PROVIDERS = ("ollama", "api", "off")

# If the best retrieved chunk scores below this cosine similarity, the pipeline
# says "not found" instead of guessing. The two embedders have different score
# ranges, so each gets its own default.
MIN_SCORE = {"tfidf": 0.08, "minilm": 0.25}
