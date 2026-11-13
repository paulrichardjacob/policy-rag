"""Tests run with the TF-IDF embedder so they need no model download."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from eval.evaluate import evaluate
from rag.bm25 import BM25, tokenize
from rag.chunker import chunk_document, split_sections, window
from rag.config import Settings
from rag.generator import NOT_FOUND, build_prompt, extractive_answer
from rag.pipeline import RAGPipeline, build_index
from rag.store import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "sample_policies"


def tfidf_settings(**overrides) -> Settings:
    settings = Settings()
    settings.embedder = "tfidf"
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("index")
    build_index(DOCS, folder, tfidf_settings())
    return folder


@pytest.fixture(scope="module")
def pipeline(index_dir):
    return RAGPipeline.from_folder(index_dir, tfidf_settings())


# ---------- chunker ----------

def test_split_sections_uses_headings():
    text = "# Title\nintro\n## Grace period\n30 days\n## Free look\n15 days"
    assert split_sections(text) == [("Title", "intro"), ("Grace period", "30 days"), ("Free look", "15 days")]


def test_window_overlap():
    pieces = window(list(range(10)), size=4, overlap=2)
    assert pieces[0] == [0, 1, 2, 3] and pieces[1] == [2, 3, 4, 5]
    assert pieces[-1][-1] == 9  # nothing dropped at the end


def test_chunks_keep_section_and_respect_size():
    text = "## Long section\n" + " ".join(f"w{i}" for i in range(300))
    chunks = chunk_document(text, "doc.md", max_words=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.section == "Long section" and len(c.text.split()) <= 100 for c in chunks)
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_document("some text", "doc.md", max_words=10, overlap=10)


# ---------- bm25 ----------

def test_bm25_ranks_matching_doc_first():
    bm25 = BM25(["ambulance charges are covered", "the grace period is 30 days", "dental treatment excluded"])
    scores = bm25.scores("what is the grace period")
    assert scores.index(max(scores)) == 1
    assert bm25.scores("zzz unknown") == [0.0, 0.0, 0.0]


def test_tokenize_drops_stopwords():
    assert tokenize("What is the Grace Period?") == ["grace", "period"]


# ---------- index ----------

def test_index_round_trip(index_dir):
    index = VectorIndex.load(index_dir)
    meta = json.loads((index_dir / "meta.json").read_text())
    assert meta["num_chunks"] == len(index.chunks) == index.embeddings.shape[0]
    assert len(meta["sources"]) == 6


@pytest.mark.parametrize("mode", ["hybrid", "dense", "bm25"])
def test_search_finds_grace_period(index_dir, mode):
    hits = VectorIndex.load(index_dir).search("grace period for paying the premium", k=3, mode=mode)
    assert hits[0].chunk.section == "Grace period"


def test_search_rejects_unknown_mode(index_dir):
    with pytest.raises(ValueError):
        VectorIndex.load(index_dir).search("anything", mode="magic")


# ---------- pipeline ----------

def test_extractive_answer_cites_source(pipeline):
    answer = pipeline.ask("How much is paid for an ambulance?", llm="off")
    assert answer.found and answer.generator == "extractive"
    assert "Rs 2,000" in answer.text and "[1]" in answer.text


def test_out_of_scope_question_is_refused(pipeline):
    answer = pipeline.ask("Who won the cricket world cup in 2011?", llm="off")
    assert not answer.found and answer.text == NOT_FOUND


def test_empty_question_raises(pipeline):
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_prompt_contains_numbered_context(pipeline):
    hits = pipeline.index.search("free-look period", k=2)
    prompt = build_prompt("What is the free-look period?", hits)
    assert "[1]" in prompt and "[2]" in prompt and "Question: What is the free-look period?" in prompt


def test_extractive_returns_not_found_without_overlap(pipeline):
    hits = pipeline.index.search("free-look period", k=1)
    assert extractive_answer("xylophone", hits) == NOT_FOUND


# ---------- Ollama (mocked with a local HTTP server) ----------

class FakeOllama(BaseHTTPRequestHandler):
    received = {}

    def do_GET(self):
        self._send({"models": []})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.received = body
        FakeOllama.path = self.path
        FakeOllama.auth = self.headers.get("Authorization")
        text = "<think>let me check</think>The grace period is 30 days [1]."
        if self.path.endswith("/chat/completions"):  # OpenAI-style API
            self._send({"choices": [{"message": {"content": text}}]})
        else:  # Ollama
            self._send({"message": {"content": text}})

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_ollama():
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_llm_answer_strips_think_tags(index_dir, fake_ollama):
    settings = tfidf_settings(ollama_host=fake_ollama, ollama_model="test-model")
    answer = RAGPipeline.from_folder(index_dir, settings).ask("What is the grace period?", llm="ollama")
    assert answer.text == "The grace period is 30 days [1]."
    assert answer.generator == "ollama:test-model"
    sent = FakeOllama.received
    assert sent["model"] == "test-model" and "Grace period" in sent["messages"][1]["content"]


def test_falls_back_to_extractive_when_ollama_is_down(index_dir):
    settings = tfidf_settings(ollama_host="http://127.0.0.1:9")  # nothing listens here
    answer = RAGPipeline.from_folder(index_dir, settings).ask("What is the grace period?", llm="ollama")
    assert answer.generator == "extractive" and "30 days" in answer.text


def test_api_answer_sends_key_and_model(index_dir, fake_ollama):
    settings = tfidf_settings(api_base=fake_ollama + "/v1", api_model="api-model", api_key="secret")
    answer = RAGPipeline.from_folder(index_dir, settings).ask("What is the grace period?", llm="api")
    assert answer.text == "The grace period is 30 days [1]."
    assert answer.generator == "api:api-model"
    assert FakeOllama.path == "/v1/chat/completions" and FakeOllama.auth == "Bearer secret"
    assert FakeOllama.received["model"] == "api-model"


def test_api_without_key_falls_back(index_dir):
    settings = tfidf_settings(api_key="")
    answer = RAGPipeline.from_folder(index_dir, settings).ask("What is the grace period?", llm="api")
    assert answer.generator == "extractive" and "API key" in answer.note


def test_auto_uses_provider_from_settings(index_dir):
    settings = tfidf_settings(llm_provider="off")
    answer = RAGPipeline.from_folder(index_dir, settings).ask("What is the grace period?")
    assert answer.generator == "extractive" and answer.note == ""


def test_unknown_provider_raises(pipeline):
    with pytest.raises(ValueError):
        pipeline.ask("What is the grace period?", llm="magic")


# ---------- evaluation ----------

def test_hybrid_retrieval_quality(index_dir):
    # Regression floor for the TF-IDF baseline. The questions are paraphrased
    # on purpose, so keyword-style retrieval is not expected to be perfect.
    results = evaluate(VectorIndex.load(index_dir))
    assert results["hybrid"]["hit@5"] >= 0.7 and results["hybrid"]["mrr"] >= 0.5
