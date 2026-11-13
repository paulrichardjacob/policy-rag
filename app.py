"""Streamlit chat app.  Run with:  streamlit run app.py"""

import tempfile
from pathlib import Path

import streamlit as st

from rag.config import Settings
from rag.generator import ollama_available
from rag.pipeline import RAGPipeline, build_index
from rag.store import MODES

SAMPLE_DOCS = Path("data/sample_policies")
INDEX_DIR = Path("index")

st.set_page_config(page_title="Policy RAG", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Building the index...")
def sample_pipeline() -> RAGPipeline:
    if not (INDEX_DIR / "meta.json").exists():
        build_index(SAMPLE_DOCS, INDEX_DIR)
    return RAGPipeline.from_folder(INDEX_DIR)


def pipeline_from_uploads(files) -> RAGPipeline:
    """Index the uploaded files in a temporary folder."""
    work = Path(tempfile.mkdtemp(prefix="policy_rag_"))
    docs = work / "docs"
    docs.mkdir()
    for f in files:
        (docs / Path(f.name).name).write_bytes(f.getvalue())
    return RAGPipeline(build_index(docs, work / "index"))


settings = Settings()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload your own policy files (optional)", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if uploads:
        key = tuple((f.name, f.size) for f in uploads)
        if st.session_state.get("upload_key") != key:
            try:
                with st.spinner("Indexing your files..."):
                    st.session_state.upload_pipeline = pipeline_from_uploads(uploads)
                st.session_state.upload_key = key
                st.session_state.messages = []
            except ValueError as error:
                st.error(str(error))
        pipeline = st.session_state.get("upload_pipeline") or sample_pipeline()
    else:
        st.session_state.pop("upload_key", None)
        st.session_state.pop("upload_pipeline", None)
        pipeline = sample_pipeline()

    sources = sorted({c.source for c in pipeline.index.chunks})
    st.caption(
        f"{len(pipeline.index.chunks)} chunks from {len(sources)} files, "
        f"embedder: {pipeline.index.embedder.name}"
    )

    st.header("Retrieval")
    mode = st.selectbox("Search mode", MODES, index=0)
    k = st.slider("Chunks to retrieve", 1, 8, settings.top_k)

    st.header("Answering")
    labels = {"ollama": "Ollama (this PC)", "api": "API (hosted model)", "off": "No LLM"}
    default = settings.llm_provider if settings.llm_provider in labels else "ollama"
    provider = st.radio(
        "Answer with", list(labels), index=list(labels).index(default), format_func=lambda p: labels[p]
    )
    if provider == "ollama":
        settings.ollama_model = st.text_input("Ollama model", settings.ollama_model)
        if ollama_available(settings.ollama_host):
            st.success(f"Ollama is running at {settings.ollama_host}")
        else:
            st.warning("Ollama is not running. Start it, or the best passage will be quoted instead.")
    elif provider == "api":
        settings.api_base = st.text_input("API base URL", settings.api_base)
        settings.api_model = st.text_input("Model", settings.api_model)
        # The key is only kept in memory for this session. It is never written to disk.
        settings.api_key = st.text_input("API key", settings.api_key, type="password")
        if not settings.api_key:
            st.warning("Enter an API key, or set LLM_API_KEY in the .env file.")
    else:
        st.info("The best matching passage is quoted from the documents.")
    pipeline.settings = settings

st.title("📄 Policy RAG")
st.caption("Ask questions about the insurance policy documents. Every answer shows the passages it came from.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(hits):
    with st.expander("Sources"):
        for i, hit in enumerate(hits, 1):
            st.markdown(
                f"**[{i}] {hit['source']} - {hit['section']}** (similarity {hit['similarity']:.2f})"
            )
            st.write(hit["text"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("hits"):
            show_sources(message["hits"])

question = st.chat_input("e.g. What is the waiting period for pre-existing diseases?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer = pipeline.ask(question, k=k, mode=mode, llm=provider)
        st.markdown(answer.text)
        st.caption(f"Answered by: {answer.generator}")
        if answer.note:
            st.warning(answer.note)
        hits = [
            {
                "source": h.chunk.source,
                "section": h.chunk.section,
                "text": h.chunk.text,
                "similarity": h.similarity,
            }
            for h in answer.hits
        ]
        if answer.found:
            show_sources(hits)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "hits": hits if answer.found else None}
    )
