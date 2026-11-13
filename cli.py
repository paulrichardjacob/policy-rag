"""Command line interface.

    python cli.py ingest                    build the index from data/sample_policies
    python cli.py ingest --docs my_folder   build it from your own documents
    python cli.py ask "What is the grace period?"                 local Ollama (default)
    python cli.py ask "What is the grace period?" --llm api       hosted API
    python cli.py ask "What is the grace period?" --llm off       no LLM, quote the passage
    python cli.py eval                      retrieval metrics for bm25 / dense / hybrid
"""

import argparse
import json
import sys

from eval.evaluate import QUESTIONS, evaluate, format_table
from rag.config import Settings
from rag.pipeline import RAGPipeline, build_index
from rag.store import MODES, VectorIndex

DEFAULT_DOCS = "data/sample_policies"
DEFAULT_INDEX = "index"


def cmd_ingest(args):
    settings = Settings()
    if args.embedder:
        settings.embedder = args.embedder
    index = build_index(args.docs, args.index, settings)
    sources = sorted({c.source for c in index.chunks})
    print(f"Indexed {len(index.chunks)} chunks from {len(sources)} files "
          f"with the '{index.embedder.name}' embedder -> {args.index}/")


def cmd_ask(args):
    pipeline = RAGPipeline.from_folder(args.index)
    answer = pipeline.ask(args.question, k=args.k, mode=args.mode, llm=args.llm)
    print(f"\nQ: {answer.question}\n\nA: {answer.text}\n")
    print(f"(answered by: {answer.generator}, retrieval: {args.mode})")
    if answer.note:
        print(f"Note: {answer.note}")
    print("\nSources:")
    for i, hit in enumerate(answer.hits, 1):
        print(f"  [{i}] {hit.chunk.source} - {hit.chunk.section}  (similarity {hit.similarity:.2f})")


def cmd_eval(args):
    index = VectorIndex.load(args.index)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    results = evaluate(index, questions)
    print(format_table(results, len(questions), index.embedder.name))
    if args.verbose:
        for mode, r in results.items():
            if r["misses"]:
                print(f"\n{mode}: not in top 3")
                for q in r["misses"]:
                    print(f"  - {q}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAG over insurance policy documents")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="chunk and index documents")
    p.add_argument("--docs", default=DEFAULT_DOCS)
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--embedder", choices=["auto", "minilm", "tfidf"])
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("-k", type=int, default=None, help="number of chunks to retrieve")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--llm", choices=["auto", "ollama", "api", "off"], default="auto",
                   help="auto uses LLM_PROVIDER from .env (default: ollama)")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="evaluate retrieval")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("-v", "--verbose", action="store_true", help="list the missed questions")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
