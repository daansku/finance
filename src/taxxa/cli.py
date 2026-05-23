"""
CLI Entrypoint — Build, index, query, and evaluate the Taxxa pipeline.

Usage:
    python -m taxxa.cli build     # Parse Finlex/HTML into JSON nodes
    python -m taxxa.cli load      # Load JSON into the graph (ChromaDB + BM25)
    python -m taxxa.cli query     # Interactive query mode
    python -m taxxa.cli eval      # Run evaluation against question_bank.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Default paths relative to the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "raw"
DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus"
DEFAULT_QUESTIONS = REPO_ROOT / "data" / "question_bank.json"
DEFAULT_CHROMADB_PATH = REPO_ROOT / "data" / "chroma"


# ---------------------------------------------------------------------------
# Build: parse Finlex + Vero HTML → JSON nodes
# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    """Parse Finlex and Vero HTML files into structured JSON DocumentTrees."""
    from .parser import parse_corpus

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing documents from {input_dir} ...")
    trees = parse_corpus(input_dir)

    if not trees:
        print("[ERROR] No documents parsed. Make sure data/raw/finlex/ contains .html files.")
        return 1

    total_sections = 0
    for tree in trees:
        out_name = (
            tree.statute.statute_id.replace("/", "_") + ".json"
            if tree.statute.statute_id
            else f"{len(tree.sections)}_sections.json"
        )
        out_path = output_dir / out_name
        out_path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")
        section_count = len(tree.sections)
        total_sections += section_count
        # For Vero/guidance docs, show text length instead of "0 sections"
        if section_count == 0 and tree.statute.text and len(tree.statute.text) > 100:
            txt_len = len(tree.statute.text)
            print(f"  → {tree.statute.title[:60]:<60s} {txt_len:>3d} chars → {out_name}")
        else:
            print(f"  → {tree.statute.title[:60]:<60s} {section_count:>3d} sections → {out_name}")

    print(f"\nDone. {len(trees)} documents, {total_sections} total sections → {output_dir}")
    return 0


# ---------------------------------------------------------------------------
# Load: load JSON nodes into the graph + index
# ---------------------------------------------------------------------------


def cmd_load(args: argparse.Namespace) -> int:
    """Load parsed JSON nodes into ChromaDB vector store and the graph."""
    from .parser import load_document_trees
    from .graph import InMemoryGraph
    from .retrieval import EmbeddingStore

    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        print(f"[ERROR] Corpus directory not found: {corpus_dir}")
        print("  Run 'build' first to parse Finlex HTML into JSON.")
        return 1

    print("Loading JSON document trees from corpus ...")
    trees = load_document_trees(str(corpus_dir))
    print(f"  Loaded {len(trees)} document trees")

    # Ingest into in-memory graph (serves as BM25 + graph)
    print("Building graph + BM25 index ...")
    graph = InMemoryGraph()
    for tree in trees:
        graph.ingest_document_tree(tree)

    print(f"  Graph: {len(graph._nodes)} nodes, {len(graph._edges)} edges")

    # Build ChromaDB vector index
    print(f"Building ChromaDB vector index at {args.chromadb_path} ...")
    chromadb_path = Path(args.chromadb_path)
    chromadb_path.mkdir(parents=True, exist_ok=True)

    embedding_store = EmbeddingStore(
        persist_dir=str(chromadb_path),
        collection_name=args.collection,
    )

    # Add all nodes to vector store
    all_nodes = list(graph._nodes.values())
    if all_nodes:
        embedding_store.add_nodes(all_nodes)
        print(f"  Vector index: {len(all_nodes)} nodes embedded")

    print("Done. Graph + vector index ready.")
    return 0


# ---------------------------------------------------------------------------
# Query: interactive question answering
# ---------------------------------------------------------------------------


def cmd_query(args: argparse.Namespace) -> int:
    """Interactive query loop using the TaxxaAgent."""
    from .graph import InMemoryGraph
    from .retrieval import EmbeddingStore, GraphRetriever, HybridRetriever
    from .parser import load_document_trees
    from .agent import TaxxaAgent, LLMClient

    # Load corpus and build graph
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"[ERROR] Corpus not found at {corpus_path}")
        print("  Run 'taxxa build' first to parse HTML → JSON.")
        return 1

    print("Loading corpus ...")
    trees = load_document_trees(str(corpus_path))
    print(f"  Loaded {len(trees)} document trees")

    print("Building graph ...")
    graph = InMemoryGraph()
    for tree in trees:
        graph.ingest_document_tree(tree)
    print(f"  Graph: {len(graph._nodes)} nodes, {len(graph._edges)} edges")

    # Load vector store
    print("Loading vector index ...")
    embedding_store = EmbeddingStore(
        persist_dir=args.chromadb_path,
        collection_name=args.collection,
    )

    # Index nodes into vector store
    all_nodes = list(graph._nodes.values())
    if all_nodes:
        embedding_store.add_nodes(all_nodes)

    # Build retriever
    graph_retriever = GraphRetriever(embedding_store)
    retriever = HybridRetriever(
        embedding_store=embedding_store,
        graph_retriever=graph_retriever,
    )

    # Build LLM client
    llm = LLMClient(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    print(f"  LLM: {args.model} @ {args.api_base}")

    # Build agent
    agent = TaxxaAgent(llm=llm, retriever=retriever)

    # Single query mode
    if args.question:
        answer = agent.answer(
            question_id="cli-query",
            question=args.question,
        )
        print(f"\n{'='*60}")
        print(f"Question: {args.question}")
        print(f"{'='*60}")
        print(f"\n{answer.answer}")
        print(f"\nConfidence: {answer.confidence:.2f} | Verified: {answer.verified}")
        if answer.citations:
            print(f"Sources: {len(answer.citations)} cited")
            for i, c in enumerate(answer.citations[:5]):
                print(f"  [{i+1}] {c.get('source_id', '?')} — {c.get('excerpt', '')[:120]}")
        return 0

    # Interactive mode
    print("\nTaxxa QA — Interactive Mode")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    qid = 0
    while True:
        try:
            question = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        qid += 1
        answer = agent.answer(
            question_id=f"q{qid:03d}",
            question=question,
        )

        print(f"\n{'─'*60}")
        print(answer.answer)
        print(f"{'─'*60}")
        print(f"Confidence: {answer.confidence:.2f} | Sources: {len(answer.citations)}")
        if answer.citations:
            for i, c in enumerate(answer.citations[:3]):
                print(f"  [{i+1}] {c.get('source_id', '?')} — {c.get('excerpt', '')[:100]}")
        print()

    return 0


# ---------------------------------------------------------------------------
# Eval: run evaluation against question_bank.json
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    """Run evaluation against question_bank.json."""
    from .graph import InMemoryGraph
    from .retrieval import EmbeddingStore, GraphRetriever, HybridRetriever
    from .parser import load_document_trees
    from .agent import TaxxaAgent, LLMClient
    from .eval import EvalRunner

    # Load corpus and build graph
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"[ERROR] Corpus not found at {corpus_path}")
        print("  Run 'taxxa build' first to parse HTML → JSON.")
        return 1

    print("Loading corpus ...")
    trees = load_document_trees(str(corpus_path))
    print(f"  Loaded {len(trees)} document trees")

    print("Building graph ...")
    graph = InMemoryGraph()
    for tree in trees:
        graph.ingest_document_tree(tree)
    print(f"  Graph: {len(graph._nodes)} nodes, {len(graph._edges)} edges")

    # Load vector store
    print("Loading vector index ...")
    embedding_store = EmbeddingStore(
        persist_dir=args.chromadb_path,
        collection_name=args.collection,
    )

    # Index nodes into vector store
    all_nodes = list(graph._nodes.values())
    if all_nodes:
        embedding_store.add_nodes(all_nodes)

    # Build retriever + agent
    graph_retriever = GraphRetriever(embedding_store)
    retriever = HybridRetriever(
        embedding_store=embedding_store,
        graph_retriever=graph_retriever,
    )
    llm = LLMClient(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    agent = TaxxaAgent(llm=llm, retriever=retriever)

    # Run evaluation
    runner = EvalRunner(agent=agent)
    question_path = args.questions or str(DEFAULT_QUESTIONS)

    if not Path(question_path).exists():
        print(f"[ERROR] Question bank not found: {question_path}")
        return 1

    results = runner.run_all(question_path)
    runner.print_table(results)

    # Save detailed results
    if args.output:
        out_path = Path(args.output)
        out_data = [
            {
                "question_id": r.question_id,
                "tier": r.tier,
                "passed": r.passed,
                "semantic_similarity": r.semantic_similarity,
                "key_facts_covered": r.key_facts_covered,
                "key_facts_missed": r.key_facts_missed,
                "citations_present": r.citations_present,
            }
            for r in results
        ]
        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nDetailed results saved to: {out_path}")

    return 0


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taxxa",
        description="Taxxa — Finnish tax law QA pipeline",
    )
    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # --- build ---
    p_build = sub.add_parser("build", help="Parse Finlex/Vero HTML into JSON nodes")
    p_build.add_argument(
        "--input", default=str(DEFAULT_INPUT),
        help=f"Directory with raw HTML (default: {DEFAULT_INPUT})",
    )
    p_build.add_argument(
        "--output", default=str(DEFAULT_CORPUS),
        help=f"Output directory for JSON nodes (default: {DEFAULT_CORPUS})",
    )

    # --- load ---
    p_load = sub.add_parser("load", help="Load JSON corpus into graph + ChromaDB")
    p_load.add_argument(
        "--corpus", default=str(DEFAULT_CORPUS),
        help=f"Corpus directory with JSON nodes (default: {DEFAULT_CORPUS})",
    )
    p_load.add_argument(
        "--chromadb-path", default=str(DEFAULT_CHROMADB_PATH),
        help="Path for ChromaDB index",
    )
    p_load.add_argument("--collection", default="taxxa", help="ChromaDB collection name")

    # --- query ---
    p_query = sub.add_parser("query", help="Interactive QA query")
    p_query.add_argument(
        "--corpus", default=str(DEFAULT_CORPUS),
        help=f"Corpus directory with JSON nodes (default: {DEFAULT_CORPUS})",
    )
    p_query.add_argument(
        "--question", default=None,
        help="Single question (otherwise interactive mode)",
    )
    p_query.add_argument(
        "--model", default="qwen2.5:14b",
        help="LLM model name",
    )
    p_query.add_argument(
        "--api-base", default="http://localhost:11434/v1",
        help="OpenAI-compatible API base URL",
    )
    p_query.add_argument(
        "--api-key", default="ollama",
        help="API key",
    )
    p_query.add_argument(
        "--chromadb-path", default=str(DEFAULT_CHROMADB_PATH),
        help="Path to ChromaDB index",
    )
    p_query.add_argument("--collection", default="taxxa", help="ChromaDB collection name")

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluate against question_bank.json")
    p_eval.add_argument(
        "--corpus", default=str(DEFAULT_CORPUS),
        help=f"Corpus directory with JSON nodes (default: {DEFAULT_CORPUS})",
    )
    p_eval.add_argument(
        "--questions", default=str(DEFAULT_QUESTIONS),
        help=f"Path to question_bank.json (default: {DEFAULT_QUESTIONS})",
    )
    p_eval.add_argument(
        "--output", default=None,
        help="Output path for detailed JSON results",
    )
    p_eval.add_argument(
        "--model", default="qwen2.5:14b",
        help="LLM model name",
    )
    p_eval.add_argument(
        "--api-base", default="http://localhost:11434/v1",
        help="OpenAI-compatible API base URL",
    )
    p_eval.add_argument(
        "--api-key", default="ollama",
        help="API key",
    )
    p_eval.add_argument(
        "--chromadb-path", default=str(DEFAULT_CHROMADB_PATH),
        help="Path to ChromaDB index",
    )
    p_eval.add_argument("--collection", default="taxxa", help="ChromaDB collection name")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        return cmd_build(args)
    elif args.command == "load":
        return cmd_load(args)
    elif args.command == "query":
        return cmd_query(args)
    elif args.command == "eval":
        return cmd_eval(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())