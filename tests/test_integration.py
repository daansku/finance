"""Integration test — end-to-end pipeline: parse → index → query → eval."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from taxxa.agent import TaxxaAgent, LLMClient
from taxxa.eval import EvalRunner, FactEvaluator
from taxxa.graph import InMemoryGraph
from taxxa.parser import FinlexParser
from taxxa.retrieval import (
    HybridRetriever,
    EmbeddingStore,
    GraphRetriever,
    RetrievedPassage,
)
from taxxa.schema import (
    AgentAnswer,
    EvalResult,
    Statute,
    Section,
    DocumentTree,
    NodeType,
    Publisher,
)


# ---------------------------------------------------------------------------
# Sample HTML
# ---------------------------------------------------------------------------

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Tuloverolaki 1535/1992</title></head>
<body>
  <h1>Tuloverolaki 1535/1992</h1>
  <p>1 luku — Yleiset säännökset</p>
  <p>§ 9 Yleinen verovelvollisuus.</p>
  <div>Verovelvollinen on se, joka verovuonna asuu Suomessa. Henkilön katsotaan asuvan Suomessa, jos hänellä on täällä varsinainen asunto ja koti.</div>
  <p>§ 32 Pääomatulo.</p>
  <div>Pääomatuloa on omaisuuden tuotto, omaisuuden luovutuksesta saatu voitto ja muu sellainen tulo, jota varallisuuden voidaan katsoa kerryttäneen.</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retrieved(node_id, text, score=1.0, node_type="Section", title="", section_number=""):
    return RetrievedPassage(
        node_id=node_id,
        text=text,
        node_type=node_type,
        title=title or f"Node {node_id}",
        section_number=section_number,
        score=score,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def html_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_HTML)
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def parsed_tree(html_file):
    """Parse sample HTML into a DocumentTree."""
    parser = FinlexParser(html_file)
    return parser.parse()


@pytest.fixture
def indexed_graph(parsed_tree):
    """Build an InMemoryGraph from a parsed DocumentTree."""
    graph = InMemoryGraph()
    graph.ingest_document_tree(parsed_tree)
    return graph


# ---------------------------------------------------------------------------
# End-to-end: parse → index → query
# ---------------------------------------------------------------------------


def test_parse_and_index_roundtrip(html_file, parsed_tree, indexed_graph):
    """Parse HTML, ingest into graph, verify nodes are reachable."""
    # Verify parsed tree
    assert parsed_tree.statute.statute_id == "1535/1992"
    assert len(parsed_tree.sections) >= 2

    # Verify graph ingestion
    sections = indexed_graph.find_by_type("Section")
    assert len(sections) >= 2

    # Verify search works — search for section text that the parser captures
    results = indexed_graph.search_text("§ 9")
    assert len(results) > 0
    # "9" may appear in the node title (e.g., "§ 9 Yleinen...") or body text
    texts_or_titles = [r.get("text", "") + " " + r.get("title", "") for r in results]
    assert any("9" in t for t in texts_or_titles)


def test_graph_neighbor_traversal(indexed_graph, parsed_tree):
    """Verify statute is connected to its sections."""
    neighbors = indexed_graph.get_neighbors(parsed_tree.statute.id, edge_type="CONTAINS")
    section_ids = {s.id for s in parsed_tree.sections}
    neighbor_ids = {n[0] for n in neighbors}
    assert section_ids.issubset(neighbor_ids)


def test_search_all_sections(indexed_graph):
    """Search for text that should appear across multiple sections."""
    all_sections = indexed_graph.find_by_type("Section")
    for section in all_sections:
        node = indexed_graph.get_node(section["id"])
        assert node is not None
        assert node["text"] != ""


# ---------------------------------------------------------------------------
# End-to-end with mocked agent
# ---------------------------------------------------------------------------


def test_full_pipeline_mocked_agent(parsed_tree):
    """Run the full pipeline with a mocked LLM agent."""
    # Build graph
    graph = InMemoryGraph()
    graph.ingest_document_tree(parsed_tree)

    # Build mock embedding store
    mock_store = Mock(spec=EmbeddingStore)
    mock_store.search.return_value = [
        _make_retrieved("n1", "Verovelvollinen on se...", score=0.9, section_number="9"),
        _make_retrieved("n2", "Pääomatuloa on omaisuuden tuotto...", score=0.8, section_number="32"),
    ]

    # Mock GraphRetriever
    mock_graph_retriever = Mock(spec=GraphRetriever)
    mock_graph_retriever.expand_context.return_value = []
    mock_graph_retriever.find_interpretations.return_value = []

    # Mock Reranker
    from taxxa.retrieval import Reranker
    mock_reranker = Mock(spec=Reranker)

    retriever = HybridRetriever(
        embedding_store=mock_store,
        graph_retriever=mock_graph_retriever,
        reranker=mock_reranker,
    )

    # Mock LLM
    mock_llm = Mock(spec=LLMClient)
    mock_llm.generate_structured.return_value = {
        "sub_questions": [
            {"text": "What is general tax liability?", "reasoning": "Need to find the rule", "priority": 1},
        ]
    }
    mock_llm.generate.return_value = "General tax liability applies to persons residing in Finland. Pääomatulo is income from property and capital gains."

    agent = TaxxaAgent(llm=mock_llm, retriever=retriever)

    # Run the agent
    answer: AgentAnswer = agent.answer(
        question_id="q1",
        question="What is general tax liability and what is capital income?",
        tier="easy",
    )
    assert answer.question_id == "q1"
    assert len(answer.answer) > 0
    assert answer.verified in (True, False)


def test_agent_answer_fields(parsed_tree):
    """AgentAnswer should include all required fields."""
    mock_llm = Mock(spec=LLMClient)
    mock_llm.generate_structured.return_value = {"sub_questions": []}
    mock_llm.generate.return_value = "Answer text"

    mock_store = Mock(spec=EmbeddingStore)
    mock_graph_retriever = Mock(spec=GraphRetriever)
    mock_reranker = Mock()
    mock_store.search.return_value = []
    mock_graph_retriever.expand_context.return_value = []
    mock_graph_retriever.find_interpretations.return_value = []

    retriever = HybridRetriever(mock_store, mock_graph_retriever, mock_reranker)
    agent = TaxxaAgent(llm=mock_llm, retriever=retriever)

    answer = agent.answer("q_test", "Test question?", "easy")
    assert isinstance(answer, AgentAnswer)
    assert answer.question_id == "q_test"
    assert answer.question == "Test question?"
    assert hasattr(answer, "answer")
    assert hasattr(answer, "citations")
    assert hasattr(answer, "confidence")


# ---------------------------------------------------------------------------
# Evaluation integration
# ---------------------------------------------------------------------------


def test_eval_runner_with_mocked_agent(parsed_tree):
    """EvalRunner should evaluate answers using key facts."""
    expected_answer = (
        "Verovelvollinen on se, joka asuu Suomessa. "
        "Pääomatuloa on omaisuuden tuotto ja omaisuuden luovutuksesta saatu voitto."
    )

    # Setup agent with mocks — return sub_questions for decompose AND draft answer
    mock_llm = Mock(spec=LLMClient)

    def canned_structured(messages, schema):
        if "sub_questions" in schema:
            return {"sub_questions": []}
        if "answer" in schema and "citations" in schema:
            return {"answer": expected_answer, "citations": [], "confidence": 0.9}
        return {}

    mock_llm.generate_structured.side_effect = canned_structured
    mock_llm.generate.return_value = expected_answer

    mock_store = Mock(spec=EmbeddingStore)
    mock_graph_retriever = Mock(spec=GraphRetriever)
    mock_reranker = Mock()
    # Return a passage so _retrieve and _draft produce an answer
    from taxxa.retrieval import RetrievedPassage
    mock_store.search.return_value = [
        RetrievedPassage(
            node_id="n1", text="Verovelvollinen on se, joka asuu Suomessa.",
            node_type="Section", title="Section 9", section_number="9", score=0.9,
        ),
    ]
    mock_graph_retriever.expand_context.return_value = []
    mock_graph_retriever.find_interpretations.return_value = []

    retriever = HybridRetriever(mock_store, mock_graph_retriever, mock_reranker)
    agent = TaxxaAgent(llm=mock_llm, retriever=retriever)

    # Write a small question bank
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        question_bank = {
            "entries": [
                {
                    "id": "q1",
                    "question": "What is general tax liability?",
                    "answer": "Tax liability applies to persons residing in Finland.",
                    "answer_key_facts": ["residing in Finland", "general tax liability"],
                    "tier": "easy",
                }
            ]
        }
        json.dump(question_bank, f)
        qb_path = Path(f.name)

    try:
        runner = EvalRunner(agent=agent)
        results = runner.run_all(str(qb_path))
        assert len(results) == 1
        assert isinstance(results[0], EvalResult)
        assert results[0].question_id == "q1"
        assert hasattr(results[0], "passed")
        assert hasattr(results[0], "semantic_similarity")
    finally:
        qb_path.unlink(missing_ok=True)


def test_fact_evaluator_exact_match():
    """FactEvaluator with no model should use exact/word-overlap matching."""
    evaluator = FactEvaluator(similarity_threshold=0.5)
    covered, missed, sim = evaluator.evaluate(
        expected_answer="Tax liability applies to persons residing in Finland",
        generated_answer="Tax liability applies to persons residing in Finland",
        key_facts=["residing in Finland", "general tax liability"],
    )
    assert len(covered) >= 1
    assert sim > 0.0


def test_fact_evaluator_no_match():
    """When answer doesn't contain key facts, they should be missed."""
    evaluator = FactEvaluator(similarity_threshold=0.5)
    covered, missed, sim = evaluator.evaluate(
        expected_answer="About capital income",
        generated_answer="Tax liability applies to persons residing in Finland",
        key_facts=["completely different topic"],
    )
    assert len(covered) == 0
    assert len(missed) >= 1


def test_fact_evaluator_empty_answer():
    """Empty generated answer should miss all key facts."""
    evaluator = FactEvaluator()
    covered, missed, sim = evaluator.evaluate(
        expected_answer="Expected answer",
        generated_answer="",
        key_facts=["fact one"],
    )
    assert len(covered) == 0
    assert sim == 0.0


# ---------------------------------------------------------------------------
# Full end-to-end (no LLM required)
# ---------------------------------------------------------------------------


def test_parse_index_retrieve_exact(indexed_graph, parsed_tree):
    """Parse → index → retrieve exact match (no LLM)."""
    # Search for known section text
    target_section = parsed_tree.sections[0]
    keyword = target_section.text.split()[0]
    results = indexed_graph.search_text(keyword)
    assert len(results) > 0
    found_ids = {r["id"] for r in results}
    assert target_section.id in found_ids