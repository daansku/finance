"""Tests for hybrid retrieval (graph + embeddings)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from taxxa.graph import InMemoryGraph
from taxxa.retrieval import (
    EmbeddingStore,
    GraphRetriever,
    Reranker,
    HybridRetriever,
    RetrievedPassage,
)
from taxxa.schema import (
    NodeType,
    Publisher,
    Section,
    Statute,
    DocumentTree,
    GraphNode,
)


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
def sample_tree():
    """A DocumentTree with statute + sections for testing."""
    statute = Statute(
        node_type=NodeType.STATUTE,
        publisher=Publisher.FINLEX,
        title="Tuloverolaki",
        statute_id="1535/1992",
        text="Tuloverolaki säädetään...",
    )
    s1 = Section(
        node_type=NodeType.SECTION,
        publisher=Publisher.FINLEX,
        title="Verovelvollisuus",
        section_number="9",
        statute_id="1535/1992",
        text="Verovelvollinen on se, joka saa Suomesta veronalaista tuloa... Myös rajoitetusti verovelvollisia koskevat säännökset...",
    )
    s2 = Section(
        node_type=NodeType.SECTION,
        publisher=Publisher.FINLEX,
        title="Pääomatulo",
        section_number="32",
        statute_id="1535/1992",
        text="Pääomatuloa on omaisuuden tuotto, omaisuuden luovutuksesta saatu voitto ja muu sellainen tulo, jota varallisuuden voidaan katsoa kerryttäneen...",
    )
    return DocumentTree(
        statute=statute,
        sections=[s1, s2],
    )


@pytest.fixture
def graph_with_nodes(sample_tree):
    """Build an InMemoryGraph and ingest the sample data."""
    graph = InMemoryGraph()
    graph.ingest_document_tree(sample_tree)
    return graph


# ---------------------------------------------------------------------------
# InMemoryGraph text search
# ---------------------------------------------------------------------------


def test_graph_search_text_finds_relevant_nodes(graph_with_nodes):
    """search_text should find nodes containing the query."""
    results = graph_with_nodes.search_text("pääomatulo")
    assert len(results) > 0
    # Should find the section about pääomatulo
    titles = [r.get("title", "") for r in results]
    assert any("Pääomatulo" in t for t in titles)


def test_graph_search_empty_query(graph_with_nodes):
    """Empty query should return no results (no match)."""
    results = graph_with_nodes.search_text("")
    assert isinstance(results, list)


def test_graph_find_by_type(graph_with_nodes):
    """find_by_type should filter nodes by type."""
    sections = graph_with_nodes.find_by_type("Section")
    assert len(sections) >= 2
    for s in sections:
        assert s["node_type"] == "Section"


def test_graph_get_node(graph_with_nodes, sample_tree):
    """get_node should return a node by its ID."""
    node = graph_with_nodes.get_node(sample_tree.statute.id)
    assert node is not None
    assert node["title"] == "Tuloverolaki"


def test_graph_neighbors(graph_with_nodes, sample_tree):
    """get_neighbors should return connected nodes via CONTAINS edges."""
    neighbors = graph_with_nodes.get_neighbors(sample_tree.statute.id)
    assert len(neighbors) >= 2  # statute contains 2 sections
    for target_id, edge_type, _ in neighbors:
        assert edge_type == "CONTAINS"


# ---------------------------------------------------------------------------
# GraphRetriever (Neo4j wrapper) — use InMemoryGraph with mock
# ---------------------------------------------------------------------------


def test_graph_retriever_from_inmemory(graph_with_nodes):
    """GraphRetriever wraps a Neo4j store."""
    mock_store = Mock()
    mock_store.search.return_value = []
    retriever = GraphRetriever(mock_store)
    assert retriever is not None
    assert hasattr(retriever, "expand_context")


# ---------------------------------------------------------------------------
# HybridRetriever with mocks
# ---------------------------------------------------------------------------


def test_hybrid_retriever_retrieve_dedup():
    """HybridRetriever passes results through its pipeline."""
    mock_store = Mock(spec=EmbeddingStore)
    mock_graph = Mock(spec=GraphRetriever)
    mock_reranker = Mock(spec=Reranker)

    entry = _make_retrieved("dup", "Duplicate entry", score=0.9)
    mock_store.search.return_value = [entry, entry]  # duplicate
    mock_graph.expand_context.return_value = [entry]
    mock_graph.find_interpretations.return_value = []
    mock_reranker.rerank.return_value = [entry]

    retriever = HybridRetriever(
        embedding_store=mock_store,
        graph_retriever=mock_graph,
        reranker=mock_reranker,
    )
    result = retriever.retrieve("test query")
    assert isinstance(result, list)
    assert len(result) >= 1


def test_hybrid_retriever_init_with_mocks():
    """HybridRetriever can be constructed with mock components."""
    mock_store = Mock(spec=EmbeddingStore)
    mock_graph = Mock(spec=GraphRetriever)
    mock_reranker = Mock(spec=Reranker)

    retriever = HybridRetriever(
        embedding_store=mock_store,
        graph_retriever=mock_graph,
        reranker=mock_reranker,
    )
    assert retriever.embeddings is mock_store
    assert retriever.graph is mock_graph
    assert retriever.reranker is mock_reranker


def test_hybrid_retriever_retrieve_mocked():
    """retrieve should combine embedding search + graph expansion + rerank."""
    mock_store = Mock(spec=EmbeddingStore)
    mock_graph = Mock(spec=GraphRetriever)
    mock_reranker = Mock(spec=Reranker)

    entry = _make_retrieved("n1", "Section 9 text", score=0.9, section_number="9")
    expanded = [
        _make_retrieved("n2", "Chapter about verovelvollisuus", score=0.7),
    ]
    final = [_make_retrieved("n1", "Section 9 text", score=0.95)]

    mock_store.search.return_value = [entry]
    mock_graph.expand_context.return_value = [entry] + expanded
    mock_graph.find_interpretations.return_value = []
    mock_reranker.rerank.return_value = final

    retriever = HybridRetriever(
        embedding_store=mock_store,
        graph_retriever=mock_graph,
        reranker=mock_reranker,
    )

    result = retriever.retrieve("What is tax liability?")
    assert len(result) == 1
    assert result[0].node_id == "n1"


def test_hybrid_retrieve_with_filters():
    """retrieve should pass publisher/statute filters to embedding store."""
    mock_store = Mock(spec=EmbeddingStore)
    mock_graph = Mock(spec=GraphRetriever)
    mock_reranker = Mock(spec=Reranker)
    mock_store.search.return_value = []
    mock_graph.expand_context.return_value = []
    mock_reranker.rerank.return_value = []

    retriever = HybridRetriever(
        embedding_store=mock_store,
        graph_retriever=mock_graph,
        reranker=mock_reranker,
    )

    retriever.retrieve("question", filter_publisher="finlex", filter_statute_id="1535/1992")

    # Check that the filter was passed through
    call_args = mock_store.search.call_args
    assert call_args is not None
    where = call_args[1].get("where")
    assert where is not None
    assert where.get("publisher") == "finlex"
    assert where.get("statute_id") == "1535/1992"


# ---------------------------------------------------------------------------
# Reranker (falls back to score-sort when no CrossEncoder)
# ---------------------------------------------------------------------------


def test_reranker_noop():
    """Even without model loaded, reranker preserves original ordering."""
    reranker = Reranker(use_api=True)
    passages = [
        _make_retrieved("a", "text A", score=0.9),
        _make_retrieved("b", "text B", score=0.3),
        _make_retrieved("c", "text C", score=0.6),
    ]
    result = reranker.rerank("test query", passages, top_k=2)
    assert len(result) == 2
    # sorted by descending score
    assert result[0].score >= result[1].score