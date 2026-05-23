"""Tests for the LangGraph agent pipeline."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from taxxa.agent import TaxxaAgent, LLMClient, AgentState, INITIAL_STATE
from taxxa.retrieval import HybridRetriever, RetrievedPassage, GraphRetriever, EmbeddingStore
from taxxa.schema import SubQuestion, AgentAnswer


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


def _make_state(question="What is the capital income tax rate?"):
    """Build a fresh AgentState dict."""
    state = dict(INITIAL_STATE)
    state["question_id"] = "q1"
    state["question"] = question
    return state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """An LLM client that returns canned JSON responses."""
    client = Mock(spec=LLMClient)

    def canned_structured(messages, schema):
        if "sub_questions" in schema:
            return {
                "sub_questions": [
                    {"text": "What is the tax rate?", "reasoning": "Need rate", "priority": 1},
                    {"text": "When does it apply?", "reasoning": "Need dates", "priority": 2},
                ]
            }
        return {}

    client.generate_structured.side_effect = canned_structured
    client.generate.return_value = "Canned LLM answer"
    return client


@pytest.fixture
def mock_retriever():
    """A HybridRetriever backed by mocked components."""
    mock_store = Mock(spec=EmbeddingStore)
    mock_graph = Mock(spec=GraphRetriever)
    retriever = Mock(spec=HybridRetriever, wraps=HybridRetriever(mock_store, mock_graph))
    retriever.retrieve.return_value = [
        _make_retrieved("n1", "The tax rate is 30%.", score=0.95, title="Section 124"),
        _make_retrieved("n2", "Applies from 2024.", score=0.85, title="Section 125"),
    ]
    return retriever


# ---------------------------------------------------------------------------
# LLMClient tests
# ---------------------------------------------------------------------------


def test_llm_client_init_defaults():
    client = LLMClient()
    assert client.model == "qwen2.5:14b"
    assert client.api_base == "http://localhost:11434/v1"
    assert client.temperature == 0.1


def test_llm_client_custom_init():
    client = LLMClient(
        model="gpt-4o",
        api_base="https://api.openrouter.ai/api/v1",
        api_key="sk-test",
        temperature=0.3,
    )
    assert client.model == "gpt-4o"
    assert client.temperature == 0.3


def test_llm_generate_mocked():
    client = LLMClient()
    with patch.object(client, "generate") as mock_gen:
        mock_gen.return_value = "Hello, world!"
        result = client.generate([{"role": "user", "content": "Hi"}])
        assert result == "Hello, world!"


def test_llm_generate_structured_mocked():
    client = LLMClient()
    with patch.object(client, "generate") as mock_gen:
        mock_gen.return_value = '{"key": "value"}'
        result = client.generate_structured(
            [{"role": "system", "content": "Extract JSON"}],
            {"key": "string"},
        )
        assert result == {"key": "value"}


def test_llm_generate_structured_bad_json():
    client = LLMClient()
    with patch.object(client, "generate") as mock_gen:
        mock_gen.return_value = "not json at all"
        result = client.generate_structured(
            [{"role": "system", "content": "Extract JSON"}],
            {"key": "string"},
        )
        assert result is None


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def test_agent_construction(mock_llm, mock_retriever):
    agent = TaxxaAgent(llm=mock_llm, retriever=mock_retriever)
    assert agent.llm is mock_llm
    assert agent.retriever is mock_retriever
    assert agent.graph is not None  # compiled LangGraph


# ---------------------------------------------------------------------------
# Decompose node
# ---------------------------------------------------------------------------


def test_decompose_node(mock_llm, mock_retriever):
    agent = TaxxaAgent(llm=mock_llm, retriever=mock_retriever)
    state = _make_state("What is the capital income tax rate above 30 000 eur?")
    state["tier"] = "medium"

    new_state = agent._decompose(state)
    assert len(new_state["sub_questions"]) == 2
    assert new_state["sub_questions"][0]["text"] == "What is the tax rate?"
    assert new_state["sub_questions"][0]["priority"] == 1


def test_decompose_fallback_on_none(mock_retriever):
    """When LLM returns None, decompose should create a single fallback sub-question."""
    llm = Mock(spec=LLMClient)
    llm.generate_structured.return_value = None

    agent = TaxxaAgent(llm=llm, retriever=mock_retriever)
    state = _make_state("What is VAT?")
    new_state = agent._decompose(state)

    assert len(new_state["sub_questions"]) == 1
    assert new_state["sub_questions"][0]["text"] == "What is VAT?"


# ---------------------------------------------------------------------------
# Retrieve node
# ---------------------------------------------------------------------------


def test_retrieve_node(mock_llm, mock_retriever):
    agent = TaxxaAgent(llm=mock_llm, retriever=mock_retriever)
    state = _make_state("What is the tax rate?")
    state["sub_questions"] = [
        {"text": "What is the tax rate?", "reasoning": "Need rate", "priority": 1}
    ]
    state["current_sub_idx"] = 0

    new_state = agent._retrieve(state)
    # _retrieve converts Pydantic to dicts
    assert len(new_state["retrieved_passages"]) == 2
    assert isinstance(new_state["retrieved_passages"][0], dict)
    assert new_state["retrieved_passages"][0]["text"] == "The tax rate is 30%."


# ---------------------------------------------------------------------------
# Draft node
# ---------------------------------------------------------------------------


def test_draft_node(mock_llm, mock_retriever):
    agent = TaxxaAgent(llm=mock_llm, retriever=mock_retriever)
    state = _make_state("What is the tax rate?")
    # _retrieve converts to dicts, so pass dicts
    state["retrieved_passages"] = [
        {"node_id": "n1", "text": "The tax rate is 30%.", "node_type": "Section",
         "title": "Section 124", "section_number": "124", "score": 0.95},
    ]
    state["sub_questions"] = [
        {"text": "What is the tax rate?", "reasoning": "Need rate", "priority": 1}
    ]

    new_state = agent._draft(state)
    assert "draft_answer" in new_state
    # When LLM returns structured result, draft_answer is set from result.get("answer")
    answer_text = new_state["draft_answer"]
    assert isinstance(answer_text, str)
    assert len(answer_text) > 0


def test_draft_empty_passages(mock_llm, mock_retriever):
    agent = TaxxaAgent(llm=mock_llm, retriever=mock_retriever)
    state = _make_state("Unknown question")
    state["retrieved_passages"] = []
    state["sub_questions"] = []

    new_state = agent._draft(state)
    # When no passages, draft_answer is set to a fallback string
    assert isinstance(new_state["draft_answer"], str)
    assert "Unable to find" in new_state["draft_answer"]


# ---------------------------------------------------------------------------
# Verify node
# ---------------------------------------------------------------------------


def test_verify_node(mock_llm, mock_retriever):
    agent = TaxxaAgent(llm=mock_llm, retriever=mock_retriever)
    state = _make_state("What is the tax rate?")
    # draft_answer is a string (set by _draft)
    state["draft_answer"] = "The tax rate is 30%."
    # _retrieve converts to dicts
    state["retrieved_passages"] = [
        {"node_id": "n1", "text": "The tax rate is 30%.", "node_type": "Section",
         "title": "Section 124", "section_number": "124", "score": 0.95},
    ]

    new_state = agent._verify(state)
    assert "verified" in new_state  # boolean
    assert "verification_results" in new_state


def test_verify_no_citations(mock_llm, mock_retriever):
    """When there are no passages or draft_answer is empty, verification fails."""
    agent = TaxxaAgent(llm=mock_llm, retriever=mock_retriever)
    state = _make_state("Something")
    state["draft_answer"] = ""
    state["retrieved_passages"] = []

    new_state = agent._verify(state)
    assert new_state["verified"] is False
    assert new_state["confidence"] == 0.0


# ---------------------------------------------------------------------------
# INITIAL_STATE
# ---------------------------------------------------------------------------


def test_initial_state_keys():
    required_keys = [
        "question_id", "question", "tier", "sub_questions",
        "current_sub_idx", "retrieved_passages", "draft_answer",
        "verification_results", "verified", "confidence", "iteration", "error",
    ]
    for key in required_keys:
        assert key in INITIAL_STATE, f"Missing key: {key}"