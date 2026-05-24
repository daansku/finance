"""Pydantic typed node models for the Finnish legal/tax knowledge graph.

Defines the canonical schema: Statute → Chapter → Section → Clause, plus
Vero guidance documents, cross-references, and amendment edges.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


class Publisher(str, Enum):
    FINLEX = "finlex"
    VERO = "vero"


class NodeType(str, Enum):
    STATUTE = "Statute"
    CHAPTER = "Chapter"
    SECTION = "Section"
    CLAUSE = "Clause"
    GUIDANCE = "Guidance"  # Vero syvennetyt ohjeet / bulletin
    CONCEPT = "Concept"


class EdgeType(str, Enum):
    CONTAINS = "CONTAINS"
    REFERENCES = "REFERENCES"
    INTERPRETS = "INTERPRETS"
    SUPERSEDES = "SUPERSEDES"
    REPEALS = "REPEALS"
    AMENDS = "AMENDS"
    CLARIFIES = "CLARIFIES"
    CITES = "CITES"
    EFFECTIVE_FROM = "EFFECTIVE_FROM"


# ---------------------------------------------------------------------------
# Base node
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """Abstract base for all graph nodes."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType
    publisher: Publisher
    title: str = ""
    text: str = ""  # full text of this node (or summary for large ones)

    class Config:
        use_enum_values = True


class Statute(GraphNode):
    """Top-level statute (e.g. AVL — Arvonlisäverolaki)."""

    node_type: NodeType = NodeType.STATUTE
    statute_id: str = ""  # e.g. "1501/1993"
    finlex_url: str = ""


class Chapter(GraphNode):
    """A chapter within a statute."""

    node_type: NodeType = NodeType.CHAPTER
    chapter_number: int = 0
    statute_id: str = ""  # parent statute reference


class Section(GraphNode):
    """A numbered section (§) within a chapter."""

    node_type: NodeType = NodeType.SECTION
    section_number: str = ""  # e.g. "102 a"
    chapter_number: int = 0
    statute_id: str = ""


class Clause(GraphNode):
    """A momentti/paragraph within a section."""

    node_type: NodeType = NodeType.CLAUSE
    clause_number: int = 0  # momentti number
    section_number: str = ""
    statute_id: str = ""


class Guidance(GraphNode):
    """A Vero guidance document (syvennetty ohje or bulletin)."""

    node_type: NodeType = NodeType.GUIDANCE
    guidance_id: str = ""  # Vero document ID
    version_date: Optional[date] = None
    vero_url: str = ""


class Concept(GraphNode):
    """A named legal/tax concept extracted from text."""

    node_type: NodeType = NodeType.CONCEPT
    concept_name: str = ""
    aliases: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Edge model
# ---------------------------------------------------------------------------


class GraphEdge(BaseModel):
    """Typed directed edge between two graph nodes."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: dict = Field(default_factory=dict)
    # Common temporal properties
    effective_from: Optional[date] = None
    effective_until: Optional[date] = None

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Cross-reference parsed from text
# ---------------------------------------------------------------------------


class CrossReference(BaseModel):
    """Parsed cross-reference like '§ 102 a momentti 2'."""

    raw_text: str
    statute_id: Optional[str] = None  # if statute is named
    section_number: Optional[str] = None
    clause_number: Optional[int] = None
    resolution_confidence: float = 0.0  # 0=regex, >0=LLM-assigned


# ---------------------------------------------------------------------------
# Document tree (parse output)
# ---------------------------------------------------------------------------


class DocumentTree(BaseModel):
    """The full parsed tree for one document."""

    statute: Statute
    chapters: list[Chapter] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    clauses: list[Clause] = Field(default_factory=list)
    cross_references: list[CrossReference] = Field(default_factory=list)
    raw_html_path: str = ""


# ---------------------------------------------------------------------------
# Query / Agent models
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """A decomposed sub-question produced by the agent."""

    text: str
    reasoning: str = ""
    priority: int = 0


class RetrievedPassage(BaseModel):
    """A passage retrieved from the knowledge graph."""

    node_id: str
    text: str
    node_type: str
    title: str
    section_number: str = ""
    score: float = 0.0  # relevance score
    source_path: str = ""  # traceable back to file/URL


class AgentAnswer(BaseModel):
    """Final answer produced by the agent loop."""

    question_id: str
    question: str
    answer: str
    citations: list[dict] = Field(default_factory=list)
    sub_answers: list[dict] = Field(default_factory=list)
    confidence: float = 0.0
    verified: bool = False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """Result of evaluating one QA pair."""

    question_id: str
    tier: str
    question: str
    expected_answer: str
    generated_answer: str
    key_facts_covered: list[str] = Field(default_factory=list)
    key_facts_missed: list[str] = Field(default_factory=list)
    semantic_similarity: float = 0.0
    passed: bool = False
    citations_present: bool = False