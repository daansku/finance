"""Tests for Finlex HTML parser and node extraction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from taxxa.parser import FinlexParser, parse_corpus
from taxxa.schema import (
    NodeType,
    Publisher,
    Statute,
    Section,
    Clause,
    DocumentTree,
    GraphNode,
)

# ---------------------------------------------------------------------------
# Sample HTML snippets
# ---------------------------------------------------------------------------

AVL_HTML = """<!DOCTYPE html>
<html>
<head><title>Arvonlisäverolaki 1501/1993</title></head>
<body>
  <h1>Arvonlisäverolaki 1501/1993</h1>
  <p>1 luku</p>
  <p>§ 102 a Maahantuonnista suoritettava vero.</p>
  <div>Maahantuontia tavaran tuonti Suomeen EU:n ulkopuolelta.</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def avl_html_file():
    """Write sample HTML to a temp file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(AVL_HTML)
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# FinlexParser tests
# ---------------------------------------------------------------------------


def test_parse_basic_html(avl_html_file):
    parser = FinlexParser(avl_html_file)
    tree = parser.parse()
    assert isinstance(tree, DocumentTree)
    assert tree.statute.node_type == NodeType.STATUTE
    assert "Arvonlisäverolaki" in tree.statute.title
    assert len(tree.sections) >= 1
    assert tree.sections[0].node_type == NodeType.SECTION


def test_parse_empty_html():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write("<html><body></body></html>")
        path = Path(f.name)
    try:
        parser = FinlexParser(path)
        tree = parser.parse()
        assert isinstance(tree, DocumentTree)
        assert tree.sections == []
        assert len(tree.chapters) == 0
    finally:
        path.unlink(missing_ok=True)


def test_parse_extracts_statute_id():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(AVL_HTML)
        path = Path(f.name)
    try:
        parser = FinlexParser(path)
        tree = parser.parse()
        assert tree.statute.statute_id == "1501/1993"
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# parse_corpus (batch parsing)
# ---------------------------------------------------------------------------


def test_parse_corpus_from_html():
    """parse_corpus reads Finlex HTML files from the raw_dir/finlex/ directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir)
        finlex_dir = raw_dir / "finlex"
        finlex_dir.mkdir()

        # Write a sample HTML file
        (finlex_dir / "test.html").write_text(
            """<!DOCTYPE html><html><head><title>Test Law 999/2024</title></head>
<body><h1>Test Law 999/2024</h1>
<p>1 luku</p>
<p>§ 1 Section One.</p><div>This is the text of section one.</div>
</body></html>""",
            encoding="utf-8",
        )

        trees = parse_corpus(raw_dir)
        # parse_corpus reads HTML from finlex/ and returns DocumentTree objects
        assert len(trees) >= 1
        tree = trees[0]
        assert isinstance(tree, DocumentTree)
        assert "Test Law" in tree.statute.title
        assert len(tree.sections) >= 1


# ---------------------------------------------------------------------------
# Round-trip: parse → dump → load → same content
# ---------------------------------------------------------------------------


def test_roundtrip_parse_dump_load(avl_html_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = Path(tmpdir) / "corpus"
        corpus_dir.mkdir()

        # Parse sample HTML
        parser = FinlexParser(avl_html_file)
        tree = parser.parse()

        # Dump to JSON
        out_path = corpus_dir / "avl.json"
        out_path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")

        # Load back
        data = json.loads(out_path.read_text(encoding="utf-8"))
        loaded = DocumentTree(**data)
        assert loaded.statute.title == tree.statute.title
        assert len(loaded.sections) == len(tree.sections)


# ---------------------------------------------------------------------------
# Node type detection
# ---------------------------------------------------------------------------


def test_document_tree_node_types(avl_html_file):
    parser = FinlexParser(avl_html_file)
    tree = parser.parse()
    assert tree.statute.node_type == NodeType.STATUTE
    for section in tree.sections:
        assert section.node_type == NodeType.SECTION
    for clause in tree.clauses:
        assert clause.node_type == NodeType.CLAUSE