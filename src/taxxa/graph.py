"""
Knowledge Graph — Neo4j ingestion with typed nodes and edges.

Handles:
- Ingesting parsed DocumentTrees into Neo4j
- Creating CONTAINS, REFERENCES, INTERPRETS, SUPERSEDES, REPEALS edges
- Amendment temporality via EFFECTIVE_FROM date properties on edges
- Cypher query patterns for graph traversal
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

from neo4j import GraphDatabase, Driver, Session, Record

from .schema import (
    Statute,
    Chapter,
    Section,
    Clause,
    Guidance,
    CrossReference,
    DocumentTree,
    GraphEdge,
    EdgeType,
    Publisher,
    NodeType,
    RetrievedPassage,
)


# ---------------------------------------------------------------------------
# Neo4j connection manager
# ---------------------------------------------------------------------------


class Neo4jStore:
    """Wraps a Neo4j driver and provides typed ingestion + query methods."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def clear(self) -> None:
        """Wipe all nodes and edges. Use with caution."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # ------------------------------------------------------------------
    # Index creation
    # ------------------------------------------------------------------

    def create_indexes(self) -> None:
        """Create useful indexes for fast lookups."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Statute) ON (n.statute_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Section) ON (n.section_number)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Clause) ON (n.clause_number)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Guidance) ON (n.guidance_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.concept_name)",
            "CREATE TEXT INDEX IF NOT EXISTS FOR (n:Section) ON (n.text)",
            "CREATE TEXT INDEX IF NOT EXISTS FOR (n:Clause) ON (n.text)",
        ]
        with self.driver.session() as session:
            for idx in indexes:
                try:
                    session.run(idx)
                except Exception as e:
                    # Some indexes may fail on community edition
                    pass

    # ------------------------------------------------------------------
    # Node ingestion
    # ------------------------------------------------------------------

    def ingest_document_tree(self, tree: DocumentTree) -> None:
        """Ingest a full DocumentTree into Neo4j."""
        with self.driver.session() as session:
            self._merge_statute(session, tree.statute)
            for ch in tree.chapters:
                self._merge_chapter(session, ch, tree.statute.id)
            for sec in tree.sections:
                self._merge_section(session, sec, tree.statute.id)
            for cl in tree.clauses:
                self._merge_clause(session, cl, tree.statute.id)
            for ref in tree.cross_references:
                self._merge_reference(session, ref)
            for sec in tree.sections:
                self._link_hierarchy_section(session, sec, tree.statute.id)

    def ingest_guidance(self, guidance: Guidance, refs: list[CrossReference]) -> None:
        """Ingest a Vero guidance document."""
        with self.driver.session() as session:
            self._merge_guidance(session, guidance)
            for ref in refs:
                self._link_guidance_ref(session, guidance.id, ref)

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    def _merge_statute(self, session: Session, s: Statute) -> None:
        session.run(
            """
            MERGE (n:Statute {id: $id})
            SET n.title = $title,
                n.statute_id = $statute_id,
                n.publisher = $publisher,
                n.text = $text,
                n.finlex_url = $finlex_url
            """,
            id=s.id,
            title=s.title,
            statute_id=s.statute_id,
            publisher=s.publisher.value if hasattr(s.publisher, "value") else str(s.publisher),
            text=s.text,
            finlex_url=s.finlex_url,
        )

    def _merge_chapter(self, session: Session, ch: Chapter, statute_id: str) -> None:
        session.run(
            """
            MERGE (n:Chapter {id: $id})
            SET n.title = $title,
                n.chapter_number = $chapter_number,
                n.statute_id = $statute_id,
                n.publisher = $publisher,
                n.text = $text
            """,
            id=ch.id,
            title=ch.title,
            chapter_number=ch.chapter_number,
            statute_id=ch.statute_id,
            publisher=ch.publisher.value if hasattr(ch.publisher, "value") else str(ch.publisher),
            text=ch.text,
        )

    def _merge_section(self, session: Session, sec: Section, statute_id: str) -> None:
        session.run(
            """
            MERGE (n:Section {id: $id})
            SET n.title = $title,
                n.section_number = $section_number,
                n.chapter_number = $chapter_number,
                n.statute_id = $statute_id,
                n.publisher = $publisher,
                n.text = $text
            """,
            id=sec.id,
            title=sec.title,
            section_number=sec.section_number,
            chapter_number=sec.chapter_number,
            statute_id=sec.statute_id,
            publisher=sec.publisher.value if hasattr(sec.publisher, "value") else str(sec.publisher),
            text=sec.text,
        )

    def _merge_clause(self, session: Session, cl: Clause, statute_id: str) -> None:
        session.run(
            """
            MERGE (n:Clause {id: $id})
            SET n.title = $title,
                n.clause_number = $clause_number,
                n.section_number = $section_number,
                n.statute_id = $statute_id,
                n.publisher = $publisher,
                n.text = $text
            """,
            id=cl.id,
            title=cl.title,
            clause_number=cl.clause_number,
            section_number=cl.section_number,
            statute_id=cl.statute_id,
            publisher=cl.publisher.value if hasattr(cl.publisher, "value") else str(cl.publisher),
            text=cl.text,
        )

    def _merge_guidance(self, session: Session, g: Guidance) -> None:
        session.run(
            """
            MERGE (n:Guidance {id: $id})
            SET n.title = $title,
                n.guidance_id = $guidance_id,
                n.version_date = $version_date,
                n.publisher = $publisher,
                n.text = $text,
                n.vero_url = $vero_url
            """,
            id=g.id,
            title=g.title,
            guidance_id=g.guidance_id,
            version_date=str(g.version_date) if g.version_date else None,
            publisher=g.publisher.value if hasattr(g.publisher, "value") else str(g.publisher),
            text=g.text,
            vero_url=g.vero_url,
        )

    def _merge_reference(self, session: Session, ref: CrossReference) -> None:
        """Store a cross-reference as a relationship or as metadata for resolution."""
        if ref.section_number and ref.statute_id:
            # Link to the target section if it exists
            session.run(
                """
                MATCH (source {id: $source_id})
                MATCH (target:Section {section_number: $section_number, statute_id: $statute_id})
                MERGE (source)-[r:REFERENCES {raw_text: $raw_text}]->(target)
                SET r.confidence = $confidence
                """,
                source_id=ref.raw_text,
                section_number=ref.section_number,
                statute_id=ref.statute_id,
                raw_text=ref.raw_text,
                confidence=ref.resolution_confidence,
            )

    def _link_hierarchy_section(self, session: Session, sec: Section, statute_id: str) -> None:
        """Link section to its parent statute and chapter."""
        # Statute -> Section via CONTAINS
        session.run(
            """
            MATCH (s:Statute {statute_id: $statute_id})
            MATCH (sec:Section {id: $section_id})
            MERGE (s)-[:CONTAINS]->(sec)
            """,
            statute_id=statute_id,
            section_id=sec.id,
        )
        # Section -> clauses via CONTAINS
        session.run(
            """
            MATCH (sec:Section {id: $section_id})
            MATCH (cl:Clause {section_number: $section_number, statute_id: $statute_id})
            MERGE (sec)-[:CONTAINS]->(cl)
            """,
            section_id=sec.id,
            section_number=sec.section_number,
            statute_id=statute_id,
        )

    def _link_guidance_ref(self, session: Session, guidance_id: str, ref: CrossReference) -> None:
        """Link guidance to a Finlex section it interprets."""
        session.run(
            """
            MATCH (g:Guidance {id: $guidance_id})
            MERGE (g)-[:INTERPRETS {raw_text: $raw_text}]->(target)
            """,
            guidance_id=guidance_id,
            raw_text=ref.raw_text,
        )

    # ------------------------------------------------------------------
    # Amendment temporality
    # ------------------------------------------------------------------

    def add_effective_date(
        self,
        source_id: str,
        target_id: str,
        effective_date: date | str,
        edge_type: str = "EFFECTIVE_FROM",
    ) -> None:
        """Add an EFFECTIVE_FROM edge with a date property."""
        effective_date_str = str(effective_date)
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (a {{id: $source_id}})
                MATCH (b {{id: $target_id}})
                MERGE (a)-[r:{edge_type}]->(b)
                SET r.date = $date
                """,
                source_id=source_id,
                target_id=target_id,
                date=effective_date_str,
            )

    def add_supersedes(
        self,
        newer_id: str,
        older_id: str,
        effective_date: date | str,
    ) -> None:
        """Mark that a newer section supersedes an older one."""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (newer {id: $newer_id})
                MATCH (older {id: $older_id})
                MERGE (newer)-[r:SUPERSEDES]->(older)
                SET r.effective_from = $date
                """,
                newer_id=newer_id,
                older_id=older_id,
                date=str(effective_date),
            )

    def add_repeals(
        self,
        repealer_id: str,
        repealed_id: str,
        effective_date: date | str,
    ) -> None:
        """Mark that one provision repeals another."""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (repealer {id: $repealer_id})
                MATCH (repealed {id: $repealed_id})
                MERGE (repealer)-[r:REPEALS]->(repealed)
                SET r.effective_from = $date
                """,
                repealer_id=repealer_id,
                repealed_id=repealed_id,
                date=str(effective_date),
            )

    # ------------------------------------------------------------------
    # Graph traversal queries
    # ------------------------------------------------------------------

    def find_section(self, section_number: str, statute_id: str = "") -> Optional[Record]:
        """Find a Section node by its section number."""
        with self.driver.session() as session:
            if statute_id:
                result = session.run(
                    """
                    MATCH (n:Section {section_number: $section_number, statute_id: $statute_id})
                    RETURN n
                    """,
                    section_number=section_number,
                    statute_id=statute_id,
                )
            else:
                result = session.run(
                    """
                    MATCH (n:Section {section_number: $section_number})
                    RETURN n
                    LIMIT 1
                    """,
                    section_number=section_number,
                )
            record = result.single()
            return record

    def get_section_with_clauses(self, section_number: str, statute_id: str) -> dict:
        """Return a section and its clauses."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (sec:Section {section_number: $section_number, statute_id: $statute_id})
                OPTIONAL MATCH (sec)-[:CONTAINS]->(cl:Clause)
                RETURN sec, collect(cl) as clauses
                """,
                section_number=section_number,
                statute_id=statute_id,
            )
            record = result.single()
            if not record:
                return {}
            sec_node = record["sec"]
            clauses = record["clauses"]
            return {
                "section": dict(sec_node),
                "clauses": [dict(cl) for cl in clauses if cl],
            }

    def walk_references(
        self,
        start_section_number: str,
        statute_id: str,
        max_depth: int = 3,
    ) -> list[dict]:
        """Walk the REFERENCES graph from a starting section, up to max_depth hops."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start:Section {section_number: $section_number, statute_id: $statute_id})
                             -[:REFERENCES*1..$depth]->(target)
                RETURN nodes(path) as nodes, relationships(path) as edges
                LIMIT 50
                """,
                section_number=start_section_number,
                statute_id=statute_id,
                depth=max_depth,
            )
            paths = []
            for record in result:
                nodes = [dict(n) for n in record["nodes"]]
                edges = [
                    {"type": r.type, "props": dict(r)}
                    for r in record["edges"]
                ]
                paths.append({"nodes": nodes, "edges": edges})
            return paths

    def find_interpretations(self, section_number: str, statute_id: str) -> list[dict]:
        """Find Vero guidance documents that interpret a given section."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (g:Guidance)-[:INTERPRETS]->(sec:Section {section_number: $section_number, statute_id: $statute_id})
                RETURN g
                """,
                section_number=section_number,
                statute_id=statute_id,
            )
            return [dict(record["g"]) for record in result]

    def query_by_date(
        self,
        section_number: str,
        statute_id: str,
        as_of_date: date | str,
    ) -> Optional[dict]:
        """Return the version of a section valid on a given date.

        Filters by EFFECTIVE_FROM edges — returns the latest version
        whose effective_from is <= as_of_date.
        """
        as_of_str = str(as_of_date)
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (sec:Section {section_number: $section_number, statute_id: $statute_id})
                OPTIONAL MATCH (newer)-[r:SUPERSEDES]->(sec)
                WHERE r.effective_from <= $as_of_date
                OPTIONAL MATCH (sec)-[r2:SUPERSEDES]->(older)
                WHERE r2.effective_from <= $as_of_date
                RETURN sec, collect(DISTINCT newer) as superseded_by, collect(DISTINCT older) as supersedes
                LIMIT 1
                """,
                section_number=section_number,
                statute_id=statute_id,
                as_of_date=as_of_str,
            )
            record = result.single()
            if not record:
                return None
            return {
                "section": dict(record["sec"]),
                "superseded_by": [dict(n) for n in record["superseded_by"] if n],
                "supersedes": [dict(n) for n in record["supersedes"] if n],
            }

    def search_by_text(self, query_text: str, limit: int = 10) -> list[RetrievedPassage]:
        """Full-text search across Section and Clause nodes."""
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('sectionText', $query)
                YIELD node, score
                RETURN node, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                query=query_text,
                limit=limit,
            )
            passages = []
            for record in result:
                node = record["node"]
                score = record["score"]
                passages.append(
                    RetrievedPassage(
                        node_id=node.get("id", ""),
                        text=node.get("text", ""),
                        node_type=list(node.labels)[0] if node.labels else "",
                        title=node.get("title", ""),
                        section_number=node.get("section_number", ""),
                        score=score,
                    )
                )
            return passages

    def get_neighborhood(
        self,
        node_id: str,
        radius: int = 2,
    ) -> list[dict]:
        """Get all nodes within `radius` hops of a given node."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n {id: $node_id})
                MATCH (n)-[*1..$radius]-(neighbor)
                RETURN DISTINCT neighbor
                LIMIT 100
                """,
                node_id=node_id,
                radius=radius,
            )
            return [dict(record["neighbor"]) for record in result]


# ---------------------------------------------------------------------------
# Fallback: in-memory graph store (networkx)
# ---------------------------------------------------------------------------


class InMemoryGraph:
    """Lightweight in-memory graph using Python dicts for prototyping.

    When Neo4j is not available, this provides the same query interface
    for development and testing.
    """

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._edges: list[dict] = []
        # Adjacency: node_id -> list of (target_id, edge_type)
        self._adj: dict[str, list[tuple[str, str, dict]]] = {}

    def add_node(self, node: dict) -> None:
        node_id = node.get("id", "")
        self._nodes[node_id] = node
        if node_id not in self._adj:
            self._adj[node_id] = []

    def add_edge(self, source_id: str, target_id: str, edge_type: str, props: dict = None) -> None:
        props = props or {}
        self._edges.append({
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "props": props,
        })
        if source_id in self._adj:
            self._adj[source_id].append((target_id, edge_type, props))
        else:
            self._adj[source_id] = [(target_id, edge_type, props)]

    def get_node(self, node_id: str) -> Optional[dict]:
        return self._nodes.get(node_id)

    def find_by_type(
        self,
        node_type: str,
        filters: dict = None,
    ) -> list[dict]:
        """Find nodes by type with optional filters."""
        results = []
        for node in self._nodes.values():
            if node.get("node_type") == node_type:
                if filters:
                    match = True
                    for k, v in filters.items():
                        if node.get(k) != v:
                            match = False
                            break
                    if match:
                        results.append(node)
                else:
                    results.append(node)
        return results

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str = None,
        direction: str = "out",
    ) -> list[tuple[str, str, dict]]:
        """Get neighbors of a node."""
        results = []
        if direction == "out":
            for target_id, etype, props in self._adj.get(node_id, []):
                if edge_type is None or etype == edge_type:
                    results.append((target_id, etype, props))
        # For "in", scan all edges
        if direction == "in":
            for edge in self._edges:
                if edge["target"] == node_id:
                    if edge_type is None or edge["type"] == edge_type:
                        results.append((edge["source"], edge["type"], edge["props"]))
        return results

    def ingest_document_tree(self, tree: DocumentTree) -> None:
        """Ingest a full DocumentTree."""
        s = tree.statute
        self.add_node({
            "id": s.id,
            "node_type": "Statute",
            "title": s.title,
            "statute_id": s.statute_id,
            "publisher": str(s.publisher),
            "text": s.text,
        })
        for ch in tree.chapters:
            self.add_node({
                "id": ch.id,
                "node_type": "Chapter",
                "title": ch.title,
                "chapter_number": ch.chapter_number,
                "statute_id": ch.statute_id,
                "publisher": str(ch.publisher),
                "text": ch.text,
            })
            self.add_edge(s.id, ch.id, "CONTAINS")
        for sec in tree.sections:
            self.add_node({
                "id": sec.id,
                "node_type": "Section",
                "title": sec.title,
                "section_number": sec.section_number,
                "chapter_number": sec.chapter_number,
                "statute_id": sec.statute_id,
                "publisher": str(sec.publisher),
                "text": sec.text,
            })
            self.add_edge(s.id, sec.id, "CONTAINS")
        for cl in tree.clauses:
            self.add_node({
                "id": cl.id,
                "node_type": "Clause",
                "title": cl.title,
                "clause_number": cl.clause_number,
                "section_number": cl.section_number,
                "statute_id": cl.statute_id,
                "publisher": str(cl.publisher),
                "text": cl.text,
            })
        for ref in tree.cross_references:
            if ref.section_number:
                # Find the target section
                targets = self.find_by_type("Section", {"section_number": ref.section_number})
                for target in targets:
                    self.add_edge(s.id, target["id"], "REFERENCES", {"raw_text": ref.raw_text})

    def search_text(self, query: str) -> list[dict]:
        """Simple substring search across all node texts."""
        import re
        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        for node_id, node in self._nodes.items():
            text = node.get("text", "") + " " + node.get("title", "")
            if pattern.search(text):
                results.append({**node, "score": 1.0})
        return results[:20]