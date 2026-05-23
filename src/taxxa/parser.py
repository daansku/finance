"""Structure-preserving parser for Finnish legal text (Finlex HTML + Vero).

Parses Finlex statute HTML into the book→chapter→section→clause tree,
and Vero guidance documents into structured Guidance nodes. Extracts
cross-references like '§ 102 a momentti 2' with regex + optional LLM pass.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Tag

from .schema import (
    Statute,
    Chapter,
    Section,
    Clause,
    Guidance,
    CrossReference,
    DocumentTree,
    Publisher,
    NodeType,
)


# ---------------------------------------------------------------------------
# Regex patterns for cross-references in Finnish legal text
# ---------------------------------------------------------------------------

# Pattern: § 102 a momentti 2, § 5, § 12.3, pykälä 102 a 2 mom
SECTION_REF_PATTERN = re.compile(
    r"(?:§|pykälä|pykälän|section)\s*"
    r"(\d+\s*[a-öA-Öä]*(?:\s*[a-öA-Öä])?)"
    r"(?:\s*(?:momentti|mom|momentin|momentissa|kohta|kohdan)\s*(\d+))?",
    re.IGNORECASE,
)

# Pattern: statute references like "AVL (1501/1993)" or "laki ... (1551/1995)"
STATUTE_REF_PATTERN = re.compile(
    r"(?:laki\s+)?(?:[A-ZÄÖÅ]{2,}(?:\s*\(\d{4}/\d{1,4}\))?)",
    re.IGNORECASE,
)

# Pattern: date references (amendments, effective dates)
DATE_PATTERN = re.compile(
    r"(\d{1,2}\.\d{1,2}\.?\d{4})",  # 1.1.2025, 1.1.25
)


# ---------------------------------------------------------------------------
# Finlex HTML parser
# ---------------------------------------------------------------------------


class FinlexParser:
    """Parse a Finlex statute HTML page into a DocumentTree.

    Finnish statutes follow a consistent structure:
      - <h1> or title: statute name + statute ID
      - Chapter headers: "1 luku", "2 luku", etc.
      - Section markers: "§ 1", "§ 2 a", etc.
      - Clause markers: "momentti" references within section text

    The parser walks the HTML tree, identifies structural markers,
    and builds the typed node hierarchy.
    """

    def __init__(self, html_path: str | Path):
        self.html_path = Path(html_path)
        self.soup: BeautifulSoup = None  # set in parse()
        self.statute: Optional[Statute] = None
        self.chapters: list[Chapter] = []
        self.sections: list[Section] = []
        self.clauses: list[Clause] = []
        self.cross_references: list[CrossReference] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def parse(self) -> DocumentTree:
        """Parse the HTML file and return a DocumentTree."""
        raw = self.html_path.read_text(encoding="utf-8")
        self.soup = BeautifulSoup(raw, "lxml")

        self._extract_statute()
        self._extract_structure()
        self._extract_cross_references()

        return DocumentTree(
            statute=self.statute,
            chapters=self.chapters,
            sections=self.sections,
            clauses=self.clauses,
            cross_references=self.cross_references,
            raw_html_path=str(self.html_path),
        )

    # ------------------------------------------------------------------
    # Statute extraction
    # ------------------------------------------------------------------

    def _extract_statute(self) -> None:
        """Extract the top-level statute metadata."""
        # Try common patterns for statute name
        title_text = ""
        statute_id = ""

        # Look for h1 or title tag
        title_tag = self.soup.find("h1") or self.soup.find("title")
        if title_tag:
            title_text = title_tag.get_text(strip=True)

        # Try to extract statute ID from the title or nearby text
        id_match = re.search(r"(\d{1,4}/\d{4})", title_text)
        if id_match:
            statute_id = id_match.group(1)

        # If not found in title, search body
        if not statute_id:
            body_text = self.soup.get_text()[:2000]
            id_match = re.search(r"(\d{1,4}/\d{4})", body_text)
            if id_match:
                statute_id = id_match.group(1)

        # Clean up title
        title_text = re.sub(r"\s+", " ", title_text).strip()

        self.statute = Statute(
            title=title_text or self.html_path.stem,
            statute_id=statute_id,
            publisher=Publisher.FINLEX,
            text=title_text,
            finlex_url=f"https://finlex.fi/fi/laki/ajantasa/{statute_id}" if statute_id else "",
        )

    # ------------------------------------------------------------------
    # Structure extraction
    # ------------------------------------------------------------------

    def _is_kho_document(self) -> bool:
        """Detect KHO (Korkein hallinto-oikeus) court decision documents.

        KHO files have a single <h1> with case ID (e.g. KHO:1980-A-I-1),
        followed by a single body <p> with the decision text, then
        reference <p> tags listing cited statutes.
        """
        return "Korkein hallinto-oikeus" in str(self.html_path)

    def _extract_structure(self) -> None:
        """Walk the document and extract chapter→section→clause hierarchy.

        For KHO court decisions, the entire body text is treated as one
        content block with cross-references extracted from the reference lines.
        """
        current_chapter: Optional[Chapter] = None
        current_section: Optional[Section] = None

        # Get all text-bearing elements in document order
        body = self.soup.find("body") or self.soup
        elements = body.find_all(["h1", "h2", "h3", "h4", "p", "div", "span", "li"])

        section_counter = 0

        # Detect KHO format
        is_kho = self._is_kho_document()

        for el in elements:
            text = el.get_text(strip=True)
            if not text or len(text) < 2:
                continue

            # ---- KHO: collect body text into a single section ----
            if is_kho:
                # After <h1> (title), first substantial <p> is body text
                if el.name == "p" and not current_section:
                    # Check if content is a reference line (short, starts with known pattern)
                    # Reference lines: "RakennusL 2 § 1 mom", "Ks. A:18.1.1977", date, diaarinumero
                    is_ref_line = bool(
                        re.match(r"^(?:[\d.]+|Ks\.|Diaarinumero|Antopäivä|Taltio|Katso)", text)
                        or re.search(r"(?:[LR]L|Laki|[Aa]setus|[Kk]aava)\b.*§", text)
                    )
                    if not is_ref_line and len(text) > 50:
                        # This is the main body text paragraph
                        sec = Section(
                            section_number="1",
                            chapter_number=1,
                            statute_id=self.statute.statute_id,
                            publisher=Publisher.FINLEX,
                            title=self.statute.title,
                            text=text,
                        )
                        self.sections.append(sec)
                        current_section = sec
                        section_counter += 1
                        continue

                # Accumulate additional body paragraphs
                if current_section and el.name == "p":
                    # Check if it's a reference line — if so, stop accumulating
                    is_ref_line = bool(
                        re.match(r"^(?:[\d.]+|Ks\.|Diaarinumero|Antopäivä|Taltio|Katso|Seutukaava|Valitus|Puheenjohtaja|Esittelijä|Asian|Esityslista|Tiedoksianto|Muutoksenhaku|KHO|Sisäasiainministeriö)", text)
                        or re.search(r"(?:[LR]L|Laki|[Aa]setus|[Kk]aava)[A-Za-z]*\s+\d+\s*[§:]", text)
                        or re.search(r"^\d+\.\d+\.\d+", text)  # Date line: "29.4.1980/2263"
                    )
                    if is_ref_line:
                        continue  # Don't mix refs into body text
                    if len(text) > 30:
                        current_section.text += " " + text
                    continue

                continue  # KHO: skip other checks below

            # Chapter detection: "X luku" or "X LUKU"
            chapter_match = re.match(r"^(\d+)\s*luku", text, re.IGNORECASE)
            if chapter_match:
                ch_num = int(chapter_match.group(1))
                ch = Chapter(
                    chapter_number=ch_num,
                    statute_id=self.statute.statute_id,
                    publisher=Publisher.FINLEX,
                    title=f"Chapter {ch_num}",
                    text=text,
                )
                self.chapters.append(ch)
                current_chapter = ch
                continue

            # Section detection: "X §" (<h3>) or "§ X ..." (<p>)
            section_match = None
            # Pattern 1: "1 §" — number before § (typical <h3>)
            m = re.match(r"^(\d+)\s*§", text)
            if m:
                section_match = m
                sec_num = m.group(1).strip()
            else:
                # Pattern 2: "§ 9 Yleinen..." or "§ 9 a ..." (§ first, as in <p>)
                m = re.match(r"^§\s*(\d+[a-öA-Öä]?(?:\.\d+)?)[\s.]", text)
                if m:
                    section_match = m
                    sec_num = m.group(1).strip()

            if section_match:
                sec = Section(
                    section_number=sec_num,
                    chapter_number=current_chapter.chapter_number if current_chapter else 1,
                    statute_id=self.statute.statute_id,
                    publisher=Publisher.FINLEX,
                    title=text,  # full header text (e.g., "1 §" or "§ 9 Yleinen verovelvollisuus.")
                    text=text,   # content text will be accumulated as paragraphs follow
                )
                self.sections.append(sec)
                current_section = sec
                section_counter += 1
                continue

            # Accumulate paragraph text under the current section
            if current_section and el.name in ("p", "div"):
                # Append paragraph text to current section's text
                if current_section.text == current_section.title:
                    # First paragraph — replace the placeholder
                    current_section.text = text
                else:
                    current_section.text += " " + text
                continue

            # Clause detection: text contains "momentti" or "mom." markers
            clause_match = re.search(
                r"(\d+)\s*(?:\.|\))?\s*(?:momentti|momentissa|momentin|mom)",
                text,
                re.IGNORECASE,
            )
            if clause_match and current_section:
                cl_num = int(clause_match.group(1))
                cl = Clause(
                    clause_number=cl_num,
                    section_number=current_section.section_number,
                    statute_id=self.statute.statute_id,
                    publisher=Publisher.FINLEX,
                    title=f"§ {current_section.section_number} mom {cl_num}",
                    text=text,
                )
                self.clauses.append(cl)

            # Also look for numbered paragraph markers (1) or 1)
            num_para = re.match(r"^\(?(\d+)\)\s", text)
            if num_para and current_section:
                p_num = int(num_para.group(1))
                cl = Clause(
                    clause_number=p_num,
                    section_number=current_section.section_number,
                    statute_id=self.statute.statute_id,
                    publisher=Publisher.FINLEX,
                    title=f"§ {current_section.section_number} mom {p_num}",
                    text=text,
                )
                self.clauses.append(cl)

    # ------------------------------------------------------------------
    # Cross-reference extraction
    # ------------------------------------------------------------------

    def _extract_cross_references(self) -> None:
        """Find all cross-references in the document text."""
        body_text = self.soup.get_text()

        for match in SECTION_REF_PATTERN.finditer(body_text):
            section_num = match.group(1).strip() if match.group(1) else None
            clause_num = int(match.group(2)) if match.group(2) else None

            ref = CrossReference(
                raw_text=match.group(0).strip(),
                statute_id=self.statute.statute_id,  # assume same statute if not otherwise specified
                section_number=section_num,
                clause_number=clause_num,
                resolution_confidence=0.0,  # regex-only, no LLM yet
            )
            self.cross_references.append(ref)

        # Deduplicate by raw_text
        seen = set()
        unique_refs = []
        for ref in self.cross_references:
            if ref.raw_text not in seen:
                seen.add(ref.raw_text)
                unique_refs.append(ref)
        self.cross_references = unique_refs


# ---------------------------------------------------------------------------
# Vero guidance parser
# ---------------------------------------------------------------------------


class VeroParser:
    """Parse a Vero guidance document (HTML) into a Guidance node.

    Vero documents have dates, titles, and cite Finlex sections. The parser
    extracts the document metadata, full text, and any Finlex references found.
    """

    def __init__(self, html_path: str | Path):
        self.html_path = Path(html_path)

    def parse(self) -> tuple[Guidance, list[CrossReference]]:
        """Parse the Vero HTML into a Guidance node + cross-references."""
        raw = self.html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw, "lxml")

        # Extract title
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else self.html_path.stem

        # Extract date
        body_text = soup.get_text()
        date_match = DATE_PATTERN.search(body_text[:2000])
        version_date = None
        if date_match:
            from datetime import datetime
            date_str = date_match.group(1)
            for fmt in ("%d.%m.%Y", "%d.%m.%y"):
                try:
                    version_date = datetime.strptime(date_str, fmt).date()
                    break
                except ValueError:
                    pass

        # Extract document ID if present
        doc_id = ""
        id_match = re.search(r"(?:Dnro|Diaarinumero|diaarinumero)\s*[:#]?\s*([\d/]+)", body_text[:2000])
        if id_match:
            doc_id = id_match.group(1)

        guidance = Guidance(
            title=title,
            text=body_text,
            guidance_id=doc_id,
            version_date=version_date,
            publisher=Publisher.VERO,
            vero_url=f"https://www.vero.fi/{self.html_path.stem}" if self.html_path.stem else "",
        )

        # Extract cross-references to Finlex
        refs: list[CrossReference] = []
        for match in SECTION_REF_PATTERN.finditer(body_text):
            section_num = match.group(1).strip() if match.group(1) else None
            clause_num = int(match.group(2)) if match.group(2) else None

            ref = CrossReference(
                raw_text=match.group(0).strip(),
                section_number=section_num,
                clause_number=clause_num,
                resolution_confidence=0.0,
            )
            refs.append(ref)

        return guidance, refs


# ---------------------------------------------------------------------------
# Bulk document processor
# ---------------------------------------------------------------------------


def parse_corpus(raw_dir: str | Path) -> list[DocumentTree]:
    """Parse all Finlex and Vero documents in the raw corpus directory.

    Walks data/raw/, identifies Finlex vs Vero documents by path,
    and returns a list of DocumentTree objects with all relationships.
    """
    raw_dir = Path(raw_dir)
    trees: list[DocumentTree] = []

    # Finlex documents
    finlex_dir = raw_dir / "finlex"
    if finlex_dir.exists():
        for html_file in finlex_dir.rglob("*.html"):
            try:
                parser = FinlexParser(html_file)
                tree = parser.parse()
                trees.append(tree)
            except Exception as e:
                print(f"  [WARN] Failed to parse Finlex {html_file}: {e}")

    # Vero documents
    vero_dir = raw_dir / "vero"
    if vero_dir.exists():
        for html_file in vero_dir.rglob("*.html"):
            try:
                parser = VeroParser(html_file)
                guidance, refs = parser.parse()
                # Wrap in a minimal DocumentTree
                tree = DocumentTree(
                    statute=Statute(
                        title=guidance.title,
                        publisher=Publisher.VERO,
                        text=guidance.text,
                    ),
                    cross_references=refs,
                    raw_html_path=str(html_file),
                )
                trees.append(tree)
            except Exception as e:
                print(f"  [WARN] Failed to parse Vero {html_file}: {e}")

    return trees


# ---------------------------------------------------------------------------
# Cross-reference resolution (LLM pass for ambiguous refs)
# ---------------------------------------------------------------------------


def load_document_trees(corpus_dir: str | Path) -> list[DocumentTree]:
    """Load parsed DocumentTree JSON files from a corpus directory.

    Reads all .json files written by ``parse_corpus`` and reconstructs
    ``DocumentTree`` objects.

    Args:
        corpus_dir: Directory containing .json files (output of ``parse_corpus``).

    Returns:
        List of ``DocumentTree`` objects.
    """
    import json

    corpus_path = Path(corpus_dir)
    trees: list[DocumentTree] = []

    for json_file in sorted(corpus_path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            tree = DocumentTree(**data)
            trees.append(tree)
        except Exception as e:
            print(f"  [WARN] Failed to load {json_file.name}: {e}")

    return trees


def resolve_ambiguous_refs_with_llm(
    refs: list[CrossReference],
    context_text: str,
    llm_call: callable,
) -> list[CrossReference]:
    """Use an LLM to resolve ambiguous cross-references.

    Args:
        refs: List of references with confidence=0 (regex-only)
        context_text: Surrounding text for context
        llm_call: Function that takes a prompt string and returns a JSON response

    Returns:
        Updated references with statute_id filled and confidence > 0
    """
    ambiguous = [r for r in refs if r.resolution_confidence == 0.0]
    if not ambiguous:
        return refs

    ref_texts = [r.raw_text for r in ambiguous]
    prompt = f"""Given the following Finnish legal text context, resolve these cross-references.
For each reference, identify:
- The statute it refers to (if named in context)
- The section number
- The clause/momentti number (if specified)

Context:
{context_text[:3000]}

References to resolve:
{json.dumps(ref_texts, ensure_ascii=False)}

Return JSON array: [{{"raw_text": "...", "statute_id": "...", "section_number": "...", "clause_number": null}}]"""

    try:
        response = llm_call(prompt)
        resolved = json.loads(response)

        # Build lookup
        resolution_map = {}
        for item in resolved:
            resolution_map[item["raw_text"]] = item

        for ref in refs:
            if ref.raw_text in resolution_map:
                resolved_data = resolution_map[ref.raw_text]
                ref.statute_id = resolved_data.get("statute_id") or ref.statute_id
                ref.section_number = resolved_data.get("section_number") or ref.section_number
                ref.clause_number = resolved_data.get("clause_number") or ref.clause_number
                ref.resolution_confidence = 1.0

    except Exception as e:
        print(f"  [WARN] LLM cross-reference resolution failed: {e}")

    return refs