"""
Hybrid Retrieval — embeddings find entry points, graph traversal walks to neighbourhood.

Components:
- EmbeddingStore: ChromaDB vector store + embedding model
- GraphRetriever: graph traversal from entry-point nodes
- Reranker: cohere-like reranking on assembled passage sets
- HybridRetriever: orchestrates the full hybrid pipeline
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Callable

# Load .env file before reading any env vars
try:
    from dotenv import load_dotenv
    _repo_root = Path(__file__).resolve().parent.parent.parent
    _env_path = _repo_root / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

# Limit CPU threads to prevent overheating on consumer hardware.
# Set TAXXA_NUM_THREADS env var to override (default: 4).
_num_threads = os.environ.get("TAXXA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", _num_threads)
os.environ.setdefault("MKL_NUM_THREADS", _num_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _num_threads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _num_threads)

# Also limit PyTorch threads if torch is available
try:
    import torch
    torch.set_num_threads(int(_num_threads))
    # Prevent torch from spawning too many interop threads
    torch.set_num_interop_threads(min(int(_num_threads), 2))
except ImportError:
    pass

import chromadb
from chromadb.config import Settings as ChromaSettings

from .schema import RetrievedPassage


# ---------------------------------------------------------------------------
# GPU detection helper
# ---------------------------------------------------------------------------

def _get_device() -> str:
    """Detect the best available device.

    Respects TAXXA_DEVICE env var (cpu, cuda, mps).
    Defaults to 'cpu' to avoid GPU overheating on laptops/consumer hardware.
    Set TAXXA_DEVICE=cuda or TAXXA_DEVICE=mps to opt into GPU acceleration.
    """
    env_device = os.environ.get("TAXXA_DEVICE", "").strip().lower()
    if env_device in ("cpu", "cuda", "mps"):
        return env_device

    # No explicit override: default to CPU for safety
    return "cpu"


# ---------------------------------------------------------------------------
# Embedding model abstraction
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Thin wrapper around sentence-transformers or OpenAI-compatible embeddings.

    Default: all-MiniLM-L6-v2 — 384-dim, ~80 MB, fast on CPU, English/Finnish OK.
    Larger option: BAAI/bge-m3 — 1024-dim, ~2 GB, slow on CPU but better quality.
    Fallback: OpenAI-compatible API (OpenRouter or local Ollama).

    Set TAXXA_USE_API_EMBED=1 to force API mode (avoids loading large local models).
    """

    def __init__(
        self,
        model_name: str | None = None,
        use_api: bool = False,
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        device: str | None = None,
    ):
        # Respect TAXXA_USE_API_EMBED env var to avoid local model loading
        if os.environ.get("TAXXA_USE_API_EMBED", "").strip() == "1":
            use_api = True
            api_base = os.environ.get("TAXXA_EMBED_API_BASE", api_base)
            model_name = os.environ.get("TAXXA_EMBED_MODEL", "nomic-embed-text")
            # Use the LLM API key for embeddings too if no separate embed key
            api_key = os.environ.get("TAXXA_EMBED_API_KEY") or os.environ.get("TAXXA_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY", api_key)

        self.model_name = model_name or os.environ.get("TAXXA_EMBED_MODEL", "BAAI/bge-m3")
        self.use_api = use_api
        self._model = None
        self._api_base = api_base
        self._api_key = api_key
        self._device = device or _get_device()

        if not use_api:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[EmbeddingModel] Loading {self.model_name} on {self._device} ...")
                self._model = SentenceTransformer(self.model_name, device=self._device)
            except ImportError:
                print("[WARN] sentence-transformers not installed, falling back to API")
                self.use_api = True
        else:
            print(f"[EmbeddingModel] Using API: {self.model_name} @ {self._api_base}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, return list of float vectors."""
        if self._model and not self.use_api:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=64,
            )
            return embeddings.tolist()

        # OpenAI-compatible API path (supports batched input)
        import httpx

        # Send all texts in one batch request if the API supports it
        try:
            response = httpx.post(
                f"{self._api_base}/embeddings",
                json={
                    "model": self.model_name,
                    "input": texts,
                },
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=120,
            )
            if response.status_code == 200:
                data = response.json()
                return [item["embedding"] for item in data["data"]]
        except Exception:
            pass

        # Fallback: one-by-one
        embeddings = []
        for text in texts:
            try:
                response = httpx.post(
                    f"{self._api_base}/embeddings",
                    json={
                        "model": self.model_name,
                        "input": text,
                    },
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    embeddings.append(data["data"][0]["embedding"])
                else:
                    embeddings.append([0.0] * 1024)
            except Exception:
                embeddings.append([0.0] * 1024)

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([query])[0]


# ---------------------------------------------------------------------------
# ChromaDB vector store
# ---------------------------------------------------------------------------


class EmbeddingStore:
    """ChromaDB-backed vector store for document nodes.

    Stores embeddings of Section and Clause text, indexed by node_id,
    with metadata (title, section_number, statute_id, publisher).
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "taxxa_sections",
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model or EmbeddingModel()

        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_nodes(self, nodes: list[dict], chunk_size: int = 2000) -> None:
        """Add nodes to the vector store in chunks to avoid OOM.

        Each node dict should have: id, text, title, section_number, statute_id, node_type.
        Text is truncated to ``max_text_len`` chars to keep embeddings fast on CPU.
        """
        if not nodes:
            return

        # Filter out nodes with empty text
        valid_nodes = []
        for node in nodes:
            text = node.get("text", "") or node.get("title", "")
            if text.strip():
                valid_nodes.append(node)

        total = len(valid_nodes)
        print(f"  Embedding {total} nodes in chunks of {chunk_size} ...")

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk = valid_nodes[chunk_start:chunk_end]

            ids = []
            documents = []
            metadatas = []
            texts_to_embed = []

            for node in chunk:
                text = node.get("text", "") or node.get("title", "")
                ids.append(node["id"])
                documents.append(text)
                metadatas.append({
                    "title": node.get("title", ""),
                    "section_number": node.get("section_number", ""),
                    "statute_id": node.get("statute_id", ""),
                    "node_type": node.get("node_type", ""),
                    "publisher": node.get("publisher", ""),
                })
                texts_to_embed.append(text)

            embeddings = self.embedding_model.embed(texts_to_embed)

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            print(f"  Chunk {chunk_start // chunk_size + 1}: {chunk_start}-{chunk_end} / {total}")

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
    ) -> list[RetrievedPassage]:
        """Search the vector store and return ranked passages."""
        query_embedding = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        passages = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                doc_text = results["documents"][0][i] if results["documents"] else ""
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Convert cosine distance to similarity score
                score = 1.0 - distance

                passages.append(
                    RetrievedPassage(
                        node_id=doc_id,
                        text=doc_text,
                        node_type=meta.get("node_type", ""),
                        title=meta.get("title", ""),
                        section_number=meta.get("section_number", ""),
                        score=score,
                    )
                )

        return passages

    def count(self) -> int:
        """Number of nodes stored."""
        return self.collection.count()


# ---------------------------------------------------------------------------
# Graph retriever
# ---------------------------------------------------------------------------


class GraphRetriever:
    """Given entry-point nodes from embeddings, traverse the graph for context.

    Uses either Neo4j or the InMemoryGraph for traversal.
    """

    def __init__(self, graph_store):
        """
        Args:
            graph_store: Neo4jStore or InMemoryGraph instance
        """
        self.graph = graph_store

    def expand_context(
        self,
        entry_passages: list[RetrievedPassage],
        radius: int = 2,
        max_total: int = 50,
    ) -> list[RetrievedPassage]:
        """Expand context by walking the graph from each entry passage.

        For each entry node, get its neighborhood within `radius` hops.
        Combine with original passages, deduplicate, and return.
        """
        seen_ids = {p.node_id for p in entry_passages}
        expanded = list(entry_passages)

        for passage in entry_passages[:5]:  # limit to top 5 to avoid explosion
            try:
                neighbors = self.graph.get_neighborhood(passage.node_id, radius=radius)
                for neighbor in neighbors:
                    nid = neighbor.get("id", "")
                    if nid and nid not in seen_ids:
                        seen_ids.add(nid)
                        expanded.append(
                            RetrievedPassage(
                                node_id=nid,
                                text=neighbor.get("text", ""),
                                node_type=neighbor.get("node_type", ""),
                                title=neighbor.get("title", ""),
                                section_number=neighbor.get("section_number", ""),
                                score=passage.score * 0.8,  # discount expanded nodes
                            )
                        )
            except Exception as e:
                # Graph may not have that node, or Neo4j may be down
                pass

            if len(expanded) >= max_total:
                break

        return expanded[:max_total]

    def traverse_references(
        self,
        section_number: str,
        statute_id: str,
        max_depth: int = 3,
    ) -> list[RetrievedPassage]:
        """Follow the REFERENCES chain from a section."""
        try:
            paths = self.graph.walk_references(section_number, statute_id, max_depth)
            passages = []
            for path in paths:
                for node in path.get("nodes", []):
                    passages.append(
                        RetrievedPassage(
                            node_id=node.get("id", ""),
                            text=node.get("text", ""),
                            node_type=node.get("node_type", ""),
                            title=node.get("title", ""),
                            section_number=node.get("section_number", ""),
                            score=0.7,
                        )
                    )
            return passages
        except Exception:
            return []

    def find_interpretations(
        self,
        section_number: str,
        statute_id: str,
    ) -> list[RetrievedPassage]:
        """Find Vero guidance interpreting a Finlex section."""
        try:
            interpretations = self.graph.find_interpretations(section_number, statute_id)
            return [
                RetrievedPassage(
                    node_id=i.get("id", ""),
                    text=i.get("text", ""),
                    node_type="Guidance",
                    title=i.get("title", ""),
                    section_number="",
                    score=0.85,
                )
                for i in interpretations
            ]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Rerank retrieved passages for final relevance scoring.

    Default: cross-encoder reranker via sentence-transformers (bge-reranker-v2-m3).
    API fallback: Cohere Rerank compatible endpoint.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_api: bool = False,
        api_base: str = None,
        api_key: str = None,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.use_api = use_api
        self._model = None
        self._api_base = api_base
        self._api_key = api_key
        self._device = device or _get_device()

        if not use_api:
            try:
                from sentence_transformers import CrossEncoder
                print(f"[Reranker] Loading {model_name} on {self._device} ...")
                self._model = CrossEncoder(model_name, device=self._device)
            except ImportError:
                print("[WARN] CrossEncoder not available, using score-based reranking")
                self.use_api = True

    def rerank(
        self,
        query: str,
        passages: list[RetrievedPassage],
        top_k: int = 10,
    ) -> list[RetrievedPassage]:
        """Rerank passages and return top_k."""
        if not passages:
            return []

        if self._model and not self.use_api:
            pairs = [(query, p.text) for p in passages]
            scores = self._model.predict(pairs, show_progress_bar=False)
            for i, passage in enumerate(passages):
                passage.score = float(scores[i])
        elif self.use_api and self._api_base:
            # API path (Cohere-compatible)
            import httpx
            try:
                response = httpx.post(
                    f"{self._api_base}/rerank",
                    json={
                        "model": self.model_name,
                        "query": query,
                        "documents": [p.text for p in passages],
                        "top_n": top_k,
                    },
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    score_map = {r["index"]: r.get("relevance_score", 0.0) for r in results}
                    for i, passage in enumerate(passages):
                        passage.score = score_map.get(i, passage.score)
            except Exception:
                pass  # keep original scores

        # Sort by score descending, return top_k
        passages.sort(key=lambda p: p.score, reverse=True)
        return passages[:top_k]


# ---------------------------------------------------------------------------
# Hybrid retriever — orchestrates the full pipeline
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Orchestrate hybrid retrieval: embeddings → graph expansion → rerank.

    Usage:
        retriever = HybridRetriever(embedding_store, graph_retriever, reranker)
        passages = retriever.retrieve("What is the capital income tax rate?")
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        graph_retriever: GraphRetriever,
        reranker: Optional[Reranker] = None,
        top_k_embedding: int = 10,
        graph_radius: int = 2,
        final_top_k: int = 10,
    ):
        self.embeddings = embedding_store
        self.graph = graph_retriever
        self.reranker = reranker or Reranker()
        self.top_k_embedding = top_k_embedding
        self.graph_radius = graph_radius
        self.final_top_k = final_top_k

    def retrieve(
        self,
        query: str,
        filter_publisher: str = None,
        filter_statute_id: str = None,
    ) -> list[RetrievedPassage]:
        """Full hybrid retrieval pipeline.

        1. Embedding search to find entry-point nodes
        2. Graph expansion from those nodes
        3. Optional: traverse cross-references and interpretations
        4. Rerank the assembled passage set
        """

        # Step 1: Embedding lookup
        where_filter = None
        if filter_publisher:
            where_filter = {"publisher": filter_publisher}
        if filter_statute_id:
            if where_filter:
                where_filter["statute_id"] = filter_statute_id
            else:
                where_filter = {"statute_id": filter_statute_id}

        entry_passages = self.embeddings.search(
            query,
            n_results=self.top_k_embedding,
            where=where_filter,
        )

        # Step 2: Graph expansion
        expanded = self.graph.expand_context(
            entry_passages,
            radius=self.graph_radius,
            max_total=50,
        )

        # Step 3: For each entry section, try to find interpretations
        for passage in entry_passages[:3]:
            if passage.section_number and passage.node_type in ("Section",):
                interpretations = self.graph.find_interpretations(
                    passage.section_number,
                    "",
                )
                expanded.extend(interpretations)

        # Step 4: Rerank
        final = self.reranker.rerank(query, expanded, top_k=self.final_top_k)

        return final

    def retrieve_with_cross_refs(
        self,
        query: str,
        section_number: str = None,
        statute_id: str = None,
    ) -> list[RetrievedPassage]:
        """Retrieve with explicit cross-reference traversal."""
        passages = self.retrieve(query)

        if section_number and statute_id:
            ref_passages = self.graph.traverse_references(section_number, statute_id)
            passages.extend(ref_passages)
            passages = self.reranker.rerank(query, passages, top_k=self.final_top_k)

        return passages