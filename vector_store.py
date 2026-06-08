"""
vector_store.py
===============
Manages the ChromaDB vector store for this demo app:
  - Creates / loads a persistent ChromaDB collection
  - Embeds documents using Sentence Transformers (local, no API key needed)
  - Provides semantic search with metadata filtering for RBAC
  - Provides BM25 keyword search for hybrid retrieval
"""

import os
import json
from typing import List, Optional, Dict, Any

from langchain.schema import Document

# Lazy imports so the module is importable even before requirements are installed
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import numpy as np
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("[!] vector_store deps not yet installed — run: pip install -r requirements.txt")


# CONFIGURATION

CHROMA_PERSIST_DIR   = "chroma_db"
COLLECTION_NAME      = "demo_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # fast, lightweight, ~80 MB
TOP_K_SEMANTIC       = 10
TOP_K_BM25           = 10


# EMBEDDING WRAPPER

class LocalEmbedder:
    """
    Wraps SentenceTransformer to produce embeddings for both
    indexing (documents) and querying (search queries).
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"[*] Loading embedding model: {model_name} …")
        self.model = SentenceTransformer(model_name)
        print(f"[✓] Embedding model ready.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts."""
        embeddings = self.model.encode(texts, show_progress_bar=True,
                                       batch_size=64, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()


# VECTOR STORE MANAGER

class DemoVectorStore:
    """
    Manages document indexing and retrieval using ChromaDB + Sentence Transformers.

    Supports:
      - Semantic search via dense embeddings
      - BM25 keyword search (stored in memory)
      - Hybrid retrieval (reciprocal rank fusion)
      - Metadata-based RBAC filtering
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        if not DEPS_AVAILABLE:
            raise ImportError("Install requirements first: pip install -r requirements.txt")

        os.makedirs(persist_dir, exist_ok=True)

        # ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Embedder
        self.embedder = LocalEmbedder()

        # Collection (creates if not exists)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )

        # BM25 index stored separately in memory (rebuilt on load)
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_docs:  List[Document] = []

        print(f"[✓] VectorStore initialised. Collection: '{COLLECTION_NAME}' "
              f"({self.collection.count()} docs)")


    def add_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        """
        Embeds and inserts documents into ChromaDB.
        Also builds the BM25 index for hybrid retrieval.
        """
        if not documents:
            print("[!] No documents to add.")
            return

        print(f"[*] Indexing {len(documents)} document chunks …")

        texts     = [doc.page_content for doc in documents]
        metadatas = [doc.metadata     for doc in documents]
        ids       = [f"doc_{i}" for i in range(len(documents))]

        # Embed in batches to avoid OOM on large corpora
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            all_embeddings.extend(self.embedder.embed_documents(batch))
            print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks …")

        # Convert metadata values to strings (ChromaDB requirement)
        clean_metas = []
        for m in metadatas:
            clean_metas.append({k: str(v) for k, v in m.items()})

        # Upsert into ChromaDB in batches
        for i in range(0, len(texts), batch_size):
            self.collection.upsert(
                ids=ids[i: i + batch_size],
                embeddings=all_embeddings[i: i + batch_size],
                documents=texts[i: i + batch_size],
                metadatas=clean_metas[i: i + batch_size],
            )

        # Build BM25 index
        self._build_bm25(documents)

        print(f"[✓] Indexed {len(documents)} chunks. "
              f"Collection size: {self.collection.count()}")

    def _build_bm25(self, documents: List[Document]) -> None:
        """Builds an in-memory BM25 index over document content."""
        self._bm25_docs = documents
        tokenised = [doc.page_content.lower().split() for doc in documents]
        self._bm25_index = BM25Okapi(tokenised)
        print(f"[✓] BM25 index built over {len(documents)} chunks.")


    def semantic_search(
        self,
        query: str,
        allowed_sources: List[str],
        top_k: int = TOP_K_SEMANTIC,
    ) -> List[Document]:
        """
        Dense vector search in ChromaDB with optional RBAC source filter.

        allowed_sources: list of source keys the caller may access.
                         Pass ["*"] for admin (no filter).
        """
        query_embedding = self.embedder.embed_query(query)

        # Build ChromaDB where-filter
        if allowed_sources and "*" not in allowed_sources:
            where_filter = {"source": {"$in": allowed_sources}}
        else:
            where_filter = None

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, max(1, self.collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        docs = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert distance (0=identical) to similarity score (1=identical)
            score = 1.0 - float(dist)
            doc = Document(page_content=text, metadata={**meta, "score": round(score, 4)})
            docs.append(doc)

        return docs


    def bm25_search(
        self,
        query: str,
        allowed_sources: List[str],
        top_k: int = TOP_K_BM25,
    ) -> List[Document]:
        """
        Keyword-based BM25 search with source filtering for RBAC.
        """
        if self._bm25_index is None or not self._bm25_docs:
            print("[!] BM25 index not built yet.")
            return []

        tokenised_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenised_query)

        # Sort descending by score
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            doc = self._bm25_docs[idx]
            source = doc.metadata.get("source", "")
            # Apply RBAC filter
            if "*" not in allowed_sources and source not in allowed_sources:
                continue
            if scores[idx] > 0:
                results.append(Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "bm25_score": round(float(scores[idx]), 4)},
                ))
            if len(results) >= top_k:
                break

        return results


    def hybrid_search(
        self,
        query: str,
        allowed_sources: List[str],
        top_k: int = 6,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF) of semantic + BM25 results.

        RRF score = Î£ 1 / (k + rank_i) for each ranked list i.
        Higher score -> better combined rank.
        """
        semantic_results = self.semantic_search(query, allowed_sources, top_k=TOP_K_SEMANTIC)
        bm25_results     = self.bm25_search(query, allowed_sources, top_k=TOP_K_BM25)

        # Assign RRF scores keyed by document content (proxy for unique ID)
        rrf_scores: Dict[str, float] = {}
        content_to_doc: Dict[str, Document] = {}

        for rank, doc in enumerate(semantic_results, start=1):
            key = doc.page_content[:200]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank)
            content_to_doc[key] = doc

        for rank, doc in enumerate(bm25_results, start=1):
            key = doc.page_content[:200]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank)
            content_to_doc[key] = doc

        # Sort by combined RRF score
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        fused = []
        for key in sorted_keys[:top_k]:
            doc = content_to_doc[key]
            doc.metadata["rrf_score"] = round(rrf_scores[key], 6)
            fused.append(doc)

        return fused


    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def document_count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Deletes and recreates the collection (use carefully)."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_index = None
        self._bm25_docs  = []
        print("[✓] Vector store reset.")


# QUICK TEST

if __name__ == "__main__":
    from document_loader import load_all_documents

    vs = DemoVectorStore()

    if vs.is_empty():
        docs = load_all_documents()
        vs.add_documents(docs)

    print("\n--- Semantic search test ---")
    results = vs.semantic_search("leave policy annual days", allowed_sources=["*"])
    for r in results[:3]:
        print(f"  [{r.metadata['source']}] {r.page_content[:120]}…")

    print("\n--- Hybrid search test ---")
    results = vs.hybrid_search("sick leave medical certificate", allowed_sources=["*"])
    for r in results[:3]:
        print(f"  [{r.metadata['source']}] score={r.metadata.get('rrf_score')} {r.page_content[:100]}…")


