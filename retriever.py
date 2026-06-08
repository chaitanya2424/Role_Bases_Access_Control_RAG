"""
retriever.py
============
Orchestrates the full retrieval pipeline:

  1. Uses QueryRouter to determine intent + target sources
  2. Uses RBACManager to validate access and filter sources
  3. Calls DemoVectorStore for hybrid retrieval
  4. Applies post-retrieval RBAC document filter (defence-in-depth)
  5. Reranks results using cross-encoder similarity
  6. Returns a RetrievalResult with documents + metadata

The reranker is a lightweight cosine-similarity reranker using the same
embedding model (no separate cross-encoder download needed).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from langchain.schema import Document

from query_router import QueryRouter, RouteResult
from rbac import RBACManager, UserContext, AccessDecision
from vector_store import DemoVectorStore


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """
    Complete output of the retrieval pipeline.

    documents    : Final ranked list of relevant documents
    route        : Routing decision metadata
    access       : RBAC decision
    query        : Original user query
    top_k        : Number of documents returned
    sources_used : Unique sources that contributed documents
    """
    documents:    List[Document]
    route:        RouteResult
    access:       AccessDecision
    query:        str
    top_k:        int
    sources_used: List[str] = field(default_factory=list)
    retrieval_trace: List[str] = field(default_factory=list)  # explainability log


# ════════════════════════════════════════════════════════════════════════════
# RETRIEVER
# ════════════════════════════════════════════════════════════════════════════

class DemoRetriever:
    """
    End-to-end retrieval with routing, RBAC, hybrid search, and reranking.
    """

    def __init__(
        self,
        vector_store: DemoVectorStore,
        rbac_manager: RBACManager,
        top_k: int = 5,
    ):
        self.vs      = vector_store
        self.rbac    = rbac_manager
        self.router  = QueryRouter()
        self.top_k   = top_k
        print("[✓] Retriever initialised.")

    # ── MAIN RETRIEVE METHOD ──────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        user_context: UserContext,
    ) -> RetrievalResult:
        """
        Full retrieval pipeline for a single query.

        Returns a RetrievalResult; if access is denied, documents=[]
        and access.allowed=False with access.reason explaining why.
        """
        trace: List[str] = []

        # ── Step 1: Route the query ───────────────────────────────────────────
        route = self.router.route(query)
        trace.append(f"🔀 **Routing:** intent=`{route.intent}` "
                     f"(confidence {route.confidence:.0%})")
        trace.append(f"   Target sources: {route.target_sources[:4]}")

        # ── Step 2: RBAC check ────────────────────────────────────────────────
        access = self.rbac.check_access(
            user_context=user_context,
            intent=route.intent,
            requested_sources=route.target_sources,
        )

        if not access.allowed:
            trace.append(f"🔒 **RBAC:** DENIED — {route.intent} not permitted for "
                         f"role `{user_context.role}`")
            return RetrievalResult(
                documents=[],
                route=route,
                access=access,
                query=query,
                top_k=0,
                retrieval_trace=trace,
            )

        trace.append(f"✅ **RBAC:** GRANTED for role `{user_context.role}`")

        # ── Step 3: Resolve effective allowed sources ─────────────────────────
        # Intersect router's target sources with RBAC allowed sources
        allowed_sources = access.allowed_sources  # ["*"] for admin
        if "*" not in allowed_sources:
            # Only search sources that are both targeted and permitted
            effective_sources = [s for s in route.target_sources
                                 if s in allowed_sources]
            if not effective_sources:
                # Fall back to all allowed sources
                effective_sources = allowed_sources
        else:
            # Admin: prioritise router's target sources but search all
            effective_sources = route.target_sources if route.target_sources else ["*"]

        trace.append(f"📂 **Effective sources:** {effective_sources[:5]}")

        # ── Step 4: Hybrid retrieval ──────────────────────────────────────────
        retrieved = self.vs.hybrid_search(
            query=query,
            allowed_sources=effective_sources,
            top_k=self.top_k * 2,  # over-retrieve, then rerank to top_k
        )
        trace.append(f"🔍 **Retrieved:** {len(retrieved)} chunks (pre-rerank)")

        # ── Step 5: Post-retrieval RBAC filter (defence in depth) ─────────────
        filtered = self.rbac.filter_documents(retrieved, user_context)
        removed = len(retrieved) - len(filtered)
        if removed > 0:
            trace.append(f"🛡️ **Post-filter:** Removed {removed} unauthorised chunks")

        # ── Step 6: Rerank ────────────────────────────────────────────────────
        reranked = self._rerank(query, filtered, top_k=self.top_k)
        trace.append(f"📊 **Reranked:** Returning top {len(reranked)} chunks")

        # ── Step 7: Collect unique sources used ───────────────────────────────
        sources_used = list({doc.metadata.get("source", "unknown")
                             for doc in reranked})

        return RetrievalResult(
            documents=reranked,
            route=route,
            access=access,
            query=query,
            top_k=len(reranked),
            sources_used=sources_used,
            retrieval_trace=trace,
        )

    # ── RERANKER ──────────────────────────────────────────────────────────────

    def _rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        """
        Lightweight reranker that combines:
          - RRF score from hybrid search
          - Query-document term overlap bonus
          - Recency bonus (not used here but placeholder exists)

        For production, replace with a cross-encoder like
        'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        """
        if not documents:
            return []

        query_tokens = set(query.lower().split())
        scored = []

        for doc in documents:
            # Base score from hybrid RRF
            rrf   = float(doc.metadata.get("rrf_score", 0.01))
            dense = float(doc.metadata.get("score", 0.5))

            # Term overlap bonus (normalised)
            doc_tokens = set(doc.page_content.lower().split())
            overlap    = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            overlap    = min(overlap, 1.0)

            # Combined score
            combined = (0.5 * rrf * 100) + (0.3 * dense) + (0.2 * overlap)
            scored.append((combined, doc))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Annotate with final rerank score
        result = []
        for rank, (score, doc) in enumerate(scored[:top_k], start=1):
            doc.metadata["rerank_score"] = round(score, 4)
            doc.metadata["rank"] = rank
            result.append(doc)

        return result

    # ── CONFIDENCE ESTIMATOR ──────────────────────────────────────────────────

    def estimate_confidence(self, retrieval_result: RetrievalResult) -> Dict[str, Any]:
        """
        Estimates response confidence based on retrieval quality.

        Returns a dict with:
          - level   : "HIGH" / "MEDIUM" / "LOW"
          - score   : float 0.0–1.0
          - factors : explanation of what drove the score
        """
        docs = retrieval_result.documents

        if not docs:
            return {"level": "NONE", "score": 0.0, "factors": ["No documents retrieved"]}

        factors = []

        # Factor 1: Number of relevant documents
        doc_count_score = min(len(docs) / self.top_k, 1.0)
        factors.append(f"{len(docs)} relevant chunks found")

        # Factor 2: Top document similarity score
        top_score = float(docs[0].metadata.get("score",
                          docs[0].metadata.get("rerank_score", 0.5)))
        factors.append(f"Top document score: {top_score:.2f}")

        # Factor 3: Source diversity
        sources = {d.metadata.get("source") for d in docs}
        diversity = min(len(sources) / 3, 1.0)
        factors.append(f"Sources: {', '.join(sources)}")

        # Factor 4: Route confidence
        route_conf = retrieval_result.route.confidence
        factors.append(f"Router confidence: {route_conf:.0%}")

        # Weighted combined score
        combined = (0.30 * doc_count_score +
                    0.35 * min(top_score, 1.0) +
                    0.15 * diversity +
                    0.20 * route_conf)

        if combined >= 0.70:
            level = "HIGH"
        elif combined >= 0.45:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {"level": level, "score": round(combined, 3), "factors": factors}


# ════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from vector_store import DemoVectorStore
    from rbac import RBACManager
    from document_loader import load_all_documents

    vs   = DemoVectorStore()
    rbac = RBACManager()

    if vs.is_empty():
        docs = load_all_documents()
        vs.add_documents(docs)

    retriever = DemoRetriever(vs, rbac)

    # Test 1: Employee asks about leave
    user = rbac.get_user_context("employee_demo")
    result = retriever.retrieve("What is the annual leave entitlement?", user)
    print(f"\nLeave query → access={result.access.allowed}, docs={len(result.documents)}")
    for doc in result.documents[:2]:
        print(f"  [{doc.metadata['source']}] {doc.page_content[:100]}…")

    # Test 2: Employee tries to access salary
    result = retriever.retrieve("Show me all salaries", user)
    print(f"\nSalary query → access={result.access.allowed}")
    print(f"  Reason: {result.access.reason[:80]}…")

    # Test 3: Admin accesses salary
    admin = rbac.get_user_context("admin_demo")
    result = retriever.retrieve("Show me all salaries", admin)
    print(f"\nAdmin salary query → access={result.access.allowed}, docs={len(result.documents)}")
