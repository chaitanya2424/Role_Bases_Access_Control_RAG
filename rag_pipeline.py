"""
rag_pipeline.py
===============
Orchestrates the complete RAG pipeline:

  User Query
    -> QueryRouter (intent + sources)
    -> RBACManager (access decision)
    -> DemoRetriever (hybrid search + rerank)
    -> PromptBuilder (construct grounded prompt)
    -> Gemini LLM (generate answer)
    -> CitationBuilder (attach source references)
    -> RAGResponse (answer + citations + confidence)

Uses Google Gemini via langchain-google-genai.
Set GEMINI_API_KEY in .env or environment variable.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from langchain.schema import Document

from retriever import DemoRetriever, RetrievalResult
from rbac import RBACManager, UserContext
from vector_store import DemoVectorStore


# CONFIGURATION

GEMINI_MODEL    = "gemini-1.5-flash"        # fast & cost-effective
MAX_CONTEXT_TOKENS = 4000                    # approx context characters
TOP_K_RETRIEVAL = 5


# DATA STRUCTURES

@dataclass
class Citation:
    """A single source citation attached to the answer."""
    index:       int
    source:      str
    doc_id:      str
    excerpt:     str        # first 150 chars of the chunk
    sensitivity: str
    owner:       str
    score:       float


@dataclass
class RAGResponse:
    """
    The complete response from the RAG pipeline.

    answer        : Generated grounded answer (or denial message)
    citations     : List of Citation objects
    confidence    : Dict with level/score/factors
    retrieval_info: The full RetrievalResult for explainability
    denied        : True if access was denied (no answer generated)
    error         : Set if a pipeline error occurred
    """
    answer:         str
    citations:      List[Citation] = field(default_factory=list)
    confidence:     Dict[str, Any] = field(default_factory=dict)
    retrieval_info: Optional[RetrievalResult] = None
    denied:         bool = False
    error:          Optional[str] = None


# LLM LOADER

def load_llm():
    """
    Loads the Gemini LLM via langchain-google-genai.
    Falls back gracefully if the API key is not set.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            print("[!] GEMINI_API_KEY not set. LLM calls will be skipped.")
            return None

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.2,     # Low temp for factual grounding
            max_output_tokens=1024,
        )
        print(f"[✓] LLM loaded: {GEMINI_MODEL}")
        return llm

    except ImportError as e:
        print(f"[!] LLM import failed: {e}")
        return None


# PROMPT BUILDER

SYSTEM_PROMPT = """You are the DemoCo Knowledge Assistant.

Your role is to answer employee questions accurately and helpfully using ONLY the provided context documents.

STRICT RULES:
1. Answer ONLY using information from the provided context. Do NOT use external knowledge.
2. If the answer is not in the context, clearly state: "I don't have enough information in the available documents to answer this question."
3. Always cite the specific document(s) you used by referring to [Source N] tags.
4. Be concise, professional, and factual.
5. Do NOT speculate or make up information.
6. If asked about restricted information you don't have access to, explain politely.

FORMAT:
- Use clear paragraphs or bullet points as appropriate.
- Reference sources as [Source 1], [Source 2], etc.
- End with a brief "Based on [Document Name]" summary if helpful.
"""

def build_prompt(query: str, documents: List[Document]) -> str:
    """
    Constructs the grounded RAG prompt with numbered context chunks.
    Each chunk is labelled [Source N] so the LLM can cite it.
    """
    if not documents:
        return f"User Question: {query}\n\nNo relevant documents found."

    context_parts = []
    for i, doc in enumerate(documents[:TOP_K_RETRIEVAL], start=1):
        source_name  = doc.metadata.get("source", "unknown")
        doc_id       = doc.metadata.get("doc_id", "")
        sensitivity  = doc.metadata.get("sensitivity", "")
        label = f"[Source {i}] — {source_name} (DocID: {doc_id}, Sensitivity: {sensitivity})"
        context_parts.append(f"{label}\n{doc.page_content}")


    prompt = f"""{SYSTEM_PROMPT}

CONTEXT DOCUMENTS:
{context_str}

USER QUESTION: {query}

ANSWER (cite sources as [Source N]):"""

    return prompt


# CITATION BUILDER

def build_citations(documents: List[Document]) -> List[Citation]:
    """Converts retrieved Document objects into Citation objects."""
    citations = []
    for i, doc in enumerate(documents[:TOP_K_RETRIEVAL], start=1):
        meta = doc.metadata
        citations.append(Citation(
            index=i,
            source=meta.get("source", "unknown"),
            doc_id=meta.get("doc_id", ""),
            excerpt=doc.page_content[:150].replace("\n", " ").strip() + "…",
            sensitivity=meta.get("sensitivity", ""),
            owner=meta.get("owner", ""),
            score=float(meta.get("rerank_score", meta.get("score", 0.5))),
        ))
    return citations


# FALLBACK ANSWER GENERATOR (when no LLM key)

def generate_fallback_answer(query: str, documents: List[Document]) -> str:
    """
    Simple extractive answer when Gemini is not available.
    Returns the most relevant chunk with source attribution.
    """
    if not documents:
        return "No relevant information found in the knowledge base."

    top_doc = documents[0]
    source  = top_doc.metadata.get("source", "document")
    content = top_doc.page_content

    # Try to find the most relevant sentence
    query_words = set(query.lower().split())
    sentences = [s.strip() for s in re.split(r'[.\n]', content) if len(s.strip()) > 30]

    best_sentence = ""
    best_overlap  = 0
    for sent in sentences:
        overlap = len(query_words & set(sent.lower().split()))
        if overlap > best_overlap:
            best_overlap  = overlap
            best_sentence = sent

    answer = best_sentence if best_sentence else content[:500]

    return (f"{answer}\n\n"
            f"*(Source: `{source}` — retrieved from the demo knowledge base. "
            f"Set GEMINI_API_KEY for full AI-generated answers.)*")


# RAG PIPELINE

class RAGPipeline:
    """
    Full RAG pipeline: routing -> RBAC -> retrieval -> generation -> citations.
    """

    def __init__(
        self,
        vector_store: DemoVectorStore,
        rbac_manager: RBACManager,
        top_k: int = TOP_K_RETRIEVAL,
    ):
        self.retriever = DemoRetriever(vector_store, rbac_manager, top_k=top_k)
        self.rbac      = rbac_manager
        self.llm       = load_llm()
        self.top_k     = top_k
        print("[✓] RAGPipeline ready.")


    def query(self, user_query: str, user_context: UserContext) -> RAGResponse:
        """
        Processes a user query end-to-end.

        Parameters
        ----------
        user_query   : Natural language question
        user_context : Authenticated user with role

        Returns
        -------
        RAGResponse with answer, citations, and confidence
        """

        retrieval = self.retriever.retrieve(user_query, user_context)

        if not retrieval.access.allowed:
            return RAGResponse(
                answer=retrieval.access.reason,
                citations=[],
                confidence={"level": "N/A", "score": 0.0, "factors": ["Access denied"]},
                retrieval_info=retrieval,
                denied=True,
            )

        if not retrieval.documents:
            return RAGResponse(
                answer=(
                    "I couldn't find relevant information in the knowledge base "
                    "for your question. Please try rephrasing or contact HR/IT support."
                ),
                citations=[],
                confidence={"level": "LOW", "score": 0.1, "factors": ["No documents retrieved"]},
                retrieval_info=retrieval,
            )

        citations = build_citations(retrieval.documents)

        confidence = self.retriever.estimate_confidence(retrieval)

        if self.llm:
            answer = self._generate_with_llm(user_query, retrieval.documents)
        else:
            answer = generate_fallback_answer(user_query, retrieval.documents)

        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            retrieval_info=retrieval,
            denied=False,
        )


    def _generate_with_llm(self, query: str, documents: List[Document]) -> str:
        """Calls the Gemini LLM with the grounded prompt."""
        try:
            prompt = build_prompt(query, documents)
            response = self.llm.invoke(prompt)

            # Extract text from LangChain response
            if hasattr(response, "content"):
                return response.content
            return str(response)

        except Exception as e:
            print(f"[!] LLM generation error: {e}")
            # Fall back to extractive answer
            return generate_fallback_answer(query, documents)


    def format_response_for_display(self, response: RAGResponse) -> Dict[str, Any]:
        """
        Formats a RAGResponse for display in the Streamlit UI.
        Returns a dict with all displayable fields.
        """
        return {
            "answer":     response.answer,
            "denied":     response.denied,
            "confidence": response.confidence,
            "citations": [
                {
                    "index":       c.index,
                    "source":      c.source,
                    "doc_id":      c.doc_id,
                    "excerpt":     c.excerpt,
                    "sensitivity": c.sensitivity,
                    "owner":       c.owner,
                    "score":       c.score,
                }
                for c in response.citations
            ],
            "trace":          response.retrieval_info.retrieval_trace if response.retrieval_info else [],
            "intent":         response.retrieval_info.route.intent if response.retrieval_info else "",
            "sources_used":   response.retrieval_info.sources_used if response.retrieval_info else [],
            "retrieved_docs": [
                {
                    "source":  d.metadata.get("source"),
                    "doc_id":  d.metadata.get("doc_id"),
                    "excerpt": d.page_content[:200],
                    "score":   d.metadata.get("rerank_score", d.metadata.get("score", 0)),
                }
                for d in (response.retrieval_info.documents if response.retrieval_info else [])
            ],
            "error": response.error,
        }


# QUICK TEST

if __name__ == "__main__":
    from vector_store import DemoVectorStore
    from rbac import RBACManager
    from document_loader import load_all_documents

    vs   = DemoVectorStore()
    rbac = RBACManager()

    if vs.is_empty():
        docs = load_all_documents()
        vs.add_documents(docs)

    pipeline = RAGPipeline(vs, rbac)

    # Test queries
    test_cases = [
        ("employee_demo", "What is the annual leave entitlement?"),
        ("employee_demo", "Show me all employee salaries"),
        ("admin_demo",    "Show me the payroll data"),
        ("manager_demo",  "What happened in the Q1 security incident?"),
    ]

    for username, query in test_cases:
        user    = rbac.get_user_context(username)
        response = pipeline.query(query, user)
        print(f"\n[{username}] Q: {query}")
        print(f"  Denied: {response.denied}")
        print(f"  Confidence: {response.confidence}")
        print(f"  Answer: {response.answer[:200]}…")




