"""
document_loader.py
==================
Loads all sample data sources (text docs, CSVs, JSONs) into
LangChain Document objects with rich metadata for RBAC filtering.

Each document chunk carries:
  - page_content : the actual text
  - metadata     : source, doc_id, sensitivity, owner, type, chunk_index
"""

import os
import json
import csv
from typing import List, Dict, Any

from langchain.schema import Document


# HELPERS

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Splits a long text into overlapping chunks.
    Simple character-level splitter — no external dependency needed.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary > start + overlap:
                end = boundary
        chunks.append(text[start:end].strip())
        start = end - overlap  # overlap between chunks
    return [c for c in chunks if c]


def _base_metadata(source_key: str, doc_id: str, doc_type: str,
                   owner: str, sensitivity: str) -> Dict[str, Any]:
    """Returns a standardised metadata dict used for RBAC checks."""
    return {
        "source":      source_key,
        "doc_id":      doc_id,
        "type":        doc_type,
        "owner":       owner,
        "sensitivity": sensitivity,
    }


# 1. TEXT / POLICY DOCUMENT LOADER

# Map filename stem -> (doc_id, source_key, doc_type, owner, sensitivity)
TEXT_DOC_MAP = {
    "leave_policy":               ("HR-POL-001", "leave_policy",               "policy",     "HR",       "low"),
    "code_of_conduct":            ("HR-POL-002", "code_of_conduct",             "policy",     "HR",       "low"),
    "performance_review_policy":  ("HR-POL-003", "performance_review_policy",   "policy",     "HR",       "low"),
    "remote_work_policy":         ("HR-POL-004", "remote_work_policy",          "policy",     "HR",       "low"),
    "data_privacy_policy":        ("COMP-001",   "data_privacy_policy",         "compliance", "Legal",    "medium"),
    "information_security_policy":("COMP-002",   "information_security_policy", "compliance", "IT",       "medium"),
    "q1_security_report":         ("SEC-RPT-Q1", "q1_security_report",          "report",     "Security", "high"),
}


def load_text_documents(docs_dir: str = "data/documents") -> List[Document]:
    """Loads .txt files, chunks them, and attaches metadata."""
    documents = []

    if not os.path.exists(docs_dir):
        print(f"[!] Documents directory not found: {docs_dir}")
        return documents

    for filename in os.listdir(docs_dir):
        if not filename.endswith(".txt"):
            continue

        stem = filename.replace(".txt", "")
        if stem not in TEXT_DOC_MAP:
            continue

        doc_id, source_key, doc_type, owner, sensitivity = TEXT_DOC_MAP[stem]
        filepath = os.path.join(docs_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            full_text = f.read()

        chunks = _chunk_text(full_text)
        for idx, chunk in enumerate(chunks):
            meta = _base_metadata(source_key, doc_id, doc_type, owner, sensitivity)
            meta.update({"chunk_index": idx, "total_chunks": len(chunks),
                         "file_path": filepath})
            documents.append(Document(page_content=chunk, metadata=meta))

    print(f"[✓] Text documents loaded: {len(documents)} chunks from {docs_dir}")
    return documents


# 2. CSV DATABASE LOADER

# Map CSV filename stem -> (doc_id, source_key, doc_type, owner, sensitivity)
CSV_DOC_MAP = {
    "employees": ("DB-EMP", "employees", "database", "HR",      "medium"),
    "payroll":   ("DB-PAY", "payroll",   "database", "Finance", "high"),
}

# Number of CSV rows to bundle into a single chunk
CSV_ROWS_PER_CHUNK = 5


def load_csv_documents(csv_dir: str = "data/csv") -> List[Document]:
    """
    Loads CSV files and converts groups of rows into text chunks.
    Each chunk represents a mini-table of rows, making it LLM-friendly.
    """
    documents = []

    if not os.path.exists(csv_dir):
        print(f"[!] CSV directory not found: {csv_dir}")
        return documents

    for filename in os.listdir(csv_dir):
        if not filename.endswith(".csv"):
            continue

        stem = filename.replace(".csv", "")
        if stem not in CSV_DOC_MAP:
            continue

        doc_id, source_key, doc_type, owner, sensitivity = CSV_DOC_MAP[stem]
        filepath = os.path.join(csv_dir, filename)

        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            for row in reader:
                rows.append(row)

        # Bundle rows into chunks
        for chunk_idx in range(0, len(rows), CSV_ROWS_PER_CHUNK):
            chunk_rows = rows[chunk_idx: chunk_idx + CSV_ROWS_PER_CHUNK]

            # Convert each row to "key: value" pairs
            row_texts = []
            for row in chunk_rows:
                pairs = " | ".join(f"{k}: {v}" for k, v in row.items())
                row_texts.append(pairs)

            chunk_text = f"[{source_key.upper()} DATA — Rows {chunk_idx+1} to {chunk_idx+len(chunk_rows)}]\n"
            chunk_text += "\n".join(row_texts)

            meta = _base_metadata(source_key, doc_id, doc_type, owner, sensitivity)
            meta.update({
                "chunk_index": chunk_idx // CSV_ROWS_PER_CHUNK,
                "row_start":   chunk_idx + 1,
                "row_end":     chunk_idx + len(chunk_rows),
                "file_path":   filepath,
                "columns":     ", ".join(headers) if headers else "",
            })
            documents.append(Document(page_content=chunk_text, metadata=meta))

    print(f"[✓] CSV documents loaded: {len(documents)} chunks from {csv_dir}")
    return documents


# 3. JSON LOG LOADER

JSON_DOC_MAP = {
    "audit_logs": ("LOG-AUDIT", "audit_logs", "log", "IT", "high"),
}

JSON_ENTRIES_PER_CHUNK = 10


def load_json_documents(json_dir: str = "data/json") -> List[Document]:
    """
    Loads JSON files (arrays of log entries) and converts them to text chunks.
    """
    documents = []

    if not os.path.exists(json_dir):
        print(f"[!] JSON directory not found: {json_dir}")
        return documents

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue

        stem = filename.replace(".json", "")
        if stem not in JSON_DOC_MAP:
            continue

        doc_id, source_key, doc_type, owner, sensitivity = JSON_DOC_MAP[stem]
        filepath = os.path.join(json_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Expect a list of log entries
        if not isinstance(data, list):
            data = [data]

        for chunk_idx in range(0, len(data), JSON_ENTRIES_PER_CHUNK):
            chunk_entries = data[chunk_idx: chunk_idx + JSON_ENTRIES_PER_CHUNK]

            lines = [f"[{source_key.upper()} — Entries {chunk_idx+1} to {chunk_idx+len(chunk_entries)}]"]
            for entry in chunk_entries:
                line = " | ".join(f"{k}: {v}" for k, v in entry.items())
                lines.append(line)

            chunk_text = "\n".join(lines)
            meta = _base_metadata(source_key, doc_id, doc_type, owner, sensitivity)
            meta.update({
                "chunk_index": chunk_idx // JSON_ENTRIES_PER_CHUNK,
                "entry_start": chunk_idx + 1,
                "entry_end":   chunk_idx + len(chunk_entries),
                "file_path":   filepath,
            })
            documents.append(Document(page_content=chunk_text, metadata=meta))

    print(f"[✓] JSON documents loaded: {len(documents)} chunks from {json_dir}")
    return documents


# 4. MASTER LOADER — combines all sources

def load_all_documents() -> List[Document]:
    """
    Loads all sample data sources and returns a single flat list
    of LangChain Document objects ready for embedding.
    """
    all_docs: List[Document] = []
    all_docs.extend(load_text_documents())
    all_docs.extend(load_csv_documents())
    all_docs.extend(load_json_documents())

    print(f"\n[✓] Total documents loaded: {len(all_docs)} chunks across all sources")
    return all_docs


# QUICK TEST

if __name__ == "__main__":
    docs = load_all_documents()
    print("\nSample document:")
    if docs:
        print(f"  Content (first 200 chars): {docs[0].page_content[:200]}")
        print(f"  Metadata: {docs[0].metadata}")



