# AI Interview Project — RAG with RBAC

![Python CI](https://github.com/chaitanya2424/Role_Bases_Access_Control_RAG/actions/workflows/python-app.yml/badge.svg)

This is a personal interview project I built as a fresher to show my understanding of retrieval-augmented generation and access control.

I built the whole pipeline in Python, including data generation, retrieval, RBAC checks, and a Streamlit demo app.

---

## Project files

- `requirements.txt` — Python dependencies
- `.env.example` — example environment template
- `.env` — local environment file for API keys
- `synthetic_data_generator.py` — creates synthetic sample data
- `document_loader.py` — loads documents into the retrieval system
- `vector_store.py` — builds the ChromaDB vector store
- `rbac.py` — role-based access control logic
- `query_router.py` — simple query routing and intent classification
- `retriever.py` — hybrid retrieval using embeddings + BM25
- `rag_pipeline.py` — prompt creation and LLM interaction
- `streamlit_app.py` — user interface for asking questions

---

## Setup

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Create a local `.env`

```powershell
copy .env.example .env
```

Open `.env` and set your key:

```env
GEMINI_API_KEY=your_key_here
```

### 3. Run the app

```powershell
streamlit run streamlit_app.py
```

---

## What this project does

- Generates synthetic business documents and policies
- Builds a semantic search index with ChromaDB
- Adds BM25 keyword search for hybrid retrieval
- Applies RBAC filters based on user role
- Uses Gemini for generative responses
- Falls back to extractive answers when the API key is missing

---

## How I implemented it

- Used `python-dotenv` to load environment variables from `.env`
- Used local embeddings and ChromaDB for vector storage
- Implemented a simple RBAC policy engine in `rbac.py`
- Created query routing and reranking logic in Python
- Built a Streamlit UI so the demo is easy to test

---

## Demo users

- `employee_demo` — employee access
- `manager_demo` — manager access
- `admin_demo` — admin access

---

## Notes

- The data is synthetic and generated locally.
- The system is intended as a simple interview demo.
- I can explain any part of the code in the interview.
