<div align="center">

# 🚀 RBAC + RAG Demo
### Retrieval-Augmented Generation with Role-Based Access Control in Python

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-orange?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![GitHub Workflow](https://github.com/chaitanya2424/Role_Bases_Access_Control_RAG/actions/workflows/python-app.yml/badge.svg)](https://github.com/chaitanya2424/Role_Bases_Access_Control_RAG/actions)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

> A Python interview demo combining document retrieval, hybrid search, and RBAC filtering, with a Streamlit user interface for secure question answering.

<br/>

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [What It Does](#-what-it-does)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Demo Users](#-demo-users)
- [Architecture](#-architecture)
- [Notes](#-notes)

---

## 🗺 Project Overview

This repo is a lightweight Python demo for a retrieval-augmented generation system that enforces role-based access control before returning answers.

The application:
- builds a synthetic knowledge corpus,
- indexes it with ChromaDB,
- uses hybrid retrieval (embeddings + BM25),
- filters results using RBAC policies,
- and serves the user through Streamlit.

---

## 💡 What It Does

- Generates synthetic documents and policies with `synthetic_data_generator.py`
- Loads documents into a retrieval pipeline with `document_loader.py`
- Builds and queries a ChromaDB vector store in `vector_store.py`
- Implements role-based access control in `rbac.py`
- Routes queries and selects retrieval strategies in `query_router.py`
- Performs hybrid retrieval using `retriever.py`
- Constructs prompts and interacts with the LLM in `rag_pipeline.py`
- Presents a user-friendly demo with `streamlit_app.py`

---

## 📁 Repository Structure

- `requirements.txt` — Python dependencies
- `.env.example` — sample environment file
- `synthetic_data_generator.py` — create synthetic documents and policies
- `document_loader.py` — load source data into the retrieval pipeline
- `vector_store.py` — build the ChromaDB vector index
- `rbac.py` — role-based access control logic and filters
- `query_router.py` — intent and query routing
- `retriever.py` — hybrid retrieval engine using embeddings + BM25
- `rag_pipeline.py` — prompt creation and LLM orchestration
- `streamlit_app.py` — Streamlit demo application
- `.github/workflows/python-app.yml` — GitHub Actions CI workflow

---

## ⚡ Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Create a local `.env`

```powershell
copy .env.example .env
```

Open `.env` and set your API key:

```env
GEMINI_API_KEY=your_key_here
```

### 3. Run the demo

```powershell
streamlit run streamlit_app.py
```

---

## 👤 Demo Users

Use the following demo users to verify role-specific behavior:

- `employee_demo` — employee access
- `manager_demo` — manager access
- `admin_demo` — admin access

---

## 🧠 Architecture

The pipeline is built around three core layers:

1. **Data generation** — synthetic corpus and policy content
2. **Retrieval** — vector search and BM25 keyword search
3. **RBAC enforcement** — filter retrieval results based on user role before generating answers

This architecture ensures the system only returns answers that match the current role's authorization level.

---

## 📝 Notes

- The data is synthetic and intended for demo/testing.
- The app is designed as an interview demo and is easy to extend.
- GitHub Actions validates the project on every push to `main`.
