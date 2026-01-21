# 📘 Annual Report RAG – AI Q&A System

A **production-grade Retrieval-Augmented Generation (RAG)** application that enables users to ask natural language questions about an **Annual Report PDF** and receive **accurate, evidence-backed answers strictly grounded in the document**.

The system is designed to **avoid hallucinations**, provide **page-level citations**, show **highlighted supporting evidence**, and display **confidence scores** that reflect the strength of the retrieved information.

---

## 🎯 Problem Statement

Annual reports are lengthy and complex, making it difficult to quickly extract precise information.  
The objective of this project is to build an AI system that:

- Answers questions **only using the uploaded PDF**
- Prevents hallucinated or assumed information
- Clearly communicates **what is known, partially known, or not available**
- Provides **transparent citations and confidence indicators**
- Handles ambiguous and missing-information queries safely

---

## ✅ Solution Overview

This application implements a **Retrieval-Augmented Generation (RAG)** pipeline where:

1. The PDF is loaded and split into meaningful text chunks  
2. Each chunk is enriched with **section-level metadata**  
3. Text chunks are converted into vector embeddings  
4. Embeddings are stored in a **Pinecone Vector Database**  
5. Relevant chunks are retrieved using **semantic similarity and section-aware logic**  
6. An LLM generates answers **strictly from the retrieved context**  
7. The UI presents:
   - Final answer
   - Confidence score
   - Page-level citations
   - Highlighted supporting evidence

---

## 🚀 Key Features

- 📄 Upload **any Annual Report PDF**
- 🔍 Semantic search using vector embeddings
- 🧠 Retrieval-Augmented Generation (RAG)
- 📌 Page-level citations *(Page X – Section)*
- 🟡 Highlighted supporting sentences
- 📊 Confidence score per answer
- ⚠️ Low-confidence warnings
- 🧩 Ambiguous question detection *(e.g. “How is the CEO?”)*
- ❌ Auto-hide evidence when information is not found
- 💰 Financial-driver prioritization:
  - Commissions
  - Advertising fees
  - User fees
  - Ticket sales
- 📚 Multi-page evidence merging with deduplication
- 🛡️ Strict hallucination control *(answers only from the PDF)*

---

## 🏗️ System Architecture

```text
[ PDF Document ]
        ↓
[ Text Chunking + Section Tagging ]
        ↓
[ Embedding Generation ]
        ↓
[ Pinecone Vector Store ]
        ↓
[ Section & Financial-Aware Retriever ]
        ↓
[ RAG Answer Generation ]
        ↓
[ Answer + Confidence + Citations + Evidence ]
```

---

## 🧩 Technologies Used

| Category | Libraries / APIs |
|--------|------------------|
| Programming Language | `Python` |
| Web Framework | `Streamlit` |
| LLM & AI Models | `OpenAI`, `OpenRouter` |
| Embeddings | `OpenAI Embeddings` |
| Retrieval-Augmented Generation (RAG) | `LangChain` |
| Vector Database | `Pinecone` |
| PDF Processing | `PyPDF` |
| Text Chunking | `langchain-text-splitters` |
| Semantic Search | `Pinecone`, `LangChain Retriever` |
| Environment Management | `python-dotenv` |
| Data Handling | `NumPy`, `JSON` |
| UI Rendering | `Streamlit Components` |
| Development Tools | `VS Code`, `Git` |

---

## 📂 Project Structure

```bash
ML PROJECT/
│
├── src/
│   ├── chunking.py          # Text chunking + section 
│   ├── config.py            # Environment configuration
│   ├── embeddings.py        # Embedding generation
│   ├── pdf_loader.py        # PDF loading logic
│   ├── pinecone_store.py    # Pinecone vector store 
│   └── rag_chain.py         # RAG pipeline logic
│
├── venv/                    # Virtual environment
├── .env                     # Environment variables 
├── .gitignore
├── app.py                   # Streamlit application
├── README.md
├── requirements.txt
└── command.txt
```

---

## ⚙️ Installation & Setup

Follow the steps below to set up and run the project locally.

1️⃣ Clone the Repository

```bash
git clone https://github.com/Krush2004/Annual-Report-RAG-System.git
cd Annual-Report-RAG-System
```

---

## 🔐 Environment Variables

Create a .env file in the project root directory and add:

```bash
OPENAI_API_KEY= your_openai_or_openrouter_key
PINECONE_API_KEY= your_pinecone_api_key
PINECONE_INDEX= swiggy-rag
```

---

▶️ Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```
----
