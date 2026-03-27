# FireGuard RAG: Automated Manual Analysis & QA

An AI-powered system that uses **Retrieval-Augmented Generation (RAG)** to analyze firefighter manuals and provide context-based explanations for exam questions.

## 🚀 Features
* **Local LLM Integration:** Uses Ollama (TinyLlama/Phi-3) for private, local processing.
* **Vector Database:** Implements FAISS for high-speed document retrieval.
* **Batch Processing:** Handles Excel/CSV question sets and outputs detailed explanations.
* **Document Intelligence:** Splits and chunks PDF data using RecursiveCharacterTextSplitter for better context matching.

## 🛠️ Tech Stack
* **Language:** Python
* **Orchestration:** LangChain
* **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
* **Storage:** FAISS
* **Inference:** Ollama
