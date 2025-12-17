# 📊 Data Lake RAG (OneDrive + FAISS + Streamlit)

A production-ready **Retrieval-Augmented Generation (RAG)** application that lets you **chat with PDFs stored in OneDrive / SharePoint** using a ChatGPT-style UI.

The system ingests documents from OneDrive, builds a **FAISS vector database**, and answers questions using OpenAI models — all through a clean Streamlit chat interface.

---

## 🚀 Features

- ✅ OneDrive / SharePoint PDF ingestion (Microsoft Graph)
- ✅ FAISS vector database (persistent on disk)
- ✅ Automatic rebuild of vector DB on every ingestion
- ✅ ChatGPT-style chat UI (Streamlit)
- ✅ Source attribution (which PDF the answer came from)
- ✅ Manual ingestion trigger from UI
- ✅ Windows & Linux compatible

---

## 🏗️ Architecture

