# RAG News QA System

A Retrieval-Augmented Generation (RAG) system for question answering on news articles.

## 📚 Learning Resources

**New to this project?** Check out our comprehensive learning guide:

👉 **[LEARNING_FUNDAMENTALS.md](LEARNING_FUNDAMENTALS.md)** - Learn what subjects you need to understand this project from the basics.

This guide covers:
- Python programming basics
- Natural Language Processing (NLP)
- Information Retrieval (BM25, FAISS)
- Machine Learning and Deep Learning fundamentals
- Vector embeddings and semantic search
- RAG architecture
- Evaluation metrics
- Recommended learning path (beginner to advanced)

## 🚀 Project Status

This is an educational RAG system implementing document retrieval and question answering on news articles.

For full documentation and implementation details, see the [implementation branch](../../tree/copilot/implement-document-retrieval-llm):
- Complete README with installation and usage instructions
- Technical report (REPORT.md)
- Implementation summary (IMPLEMENTATION_SUMMARY.md)
- Working code examples in `src/` and `examples/` directories

## 🎯 Quick Overview

**What this project does:**
- Preprocesses news article datasets
- Builds retrieval indexes (BM25 and FAISS)
- Retrieves relevant documents based on queries
- Combines retrieval with LLMs for answer generation
- Evaluates retrieval and generation quality

**Technologies used:**
- Python, NumPy, Pandas
- BM25 for sparse retrieval
- FAISS for dense vector search
- Sentence Transformers for embeddings
- HuggingFace Transformers for LLMs
