# RAG News QA System

A Retrieval-Augmented Generation (RAG) system for question answering on news articles using the News Category Dataset from Kaggle.

## Overview

This system implements a complete RAG pipeline that:
- Preprocesses news article dataset
- Builds retrieval indexes (BM25 and FAISS)
- Retrieves relevant documents based on queries
- Combines retrieval with LLM for answer generation
- Evaluates retrieval and generation quality

## Features

- **Multiple Retrieval Methods**:
  - BM25 (sparse retrieval)
  - FAISS (dense retrieval with sentence embeddings)
  - Hybrid (combination of BM25 and FAISS)

- **Document Processing**:
  - Text cleaning and preprocessing
  - Metadata extraction
  - Efficient indexing

- **Evaluation Metrics**:
  - Retrieval: Precision@K, Recall@K, MRR, NDCG@K
  - Generation: Exact Match, Token F1

- **Interactive and Batch Modes**:
  - Interactive QA interface
  - Batch query processing

## Installation

### Requirements
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DenyYusufMarcheno/rag-news-qa-system.git
cd rag-news-qa-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
python examples/download_data.py
```

**Note**: You need Kaggle API credentials. See [Kaggle API documentation](https://github.com/Kaggle/kaggle-api) for setup instructions.

Alternatively, manually download from: https://www.kaggle.com/datasets/rmisra/news-category-dataset

## Usage

### Main Application

Run the interactive QA system:
```bash
python main.py --retriever bm25
```

Options:
- `--data`: Path to dataset (default: `data/News_Category_Dataset_v3.json`)
- `--retriever`: Retriever type (`bm25`, `faiss`, `hybrid`) (default: `bm25`)
- `--max-docs`: Maximum documents to load (default: 10000)
- `--mode`: Application mode (`interactive`, `batch`) (default: `interactive`)
- `--queries`: Queries for batch mode

Example batch mode:
```bash
python main.py --mode batch --queries "What is happening in politics?" "Tell me about technology news"
```

### Demo Scripts

#### BM25 Retrieval Demo
```bash
python examples/demo_bm25.py
```

#### FAISS Retrieval Demo
```bash
python examples/demo_faiss.py
```

#### Evaluation Demo
```bash
python examples/demo_evaluation.py
```

## Architecture

### Components

1. **Preprocessing Module** (`src/preprocessing.py`)
   - Loads and cleans news dataset
   - Combines headlines and descriptions
   - Extracts metadata

2. **Retrieval Module** (`src/retrieval.py`)
   - BM25Retriever: Sparse retrieval using BM25 algorithm
   - FAISSRetriever: Dense retrieval using sentence embeddings
   - HybridRetriever: Combined approach

3. **RAG Pipeline** (`src/rag_pipeline.py`)
   - RAGPipeline: Full RAG with LLM integration
   - SimpleRAGPipeline: Retrieval-only pipeline for demonstration

4. **Evaluation Module** (`src/evaluation.py`)
   - RetrievalEvaluator: Metrics for retrieval quality
   - GenerationEvaluator: Metrics for answer generation
   - RAGEvaluator: Combined evaluation

## Project Structure

```
rag-news-qa-system/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data preprocessing
│   ├── retrieval.py          # Retrieval implementations
│   ├── rag_pipeline.py       # RAG pipeline
│   └── evaluation.py         # Evaluation metrics
├── examples/
│   ├── download_data.py      # Dataset downloader
│   ├── demo_bm25.py          # BM25 demo
│   ├── demo_faiss.py         # FAISS demo
│   └── demo_evaluation.py    # Evaluation demo
├── main.py                   # Main application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Dataset

**News Category Dataset v3**
- Source: https://www.kaggle.com/datasets/rmisra/news-category-dataset
- Size: ~200k news articles
- Format: JSON (one article per line)
- Fields: headline, short_description, category, date, authors, link

## Evaluation

The system includes comprehensive evaluation capabilities:

### Retrieval Metrics
- **Precision@K**: Proportion of relevant documents in top-K results
- **Recall@K**: Proportion of relevant documents retrieved
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant document
- **NDCG@K**: Normalized Discounted Cumulative Gain

### Generation Metrics
- **Exact Match**: Binary match between generated and reference answers
- **Token F1**: F1 score of token overlap

## Examples

### Interactive Mode
```bash
$ python main.py --retriever bm25
RAG News QA System
==================================================================
Loading and preprocessing data...
Loaded 10000 documents

Building BM25 index...
BM25 index built successfully!

Interactive QA Mode
Type 'quit' or 'exit' to stop
==================================================================

Your question: What are the latest developments in climate change?

Searching for relevant documents...

Query: What are the latest developments in climate change?
------------------------------------------------------------------

Retrieved 3 relevant documents:

1. (Score: 15.2341)
   Climate scientists warn of accelerating global warming...

2. (Score: 13.8952)
   New climate policy aims to reduce carbon emissions...

3. (Score: 12.4563)
   Environmental groups push for stronger climate action...
```

### Batch Mode
```bash
python main.py --mode batch --retriever faiss --queries \
    "What is happening in politics?" \
    "Tell me about technology innovations" \
    "Latest sports updates"
```

## Performance Considerations

- **BM25**: Fast, works well for keyword matching, lower memory usage
- **FAISS**: Better semantic understanding, higher memory usage, slower indexing
- **Hybrid**: Best accuracy, combines both approaches

For large datasets (>100k documents):
- Use BM25 for fast prototyping
- Use FAISS with GPU for production
- Consider document chunking for very long texts

## Future Enhancements

- [ ] Integration with larger LLMs (LLaMA, GPT)
- [ ] Query expansion and reformulation
- [ ] Re-ranking strategies
- [ ] Caching for frequent queries
- [ ] Web interface
- [ ] Multi-language support
- [ ] Real-time news updates

## License

MIT License

## Citation

If you use the News Category Dataset, please cite:
```
Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022).
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
