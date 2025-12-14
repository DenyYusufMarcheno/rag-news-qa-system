"""Demo script for FAISS-based retrieval."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import NewsDataPreprocessor
from src.retrieval import FAISSRetriever
from src.rag_pipeline import SimpleRAGPipeline


def main():
    """Run FAISS retrieval demo."""
    
    print("=" * 60)
    print("FAISS Retrieval Demo")
    print("=" * 60)
    print()
    
    # Path to dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'News_Category_Dataset_v3.json')
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please run 'python examples/download_data.py' first")
        return
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    preprocessor = NewsDataPreprocessor()
    preprocessor.load_data(data_path)
    preprocessor.preprocess()
    
    # Limit to first 5000 documents for demo (FAISS is slower)
    documents = preprocessor.get_documents()[:5000]
    print(f"Loaded {len(documents)} documents")
    print()
    
    # Build FAISS index
    print("Building FAISS index (this may take a few minutes)...")
    retriever = FAISSRetriever()
    retriever.build_index(documents)
    print("Index built successfully!")
    print()
    
    # Create RAG pipeline
    rag = SimpleRAGPipeline(retriever)
    
    # Example queries
    queries = [
        "What are the latest developments in climate change?",
        "Tell me about technology and innovation",
        "What happened in politics recently?",
        "Sports news and updates"
    ]
    
    # Run queries
    print("=" * 60)
    print("Running Example Queries")
    print("=" * 60)
    print()
    
    for query in queries:
        print(f"Query: {query}")
        print("-" * 60)
        
        result = rag.generate_answer(query, top_k=3)
        
        print(f"Retrieved {result['num_retrieved']} documents:")
        print()
        
        for i, doc in enumerate(result['retrieved_documents'], 1):
            print(f"Document {i} (Distance: {doc['score']:.4f}):")
            print(f"  {doc['text'][:150]}...")
            print()
        
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
