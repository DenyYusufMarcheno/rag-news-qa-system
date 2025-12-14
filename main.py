"""Main application for RAG News QA System."""

import os
import sys
import argparse
from src.preprocessing import NewsDataPreprocessor
from src.retrieval import BM25Retriever, FAISSRetriever, HybridRetriever
from src.rag_pipeline import SimpleRAGPipeline


def setup_retriever(retriever_type: str, documents):
    """Setup and build retriever index."""
    print(f"Building {retriever_type} index...")
    
    if retriever_type == "bm25":
        retriever = BM25Retriever()
    elif retriever_type == "faiss":
        retriever = FAISSRetriever()
    elif retriever_type == "hybrid":
        retriever = HybridRetriever()
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    retriever.build_index(documents)
    print(f"{retriever_type.upper()} index built successfully!")
    return retriever


def interactive_mode(rag_pipeline):
    """Run interactive QA mode."""
    print("\n" + "=" * 60)
    print("Interactive QA Mode")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("\nSearching for relevant documents...")
            result = rag_pipeline.generate_answer(query, top_k=3)
            
            print("\n" + "-" * 60)
            print(f"Query: {result['query']}")
            print("-" * 60)
            print(f"\nRetrieved {result['num_retrieved']} relevant documents:\n")
            
            for i, doc in enumerate(result['retrieved_documents'], 1):
                print(f"{i}. (Score: {doc['score']:.4f})")
                print(f"   {doc['text'][:200]}...")
                print()
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def batch_mode(rag_pipeline, queries):
    """Run batch QA mode."""
    print("\n" + "=" * 60)
    print("Batch QA Mode")
    print("=" * 60 + "\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query: {query}")
        print("-" * 60)
        
        result = rag_pipeline.generate_answer(query, top_k=3)
        
        print(f"Retrieved {result['num_retrieved']} documents:\n")
        
        for j, doc in enumerate(result['retrieved_documents'], 1):
            print(f"  {j}. (Score: {doc['score']:.4f})")
            print(f"     {doc['text'][:150]}...")
            print()
        
        print("-" * 60)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="RAG News QA System")
    parser.add_argument(
        "--data",
        type=str,
        default="data/News_Category_Dataset_v3.json",
        help="Path to news dataset"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bm25", "faiss", "hybrid"],
        default="bm25",
        help="Retriever type"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10000,
        help="Maximum number of documents to load"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch"],
        default="interactive",
        help="Application mode"
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        help="Queries for batch mode"
    )
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}")
        print("\nPlease download the dataset first:")
        print("  python examples/download_data.py")
        print("\nOr manually download from:")
        print("  https://www.kaggle.com/datasets/rmisra/news-category-dataset")
        return 1
    
    # Load and preprocess data
    print("=" * 60)
    print("RAG News QA System")
    print("=" * 60)
    print(f"\nLoading and preprocessing data from {args.data}...")
    
    preprocessor = NewsDataPreprocessor()
    preprocessor.load_data(args.data)
    preprocessor.preprocess()
    
    documents = preprocessor.get_documents()[:args.max_docs]
    print(f"Loaded {len(documents)} documents\n")
    
    # Build retriever
    retriever = setup_retriever(args.retriever, documents)
    
    # Create RAG pipeline
    rag_pipeline = SimpleRAGPipeline(retriever)
    print("\nRAG pipeline ready!")
    
    # Run in appropriate mode
    if args.mode == "interactive":
        interactive_mode(rag_pipeline)
    elif args.mode == "batch":
        if not args.queries:
            print("ERROR: --queries required for batch mode")
            return 1
        batch_mode(rag_pipeline, args.queries)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
