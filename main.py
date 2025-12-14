"""
Main application for RAG News QA System. 
"""

import os
import sys
import argparse
import pickle
from typing import List, Dict, Tuple, Optional

try:
    from src.preprocessing import NewsDataPreprocessor
    from src.retrieval import BM25Retriever, FAISSRetriever, HybridRetriever
    from src.query_processor import QueryProcessor
except ImportError as e:
    print(f"❌ Import Error:  {e}")
    print("Make sure all required modules are in src/ folder")
    sys.exit(1)


def setup_retriever(retriever_type: str, documents: List[Dict]) -> object:
    """Setup and build retriever index. 
    
    Args:
        retriever_type: Type of retriever ('bm25', 'faiss', 'hybrid')
        documents: List of processed documents
        
    Returns:  
        Initialized retriever object
    """
    print(f"🏗️ Building {retriever_type. upper()} index...")
    
    if retriever_type == "bm25":
        retriever = BM25Retriever(documents)
        retriever.build_index()
    elif retriever_type == "faiss":
        retriever = FAISSRetriever(documents)
        retriever.build_index()
    elif retriever_type == "hybrid":
        retriever = HybridRetriever(documents)
        retriever.build_index()
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    print(f"✅ {retriever_type.upper()} index built successfully")
    
    return retriever


def format_results(results: List[Tuple[int, float]], documents: List[Dict], show_category: bool = True):
    """Format and display retrieval results.
    
    Args:
        results: List of (index, score) tuples
        documents: List of documents
        show_category: Whether to show category information
    """
    print(f"\nRetrieved {len(results)} relevant documents:\n")
    
    for rank, (idx, score) in enumerate(results, 1):
        doc = documents[idx]
        text = doc. get('text', '')
        category = doc.get('category', 'UNKNOWN')
        headline = doc.get('headline', '')
        
        # Truncate text for display
        display_text = text[:200] + "..." if len(text) > 200 else text
        
        print(f"{rank}. (Score: {score:.4f})")
        if show_category:
            print(f"   Category: {category}")
        if headline:
            print(f"   Headline: {headline}")
        print(f"   {display_text}")
        print()


def process_single_query(query: str, retriever: object, documents:  List[Dict], 
                        top_k: int, query_processor):
    """Process a single query. 
    
    Args:
        query: User query string
        retriever: Retriever object
        documents: List of documents
        top_k: Number of results to retrieve
        query_processor: Query processor object
    """
    print(f"\nQuery: {query}")
    print("=" * 60)
    
    # Process query
    processed = query_processor.process(query)
    
    print(f"\n🔍 Query Analysis:")
    print(f"   Topic: {processed['topic']}")
    print(f"   Keywords: {', '.join(processed['keywords'])}")
    if processed['category_filters']:
        print(f"   Target Categories: {', '.join(processed['category_filters'])}")
    
    # Retrieve documents
    print(f"\n🔎 Retrieving documents...")
    results = retriever. retrieve(query, top_k=top_k)
    
    # Display results
    print("-" * 60)
    format_results(results, documents)
    print("-" * 60)


def interactive_mode(retriever:  object, documents: List[Dict], top_k: int):
    """Run in interactive mode.
    
    Args:
        retriever: Retriever object
        documents: List of documents
        top_k: Number of results to retrieve
    """
    query_processor = QueryProcessor()
    
    print("\n" + "=" * 60)
    print("🤖 RAG News QA System - Interactive Mode")
    print("=" * 60)
    print("Enter your questions about news.  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            # Get user input
            query = input("💬 Your question: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 Goodbye!")
                break
            
            # Process query
            process_single_query(query, retriever, documents, top_k, query_processor)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def show_welcome_menu():
    """Show welcome menu for easy access."""
    print("\n" + "=" * 60)
    print("🤖 Welcome to RAG News QA System!")
    print("=" * 60)
    print("\nWhat would you like to do?")
    print("1. Start Interactive Q&A")
    print("2. Ask a Single Question")
    print("3. Show Dataset Statistics")
    print("4. Exit")
    print("=" * 60)
    
    choice = input("\nYour choice (1-4): ").strip()
    return choice


def main():
    """Main application entry point."""
    
    # Check if no arguments provided - show menu
    if len(sys.argv) == 1:
        choice = show_welcome_menu()
        
        if choice == "1": 
            sys.argv.extend(["--interactive"])
        elif choice == "2":
            query = input("\n💬 Enter your question: ").strip()
            if query:
                sys.argv.extend(["--query", query])
            else:
                print("❌ No query provided!")
                return
        elif choice == "3":
            sys.argv.extend(["--show-stats", "--interactive"])
        elif choice == "4":
            print("\n👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice!")
            return
    
    parser = argparse.ArgumentParser(
        description='RAG News QA System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python main.py
  
  # Interactive mode with hybrid retrieval
  python main.py --strategy hybrid --interactive
  
  # Single query with BM25
  python main. py --query "What is the latest technology news?"
  
  # Use preprocessed data
  python main. py --load-processed data/processed/processed_news.pkl
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data', type=str,
                       default='data/News_Category_Dataset_v3.json',
                       help='Path to dataset (default: data/News_Category_Dataset_v3.json)')
    parser.add_argument('--query', type=str, help='Single query to process')
    parser.add_argument('--strategy', type=str,
                       choices=['bm25', 'faiss', 'hybrid'],
                       default='bm25',
                       help='Retrieval strategy (default: bm25)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of documents to retrieve (default: 5)')
    parser.add_argument('--save-processed', type=str,
                       help='Save processed data to this path')
    parser.add_argument('--load-processed', type=str,
                       help='Load preprocessed data from this path')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--show-stats', action='store_true',
                       help='Show dataset statistics')
    
    args = parser.parse_args()
    
    # If query is provided, disable interactive mode
    if args.query:
        args.interactive = False
    
    # If nothing specified, default to interactive
    if not args.query and not args.interactive:
        args.interactive = True
    
    # Print header
    print("\n" + "=" * 60)
    print("🚀 RAG News QA System")
    print("=" * 60 + "\n")
    
    try:
        # Initialize preprocessor
        preprocessor = NewsDataPreprocessor()
        
        # Load or process data
        if args.load_processed and os.path.exists(args. load_processed):
            print(f"📥 Loading preprocessed data from: {args.load_processed}")
            documents = preprocessor.load_processed_data(args.load_processed)
        else:
            print(f"📥 Loading and processing dataset from: {args.data}")
            documents = preprocessor.process_dataset(
                args.data,
                output_path=args.save_processed
            )
        
        # Show statistics if requested
        if args.show_stats:
            preprocessor.print_statistics()
        
        # Setup retriever
        retriever = setup_retriever(args.strategy, documents)
        
        # Initialize query processor
        query_processor = QueryProcessor()
        
        # Process based on mode
        if args.interactive:
            # Interactive mode
            interactive_mode(retriever, documents, args.top_k)
        elif args.query:
            # Single query mode
            process_single_query(args.query, retriever, documents,
                               args.top_k, query_processor)
        else:
            # Fallback to interactive
            print("ℹ️ No query provided. Starting interactive mode...")
            interactive_mode(retriever, documents, args. top_k)
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure the dataset is at: data/News_Category_Dataset_v3.json")
        print("\nOr download it using:")
        print("  python examples/download_data.py")
        print("\nOr manually download from:")
        print("  https://www.kaggle.com/datasets/rmisra/news-category-dataset")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()