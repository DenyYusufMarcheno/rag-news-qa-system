"""Demo script showing the RAG News QA System in action."""

import json
from src.preprocessing import DocumentPreprocessor
from src.query_processor import QueryProcessor
from src.retrieval import BM25Retriever


def main():
    """Run a simple demo of the system."""
    print("="*80)
    print("RAG NEWS QA SYSTEM - DEMO")
    print("="*80)
    
    # Load sample data
    print("\n1. Loading sample news data...")
    with open('data/sample_news.json', 'r') as f:
        raw_docs = json.load(f)
    print(f"   Loaded {len(raw_docs)} news articles")
    
    # Initialize components
    print("\n2. Initializing system components...")
    doc_preprocessor = DocumentPreprocessor()
    query_processor = QueryProcessor()
    bm25_retriever = BM25Retriever()
    
    # Preprocess documents
    print("\n3. Preprocessing documents...")
    documents = doc_preprocessor.preprocess_documents(raw_docs)
    print(f"   Preprocessed {len(documents)} documents")
    
    # Index documents
    print("\n4. Creating BM25 search index...")
    bm25_retriever.index_documents(documents)
    print("   Index created successfully")
    
    # Demo queries
    print("\n" + "="*80)
    print("TESTING DIFFERENT QUERIES")
    print("="*80)
    
    test_queries = [
        ("How is the economy performing this year?", ['BUSINESS', 'MONEY', 'POLITICS']),
        ("What are the latest technology developments?", ['TECH', 'SCIENCE']),
        ("Tell me about recent sports events", ['SPORTS']),
    ]
    
    for query, relevant_categories in test_queries:
        print("\n" + "-"*80)
        print(f"Query: {query}")
        print("-"*80)
        
        # Process query
        processed = query_processor.process_query(query)
        print(f"\nDetected Intent: {processed['intent']}")
        print(f"Confidence: {processed['confidence']:.2f}")
        print(f"Relevant Categories: {', '.join(processed['categories']) if processed['categories'] else 'None'}")
        print(f"Expanded Query: {processed['expanded']}")
        
        # Search WITHOUT filtering
        print("\n📋 Results WITHOUT Category Filtering:")
        results_no_filter = bm25_retriever.search(processed['keywords'], top_k=3)
        
        for i, (doc, score) in enumerate(results_no_filter, 1):
            category = doc['category']
            is_relevant = "✅" if category.upper() in [c.upper() for c in relevant_categories] else "❌"
            print(f"\n{i}. {is_relevant} [{category}] (Score: {score:.4f})")
            print(f"   {doc['headline']}")
            if doc.get('short_description'):
                print(f"   {doc['short_description'][:100]}...")
        
        # Calculate precision
        relevant_count = sum(1 for doc, _ in results_no_filter[:3]
                           if doc['category'].upper() in [c.upper() for c in relevant_categories])
        precision_no_filter = (relevant_count / 3) * 100
        
        # Search WITH filtering
        print("\n\n📋 Results WITH Category Filtering:")
        results_filtered = bm25_retriever.search(
            processed['keywords'], 
            top_k=3,
            category_filter=processed['categories']
        )
        
        for i, (doc, score) in enumerate(results_filtered, 1):
            category = doc['category']
            is_relevant = "✅" if category.upper() in [c.upper() for c in relevant_categories] else "❌"
            print(f"\n{i}. {is_relevant} [{category}] (Score: {score:.4f})")
            print(f"   {doc['headline']}")
            if doc.get('short_description'):
                print(f"   {doc['short_description'][:100]}...")
        
        # Calculate precision
        relevant_count_filtered = sum(1 for doc, _ in results_filtered[:3]
                                     if doc['category'].upper() in [c.upper() for c in relevant_categories])
        precision_filtered = (relevant_count_filtered / len(results_filtered)) * 100 if results_filtered else 0
        
        print(f"\n📊 Precision Comparison:")
        print(f"   Without filtering: {precision_no_filter:.1f}%")
        print(f"   With filtering:    {precision_filtered:.1f}%")
        print(f"   Improvement:       {precision_filtered - precision_no_filter:+.1f}%")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("✓ Query preprocessing and intent detection")
    print("✓ Query expansion with domain keywords")
    print("✓ Category-based filtering for improved precision")
    print("✓ BM25 lexical retrieval")
    print("\nFor the complete system with FAISS and reranking, use main.py")
    print("Example: python main.py --data data/sample_news.json --query 'economy' --strategy hybrid")


if __name__ == '__main__':
    main()
