"""Comprehensive demonstration of the RAG News QA System."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import NewsDataPreprocessor
from src.retrieval import BM25Retriever, FAISSRetriever, HybridRetriever
from src.rag_pipeline import SimpleRAGPipeline
from src.evaluation import RetrievalEvaluator
from src.utils import DEFAULT_DATASET_PATH


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


def demo_preprocessing():
    """Demonstrate data preprocessing."""
    print_section("STEP 1: Data Preprocessing")
    
    data_path = DEFAULT_DATASET_PATH
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please run 'python examples/download_data.py' first")
        return None, None
    
    print("Loading News Category Dataset...")
    preprocessor = NewsDataPreprocessor()
    df = preprocessor.load_data(data_path)
    
    print(f"Total articles loaded: {len(df)}")
    print(f"\nSample categories: {df['category'].value_counts().head()}")
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    
    print("\nPreprocessing data...")
    preprocessor.preprocess()
    
    documents = preprocessor.get_documents()[:10000]  # Use 10k for demo
    print(f"Documents prepared for indexing: {len(documents)}")
    
    print("\nSample processed document:")
    print(f"  {documents[0][:200]}...")
    
    return documents, df['category'].tolist()[:10000]


def demo_bm25_retrieval(documents):
    """Demonstrate BM25 retrieval."""
    print_section("STEP 2: BM25 Retrieval")
    
    print("Building BM25 index...")
    retriever = BM25Retriever()
    retriever.build_index(documents)
    print("BM25 index built successfully!")
    
    # Test queries
    test_queries = [
        "climate change environmental issues",
        "artificial intelligence technology",
        "presidential election politics"
    ]
    
    print("\nTesting BM25 retrieval with sample queries:\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        results = retriever.retrieve(query, top_k=3)
        
        print("Top 3 results:")
        for i, (idx, score, text) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.4f}")
            print(f"     {text[:100]}...")
        print()
    
    return retriever


def demo_faiss_retrieval(documents):
    """Demonstrate FAISS retrieval."""
    print_section("STEP 3: FAISS Dense Retrieval")
    
    print("Building FAISS index with sentence embeddings...")
    print("(This may take a few minutes...)")
    retriever = FAISSRetriever()
    retriever.build_index(documents[:5000])  # Use fewer docs for speed
    print("FAISS index built successfully!")
    
    # Test semantic query
    query = "What are the environmental impacts of global warming?"
    
    print(f"\nSemantic search query: '{query}'")
    results = retriever.retrieve(query, top_k=3)
    
    print("Top 3 semantic matches:")
    for i, (idx, dist, text) in enumerate(results, 1):
        print(f"  {i}. Distance: {dist:.4f}")
        print(f"     {text[:100]}...")
    print()
    
    return retriever


def demo_hybrid_retrieval(documents):
    """Demonstrate hybrid retrieval."""
    print_section("STEP 4: Hybrid Retrieval")
    
    print("Building hybrid index (BM25 + FAISS)...")
    retriever = HybridRetriever(bm25_weight=0.5, faiss_weight=0.5)
    retriever.build_index(documents[:5000])
    print("Hybrid index built successfully!")
    
    query = "technology innovation artificial intelligence"
    
    print(f"\nHybrid search query: '{query}'")
    results = retriever.retrieve(query, top_k=3)
    
    print("Top 3 hybrid results:")
    for i, (idx, score, text) in enumerate(results, 1):
        print(f"  {i}. Combined Score: {score:.4f}")
        print(f"     {text[:100]}...")
    print()
    
    return retriever


def demo_rag_pipeline(retriever):
    """Demonstrate RAG pipeline."""
    print_section("STEP 5: RAG Pipeline")
    
    print("Creating RAG pipeline...")
    rag = SimpleRAGPipeline(retriever)
    print("Pipeline ready!")
    
    # Example QA scenarios
    questions = [
        "What are the latest developments in space exploration?",
        "Tell me about recent political events",
        "What's happening in the world of sports?"
    ]
    
    print("\nQuestion Answering Examples:\n")
    
    for question in questions:
        print(f"Q: {question}")
        result = rag.generate_answer(question, top_k=3)
        
        print(f"Retrieved {result['num_retrieved']} relevant documents:")
        for i, doc in enumerate(result['retrieved_documents'], 1):
            print(f"  {i}. (Score: {doc['score']:.4f})")
            print(f"     {doc['text'][:120]}...")
        print()


def demo_evaluation(retriever, categories):
    """Demonstrate evaluation metrics."""
    print_section("STEP 6: Evaluation Metrics")
    
    print("Evaluating retrieval quality...\n")
    
    # Create test cases based on categories
    test_cases = [
        ("climate change global warming", "ENVIRONMENT"),
        ("technology artificial intelligence", "TECH"),
        ("presidential election voting", "POLITICS"),
    ]
    
    retrieved_docs_list = []
    relevant_docs_list = []
    
    for query, target_category in test_cases:
        # Retrieve documents
        results = retriever.retrieve(query, top_k=10)
        retrieved_indices = [r[0] for r in results]
        
        # Get relevant documents (same category)
        relevant_indices = [i for i, cat in enumerate(categories) if cat == target_category][:20]
        
        retrieved_docs_list.append(retrieved_indices)
        relevant_docs_list.append(relevant_indices)
        
        # Show metrics for this query
        precision = RetrievalEvaluator.precision_at_k(retrieved_indices, relevant_indices, k=5)
        recall = RetrievalEvaluator.recall_at_k(retrieved_indices, relevant_indices, k=5)
        
        print(f"Query: '{query}' (Target: {target_category})")
        print(f"  Precision@5: {precision:.4f}")
        print(f"  Recall@5: {recall:.4f}")
        print()
    
    # Overall metrics
    print("Overall Performance Metrics:")
    metrics = RetrievalEvaluator.evaluate_retrieval(
        retrieved_docs_list, relevant_docs_list, k_values=[3, 5, 10]
    )
    
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def main():
    """Run full demonstration."""
    print("\n" + "=" * 70)
    print("RAG NEWS QA SYSTEM - COMPLETE DEMONSTRATION")
    print("=" * 70)
    
    # Step 1: Preprocessing
    documents, categories = demo_preprocessing()
    if documents is None:
        return
    
    # Step 2: BM25 Retrieval
    bm25_retriever = demo_bm25_retrieval(documents)
    
    # Step 3: FAISS Retrieval (optional, slower)
    print("\n" + "-" * 70)
    response = input("Run FAISS demo? (slower, takes a few minutes) [y/N]: ")
    if response.lower() == 'y':
        faiss_retriever = demo_faiss_retrieval(documents)
        
        # Step 4: Hybrid Retrieval
        print("\n" + "-" * 70)
        response = input("Run Hybrid retrieval demo? [y/N]: ")
        if response.lower() == 'y':
            hybrid_retriever = demo_hybrid_retrieval(documents)
            demo_rag_pipeline(hybrid_retriever)
            demo_evaluation(hybrid_retriever, categories)
        else:
            demo_rag_pipeline(faiss_retriever)
            demo_evaluation(faiss_retriever, categories)
    else:
        # Use BM25 for remaining demos
        demo_rag_pipeline(bm25_retriever)
        demo_evaluation(bm25_retriever, categories)
    
    # Summary
    print_section("DEMONSTRATION COMPLETE")
    print("This demonstration covered:")
    print("  ✓ Data preprocessing and cleaning")
    print("  ✓ Building retrieval indexes (BM25/FAISS/Hybrid)")
    print("  ✓ Document retrieval")
    print("  ✓ RAG pipeline for question answering")
    print("  ✓ Evaluation metrics (Precision, Recall, MRR, NDCG)")
    print("\nFor interactive usage, run:")
    print("  python main.py --retriever bm25")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
