"""Demo script for evaluation metrics."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import NewsDataPreprocessor
from src.retrieval import BM25Retriever, FAISSRetriever
from src.evaluation import RetrievalEvaluator, GenerationEvaluator
import random


def main():
    """Run evaluation demo."""
    
    print("=" * 60)
    print("Evaluation Metrics Demo")
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
    df = preprocessor.load_data(data_path)
    preprocessor.preprocess()
    
    # Limit to first 5000 documents for demo
    documents = preprocessor.get_documents()[:5000]
    categories = df['category'].tolist()[:5000]
    print(f"Loaded {len(documents)} documents")
    print()
    
    # Build retrievers
    print("Building BM25 index...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(documents)
    print("BM25 index built!")
    print()
    
    # Create test queries based on categories
    test_queries = [
        ("climate change environmental issues", "ENVIRONMENT"),
        ("latest technology innovations", "TECH"),
        ("political news updates", "POLITICS"),
        ("sports games results", "SPORTS"),
        ("entertainment movies shows", "ENTERTAINMENT")
    ]
    
    print("=" * 60)
    print("Retrieval Evaluation")
    print("=" * 60)
    print()
    
    retrieved_docs_list = []
    relevant_docs_list = []
    
    for query, target_category in test_queries:
        print(f"Query: {query} (Target category: {target_category})")
        
        # Retrieve documents
        results = bm25_retriever.retrieve(query, top_k=10)
        retrieved_indices = [r[0] for r in results]
        
        # Get "relevant" documents (same category)
        relevant_indices = [i for i, cat in enumerate(categories) if cat == target_category][:10]
        
        retrieved_docs_list.append(retrieved_indices)
        relevant_docs_list.append(relevant_indices)
        
        # Calculate metrics for this query
        precision = RetrievalEvaluator.precision_at_k(retrieved_indices, relevant_indices, k=5)
        recall = RetrievalEvaluator.recall_at_k(retrieved_indices, relevant_indices, k=5)
        ndcg = RetrievalEvaluator.ndcg_at_k(retrieved_indices, relevant_indices, k=5)
        
        print(f"  Precision@5: {precision:.4f}")
        print(f"  Recall@5: {recall:.4f}")
        print(f"  NDCG@5: {ndcg:.4f}")
        print()
    
    # Overall metrics
    print("-" * 60)
    overall_metrics = RetrievalEvaluator.evaluate_retrieval(
        retrieved_docs_list, relevant_docs_list, k_values=[3, 5, 10]
    )
    
    print("Overall Retrieval Metrics:")
    for metric, value in overall_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Generation evaluation example
    print("=" * 60)
    print("Generation Evaluation")
    print("=" * 60)
    print()
    
    # Example generated and reference answers
    generated_answers = [
        "Climate change is affecting global temperatures and weather patterns.",
        "New technology innovations include AI and machine learning advancements.",
        "Political developments include new policies and legislation."
    ]
    
    reference_answers = [
        "Climate change is causing rising temperatures and extreme weather.",
        "Technology innovations focus on artificial intelligence and automation.",
        "Recent political news covers new policies and government actions."
    ]
    
    gen_metrics = GenerationEvaluator.evaluate_generation(generated_answers, reference_answers)
    
    print("Generation Metrics:")
    for metric, value in gen_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
