"""Main application for RAG News QA System."""

import argparse
import os
import sys
import yaml
import json
from typing import List, Dict, Any, Optional

from src.preprocessing import DocumentPreprocessor
from src.query_processor import QueryProcessor
from src.retrieval import BM25Retriever, FAISSRetriever, HybridRetriever
from src.reranker import Reranker


class RAGNewsQA:
    """RAG News QA System with enhanced retrieval."""
    
    def __init__(self, config_path: str = "configs/retrieval_config.yaml"):
        """Initialize the RAG system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.doc_preprocessor = DocumentPreprocessor()
        self.query_processor = QueryProcessor(config_path)
        
        # Initialize retrievers
        bm25_config = self.config.get('bm25', {})
        self.bm25_retriever = BM25Retriever(
            k1=bm25_config.get('k1', 1.5),
            b=bm25_config.get('b', 0.75)
        )
        
        faiss_config = self.config.get('faiss', {})
        self.faiss_retriever = FAISSRetriever(
            model_name=faiss_config.get('embedding_model', 
                                       'sentence-transformers/all-MiniLM-L6-v2'),
            normalize_embeddings=faiss_config.get('normalize_embeddings', True)
        )
        
        hybrid_config = self.config.get('hybrid', {})
        self.hybrid_retriever = HybridRetriever(
            self.bm25_retriever,
            self.faiss_retriever,
            alpha=hybrid_config.get('alpha', 0.5),
            normalize_scores=hybrid_config.get('normalize_scores', True)
        )
        
        # Initialize reranker if enabled
        rerank_config = self.config.get('reranking', {})
        self.use_reranking = rerank_config.get('enabled', True)
        if self.use_reranking:
            self.reranker = Reranker(
                model_name=rerank_config.get('model', 
                                            'cross-encoder/ms-marco-MiniLM-L-6-v2')
            )
        else:
            self.reranker = None
        
        self.documents = []
        self.indexed = False
    
    def load_documents(self, data_path: str):
        """Load documents from JSON file.
        
        Args:
            data_path: Path to JSON file with documents
        """
        with open(data_path, 'r') as f:
            raw_docs = json.load(f)
        
        # Preprocess documents
        print(f"Preprocessing {len(raw_docs)} documents...")
        self.documents = self.doc_preprocessor.preprocess_documents(raw_docs)
        print(f"Preprocessed {len(self.documents)} documents")
    
    def index_documents(self):
        """Index documents for retrieval."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        print("Indexing documents...")
        
        # Index for BM25
        print("Creating BM25 index...")
        self.bm25_retriever.index_documents(self.documents)
        
        # Index for FAISS
        print("Creating FAISS index...")
        self.faiss_retriever.index_documents(self.documents)
        
        self.indexed = True
        print("Indexing complete!")
    
    def retrieve(self, query: str, top_k: Optional[int] = None,
                strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (uses config default if None)
            strategy: Retrieval strategy ("bm25", "faiss", "hybrid")
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.indexed:
            raise ValueError("Documents not indexed. Call index_documents() first.")
        
        # Get configuration
        if top_k is None:
            top_k = self.config.get('top_k', 10)
        
        if strategy is None:
            strategy = self.config.get('retrieval_strategy', 'hybrid')
        
        # Process query
        query_config = self.config.get('query_preprocessing', {})
        processed_query = self.query_processor.process_query(
            query,
            expand=query_config.get('query_expansion', True),
            extract_kw=query_config.get('keyword_extraction', True)
        )
        
        print(f"\nQuery: {query}")
        print(f"Intent: {processed_query['intent']} "
              f"(confidence: {processed_query['confidence']:.2f})")
        if processed_query['categories']:
            print(f"Relevant categories: {', '.join(processed_query['categories'])}")
        
        # Get category filter if enabled
        category_config = self.config.get('category_filtering', {})
        category_filter = None
        if category_config.get('enabled', True):
            category_filter = processed_query['categories']
        
        # Determine number of documents to retrieve before reranking
        rerank_config = self.config.get('reranking', {})
        retrieve_k = rerank_config.get('top_k', 20) if self.use_reranking else top_k
        
        # Retrieve based on strategy
        print(f"\nRetrieving with strategy: {strategy}")
        
        if strategy == 'bm25':
            results = self.bm25_retriever.search(
                processed_query['keywords'],
                retrieve_k,
                category_filter
            )
        elif strategy == 'faiss':
            # Use expanded query for FAISS
            query_text = processed_query['expanded']
            results = self.faiss_retriever.search(
                query_text,
                retrieve_k,
                category_filter
            )
        elif strategy == 'hybrid':
            query_text = processed_query['expanded']
            results = self.hybrid_retriever.search(
                query_text,
                processed_query['keywords'],
                retrieve_k,
                category_filter
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Apply reranking if enabled
        if self.use_reranking and self.reranker and len(results) > 0:
            print(f"Reranking top {len(results)} documents...")
            
            # Prepare documents for reranking
            docs_for_rerank = [doc for doc, _ in results]
            
            final_k = rerank_config.get('final_k', top_k)
            score_threshold = rerank_config.get('score_threshold', 0.0)
            
            reranked = self.reranker.rerank(
                query,
                docs_for_rerank,
                final_k,
                score_threshold
            )
            
            # Convert back to list of dicts with scores
            results = reranked
        else:
            # Just take top_k
            results = results[:top_k]
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'headline': doc.get('headline', ''),
                'short_description': doc.get('short_description', ''),
                'category': doc.get('category', ''),
                'score': float(score),
                'link': doc.get('link', ''),
                'authors': doc.get('authors', ''),
                'date': doc.get('date', '')
            })
        
        return formatted_results
    
    def save_index(self, path_prefix: str):
        """Save indices to disk.
        
        Args:
            path_prefix: Path prefix for saving indices
        """
        os.makedirs(os.path.dirname(path_prefix) or '.', exist_ok=True)
        
        print(f"Saving BM25 index to {path_prefix}_bm25.pkl...")
        self.bm25_retriever.save(f"{path_prefix}_bm25.pkl")
        
        print(f"Saving FAISS index to {path_prefix}_faiss...")
        self.faiss_retriever.save(f"{path_prefix}_faiss")
        
        print("Indices saved!")
    
    def load_index(self, path_prefix: str):
        """Load indices from disk.
        
        Args:
            path_prefix: Path prefix for loading indices
        """
        print(f"Loading BM25 index from {path_prefix}_bm25.pkl...")
        self.bm25_retriever.load(f"{path_prefix}_bm25.pkl")
        
        print(f"Loading FAISS index from {path_prefix}_faiss...")
        self.faiss_retriever.load(f"{path_prefix}_faiss")
        
        self.documents = self.bm25_retriever.documents
        self.indexed = True
        print("Indices loaded!")


def format_results(results: List[Dict[str, Any]]) -> str:
    """Format results for display.
    
    Args:
        results: List of retrieved documents
        
    Returns:
        Formatted string
    """
    output = []
    output.append("\n" + "="*80)
    output.append(f"Retrieved {len(results)} documents:")
    output.append("="*80)
    
    for i, doc in enumerate(results, 1):
        output.append(f"\n{i}. [{doc['category']}] (Score: {doc['score']:.4f})")
        output.append(f"   Headline: {doc['headline']}")
        if doc['short_description']:
            output.append(f"   Description: {doc['short_description'][:150]}...")
        output.append("")
    
    return "\n".join(output)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG News QA System - Enhanced Retrieval"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/retrieval_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to JSON file with news articles'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Query to search for'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['bm25', 'faiss', 'hybrid'],
        help='Retrieval strategy (overrides config)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        help='Number of documents to retrieve (overrides config)'
    )
    
    parser.add_argument(
        '--save-index',
        type=str,
        help='Save indices to this path prefix'
    )
    
    parser.add_argument(
        '--load-index',
        type=str,
        help='Load indices from this path prefix'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing RAG News QA System...")
    system = RAGNewsQA(args.config)
    
    # Load or create index
    if args.load_index:
        system.load_index(args.load_index)
    else:
        system.load_documents(args.data)
        system.index_documents()
        
        if args.save_index:
            system.save_index(args.save_index)
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*80)
        print("Interactive Mode - Enter 'quit' to exit")
        print("="*80)
        
        while True:
            try:
                query = input("\nEnter query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                results = system.retrieve(query, args.top_k, args.strategy)
                print(format_results(results))
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # Single query mode
    elif args.query:
        results = system.retrieve(args.query, args.top_k, args.strategy)
        print(format_results(results))
    
    else:
        print("\nNo query provided. Use --query or --interactive mode.")
        print("Example: python main.py --data data/news.json --query 'economy performance'")


if __name__ == '__main__':
    main()
