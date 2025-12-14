"""
RAG Pipeline module - Enhanced with LLM integration.
"""

from typing import List, Dict, Optional
from src.llm_integration import create_llm


class EnhancedRAGPipeline:
    """RAG Pipeline with LLM generation."""
    
    def __init__(
        self, 
        retriever, 
        llm_provider: str = "groq",
        llm_model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_k: int = 5
    ):
        """Initialize RAG pipeline.
        
        Args:
            retriever: Document retriever (BM25, FAISS, or Hybrid)
            llm_provider:  LLM provider name
            llm_model: Model name for the LLM
            temperature: LLM temperature
            max_tokens: Max tokens for generation
            top_k: Number of documents to retrieve
        """
        self.retriever = retriever
        self. top_k = top_k
        
        # Initialize LLM
        try:
            self.llm = create_llm(
                provider=llm_provider,
                model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.llm_enabled = True
        except Exception as e:
            print(f"⚠️  LLM initialization failed: {e}")
            print("   Running in retrieval-only mode")
            self.llm = None
            self.llm_enabled = False
    
    def query(self, query: str, top_k: Optional[int] = None) -> Dict:
        """Process query through full RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve (overrides default)
            
        Returns: 
            Dictionary with answer and metadata
        """
        k = top_k or self.top_k
        
        # Step 1: Retrieve documents
        print(f"🔎 Retrieving top-{k} documents...")
        results = self.retriever.retrieve(query, top_k=k)
        
        # Get full documents
        retrieved_docs = []
        for idx, score in results:
            doc = self.retriever.documents[idx]
            doc['retrieval_score'] = score
            retrieved_docs.append(doc)
        
        print(f"✅ Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Generate answer with LLM (if enabled)
        if self.llm_enabled and self.llm:
            print(f"🤖 Generating answer with LLM...")
            response = self.llm.answer_query(query, retrieved_docs)
            print(f"✅ Answer generated")
            return response
        else:
            # Retrieval-only mode
            return {
                'query': query,
                'answer': None,
                'retrieved_docs': retrieved_docs,
                'num_sources': len(retrieved_docs),
                'mode': 'retrieval_only'
            }
    
    def batch_query(self, queries: List[str], top_k: Optional[int] = None) -> List[Dict]:
        """Process multiple queries. 
        
        Args:
            queries: List of queries
            top_k: Number of documents per query
            
        Returns:
            List of responses
        """
        responses = []
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}/{len(queries)} ---")
            response = self.query(query, top_k)
            responses.append(response)
        
        return responses


class SimpleRAGPipeline:
    """Simple RAG Pipeline (retrieval only, no LLM)."""
    
    def __init__(self, retriever, top_k: int = 5):
        """Initialize simple pipeline.
        
        Args:
            retriever: Document retriever
            top_k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.top_k = top_k
    
    def query(self, query: str, top_k: Optional[int] = None) -> Dict:
        """Retrieve documents for query.
        
        Args:
            query: User query
            top_k: Number of documents
            
        Returns:
            Dictionary with retrieved documents
        """
        k = top_k or self.top_k
        results = self.retriever.retrieve(query, top_k=k)
        
        retrieved_docs = []
        for idx, score in results:
            doc = self.retriever.documents[idx]
            doc['retrieval_score'] = score
            retrieved_docs.append(doc)
        
        return {
            'query': query,
            'retrieved_docs': retrieved_docs,
            'num_sources': len(retrieved_docs)
        }