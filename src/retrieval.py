"""
Retrieval module for RAG News QA System. 
Implements BM25, FAISS, and Hybrid retrieval strategies.
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available. Install with: pip install faiss-cpu")

from src.query_processor import QueryProcessor


class BM25Retriever:
    """BM25-based retrieval system."""
    
    def __init__(self, documents: List[Dict] = None):
        """Initialize BM25 retriever.
        
        Args:
            documents:  List of processed documents
        """
        self. documents = documents or []
        self.bm25 = None
        self.tokenized_docs = None
        self.query_processor = QueryProcessor()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns: 
            List of tokens
        """
        return text.lower().split()
    
    def build_index(self):
        """Build BM25 index from documents."""
        if not self.documents:
            raise ValueError("No documents provided")
        
        print(f"   Tokenizing {len(self.documents)} documents...")
        
        # Extract text and tokenize
        self.tokenized_docs = []
        for doc in self.documents:
            text = doc.get('text', '')
            tokens = self.tokenize(text)
            self.tokenized_docs.append(tokens)
        
        print(f"   Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        print(f"   ✅ BM25 index built with {len(self.tokenized_docs)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns: 
            List of (document_index, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Process query
        processed = self.query_processor.process(query)
        
        # Use expanded query for better results
        enhanced_query = processed['expanded']
        tokenized_query = self.tokenize(enhanced_query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np. argsort(scores)[::-1][:top_k * 3]  # Get more candidates
        
        # Filter by category if applicable
        category_filters = processed['category_filters']
        if category_filters: 
            filtered_results = []
            
            for idx in top_indices:
                if idx >= len(self.documents):
                    continue
                    
                doc = self.documents[idx]
                doc_category = doc.get('category', '')
                
                # Check if document category matches filter
                if doc_category in category_filters:
                    filtered_results.append((int(idx), float(scores[idx])))
                
                if len(filtered_results) >= top_k:
                    break
            
            # If not enough filtered results, add unfiltered ones
            if len(filtered_results) < top_k: 
                for idx in top_indices:
                    if idx >= len(self. documents):
                        continue
                    if (int(idx), float(scores[idx])) not in filtered_results: 
                        filtered_results.append((int(idx), float(scores[idx])))
                    if len(filtered_results) >= top_k:
                        break
            
            results = filtered_results[: top_k]
        else: 
            results = [(int(idx), float(scores[idx])) for idx in top_indices[: top_k]]
        
        return results
    
    def save_index(self, path: str):
        """Save index to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_docs':  self.tokenized_docs
            }, f)
        print(f"💾 BM25 index saved to {path}")
    
    def load_index(self, path: str):
        """Load index from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.tokenized_docs = data['tokenized_docs']
        print(f"📥 BM25 index loaded from {path}")


class FAISSRetriever:
    """FAISS-based dense retrieval system."""
    
    def __init__(self, documents:  List[Dict] = None, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize FAISS retriever.
        
        Args:
            documents: List of processed documents
            model_name: Name of sentence transformer model
        """
        if not FAISS_AVAILABLE: 
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.documents = documents or []
        self.model_name = model_name
        self.index = None
        self.embeddings = None
        self.model = None
        self.query_processor = QueryProcessor()
    
    def load_model(self):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"   Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self. model_name)
            print(f"   ✅ Model loaded")
        except ImportError:
            raise ImportError("sentence-transformers not installed.  Install with: pip install sentence-transformers")
    
    def build_index(self):
        """Build FAISS index from documents."""
        if not self.documents:
            raise ValueError("No documents provided")
        
        # Load model
        if self.model is None:
            self.load_model()
        
        print(f"   Encoding {len(self.documents)} documents...")
        
        # Extract texts
        texts = [doc.get('text', '') for doc in self.documents]
        
        # Generate embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Build FAISS index
        print(f"   Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add to index
        self.index.add(self.embeddings)
        
        print(f"   ✅ FAISS index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]: 
        """Retrieve documents using FAISS.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns: 
            List of (document_index, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Process query
        processed = self.query_processor.process(query)
        enhanced_query = processed['expanded']
        
        # Encode query
        query_embedding = self.model.encode([enhanced_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k * 3)
        
        # Filter by category
        category_filters = processed['category_filters']
        if category_filters:
            filtered_results = []
            
            for idx, score in zip(indices[0], scores[0]):
                if idx >= len(self.documents):
                    continue
                    
                doc = self.documents[idx]
                doc_category = doc. get('category', '')
                
                if doc_category in category_filters: 
                    filtered_results.append((int(idx), float(score)))
                
                if len(filtered_results) >= top_k:
                    break
            
            # Fill with unfiltered if needed
            if len(filtered_results) < top_k:
                for idx, score in zip(indices[0], scores[0]):
                    if idx >= len(self.documents):
                        continue
                    if (int(idx), float(score)) not in filtered_results: 
                        filtered_results.append((int(idx), float(score)))
                    if len(filtered_results) >= top_k:
                        break
            
            results = filtered_results[:top_k]
        else:
            results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])][:top_k]
        
        return results
    
    def save_index(self, path: str):
        """Save FAISS index."""
        faiss.write_index(self. index, f"{path}.index")
        np.save(f"{path}.npy", self.embeddings)
        print(f"💾 FAISS index saved to {path}")
    
    def load_index(self, path: str):
        """Load FAISS index."""
        self.index = faiss.read_index(f"{path}.index")
        self.embeddings = np.load(f"{path}.npy")
        if self.model is None:
            self.load_model()
        print(f"📥 FAISS index loaded from {path}")


class HybridRetriever:
    """Hybrid retrieval combining BM25 and FAISS."""
    
    def __init__(self, documents: List[Dict] = None, alpha: float = 0.5):
        """Initialize hybrid retriever. 
        
        Args:
            documents: List of processed documents
            alpha: Weight for BM25 (1-alpha for FAISS)
        """
        self.documents = documents or []
        self.alpha = alpha
        self.bm25_retriever = BM25Retriever(documents)
        
        if FAISS_AVAILABLE: 
            self.faiss_retriever = FAISSRetriever(documents)
        else:
            print("⚠️ FAISS not available, using BM25 only")
            self.faiss_retriever = None
        
        self.query_processor = QueryProcessor()
    
    def build_index(self):
        """Build both BM25 and FAISS indices."""
        print("   Building BM25 component...")
        self.bm25_retriever. build_index()
        
        if self.faiss_retriever:
            print("   Building FAISS component...")
            self.faiss_retriever.build_index()
        
        print(f"   ✅ Hybrid index built")
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve using hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns: 
            List of (document_index, score) tuples
        """
        # Get BM25 results
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        
        if not self.faiss_retriever:
            return bm25_results[: top_k]
        
        # Get FAISS results
        faiss_results = self.faiss_retriever.retrieve(query, top_k=top_k * 2)
        
        # Combine scores
        combined_scores = {}
        
        # Add BM25 scores
        bm25_scores = np.array([score for _, score in bm25_results])
        bm25_scores_norm = self.normalize_scores(bm25_scores)
        
        for (idx, _), norm_score in zip(bm25_results, bm25_scores_norm):
            combined_scores[idx] = self.alpha * norm_score
        
        # Add FAISS scores
        faiss_scores = np.array([score for _, score in faiss_results])
        faiss_scores_norm = self.normalize_scores(faiss_scores)
        
        for (idx, _), norm_score in zip(faiss_results, faiss_scores_norm):
            if idx in combined_scores:
                combined_scores[idx] += (1 - self.alpha) * norm_score
            else: 
                combined_scores[idx] = (1 - self.alpha) * norm_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = [(int(idx), float(score)) for idx, score in sorted_results[:top_k]]
        
        return results
    
    def save_index(self, path:  str):
        """Save both indices."""
        self.bm25_retriever.save_index(f"{path}_bm25.pkl")
        if self.faiss_retriever:
            self. faiss_retriever.save_index(f"{path}_faiss")
    
    def load_index(self, path: str):
        """Load both indices."""
        self. bm25_retriever.load_index(f"{path}_bm25.pkl")
        if self.faiss_retriever:
            self.faiss_retriever.load_index(f"{path}_faiss")


# Utility functions
def create_retriever(retriever_type: str, documents: List[Dict], **kwargs):
    """Factory function to create retriever. 
    
    Args:
        retriever_type: Type of retriever ('bm25', 'faiss', 'hybrid')
        documents: List of documents
        **kwargs: Additional arguments
        
    Returns:
        Retriever object
    """
    if retriever_type == 'bm25':
        return BM25Retriever(documents)
    elif retriever_type == 'faiss':
        return FAISSRetriever(documents, **kwargs)
    elif retriever_type == 'hybrid': 
        return HybridRetriever(documents, **kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")