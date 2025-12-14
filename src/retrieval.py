"""Document retrieval module using BM25 and FAISS."""

from typing import List, Tuple, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer


class BM25Retriever:
    """BM25-based document retriever."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        self.bm25 = None
        self.documents = None
        self.tokenized_docs = None
        
    def build_index(self, documents: List[str]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        # Simple tokenization by splitting on whitespace
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Retrieve top-k most relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples (document_index, score, document_text)
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(scores[idx]), self.documents[idx]))
        
        return results


class FAISSRetriever:
    """FAISS-based dense retriever using sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize FAISS retriever.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = None
        self.embeddings = None
        
    def build_index(self, documents: List[str]):
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        self.embeddings = self.model.encode(
            documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Validate embeddings
        if len(self.embeddings.shape) != 2 or self.embeddings.shape[1] == 0:
            raise ValueError(f"Invalid embedding shape: {self.embeddings.shape}. Expected 2D array with non-zero dimension.")
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Retrieve top-k most relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples (document_index, distance, document_text)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            results.append((int(idx), float(dist), self.documents[idx]))
        
        return results
    
    def save_index(self, filepath: str):
        """
        Save FAISS index to file.
        
        Args:
            filepath: Output file path
        """
        if self.index is None:
            raise ValueError("No index to save.")
        
        faiss.write_index(self.index, filepath)
        
    def load_index(self, filepath: str):
        """
        Load FAISS index from file.
        
        Args:
            filepath: Input file path
        """
        self.index = faiss.read_index(filepath)


class HybridRetriever:
    """Hybrid retriever combining BM25 and FAISS."""
    
    def __init__(self, bm25_weight: float = 0.5, faiss_weight: float = 0.5):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_weight: Weight for BM25 scores
            faiss_weight: Weight for FAISS scores
        """
        self.bm25_retriever = BM25Retriever()
        self.faiss_retriever = FAISSRetriever()
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        
    def build_index(self, documents: List[str]):
        """
        Build both BM25 and FAISS indexes.
        
        Args:
            documents: List of document strings
        """
        self.bm25_retriever.build_index(documents)
        self.faiss_retriever.build_index(documents)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples (document_index, combined_score, document_text)
        """
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k*2)
        faiss_results = self.faiss_retriever.retrieve(query, top_k=top_k*2)
        
        # Normalize scores
        def normalize_scores(results):
            if not results:
                return {}
            scores = [r[1] for r in results]
            min_score, max_score = min(scores), max(scores)
            if max_score - min_score == 0:
                return {r[0]: 0.5 for r in results}
            return {r[0]: (r[1] - min_score) / (max_score - min_score) for r in results}
        
        def invert_distances_to_scores(results):
            """Convert FAISS distances (lower is better) to scores (higher is better)."""
            return [(r[0], -r[1], r[2]) for r in results]
        
        bm25_scores = normalize_scores(bm25_results)
        # For FAISS, lower distance is better, so invert before normalizing
        faiss_scores = normalize_scores(invert_distances_to_scores(faiss_results))
        
        # Combine scores
        all_doc_indices = set(bm25_scores.keys()) | set(faiss_scores.keys())
        combined_scores = {}
        
        for idx in all_doc_indices:
            bm25_score = bm25_scores.get(idx, 0)
            faiss_score = faiss_scores.get(idx, 0)
            combined_scores[idx] = (
                self.bm25_weight * bm25_score + 
                self.faiss_weight * faiss_score
            )
        
        # Get top-k
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        documents = self.bm25_retriever.documents
        for idx, score in sorted_indices:
            results.append((idx, score, documents[idx]))
        
        return results
