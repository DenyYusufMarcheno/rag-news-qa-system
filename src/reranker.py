"""Reranking utilities using cross-encoder models."""

from typing import List, Dict, Any, Tuple
import numpy as np


class Reranker:
    """Rerank retrieved documents using cross-encoder model."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker with cross-encoder model.
        
        Args:
            model_name: Name of the cross-encoder model from sentence-transformers
        """
        self.model_name = model_name
        self.model = None
    
    def _load_model(self):
        """Lazy load the model when first needed."""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
            except Exception as e:
                print(f"Warning: Could not load cross-encoder model: {e}")
                print("Reranking will be disabled.")
                self.model = False  # Mark as failed to load
    
    def _prepare_document_text(self, doc: Dict[str, Any]) -> str:
        """Prepare document text for reranking.
        
        Args:
            doc: Document dictionary
            
        Returns:
            Formatted document text
        """
        headline = doc.get('headline', '')
        description = doc.get('short_description', '')
        return f"{headline}. {description}"
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: int = 5, score_threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """Rerank documents using cross-encoder.
        
        Args:
            query: Query string
            documents: List of retrieved documents
            top_k: Number of top documents to return
            score_threshold: Minimum score threshold for documents
            
        Returns:
            List of tuples (document, score) sorted by relevance
        """
        if not documents:
            return []
        
        # Load model if not already loaded
        self._load_model()
        
        # If model failed to load, return documents with original scores
        if self.model is False:
            return [(doc, doc.get('score', 0.0)) for doc in documents[:top_k]]
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        for doc in documents:
            doc_text = self._prepare_document_text(doc)
            pairs.append([query, doc_text])
        
        # Get scores from cross-encoder
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"Warning: Cross-encoder prediction failed: {e}")
            # Fallback to original scores
            return [(doc, doc.get('score', 0.0)) for doc in documents[:top_k]]
        
        # Combine documents with scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and return top-k
        filtered = [(doc, float(score)) for doc, score in doc_scores 
                   if score >= score_threshold]
        
        return filtered[:top_k]
    
    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair.
        
        Args:
            query: Query string
            document: Document text
            
        Returns:
            Relevance score
        """
        # Load model if not already loaded
        self._load_model()
        
        if self.model is False:
            return 0.0
        
        try:
            score = self.model.predict([[query, document]])[0]
            return float(score)
        except Exception as e:
            print(f"Warning: Scoring failed: {e}")
            return 0.0


def reciprocal_rank_fusion(rankings: List[List[Any]], k: int = 60) -> List[Tuple[Any, float]]:
    """Combine multiple rankings using Reciprocal Rank Fusion.
    
    Args:
        rankings: List of ranked lists (each list is in order of relevance)
        k: Constant for RRF formula (default 60)
        
    Returns:
        Combined ranking as list of (item, score) tuples
    """
    scores = {}
    
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            # Create hashable key from item
            item_id = id(item)
            if item_id not in scores:
                scores[item_id] = {'item': item, 'score': 0.0}
            scores[item_id]['score'] += 1.0 / (k + rank)
    
    # Sort by score
    result = [(v['item'], v['score']) for v in scores.values()]
    result.sort(key=lambda x: x[1], reverse=True)
    
    return result


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range using min-max normalization.
    
    Args:
        scores: List of scores
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    scores = np.array(scores)
    min_score = scores.min()
    max_score = scores.max()
    
    # Avoid division by zero
    if max_score - min_score == 0:
        return [1.0] * len(scores)
    
    normalized = (scores - min_score) / (max_score - min_score)
    return normalized.tolist()


def combine_scores(scores1: List[float], scores2: List[float], 
                   alpha: float = 0.5) -> List[float]:
    """Combine two score lists with weighted average.
    
    Args:
        scores1: First list of scores
        scores2: Second list of scores
        alpha: Weight for first scores (1-alpha for second)
        
    Returns:
        Combined scores
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score lists must have same length")
    
    return [alpha * s1 + (1 - alpha) * s2 for s1, s2 in zip(scores1, scores2)]
